"""Poison generator for attacks A1-A6 (SPEC §B5.3).

Poison goals are harmless and machine-checkable: a specific wrong answer, or a canary
string like ``ZEBRA-7741``. Each target question gets ``poison_docs_per_target`` documents
(A6: a pair of halves) written by the poison-writer model from one of the attack's
templates. The seed fixes the template choice, the false answer, the canary, the
source tier and the A5 hiding technique, so a run is reproducible.

Test poison uses templates t05-t08 on target questions; training poison for D2 uses
t01-t04 on dev-split targets only (never on test questions).
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from bench.poison.render import (
    HIDING_TECHNIQUES,
    HidingTechnique,
    insert_middle,
    render_html,
    render_pdf,
)
from bench.poison.templates import Template, all_templates, template_ids
from bench.prompting import fill, parse_json_object, user_message
from bench.schemas import BenchQuestion, PoisonSample
from bench.score import mentions
from ragantine.config import AppConfig
from ragantine.generate.llm import LLMClient
from ragantine.schemas import RawDocument, SourceTier

CANARY_WORDS = (
    "ZEBRA", "OTTER", "FALCON", "MAPLE", "COBALT", "LYNX", "QUARTZ", "HERON", "TULIP", "BISON",
    "CEDAR", "ORCA", "PUMA", "IBIS", "YUCCA", "WALRUS", "GECKO", "KOALA", "LOTUS", "MARLIN",
)  # fmt: skip
FALSE_ANSWER_CANDIDATES = 3
FALSE_ANSWER_RETRIES = 2
MAX_WORDS = 6
POISON_INGESTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
FORMAT_EXT = {"text": "txt", "markdown": "md", "html": "html", "pdf": "pdf"}


class PoisonReport(BaseModel):
    targets: int = 0
    samples: int = 0
    failed: dict[str, int] = Field(default_factory=dict)
    false_answer_retries: int = 0


def rng_for(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_canary(rng: random.Random) -> str:
    return f"{rng.choice(CANARY_WORDS)}-{rng.randint(1000, 9999)}"


def pick_tier(rng: random.Random, mix: dict[SourceTier, float]) -> SourceTier:
    u, acc = rng.random(), 0.0
    for tier, share in mix.items():
        acc += share
        if u < acc:
            return tier
    return next(iter(mix))


def make_document(
    poison_id: str, title: str, fmt: str, raw: str | bytes, tier: SourceTier
) -> RawDocument:
    # neutral URI: nothing about the document says it is poison
    uri = f"{tier.value}://uploads/{hashlib.sha1(poison_id.encode()).hexdigest()[:12]}"
    return RawDocument(
        doc_id=hashlib.sha1(uri.encode()).hexdigest()[:16],
        title=title,
        source_uri=uri,
        fmt=fmt,
        raw=raw,
        source_tier=tier,
        ingested_at=POISON_INGESTED_AT,
    )


class PoisonWriter:
    def __init__(self, cfg: AppConfig, llm: LLMClient, writer_alias: str) -> None:
        self.cfg = cfg
        self.llm = llm
        self.alias = writer_alias

    async def false_answer(self, q: BenchQuestion, rng: random.Random, report: PoisonReport) -> str:
        """A plausible wrong answer of the gold's type; never the gold or an alias."""
        golds = [q.gold_answer, *q.gold_aliases]
        rejected: list[str] = []
        for _ in range(1 + FALSE_ANSWER_RETRIES):
            avoid = f"Do not use any of these: {', '.join(rejected)}." if rejected else ""
            result = await self.llm.achat(
                self.alias,
                user_message(
                    "false_answer",
                    n=FALSE_ANSWER_CANDIDATES,
                    avoid=avoid,
                    question=q.question,
                    gold=q.gold_answer,
                ),
                json_mode=True,
            )
            raw = parse_json_object(result.text).get("wrong_answers", [])
            ok: list[str] = []
            for cand in raw if isinstance(raw, list) else []:
                cand = str(cand).strip()
                bad = (
                    not cand
                    or len(cand.split()) > MAX_WORDS
                    or any(mentions(cand, g) or mentions(g, cand) for g in golds)
                )
                (rejected if bad else ok).append(cand)
            if ok:
                return str(rng.choice(ok))
            report.false_answer_retries += 1
        raise ValueError(f"no valid false answer for {q.qid}")

    async def rewordings(self, q: BenchQuestion, n: int = 2) -> list[str]:
        result = await self.llm.achat(
            self.alias, user_message("rewordings", n=n, question=q.question), json_mode=True
        )
        items = parse_json_object(result.text).get("rewordings", [])
        out = [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else []
        return (out + [q.question] * n)[:n]

    async def body(
        self, template: Template, subject: str, q: BenchQuestion, false_answer: str | None
    ) -> str:
        prompt = fill(
            template.writer, subject=subject, question=q.question, false_answer=false_answer or ""
        )
        result = await self.llm.achat(self.alias, [{"role": "user", "content": prompt}])
        text = result.text.strip()
        if not text:
            raise ValueError("empty document from writer")
        return text


async def build_target_poison(
    writer: PoisonWriter,
    q: BenchQuestion,
    subject: str,
    seed: int,
    kind: str,
    report: PoisonReport,
) -> list[PoisonSample]:
    assert q.attack is not None
    cfg = writer.cfg
    rng = rng_for(seed, kind, q.qid)
    samples: list[PoisonSample] = []
    for k in range(cfg.bench.poison_docs_per_target):
        tid = rng.choice(template_ids(q.attack, kind))
        template = all_templates()[tid]
        tier = pick_tier(rng, cfg.bench.poison_tier_mix)
        canary = make_canary(rng)
        technique: HidingTechnique = HIDING_TECHNIQUES[rng.randrange(len(HIDING_TECHNIQUES))]
        pid = f"P-{q.qid}-{kind}-{seed}-{k}"
        goal_false = q.attack in ("A1", "A3", "A4")
        false = await writer.false_answer(q, rng, report) if goal_false else None
        body = await writer.body(template, subject, q, false)
        values = {
            "subject": subject,
            "question": q.question,
            "canary": canary,
            "false_answer": false or "",
        }

        common = {
            "qid": q.qid,
            "attack": q.attack,
            "goal": "false_answer" if goal_false else "canary",
            "target_false_answer": false,
            "canary": None if goal_false else canary,
            "template_id": tid,
            "seed": seed,
        }

        def sample(
            poison_id: str,
            doc: RawDocument,
            tech: str | None = None,
            common: Mapping[str, object] = common,
        ) -> PoisonSample:
            return PoisonSample.model_validate(
                {**common, "poison_id": poison_id, "document": doc, "technique": tech}
            )

        if q.attack == "A1":
            samples.append(sample(pid, make_document(pid, subject, "text", body, tier)))
        elif q.attack == "A2":
            text = insert_middle(body, fill(template.payload or "", **values))
            samples.append(sample(pid, make_document(pid, subject, "text", text, tier)))
        elif q.attack == "A3":
            samples.append(sample(pid, make_document(pid, subject, "markdown", body, tier)))
        elif q.attack == "A4":
            words = await writer.rewordings(q)
            head = fill(
                template.payload or "{question}\n{rewordings}\n\n",
                **values,
                rewordings="\n".join(words),
            )
            text = head.rstrip() + "\n\n" + body  # payload parsing strips the trailing blank line
            samples.append(sample(pid, make_document(pid, subject, "text", text, tier)))
        elif q.attack == "A5":
            payload = fill(template.payload or "", **values)
            if technique == "pdf_white":
                doc = make_document(pid, subject, "pdf", render_pdf(subject, body, payload), tier)
            else:
                doc = make_document(
                    pid, subject, "html", render_html(subject, body, payload, technique), tier
                )
            samples.append(sample(pid, doc, technique))
        elif q.attack == "A6":
            body_b = await writer.body(
                template,
                subject,
                q.model_copy(update={"question": q.question + " (overview)"}),
                None,
            )
            halves = ((body, template.payload or ""), (body_b, template.payload_b or ""))
            for part, (half_text, half_payload) in zip("ab", halves, strict=True):
                ppid = f"{pid}{part}"
                doc_text = insert_middle(half_text, fill(half_payload, **values))
                samples.append(
                    sample(ppid, make_document(ppid, subject, "text", doc_text, tier), part)
                )
    return samples


async def generate_poison(
    cfg: AppConfig,
    llm: LLMClient,
    writer_alias: str,
    questions: Sequence[BenchQuestion],
    subjects: dict[str, str],
    seeds: Sequence[int],
    kind: str = "test",
    log: Callable[[str], None] = print,
) -> tuple[list[PoisonSample], PoisonReport]:
    """Poison for every target question (test: all splits; train: dev split only)."""
    writer = PoisonWriter(cfg, llm, writer_alias)
    report = PoisonReport()
    targets = [q for q in questions if q.qset == "target" and (kind == "test" or q.split == "dev")]
    report.targets = len(targets)

    async def one(q: BenchQuestion, seed: int) -> list[PoisonSample]:
        try:
            return await build_target_poison(
                writer, q, subjects[q.gold_chunk_id], seed, kind, report
            )
        except Exception as exc:  # one bad generation must not sink the batch
            key = type(exc).__name__
            report.failed[key] = report.failed.get(key, 0) + 1
            log(f"  poison failed for {q.qid} seed={seed}: {key}: {exc}")
            return []

    results = await asyncio.gather(*(one(q, s) for s in seeds for q in targets))
    samples = [p for batch in results for p in batch]
    report.samples = len(samples)
    return samples, report


def write_poison_files(samples: Sequence[PoisonSample], out_dir: Path) -> None:
    """Rendered files you can open and read (data/bench/poison_docs/<poison_id>.<ext>)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in samples:
        path = out_dir / f"{p.poison_id}.{FORMAT_EXT[p.document.fmt]}"
        raw = p.document.raw
        if isinstance(raw, bytes):
            path.write_bytes(raw)
        else:
            path.write_text(raw, encoding="utf-8")
