"""Question generation and filtering (SPEC §B5.2).

1. One candidate chunk per article (deterministic, seeded). The poison-writer model writes
   one short-answer question per chunk (JSON).
2. Keep a question only if (a) its gold chunk is in the clean-index top ``fused_k`` and
   (b) the default generator answers it correctly on the clean index (rule or judge), so
   every point of ASR is caused by poison, not by a weak baseline. Rejections are logged.
3. Assign kept questions: target (attacks round-robin), then clean, then paraphrases of
   target questions. 4. Split each set into dev/test, stratified by attack.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from bench.prompting import parse_json_object, user_message
from bench.schemas import AttackType, BenchQuestion
from bench.score import ajudge
from ragantine.config import AppConfig
from ragantine.graph import Pipeline, arun_query
from ragantine.schemas import Chunk

ATTACKS: tuple[AttackType, ...] = ("A1", "A2", "A3", "A4", "A5", "A6")
MAX_ANSWER_WORDS = 6
CANDIDATE_BATCH = 24


class QuestionsReport(BaseModel):
    candidates: int = 0
    kept: int = 0
    rejected: dict[str, int] = Field(default_factory=dict)
    targets: int = 0
    clean: int = 0
    paraphrases: int = 0
    by_attack: dict[str, int] = Field(default_factory=dict)
    by_split: dict[str, int] = Field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        return 0.0 if not self.candidates else 1 - self.kept / self.candidates


class SetSizes(BaseModel):
    targets: int
    clean: int
    paraphrases: int


def set_sizes(cfg: AppConfig, limit: int | None) -> SetSizes:
    """Full sizes from config, or ``limit`` target questions with clean/paraphrase scaled."""
    full_targets = len(cfg.bench.attacks) * cfg.bench.targets_per_attack
    if limit is None or limit >= full_targets:
        return SetSizes(
            targets=full_targets,
            clean=cfg.bench.n_clean_questions,
            paraphrases=cfg.bench.n_paraphrase,
        )
    scale = limit / full_targets
    return SetSizes(
        targets=limit,
        clean=max(1, round(cfg.bench.n_clean_questions * scale)),
        paraphrases=max(1, round(cfg.bench.n_paraphrase * scale)),
    )


def _h(seed: int, key: str) -> str:
    return hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()


def candidate_chunks(chunks: Sequence[Chunk], seed: int) -> list[Chunk]:
    """One chunk per article, articles in seeded hash order."""
    best: dict[str, Chunk] = {}
    for c in chunks:
        cur = best.get(c.doc_id)
        if cur is None or _h(seed, c.chunk_id) < _h(seed, cur.chunk_id):
            best[c.doc_id] = c
    return sorted(best.values(), key=lambda c: _h(seed, c.doc_id))


class Candidate(BaseModel):
    order: int
    chunk: Chunk
    question: str
    answer: str
    aliases: list[str]


async def write_question(
    pipeline: Pipeline, writer_alias: str, order: int, chunk: Chunk
) -> Candidate | str:
    """Returns a Candidate, or a rejection reason."""
    result = await pipeline.llm.achat(
        writer_alias, user_message("question_writer", chunk_text=chunk.text), json_mode=True
    )
    try:
        data = parse_json_object(result.text)
    except ValueError:
        return "bad_json"
    question, answer = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
    aliases = [str(a).strip() for a in data.get("aliases", []) or [] if str(a).strip()]
    if not question or not answer:
        return "bad_json"
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return "answer_too_long"
    return Candidate(order=order, chunk=chunk, question=question, answer=answer, aliases=aliases)


async def check_candidate(
    pipeline: Pipeline, cand: Candidate, generator: str, judge: str
) -> str | None:
    """None if the clean system retrieves the gold chunk and answers correctly; else a reason."""
    hits = await asyncio.to_thread(pipeline.retriever.search, cand.question)
    if cand.chunk.chunk_id not in {
        h.chunk.chunk_id for h in hits[: pipeline.cfg.retrieval.fused_k]
    }:
        return "gold_not_retrieved"
    answer = await arun_query(cand.question, pipeline, defenses="none", model_alias=generator)
    probe = BenchQuestion(
        qid="probe", question=cand.question, gold_answer=cand.answer, gold_aliases=cand.aliases,
        gold_chunk_id=cand.chunk.chunk_id, qset="clean", split="dev",
    )  # fmt: skip
    verdict = await ajudge(answer.text, probe, [], pipeline.llm, judge)
    return None if verdict.verdict == "correct" else f"generator_{verdict.verdict}"


def stratified_split(
    items: list[BenchQuestion], seed: int, dev_fraction: float
) -> list[BenchQuestion]:
    groups: dict[str | None, list[BenchQuestion]] = {}
    for q in items:
        groups.setdefault(q.attack, []).append(q)
    out = []
    for group in groups.values():
        ordered = sorted(group, key=lambda q: _h(seed, q.qid))
        n_dev = round(len(ordered) * dev_fraction)
        out += [
            q.model_copy(update={"split": "dev" if i < n_dev else "test"})
            for i, q in enumerate(ordered)
        ]
    return out


async def generate_questions(
    cfg: AppConfig,
    pipeline: Pipeline,
    chunks: Sequence[Chunk],
    writer_alias: str,
    generator_alias: str,
    judge_alias: str,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[BenchQuestion], QuestionsReport]:
    sizes = set_sizes(cfg, limit)
    needed = sizes.targets + sizes.clean
    report = QuestionsReport()
    rejected: Counter[str] = Counter()
    kept: list[Candidate] = []
    pool = candidate_chunks(chunks, cfg.seed)
    pos = 0

    async def process(order: int, chunk: Chunk) -> Candidate | None:
        try:
            cand = await write_question(pipeline, writer_alias, order, chunk)
            if isinstance(cand, str):
                rejected[cand] += 1
                return None
            reason = await check_candidate(pipeline, cand, generator_alias, judge_alias)
        except Exception as exc:  # keep going; count it
            rejected[f"error_{type(exc).__name__}"] += 1
            log(f"  candidate {chunk.chunk_id} failed: {type(exc).__name__}: {exc}")
            return None
        if reason:
            rejected[reason] += 1
            return None
        return cand

    while len(kept) < needed and pos < len(pool):
        # oversample a little: roughly half of candidates survive both filters
        batch = pool[pos : pos + max(CANDIDATE_BATCH, 2 * (needed - len(kept)))]
        report.candidates += len(batch)
        results = await asyncio.gather(*(process(pos + i, c) for i, c in enumerate(batch)))
        pos += len(batch)
        kept += [c for c in results if c is not None]
        log(f"  {report.candidates} candidates -> {len(kept)} kept (need {needed})")
    kept.sort(key=lambda c: c.order)
    if len(kept) < needed:
        log(f"warning: only {len(kept)} usable questions (wanted {needed}); corpus too small?")
    report.kept = len(kept)  # survivors beyond `needed` are simply unused
    report.rejected = dict(rejected)

    def bq(
        c: Candidate, qset: Literal["target", "clean"], attack: AttackType | None
    ) -> BenchQuestion:
        return BenchQuestion(
            qid=f"q{_h(cfg.seed, c.chunk.chunk_id)[:10]}", question=c.question, gold_answer=c.answer,
            gold_aliases=c.aliases, gold_chunk_id=c.chunk.chunk_id, qset=qset,
            split="dev", attack=attack,
        )  # fmt: skip

    attacks = [a for a in ATTACKS if a in cfg.bench.attacks]
    targets = [
        bq(c, "target", attacks[i % len(attacks)]) for i, c in enumerate(kept[: sizes.targets])
    ]
    clean = [bq(c, "clean", None) for c in kept[sizes.targets : needed]]
    targets = stratified_split(targets, cfg.seed, cfg.bench.dev_fraction)
    clean = stratified_split(clean, cfg.seed, cfg.bench.dev_fraction)
    paraphrases = await make_paraphrases(pipeline, writer_alias, targets, sizes.paraphrases, log)

    questions = [*targets, *clean, *paraphrases]
    report.targets, report.clean, report.paraphrases = len(targets), len(clean), len(paraphrases)
    report.by_attack = dict(
        Counter(q.attack or "clean" for q in questions if q.qset != "paraphrase")
    )
    report.by_split = dict(Counter(f"{q.qset}/{q.split}" for q in questions))
    return questions, report


async def make_paraphrases(
    pipeline: Pipeline,
    writer_alias: str,
    targets: list[BenchQuestion],
    n: int,
    log: Callable[[str], None],
) -> list[BenchQuestion]:
    """Reworded copies of target questions, spread across attacks; they inherit the split."""
    by_attack: dict[str | None, list[BenchQuestion]] = {}
    for q in sorted(targets, key=lambda q: q.qid):
        by_attack.setdefault(q.attack, []).append(q)
    chosen: list[BenchQuestion] = []
    while len(chosen) < min(n, len(targets)):
        for group in by_attack.values():
            if group and len(chosen) < n:
                chosen.append(group.pop(0))

    async def one(q: BenchQuestion) -> BenchQuestion | None:
        try:
            result = await pipeline.llm.achat(
                writer_alias, user_message("paraphrase", question=q.question), json_mode=True
            )
            text = str(parse_json_object(result.text).get("question", "")).strip()
        except Exception as exc:
            log(f"  paraphrase of {q.qid} failed: {type(exc).__name__}")
            return None
        if not text or text.lower() == q.question.lower():
            return None
        return q.model_copy(
            update={"qid": f"{q.qid}p", "question": text, "qset": "paraphrase", "parent_qid": q.qid}
        )

    return [p for p in await asyncio.gather(*(one(q) for q in chosen)) if p is not None]
