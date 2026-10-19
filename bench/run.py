"""Benchmark run protocol (SPEC §B5.4).

1. Each distinct set of ingestion defenses gets its own index; clean embeddings are reused.
2. The clean corpus + all poison docs of the split (for the run's seed) are ingested
   together, as in a real corpus.
3. Every selected question goes through ``arun_query``; answers are scored rule-first.
4. ``results/<run_id>/`` gets config.yaml, answers.jsonl, decisions.jsonl, metrics.json,
   report.md and human_check.csv.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel

from bench.io import write_jsonl
from bench.metrics import IngestionFacts, Metrics, TokenUsage, compute_metrics
from bench.schemas import BenchQuestion, JudgedAnswer, PoisonSample
from bench.score import ajudge
from ragantine import __version__
from ragantine.config import AppConfig, ModelsConfig
from ragantine.defenses.base import parse_defense_ids
from ragantine.generate.llm import QuotaExhaustedError
from ragantine.graph import Pipeline, arun_query
from ragantine.schemas import Answer

QSet = Literal["target", "clean", "paraphrase"]
HUMAN_CHECK_FRACTION = 0.10


class RunSpec(BaseModel):
    run_id: str
    defenses: list[str]
    model_alias: str
    sets: list[QSet]
    split: Literal["dev", "test"]
    seed: int
    limit: int | None = None
    use_cache: bool = True
    label: str | None = None  # configuration label for reports (e.g. "Full stack")
    fixture: str | None = None  # set for offline fixture runs (CI smoke)
    index_tag: str | None = None  # forces a separate index (adaptive attacker mutates its index)


def make_run_id(defenses: Sequence[str], model_alias: str, split: str, seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tag = "-".join(defenses) if defenses else "d0"
    return f"{stamp}-{model_alias}-{tag}-{split}-s{seed}"


class Workspace(Protocol):
    """Builds (or reuses) the index for a run and returns a ready pipeline."""

    def prepare(
        self, spec: RunSpec, poison: Sequence[PoisonSample]
    ) -> tuple[Pipeline, IngestionFacts]: ...


def select_questions(
    questions: Sequence[BenchQuestion], sets: Sequence[str], split: str, limit: int | None
) -> list[BenchQuestion]:
    """Questions of the given sets/split; ``limit`` keeps the mix balanced across sets and attacks."""
    pool = [q for q in questions if q.qset in sets and q.split == split]
    if limit is None or limit >= len(pool):
        return pool
    groups: dict[tuple[str, str | None], list[BenchQuestion]] = {}
    for q in pool:
        groups.setdefault((q.qset, q.attack), []).append(q)
    ordered = [groups[k] for k in sorted(groups, key=lambda k: (k[0], k[1] or ""))]
    out: list[BenchQuestion] = []
    while len(out) < limit:
        for g in ordered:
            if g and len(out) < limit:
                out.append(g.pop(0))
    return out


def poison_lookup(
    poison: Sequence[PoisonSample], questions: Sequence[BenchQuestion]
) -> dict[str, list[PoisonSample]]:
    """qid -> poison samples planted for it (paraphrases use their parent's poison)."""
    by_qid: dict[str, list[PoisonSample]] = {}
    for p in poison:
        by_qid.setdefault(p.qid, []).append(p)
    out: dict[str, list[PoisonSample]] = {}
    for q in questions:
        key = q.parent_qid if q.qset == "paraphrase" and q.parent_qid else q.qid
        out[q.qid] = by_qid.get(key, []) if q.qset != "clean" else []
    return out


def _error_answer(q: BenchQuestion, spec: RunSpec, exc: BaseException) -> Answer:
    return Answer(
        question=q.question, text="", citations=[], decisions=[], retrieved=[],
        model_alias=spec.model_alias, latency_ms=0, prompt_tokens=0, completion_tokens=0,
    )  # fmt: skip


async def answer_and_judge(
    pipeline: Pipeline,
    spec: RunSpec,
    questions: Sequence[BenchQuestion],
    poison_by_qid: dict[str, list[PoisonSample]],
    judge_alias: str,
    log: Callable[[str], None],
) -> tuple[list[JudgedAnswer], TokenUsage, int]:
    judge_usage = TokenUsage()
    errors = 0
    done = 0

    async def one(q: BenchQuestion) -> JudgedAnswer:
        nonlocal errors, done
        try:
            answer = await arun_query(q.question, pipeline, spec.defenses, spec.model_alias)
        except QuotaExhaustedError:
            raise
        except Exception as exc:
            errors += 1
            log(f"  {q.qid} failed: {type(exc).__name__}: {exc}")
            return JudgedAnswer(
                qid=q.qid, run_id=spec.run_id, seed=spec.seed, answer=_error_answer(q, spec, exc),
                verdict="other", judge_reason=f"error: {type(exc).__name__}", matched_by="rule",
            )  # fmt: skip
        j = await ajudge(answer.text, q, poison_by_qid.get(q.qid, []), pipeline.llm, judge_alias)
        judge_usage.prompt_tokens += j.prompt_tokens
        judge_usage.completion_tokens += j.completion_tokens
        done += 1
        if done % 25 == 0:
            log(f"  {done}/{len(questions)} answered")
        return JudgedAnswer(
            qid=q.qid, run_id=spec.run_id, seed=spec.seed, answer=answer,
            verdict=j.verdict, judge_reason=j.reason, matched_by=j.matched_by,
        )  # fmt: skip

    judged = await asyncio.gather(*(one(q) for q in questions))
    return list(judged), judge_usage, errors


def fingerprint(items: Sequence[BaseModel]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.model_dump_json().encode())
    return h.hexdigest()[:12]


async def execute_run(
    cfg: AppConfig,
    models: ModelsConfig,
    spec: RunSpec,
    questions: Sequence[BenchQuestion],
    poison: Sequence[PoisonSample],
    workspace: Workspace,
    log: Callable[[str], None] = print,
) -> tuple[Metrics, Path]:
    """Run one configuration end to end and write results/<run_id>/."""
    spec = spec.model_copy(update={"defenses": parse_defense_ids(spec.defenses)})
    selected = select_questions(questions, spec.sets, spec.split, spec.limit)
    # all poison of this split and seed is ingested, even for questions outside --limit
    split_targets = {q.qid for q in questions if q.qset == "target" and q.split == spec.split}
    run_poison = [p for p in poison if p.seed == spec.seed and p.qid in split_targets]
    pipeline, facts = workspace.prepare(spec, run_poison)
    judge_alias = models.judge_for(spec.model_alias)
    log(
        f"run {spec.run_id}: {len(selected)} questions, defenses={spec.defenses or 'none'}, judge={judge_alias}"
    )
    hits_before, calls_before = pipeline.llm.cache_hits, pipeline.llm.calls
    judged, judge_usage, errors = await answer_and_judge(
        pipeline, spec, selected, poison_lookup(run_poison, selected), judge_alias, log
    )
    judge_entry = models.models.get(judge_alias)
    judge_usage.cost_usd = (
        judge_entry.cost_usd(judge_usage.prompt_tokens, judge_usage.completion_tokens)
        if judge_entry
        else None
    )
    metrics = compute_metrics(
        selected, judged, run_poison, facts, models, judge_usage, seed=spec.seed,
        cache_hits=pipeline.llm.cache_hits - hits_before, llm_calls=pipeline.llm.calls - calls_before,
        errors=errors,
    )  # fmt: skip
    out = cfg.paths.results_dir / spec.run_id
    write_outputs(out, cfg, models, spec, selected, judged, run_poison, facts, metrics)
    return metrics, out


def write_outputs(
    out: Path,
    cfg: AppConfig,
    models: ModelsConfig,
    spec: RunSpec,
    questions: Sequence[BenchQuestion],
    judged: Sequence[JudgedAnswer],
    poison: Sequence[PoisonSample],
    facts: IngestionFacts,
    metrics: Metrics,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "spec": spec.model_dump(mode="json"),
        "ragantine_version": __version__,
        "data": {"questions": fingerprint(questions), "poison": fingerprint(poison)},
        "config": cfg.model_dump(mode="json"),
        "models": models.model_dump(mode="json"),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True))
    write_jsonl(out / "answers.jsonl", judged)
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for d in facts.decisions:
            f.write(json.dumps({"qid": None, **d.model_dump(mode="json")}) + "\n")
        for j in judged:
            for d in j.answer.decisions:
                f.write(json.dumps({"qid": j.qid, **d.model_dump(mode="json")}) + "\n")
    (out / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    write_human_check(out / "human_check.csv", questions, judged, poison, spec.seed)
    from bench.report import single_run_markdown

    (out / "report.md").write_text(single_run_markdown(spec, metrics))


HUMAN_CHECK_COLUMNS = [
    "qid", "qset", "attack", "question", "gold_answer", "planted_false_answer", "planted_canary",
    "answer", "verdict", "matched_by", "human_label",
]  # fmt: skip


def write_human_check(
    path: Path,
    questions: Sequence[BenchQuestion],
    judged: Sequence[JudgedAnswer],
    poison: Sequence[PoisonSample],
    seed: int,
) -> None:
    """A random 10% of judged items for hand-labelling (fill the human_label column)."""
    by_qid = {q.qid: q for q in questions}
    lookup = poison_lookup(poison, questions)
    pool = sorted(judged, key=lambda j: j.qid)
    k = max(1, round(len(pool) * HUMAN_CHECK_FRACTION)) if pool else 0
    sample = random.Random(seed).sample(pool, k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HUMAN_CHECK_COLUMNS)
        w.writeheader()
        for j in sorted(sample, key=lambda j: j.qid):
            q = by_qid[j.qid]
            ps = lookup.get(j.qid, [])
            w.writerow(
                {
                    "qid": j.qid, "qset": q.qset, "attack": q.attack or "", "question": q.question,
                    "gold_answer": " / ".join([q.gold_answer, *q.gold_aliases]),
                    "planted_false_answer": next((p.target_false_answer for p in ps if p.target_false_answer), ""),
                    "planted_canary": next((p.canary for p in ps if p.canary), ""),
                    "answer": j.answer.text, "verdict": j.verdict, "matched_by": j.matched_by,
                    "human_label": "",
                }
            )  # fmt: skip
