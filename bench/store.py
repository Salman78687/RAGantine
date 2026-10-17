"""Benchmark data files under data/bench/ (SPEC §B3 storage layout).

``poison.jsonl`` is the ground truth. Only ``bench/`` reads it; ``ragantine/`` never does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bench.io import read_jsonl
from bench.schemas import BenchQuestion, PoisonSample
from ragantine.config import AppConfig
from ragantine.schemas import RawDocument


def bench_paths(cfg: AppConfig) -> dict[str, Path]:
    d = cfg.paths.bench_dir
    return {
        "questions": d / "questions.jsonl",
        "questions_report": d / "questions_report.json",
        "poison": d / "poison.jsonl",
        "poison_train": d / "poison_train.jsonl",
        "poison_docs": d / "poison_docs",
    }


def load_questions(cfg: AppConfig) -> list[BenchQuestion]:
    path = bench_paths(cfg)["questions"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `ragantine bench questions` first")
    return list(read_jsonl(path, BenchQuestion))


def load_poison(cfg: AppConfig, kind: str = "test") -> list[PoisonSample]:
    path = bench_paths(cfg)["poison" if kind == "test" else "poison_train"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `ragantine bench poison` first")
    return list(read_jsonl(path, PoisonSample))


def poison_for(
    cfg: AppConfig, split: str | None, seed: int, samples: list[PoisonSample] | None = None
) -> list[PoisonSample]:
    """Test-template poison for target questions in ``split`` (None = all splits) and seed."""
    samples = samples if samples is not None else load_poison(cfg)
    if split is None:
        return [p for p in samples if p.seed == seed]
    qids = {q.qid for q in load_questions(cfg) if q.qset == "target" and q.split == split}
    return [p for p in samples if p.seed == seed and p.qid in qids]


def poison_set_fingerprint(cfg: AppConfig, split: str, seed: int) -> str:
    ids = sorted(p.poison_id for p in poison_for(cfg, split, seed))
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:8]


def load_poison_documents(cfg: AppConfig, split: str, seed: int) -> list[RawDocument]:
    return [p.document for p in poison_for(cfg, split, seed)]
