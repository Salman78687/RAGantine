"""Adaptive attacker (FR-14) on the mini corpus with the scripted fixture model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bench.adaptive import MAX_ROUNDS, run_adaptive
from bench.run import RunSpec
from bench.workspace import MemoryWorkspace
from ragantine.defenses.base import build_defenses
from ragantine.fakes import (
    HashingEmbedder,
    KeywordInjectionScorer,
    OverlapReranker,
    fake_models_config,
)
from tests.fakes_bench import build_bench_data
from tests.helpers import word_count


def test_adaptive_rounds_are_recorded_and_bounded(tmp_path: Path) -> None:
    cfg, llm, docs, questions, poison = build_bench_data(limit=12)
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"results_dir": tmp_path})})
    d2 = build_defenses(["d2"], cfg)[0]
    d2.scorer = KeywordInjectionScorer()  # type: ignore[attr-defined]
    ws = MemoryWorkspace(cfg, llm, docs, HashingEmbedder(cfg.embedding.dimension), OverlapReranker(),
                         defense_overrides={"d2": d2}, count=word_count)  # fmt: skip
    spec = RunSpec(
        run_id="adaptive-test",
        defenses=["d1", "d2", "d3"],
        model_alias="fake",
        sets=["target"],
        split="test",
        seed=cfg.seed,
    )  # type: ignore[arg-type]
    report, metrics, out = asyncio.run(
        run_adaptive(
            cfg,
            fake_models_config(),
            spec,
            questions,
            poison,
            ws,
            "fake",
            rounds=5,
            log=lambda _m: None,
        )
    )
    assert [r.round for r in report.rounds] == list(range(MAX_ROUNDS + 1))  # capped at 3 rounds
    asrs = [r.asr or 0.0 for r in report.rounds]
    assert asrs == sorted(asrs)  # cumulative: a question once poisoned stays counted
    assert report.rounds[1].regenerated > 0
    assert json.loads((out / "adaptive.json").read_text())["run_id"] == "adaptive-test"
    assert (out / "metrics.json").exists() and metrics.asr.n == report.rounds[-1].n
    assert "adaptive attacker" in (out / "report.md").read_text()
