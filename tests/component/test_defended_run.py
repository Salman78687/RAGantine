"""Full defense stack vs baseline on the mini corpus (FakeLLM + heuristic model stand-ins).

This checks the plumbing (hooks, quarantine routing, metrics), not the real models'
accuracy: D2 and D5 use KeywordInjectionScorer / LexicalNLI here.
"""

from __future__ import annotations

import asyncio

import pytest

from bench.run import RunSpec, execute_run
from bench.workspace import MemoryWorkspace
from ragantine.defenses.base import build_defenses
from ragantine.fakes import (
    HashingEmbedder,
    KeywordInjectionScorer,
    LexicalNLI,
    OverlapReranker,
    fake_models_config,
)
from tests.fakes_bench import build_bench_data
from tests.helpers import word_count


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    cfg, llm, docs, questions, poison = build_bench_data(limit=24, seeds=(1337,))
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"results_dir": tmp_path_factory.mktemp("results")}
            )
        }
    )
    d2, d5 = build_defenses(["d2", "d5"], cfg)
    d2.scorer = KeywordInjectionScorer()  # type: ignore[attr-defined]
    d5.nli = LexicalNLI()  # type: ignore[attr-defined]
    ws = MemoryWorkspace(cfg, llm, docs, HashingEmbedder(cfg.embedding.dimension), OverlapReranker(),
                         defense_overrides={"d2": d2, "d5": d5}, count=word_count)  # fmt: skip
    out = {}
    for name, ids in {
        "d0": [],
        "d1": ["d1"],
        "d2": ["d2"],
        "d4": ["d4"],
        "full": ["d1", "d2", "d3", "d4", "d5"],
    }.items():
        spec = RunSpec(
            run_id=f"t-{name}",
            defenses=ids,
            model_alias="fake",
            sets=["target", "clean"],
            split="test",
            seed=1337,
        )  # type: ignore[arg-type]
        out[name] = asyncio.run(
            execute_run(cfg, fake_models_config(), spec, questions, poison, ws, log=lambda _m: None)
        )[0]
    return out


def test_baseline_is_vulnerable(runs) -> None:  # type: ignore[no-untyped-def]
    assert runs["d0"].asr.value is not None and runs["d0"].asr.value >= 0.5


def test_full_stack_cuts_asr_and_detects_poison(runs) -> None:  # type: ignore[no-untyped-def]
    base, full = runs["d0"], runs["full"]
    assert full.asr.value is not None and base.asr.value is not None
    assert full.asr.value < base.asr.value
    assert full.detection_recall.value is not None and full.detection_recall.value > 0
    assert full.ingestion_false_quarantine.value == 0  # mini corpus is clean prose


def test_d1_stops_hidden_text(runs) -> None:  # type: ignore[no-untyped-def]
    assert runs["d1"].asr_by_attack["A5"].k <= runs["d0"].asr_by_attack["A5"].k
    assert runs["d1"].detection_recall.k > 0


def test_d2_flags_instruction_attacks(runs) -> None:  # type: ignore[no-untyped-def]
    for attack in ("A2", "A5"):
        assert runs["d2"].asr_by_attack[attack].k <= runs["d0"].asr_by_attack[attack].k
