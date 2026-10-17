"""Run protocol, metrics and report on the mini corpus with a compliant FakeLLM (SPEC §B5.4-B5.6)."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from bench.report import build_report, config_label
from bench.run import RunSpec, execute_run, select_questions
from bench.workspace import MemoryWorkspace
from ragantine.fakes import HashingEmbedder, OverlapReranker, fake_models_config
from tests.fakes_bench import build_bench_data
from tests.helpers import word_count


@pytest.fixture(scope="module")
def data():  # type: ignore[no-untyped-def]
    return build_bench_data(limit=12, seeds=(1337, 7))


def _run(
    data,
    tmp: Path,
    defenses: list[str],
    seed: int = 1337,
    split: str = "dev",
    limit: int | None = None,
):  # type: ignore[no-untyped-def]
    cfg, llm, docs, questions, poison = data
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"results_dir": tmp})})
    ws = MemoryWorkspace(
        cfg,
        llm,
        docs,
        HashingEmbedder(cfg.embedding.dimension),
        OverlapReranker(),
        count=word_count,
    )
    spec = RunSpec(
        run_id=f"run-{'-'.join(defenses) or 'd0'}-{split}-{seed}",
        defenses=defenses,
        model_alias="fake",
        sets=["target", "clean", "paraphrase"],
        split=split,
        seed=seed,
        limit=limit,
    )  # type: ignore[arg-type]
    return asyncio.run(
        execute_run(cfg, fake_models_config(), spec, questions, poison, ws, log=lambda _m: None)
    )


def test_baseline_run_outputs_and_metrics(data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    metrics, out = _run(data, tmp_path, [])
    for name in (
        "config.yaml",
        "answers.jsonl",
        "decisions.jsonl",
        "metrics.json",
        "report.md",
        "human_check.csv",
    ):
        assert (out / name).exists(), name
    # the scripted model obeys injected canaries and repeats planted text: the baseline is vulnerable
    assert metrics.asr.n == 6 and metrics.asr.value is not None and metrics.asr.value > 0
    assert set(metrics.asr_by_attack) == {"A1", "A2", "A3", "A4", "A5", "A6"}
    assert metrics.clean_accuracy.n > 0
    assert (
        metrics.detection_recall.k == 0 and metrics.detection_recall.n > 0
    )  # no defenses -> nothing flagged
    assert metrics.ingestion_false_quarantine.value == 0
    assert metrics.latency_p95_ms is not None and metrics.generator.prompt_tokens > 0
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["asr"]["k"] == metrics.asr.k
    rows = list(csv.DictReader((out / "human_check.csv").open()))
    assert len(rows) >= 1 and rows[0]["human_label"] == ""


def test_conditional_asr_only_counts_poison_that_reached_the_prompt(data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    metrics, _ = _run(data, tmp_path, [])
    assert metrics.conditional_asr.n <= metrics.asr.n
    assert (
        metrics.conditional_asr.k == metrics.asr.k
    )  # a planted answer can only win if it was retrieved


def test_limit_keeps_sets_balanced(data) -> None:  # type: ignore[no-untyped-def]
    _, _, _, questions, _ = data
    chosen = select_questions(questions, ["target", "clean"], "test", 4)
    assert {q.qset for q in chosen} == {"target", "clean"}


def test_report_pools_seeds_and_writes_table_csv_chart(data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _run(data, tmp_path, [], seed=1337)
    _run(data, tmp_path, [], seed=7)
    text, out = build_report(tmp_path, "all", out_dir=tmp_path / "report")
    assert "| D0 Baseline | fake |" in text and "| 2 |" in text  # one row, two seeds
    assert (out / "per_attack.csv").exists() and (
        out / "asr_by_attack_fake.png"
    ).stat().st_size > 1000
    rows = list(csv.DictReader((out / "per_attack.csv").open()))
    assert {r["attack"] for r in rows} == {"A1", "A2", "A3", "A4", "A5", "A6"}
    assert all(int(r["n"]) == 2 for r in rows)  # one dev target per attack per seed


def test_config_labels() -> None:
    assert config_label([]) == "D0 Baseline"
    assert config_label(["d1"]) == "+ D1 Sanitizer"
    assert config_label(["d1", "d2", "d3", "d4", "d5"]) == "Full stack"
    assert config_label(["d1", "d2", "d4", "d5"]) == "Full stack − D3"
    assert config_label(["d1", "d3"]) == "+ D1 + D3"
