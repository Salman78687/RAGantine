"""D2 data: train-only templates, negatives ratio, threshold selection on dev (SPEC §B6)."""

from pathlib import Path

import pytest

from bench.d2 import D2Example, build_training_set, pick_threshold, write_threshold
from tests.fakes_bench import build_bench_data


def _ex(label: int, attack: str | None = None) -> D2Example:
    return D2Example(
        text="t", label=label, source="poison" if label else "clean", split="dev", attack=attack
    )


def test_pick_threshold_respects_false_quarantine_budget() -> None:
    clean = [0.009 * i for i in range(100)]  # 0.000 .. 0.891
    poison = [0.95, 0.97, 0.5, 0.99]
    choice = pick_threshold(clean + poison, [_ex(0)] * 100 + [_ex(1, "A2")] * 4)
    assert choice.false_quarantine <= 0.02
    assert choice.recall == pytest.approx(0.75)  # 0.5 is below the clean noise floor
    with pytest.raises(ValueError):
        pick_threshold([0.5], [_ex(1)])


def test_write_threshold_keeps_comments(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "defenses:\n  d2: { model_path: m, threshold: 0.85, apply_at: [x] }   # keep me\n"
    )
    write_threshold(str(cfg), 0.913)
    assert "threshold: 0.913" in cfg.read_text() and "# keep me" in cfg.read_text()


def test_training_set_uses_train_templates_and_negative_ratio() -> None:
    import asyncio

    from bench.poison.generator import generate_poison

    cfg, llm, docs, questions, _ = build_bench_data(limit=12)
    from tests.helpers import memory_pipeline

    pipeline, _ = memory_pipeline(llm, docs=docs, cfg=cfg)
    chunks = pipeline.backend.all_chunks()  # type: ignore[attr-defined]
    subjects = {c.chunk_id: c.title for c in chunks}
    train, _ = asyncio.run(
        generate_poison(cfg, llm, "fake", questions, subjects, [cfg.seed], "train", lambda _m: None)
    )
    examples = build_training_set(cfg, train, chunks)
    pos = [e for e in examples if e.label == 1]
    neg = [e for e in examples if e.label == 0]
    assert pos and all(int(str(e.template_id)[-2:]) <= 4 for e in pos)
    assert len(neg) <= 6 * len(pos) and len([e for e in neg if e.source == "clean"]) == min(
        len(chunks), 5 * len(pos)
    )
    with pytest.raises(ValueError, match="t01-t04"):
        bad = train[0].model_copy(update={"template_id": train[0].template_id[:-2] + "05"})
        build_training_set(cfg, [bad], chunks)
