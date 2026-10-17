"""Metric formulas and bootstrap CIs (SPEC §B5.6)."""

from bench.metrics import bootstrap_ci, flagged_chunks, percentile, rate
from ragantine.schemas import DefenseDecision


def test_rate_and_ci_bounds() -> None:
    r = rate([True] * 30 + [False] * 70, seed=1)
    assert (r.value, r.n, r.k) == (0.3, 100, 30)
    assert r.ci_low is not None and r.ci_high is not None
    assert r.ci_low < 0.3 < r.ci_high and r.ci_high - r.ci_low < 0.25


def test_bootstrap_is_deterministic_and_degenerate_cases() -> None:
    xs = [True, False, True, True]
    assert bootstrap_ci(xs, seed=7) == bootstrap_ci(xs, seed=7)
    assert bootstrap_ci([True] * 10, seed=1) == (1.0, 1.0)
    assert bootstrap_ci([], seed=1) == (None, None)
    assert rate([], seed=1).value is None


def test_percentiles() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0, 1000.0], 50) == 30.0
    assert percentile([], 95) is None


def test_document_level_decisions_flag_every_chunk() -> None:
    doc_chunks = {"doc1": ["doc1:0000", "doc1:0001"], "doc2": ["doc2:0000"]}
    decisions = [
        DefenseDecision(
            defense_id="d1",
            stage="ingestion",
            target_id="doc1",
            action="quarantine",
            reason="hidden_content",
        ),
        DefenseDecision(
            defense_id="d2",
            stage="retrieval",
            target_id="doc2:0000",
            action="filter",
            reason="injection",
        ),
        DefenseDecision(
            defense_id="d3",
            stage="retrieval",
            target_id="doc9:0000",
            action="downweight",
            reason="low tier",
        ),
    ]
    assert flagged_chunks(decisions, doc_chunks) == {"doc1:0000", "doc1:0001", "doc2:0000"}
