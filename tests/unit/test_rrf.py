"""Reciprocal rank fusion math (SPEC §B4.3)."""

import pytest

from ragantine.retrieve.rrf import rrf_fuse


def test_rrf_scores_match_formula() -> None:
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_orders_by_score_and_truncates() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60, top_k=2)
    assert [k for k, _ in fused] == ["b", "a"]


def test_rrf_ties_break_by_first_appearance() -> None:
    fused = rrf_fuse([["x", "y"], ["y", "x"]], rrf_k=60)
    assert [k for k, _ in fused] == ["x", "y"]


def test_rrf_empty() -> None:
    assert rrf_fuse([[], []], rrf_k=60) == []
