"""Reranker score cache: each (model, query, passage) is scored once, across instances."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ragantine.retrieve.reranker import CrossEncoderReranker


class CountingReranker(CrossEncoderReranker):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__("fake-model", cache_dir=cache_dir)
        self.scored: list[str] = []

    def _predict(self, query: str, texts: Sequence[str]) -> list[float]:
        self.scored += list(texts)
        return [float(len(t)) for t in texts]


def test_scores_are_cached_on_disk_and_reused(tmp_path: Path) -> None:
    first = CountingReranker(tmp_path)
    assert first.score("q", ["aa", "b"]) == [2.0, 1.0]
    second = CountingReranker(tmp_path)  # a later run (new process) sharing the cache
    assert second.score("q", ["b", "cccc", "aa"]) == [1.0, 4.0, 2.0]
    assert second.scored == ["cccc"]  # only the new passage hits the model
    assert second.score("other query", ["b"]) == [1.0] and second.scored[-1] == "b"
