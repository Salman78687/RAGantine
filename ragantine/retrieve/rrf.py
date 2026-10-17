"""Reciprocal rank fusion: score = sum(1 / (rrf_k + rank)) (SPEC §B4.3)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def rrf_fuse(
    rankings: Sequence[Sequence[K]], rrf_k: int, top_k: int | None = None
) -> list[tuple[K, float]]:
    """Fuse ranked lists (best first, ranks are 1-based) into ``(key, score)`` best first.

    Ties are broken by first appearance across the input lists, so fusion is deterministic.
    """
    scores: dict[K, float] = {}
    order: dict[K, int] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            order.setdefault(key, len(order))
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    return fused if top_k is None else fused[:top_k]
