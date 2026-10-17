"""Builders for defense unit tests."""

from __future__ import annotations

from typing import Any

from ragantine.defenses.base import Defense, build_defenses
from ragantine.schemas import Chunk, RetrievedChunk, SourceTier
from tests.helpers import NOW, small_config

CFG = small_config()


def defense(did: str, **overrides: Any) -> Defense:
    d = build_defenses([did], CFG)[0]
    d.cfg.update(overrides)
    return d


def chunk(
    doc: str, text: str, tier: SourceTier | str = "curated", ordinal: int = 0, title: str = "T"
) -> Chunk:
    doc_id = (doc * 16)[:16]
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal:04d}", doc_id=doc_id, title=title, text=f"Title: {title}\n\n{text}",
        ordinal=ordinal, n_tokens=len(text.split()), source_tier=SourceTier(tier),
        content_hash=f"{doc}{ordinal}{text}"[:64], ingested_at=NOW,
    )  # fmt: skip


def hit(c: Chunk, rerank: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=c, rrf_score=0.01, rerank_score=rerank, final_score=rerank)
