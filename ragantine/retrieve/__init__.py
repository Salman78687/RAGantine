"""Hybrid retrieval: BM25 + dense, RRF fusion, cross-encoder rerank (SPEC §B4.3).

``HybridRetriever.search`` returns the reranked ``fused_k`` candidates with every rank
and score filled in for tracing; the query graph's ``filter_context`` node applies the
retrieval-time defenses and keeps ``final_k`` of them for the prompt.
"""

from __future__ import annotations

from typing import Any, Protocol

from ragantine.config import RetrievalConfig
from ragantine.index.base import IndexBackend
from ragantine.retrieve.dense import Embedder
from ragantine.retrieve.reranker import Reranker
from ragantine.retrieve.rrf import rrf_fuse
from ragantine.schemas import Chunk, RetrievedChunk


class Retriever(Protocol):
    """What the query graph needs from a retriever (HybridRetriever, AdapterRetriever)."""

    backend: IndexBackend  # holds the quarantine store
    embedder: Any

    def search(self, query: str) -> list[RetrievedChunk]: ...


class HybridRetriever:
    def __init__(
        self,
        backend: IndexBackend,
        embedder: Embedder,
        reranker: Reranker,
        cfg: RetrievalConfig,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg

    def search(self, query: str) -> list[RetrievedChunk]:
        bm25_hits = self.backend.bm25(query, self.cfg.bm25_k)
        knn_hits = self.backend.knn(self.embedder.embed_query(query), self.cfg.knn_k)
        by_id: dict[str, Chunk] = {c.chunk_id: c for c in [*bm25_hits, *knn_hits]}
        bm25_rank = {c.chunk_id: r for r, c in enumerate(bm25_hits, start=1)}
        knn_rank = {c.chunk_id: r for r, c in enumerate(knn_hits, start=1)}
        fused = rrf_fuse(
            [[c.chunk_id for c in bm25_hits], [c.chunk_id for c in knn_hits]],
            rrf_k=self.cfg.rrf_k,
            top_k=self.cfg.fused_k,
        )
        if not fused:
            return []
        rerank = self.reranker.score(query, [by_id[cid].text for cid, _ in fused])
        hits = [
            RetrievedChunk(
                chunk=by_id[cid],
                bm25_rank=bm25_rank.get(cid),
                knn_rank=knn_rank.get(cid),
                rrf_score=rrf,
                rerank_score=score,
                final_score=score,
            )
            for (cid, rrf), score in zip(fused, rerank, strict=True)
        ]
        hits.sort(key=lambda h: -h.final_score)
        return hits
