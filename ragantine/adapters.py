"""Benchmark your own retriever (FR-10).

Wrap any retriever so RAGantine's query graph, defenses and benchmark can run on top of it:

    from ragantine.adapters import AdapterRetriever
    retriever = AdapterRetriever.from_langchain(my_lc_retriever)      # .invoke(q) -> Documents
    retriever = AdapterRetriever.from_llamaindex(my_li_retriever)     # .retrieve(q) -> NodeWithScore
    retriever = AdapterRetriever(lambda q: [ExternalHit(text=..., doc_id=..., title=...)])

Hits are converted to ``RetrievedChunk``. Keep RAGantine's ``doc_id`` and ``source_tier`` in
your documents' metadata so trust scoring (D3) and the benchmark's poison tracking work.
LLMs need no adapter: any OpenAI-compatible endpoint (vLLM, Ollama, Azure, ...) is an entry in
configs/models.yaml. Neither LangChain nor LlamaIndex is a dependency; objects are duck-typed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from ragantine.retrieve.memory import MemoryIndex
from ragantine.schemas import Chunk, RetrievedChunk, SourceTier

UNKNOWN_DATE = datetime(1970, 1, 1, tzinfo=UTC)
QUARANTINE_ONLY_DIM = 1  # the adapter's local MemoryIndex only holds the quarantine store


class ExternalHit(BaseModel):
    text: str
    doc_id: str | None = None
    title: str = ""
    source_tier: SourceTier = SourceTier.WEB  # unknown provenance is not trusted
    score: float | None = None


def _hit_from_metadata(text: str, metadata: dict[str, Any], score: float | None) -> ExternalHit:
    tier = metadata.get("source_tier")
    return ExternalHit(
        text=text,
        doc_id=metadata.get("doc_id") or metadata.get("id") or metadata.get("source"),
        title=str(metadata.get("title", "")),
        source_tier=SourceTier(tier) if tier in {t.value for t in SourceTier} else SourceTier.WEB,
        score=score,
    )


def to_retrieved(hits: Sequence[ExternalHit]) -> list[RetrievedChunk]:
    """Rank-ordered external hits -> RetrievedChunk with stable ids per document."""
    out = []
    ordinals: dict[str, int] = {}
    for rank, h in enumerate(hits, start=1):
        doc_id = (
            (h.doc_id and hashlib.sha1(h.doc_id.encode()).hexdigest()[:16])
            if h.doc_id
            else hashlib.sha1(h.text.encode()).hexdigest()[:16]
        )
        if h.doc_id and len(h.doc_id) == 16 and all(c in "0123456789abcdef" for c in h.doc_id):
            doc_id = h.doc_id  # already a RAGantine doc_id: keep it so poison can be tracked
        ordinal = ordinals.get(doc_id, 0)
        ordinals[doc_id] = ordinal + 1
        chunk = Chunk(
            chunk_id=f"{doc_id}:{ordinal:04d}", doc_id=doc_id, title=h.title or doc_id, text=h.text,
            ordinal=ordinal, n_tokens=len(h.text.split()), source_tier=h.source_tier,
            content_hash=hashlib.sha256(h.text.encode()).hexdigest(), ingested_at=UNKNOWN_DATE,
        )  # fmt: skip
        score = h.score if h.score is not None else 1.0 / rank
        out.append(
            RetrievedChunk(chunk=chunk, rrf_score=1.0 / rank, rerank_score=score, final_score=score)
        )
    return out


class AdapterRetriever:
    """Drop-in for ``HybridRetriever``: ``search(question) -> list[RetrievedChunk]``."""

    def __init__(self, fn: Callable[[str], Sequence[ExternalHit]]) -> None:
        self.fn = fn
        self.backend = MemoryIndex(QUARANTINE_ONLY_DIM)  # quarantine review queue for this adapter
        self.embedder: Any = None  # D6 hub statistics need an embedder; unavailable for adapters

    def search(self, query: str) -> list[RetrievedChunk]:
        return to_retrieved(self.fn(query))

    @classmethod
    def from_langchain(cls, retriever: Any) -> AdapterRetriever:
        def fn(q: str) -> list[ExternalHit]:
            docs = retriever.invoke(q)
            return [_hit_from_metadata(d.page_content, dict(d.metadata or {}), None) for d in docs]

        return cls(fn)

    @classmethod
    def from_llamaindex(cls, retriever: Any) -> AdapterRetriever:
        def fn(q: str) -> list[ExternalHit]:
            nodes = retriever.retrieve(q)
            return [
                _hit_from_metadata(
                    n.node.get_content(), dict(getattr(n.node, "metadata", {}) or {}), n.score
                )
                for n in nodes
            ]

        return cls(fn)
