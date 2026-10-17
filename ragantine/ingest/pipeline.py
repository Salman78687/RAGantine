"""``ingest(docs, defenses) -> IngestReport`` (SPEC §B4.1).

Order: load -> (D1 on_load) -> chunk -> (D2 on_chunks) -> embed -> bulk index.
Chunks with any ``quarantine`` decision go to the quarantine store instead of the main
index. Ingestion is idempotent by ``content_hash``, and precomputed embeddings (keyed by
content_hash) are reused so clean chunks are never re-embedded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from itertools import islice

import numpy as np
from pydantic import BaseModel, Field

from ragantine.config import ChunkingConfig
from ragantine.defenses.base import Defense
from ragantine.index.base import IndexBackend
from ragantine.ingest.chunker import TokenCounter, chunk_document
from ragantine.ingest.loaders import load_text
from ragantine.retrieve.dense import Embedder
from ragantine.schemas import Chunk, DefenseDecision, QuarantineRecord, RawDocument

DOC_BATCH = 256


class IngestReport(BaseModel):
    n_docs: int = 0
    n_chunks: int = 0
    n_indexed: int = 0
    n_quarantined: int = 0
    n_skipped_existing: int = 0
    n_embedded: int = 0
    n_reused_embeddings: int = 0
    decisions: list[DefenseDecision] = Field(default_factory=list)
    quarantined_chunk_ids: list[str] = Field(default_factory=list)

    def merge(self, other: IngestReport) -> None:
        for name in type(self).model_fields:
            mine, theirs = getattr(self, name), getattr(other, name)
            setattr(self, name, mine + theirs)


def _quarantined(decisions: Iterable[DefenseDecision]) -> bool:
    return any(d.action == "quarantine" for d in decisions)


def prepare_document(
    doc: RawDocument,
    defenses: list[Defense],
    chunking: ChunkingConfig,
    count: TokenCounter | None = None,
) -> tuple[list[Chunk], dict[str, list[DefenseDecision]], list[DefenseDecision]]:
    """Load + on_load + chunk + on_chunks for one document.

    Returns (chunks, per-chunk quarantine decisions, all decisions).
    """
    text = load_text(doc)
    doc_decisions: list[DefenseDecision] = []
    for d in defenses:
        text, dec = d.on_load(doc, text)
        doc_decisions += dec
    chunks = chunk_document(
        doc.doc_id, doc.title, text, doc.source_tier, doc.ingested_at, chunking, count
    )
    all_decisions = list(doc_decisions)
    per_chunk: dict[str, list[DefenseDecision]] = {}
    if _quarantined(doc_decisions):
        # Whole-document quarantine (e.g. D1 hidden_content): every chunk is held for review.
        for c in chunks:
            per_chunk[c.chunk_id] = [d for d in doc_decisions if d.action == "quarantine"]
        return chunks, per_chunk, all_decisions
    for d in defenses:
        chunks_after, dec = d.on_chunks(chunks)
        all_decisions += dec
        for x in dec:
            if x.action == "quarantine":
                per_chunk.setdefault(x.target_id, []).append(x)
        # a defense may drop chunks it quarantined; keep them for the quarantine store
        kept = {c.chunk_id for c in chunks_after}
        chunks = chunks_after + [c for c in chunks if c.chunk_id not in kept]
    chunks.sort(key=lambda c: c.ordinal)
    return chunks, per_chunk, all_decisions


def ingest(
    docs: Iterable[RawDocument],
    defenses: list[Defense],
    *,
    backend: IndexBackend,
    embedder: Embedder,
    chunking: ChunkingConfig,
    precomputed: Mapping[str, np.ndarray] | None = None,
    count: TokenCounter | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    on_batch: Callable[[IngestReport], None] | None = None,
) -> IngestReport:
    report = IngestReport()
    it = iter(docs)
    while batch := list(islice(it, DOC_BATCH)):
        part = _ingest_batch(batch, defenses, backend, embedder, chunking, precomputed, count, now)
        report.merge(part)
        if on_batch:
            on_batch(report)
    return report


def _ingest_batch(
    docs: list[RawDocument],
    defenses: list[Defense],
    backend: IndexBackend,
    embedder: Embedder,
    chunking: ChunkingConfig,
    precomputed: Mapping[str, np.ndarray] | None,
    count: TokenCounter | None,
    now: Callable[[], datetime],
) -> IngestReport:
    report = IngestReport(n_docs=len(docs))
    chunks: list[Chunk] = []
    holds: dict[str, list[DefenseDecision]] = {}
    for doc in docs:
        doc_chunks, per_chunk, decisions = prepare_document(doc, defenses, chunking, count)
        chunks += doc_chunks
        holds.update(per_chunk)
        report.decisions += decisions
    report.n_chunks = len(chunks)

    existing = backend.existing_hashes([c.content_hash for c in chunks])
    seen: set[str] = set()
    fresh: list[Chunk] = []
    for c in chunks:
        if c.content_hash in existing or c.content_hash in seen:
            report.n_skipped_existing += 1
            continue
        seen.add(c.content_hash)
        fresh.append(c)
    if not fresh:
        return report

    vectors = _embed(fresh, embedder, precomputed, report)
    main_idx = [i for i, c in enumerate(fresh) if c.chunk_id not in holds]
    held_idx = [i for i, c in enumerate(fresh) if c.chunk_id in holds]
    if main_idx:
        report.n_indexed = backend.add([fresh[i] for i in main_idx], vectors[main_idx])
    for i in held_idx:
        c = fresh[i]
        backend.quarantine(
            QuarantineRecord(chunk=c, decisions=holds[c.chunk_id], quarantined_at=now()),
            vectors[i],
        )
        report.quarantined_chunk_ids.append(c.chunk_id)
    report.n_quarantined = len(held_idx)
    return report


def _embed(
    chunks: list[Chunk],
    embedder: Embedder,
    precomputed: Mapping[str, np.ndarray] | None,
    report: IngestReport,
) -> np.ndarray:
    vectors = np.zeros((len(chunks), embedder.dimension), dtype=np.float32)
    todo = []
    for i, c in enumerate(chunks):
        vec = precomputed.get(c.content_hash) if precomputed is not None else None
        if vec is None:
            todo.append(i)
        else:
            vectors[i] = vec
    report.n_reused_embeddings = len(chunks) - len(todo)
    if todo:
        vectors[todo] = embedder.embed_passages([chunks[i].text for i in todo])
    report.n_embedded = len(todo)
    return vectors
