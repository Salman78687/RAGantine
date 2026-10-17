"""Pydantic v2 data contracts passed across every module boundary (SPEC §B3).

Benchmark-only models (questions, poison ground truth, judged answers) live in
``bench/schemas.py`` so that nothing in ``ragantine/`` can see ground truth.
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator


class SourceTier(StrEnum):
    CURATED = "curated"
    PARTNER = "partner"
    WEB = "web"
    USER_UPLOAD = "user_upload"


DefenseId = Literal["d1", "d2", "d3", "d4", "d5", "d6"]
Stage = Literal["ingestion", "retrieval", "generation", "verification"]
DefenseAction = Literal["allow", "quarantine", "filter", "downweight", "rewrite", "hedge"]


class RawDocument(BaseModel):
    doc_id: str  # sha1(source_uri)[:16]
    title: str
    source_uri: str
    fmt: Literal["text", "html", "markdown", "pdf"]
    raw: str | bytes  # original content, untouched
    source_tier: SourceTier
    ingested_at: datetime

    # Binary formats (pdf) round-trip through JSON as base64 strings.
    @field_serializer("raw", when_used="json")
    def _raw_to_json(self, raw: str | bytes) -> str:
        return base64.b64encode(raw).decode("ascii") if isinstance(raw, bytes) else raw

    @model_validator(mode="before")
    @classmethod
    def _raw_from_json(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("fmt") == "pdf" and isinstance(data.get("raw"), str):
            data = {**data, "raw": base64.b64decode(data["raw"])}
        return data


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}:{ordinal:04d}"
    doc_id: str
    title: str
    text: str  # text that goes to the LLM
    ordinal: int = Field(ge=0)
    n_tokens: int = Field(ge=0)
    source_tier: SourceTier
    content_hash: str  # sha256(text)
    ingested_at: datetime

    @model_validator(mode="after")
    def _chunk_id_matches(self) -> Chunk:
        expected = f"{self.doc_id}:{self.ordinal:04d}"
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id must be {expected!r}, got {self.chunk_id!r}")
        return self


class RetrievedChunk(BaseModel):
    chunk: Chunk
    bm25_rank: int | None = None
    knn_rank: int | None = None
    rrf_score: float
    rerank_score: float | None = None
    final_score: float  # after D3 re-weighting if enabled


class DefenseDecision(BaseModel):
    defense_id: DefenseId
    stage: Stage
    target_id: str  # chunk_id or "answer"
    action: DefenseAction
    score: float | None = None
    reason: str


class QuarantineRecord(BaseModel):
    chunk: Chunk
    decisions: list[DefenseDecision]
    quarantined_at: datetime
    status: Literal["quarantined", "released"] = "quarantined"
    released_by: str | None = None


class Citation(BaseModel):
    chunk_id: str
    title: str
    source_tier: SourceTier


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation]
    decisions: list[DefenseDecision]
    retrieved: list[RetrievedChunk]
    model_alias: str
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    trace_id: str | None = None
    # Answer trace (FR-9): every fused candidate considered before retrieval-time defenses,
    # and the draft before D5 verification (equal to `text` unless D5 rewrote it).
    candidate_ids: list[str] = Field(default_factory=list)
    draft: str | None = None
