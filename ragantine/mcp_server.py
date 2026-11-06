"""MCP server exposing RAGantine to agents (FR-15, P2).

Tools:
* ``safe_retrieve(question, defenses)``: hybrid retrieval through the retrieval-time
  defenses (D2 re-score, D3 trust, D6 hubs). Quarantined chunks are never returned.
* ``audit_corpus(documents, defenses)``: run the ingestion defenses (D1 sanitizer, D2
  classifier) over documents an agent is about to add, without indexing them, and report
  what would be quarantined and why.

Run over stdio: ``ragantine mcp``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from ragantine.graph import Pipeline, asafe_retrieve
from ragantine.ingest.pipeline import prepare_document
from ragantine.local_models import run_model
from ragantine.schemas import RawDocument, SourceTier

DEFAULT_RETRIEVAL_DEFENSES = ["d2", "d3"]
DEFAULT_INGESTION_DEFENSES = ["d1", "d2"]
MAX_AUDIT_DOCS = 50


class AuditDocument(BaseModel):
    title: str = Field(max_length=300)
    text: str = Field(max_length=200_000, description="Raw document content")
    fmt: Literal["text", "html", "markdown"] = "text"
    source_tier: SourceTier = SourceTier.USER_UPLOAD


def build_server(get_pipeline: Callable[[], Pipeline]) -> MCPServer:
    server = MCPServer(
        name="ragantine",
        instructions="Poisoning-aware retrieval. Use safe_retrieve for context and audit_corpus before adding documents.",
    )

    @server.tool(
        description="Retrieve context for a question through RAGantine's retrieval-time defenses."
    )
    async def safe_retrieve(question: str, defenses: list[str] | None = None) -> dict[str, Any]:
        hits, decisions = await asafe_retrieve(
            question,
            get_pipeline(),
            defenses if defenses is not None else DEFAULT_RETRIEVAL_DEFENSES,
        )
        return {
            "chunks": [
                {"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "source_tier": h.chunk.source_tier.value,
                 "score": h.final_score, "text": h.chunk.text}
                for h in hits
            ],
            "decisions": [d.model_dump(mode="json") for d in decisions],
        }  # fmt: skip

    @server.tool(
        description="Audit documents with the ingestion defenses (no indexing); reports quarantines."
    )
    async def audit_corpus(
        documents: list[AuditDocument], defenses: list[str] | None = None
    ) -> dict[str, Any]:
        if len(documents) > MAX_AUDIT_DOCS:
            raise ValueError(f"at most {MAX_AUDIT_DOCS} documents per call")
        pipeline = get_pipeline()
        active = pipeline.defenses(defenses if defenses is not None else DEFAULT_INGESTION_DEFENSES)
        report = []
        for doc in documents:
            uri = f"audit://{hashlib.sha1((doc.title + doc.text).encode()).hexdigest()[:16]}"
            raw = RawDocument(
                doc_id=hashlib.sha1(uri.encode()).hexdigest()[:16], title=doc.title, source_uri=uri,
                fmt=doc.fmt, raw=doc.text, source_tier=doc.source_tier, ingested_at=datetime.now(UTC),
            )  # fmt: skip
            chunks, held, decisions = await run_model(
                prepare_document, raw, active, pipeline.cfg.chunking
            )
            report.append(
                {"title": doc.title, "chunks": len(chunks), "quarantined_chunks": len(held),
                 "would_quarantine": bool(held), "decisions": [d.model_dump(mode="json") for d in decisions]}
            )  # fmt: skip
        return {"documents": report, "flagged": sum(r["would_quarantine"] for r in report)}

    return server


def main() -> None:
    from ragantine.config import Settings, load_config, load_models_config

    state: dict[str, Pipeline] = {}

    def get_pipeline() -> Pipeline:
        if "p" not in state:
            state["p"] = Pipeline.from_config(load_config(), load_models_config(), Settings())
        return state["p"]

    build_server(get_pipeline).run("stdio")


if __name__ == "__main__":
    main()
