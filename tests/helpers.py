"""Shared builders for unit/component tests (no network, no OpenSearch)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from ragantine.config import AppConfig, load_config
from ragantine.defenses.base import Defense
from ragantine.fakes import FakeLLM, HashingEmbedder, OverlapReranker
from ragantine.graph import Pipeline
from ragantine.ingest.pipeline import IngestReport, ingest
from ragantine.retrieve.memory import MemoryIndex
from ragantine.schemas import RawDocument, SourceTier

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2023, 11, 1, tzinfo=UTC)


def word_count(text: str) -> int:
    """Whitespace token counter: deterministic and offline (tiktoken downloads its vocab)."""
    return len(text.split())


def small_config() -> AppConfig:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return cfg.model_copy(
        update={
            "chunking": cfg.chunking.model_copy(
                update={"target_tokens": 60, "overlap_tokens": 10, "min_tokens": 5}
            ),
            "embedding": cfg.embedding.model_copy(update={"dimension": 64}),
            "retrieval": cfg.retrieval.model_copy(
                update={"bm25_k": 10, "knn_k": 10, "fused_k": 8, "final_k": 3}
            ),
        }
    )


def make_doc(
    title: str,
    raw: str | bytes,
    fmt: str = "text",
    tier: SourceTier | str = SourceTier.CURATED,
) -> RawDocument:
    uri = f"fixture://{title.lower().replace(' ', '-')}"
    return RawDocument(
        doc_id=hashlib.sha1(uri.encode()).hexdigest()[:16],
        title=title,
        source_uri=uri,
        fmt=fmt,  # type: ignore[arg-type]
        raw=raw,
        source_tier=SourceTier(tier),
        ingested_at=NOW,
    )


SAMPLE_DOCS = [
    make_doc(
        "Mount Everest",
        "Mount Everest is the highest mountain on Earth above sea level. It stands at 8,849 metres "
        "on the border between Nepal and China.\n\nThe first confirmed ascent was made in 1953 by "
        "Edmund Hillary and Tenzing Norgay.",
    ),
    make_doc(
        "Eiffel Tower",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in "
        "1889 as the entrance arch to the World's Fair.\n\nThe tower was designed by the company of "
        "Gustave Eiffel and is about 330 metres tall.",
    ),
    make_doc(
        "Lake Baikal",
        "Lake Baikal is a rift lake in Siberia, Russia. It is the deepest lake in the world, with a "
        "maximum depth of about 1,642 metres.\n\nBaikal holds roughly a fifth of the world's "
        "unfrozen fresh surface water.",
        tier=SourceTier.PARTNER,
    ),
]


def memory_pipeline(
    llm: FakeLLM,
    docs: list[RawDocument] | None = None,
    defenses: list[Defense] | None = None,
    cfg: AppConfig | None = None,
) -> tuple[Pipeline, IngestReport]:
    cfg = cfg or small_config()
    backend = MemoryIndex(cfg.embedding.dimension)
    embedder = HashingEmbedder(cfg.embedding.dimension)
    report = ingest(
        docs if docs is not None else SAMPLE_DOCS,
        defenses or [],
        backend=backend,
        embedder=embedder,
        chunking=cfg.chunking,
        count=word_count,
    )
    return Pipeline.from_parts(cfg, llm, backend, embedder, OverlapReranker()), report
