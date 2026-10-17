"""Index builds for the benchmark: clean corpus plus optional poison docs (SPEC §B5.4)."""

from __future__ import annotations

from collections.abc import Iterable

from bench.corpus.build import corpus_paths, load_articles, load_embedding_lookup
from bench.store import load_poison_documents, poison_set_fingerprint
from ragantine.config import AppConfig, Settings
from ragantine.defenses.base import build_defenses, parse_defense_ids
from ragantine.index.naming import index_names, ingestion_defenses
from ragantine.index.opensearch import OpenSearchIndex
from ragantine.ingest.pipeline import IngestReport, ingest
from ragantine.retrieve.dense import Embedder, SentenceTransformerEmbedder
from ragantine.schemas import RawDocument


def poison_variant(cfg: AppConfig, split: str, seed: int) -> str:
    """Name of the document-set variant 'clean corpus + test poison of <split> for <seed>'."""
    return f"poison-{split}-{seed}-{poison_set_fingerprint(cfg, split, seed)}"


def build_index(
    cfg: AppConfig,
    settings: Settings,
    defenses: str | Iterable[str],
    *,
    with_poison: bool,
    split: str = "test",
    seed: int | None = None,
    drop: bool = False,
    embedder: Embedder | None = None,
) -> tuple[IngestReport, tuple[str, str]]:
    ids = parse_defense_ids(defenses)
    ingestion_ids = ingestion_defenses(cfg, ids)
    seed = cfg.seed if seed is None else seed
    variant = poison_variant(cfg, split, seed) if with_poison else None
    names = index_names(cfg, ingestion_ids, variant)
    backend = OpenSearchIndex.connect(
        settings.opensearch_url,
        cfg.index,
        cfg.embedding.dimension,
        name=names[0],
        quarantine_name=names[1],
    )
    if drop:
        backend.drop()
    backend.ensure()
    docs: list[RawDocument] = load_articles(cfg)
    if with_poison:
        docs += load_poison_documents(cfg, split, seed)
    report = ingest(
        docs,
        build_defenses(ingestion_ids, cfg),
        backend=backend,
        embedder=embedder or SentenceTransformerEmbedder(cfg.embedding),
        chunking=cfg.chunking,
        precomputed=load_embedding_lookup(corpus_paths(cfg)),
    )
    return report, names
