"""Workspaces build (or reuse) the index for a run and hand back a ready pipeline.

``OpenSearchWorkspace`` is the real thing: one index pair per ingestion config + poison set,
with a manifest under data/indexes/ so later runs reuse it without re-ingesting.
``MemoryWorkspace`` runs fully in-process for tests and the CI smoke benchmark.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from bench.corpus.build import corpus_paths, load_articles, load_embedding_lookup
from bench.metrics import IngestionFacts
from bench.run import RunSpec
from bench.schemas import PoisonSample
from ragantine.config import AppConfig, ModelsConfig, Settings
from ragantine.defenses.base import Defense, build_defenses
from ragantine.defenses.d6_anomaly import AnomalyDetector, build_hubs
from ragantine.generate.llm import LLMClient
from ragantine.graph import Pipeline
from ragantine.index.base import IndexBackend
from ragantine.index.naming import index_names, ingestion_defenses
from ragantine.ingest.chunker import TokenCounter
from ragantine.ingest.pipeline import ingest
from ragantine.retrieve import HybridRetriever, Retriever
from ragantine.retrieve.dense import Embedder
from ragantine.retrieve.memory import MemoryIndex
from ragantine.retrieve.reranker import Reranker
from ragantine.schemas import RawDocument


def poison_variant(spec: RunSpec, poison: Sequence[PoisonSample]) -> str:
    ids = "\n".join(sorted(p.poison_id for p in poison))
    tag = f"-{spec.index_tag}" if spec.index_tag else ""
    return f"poison-{spec.split}-{spec.seed}-{hashlib.sha256(ids.encode()).hexdigest()[:8]}{tag}"


class _DefensePool:
    """One instance per defense id, shared across runs so models load once."""

    def __init__(
        self, cfg: AppConfig, llm: LLMClient, overrides: dict[str, Defense] | None = None
    ) -> None:
        self.cfg, self.llm = cfg, llm
        self.instances: dict[str, Defense] = dict(overrides or {})
        self._hubs: dict[str, dict[str, float]] = {}

    def get(self, ids: Sequence[str]) -> list[Defense]:
        missing = [d for d in ids if d not in self.instances]
        for d in build_defenses(missing, self.cfg, self.llm):
            self.instances[d.id] = d
        return [self.instances[d] for d in sorted(ids)]

    def prepare_d6(
        self,
        ids: Sequence[str],
        backend: IndexBackend,
        embedder: Embedder,
        titles: Sequence[str],
        key: str,
    ) -> None:
        """D6 needs hub statistics of the exact index it guards; build them once per index."""
        if "d6" not in ids:
            return
        d6 = self.get(["d6"])[0]
        assert isinstance(d6, AnomalyDetector)
        if key not in self._hubs:
            self._hubs[key] = build_hubs(backend, embedder, titles, d6.cfg, self.cfg.seed)
        d6.hubs = self._hubs[key]

    def pipeline(self, retriever: Retriever, ids: Sequence[str]) -> Pipeline:
        p = Pipeline(self.cfg, self.llm, retriever)
        for d in self.get(ids):
            p.set_defense(d)
        return p


class OpenSearchWorkspace:
    def __init__(
        self,
        cfg: AppConfig,
        models: ModelsConfig,
        settings: Settings,
        use_cache: bool = True,
        log: Callable[[str], None] = print,
    ) -> None:
        from ragantine.retrieve.dense import SentenceTransformerEmbedder
        from ragantine.retrieve.reranker import CrossEncoderReranker

        self.cfg, self.settings, self.log = cfg, settings, log
        self.llm = LLMClient(
            models, settings, cache_dir=cfg.paths.llm_cache_dir, use_cache=use_cache
        )
        self.embedder: Embedder = SentenceTransformerEmbedder(cfg.embedding)
        self.reranker: Reranker = CrossEncoderReranker(cfg.retrieval.rerank_model)
        self.defenses = _DefensePool(cfg, self.llm)

    def manifest_path(self, index_name: str) -> Path:
        return self.cfg.paths.data_dir / "indexes" / f"{index_name}.json"

    def prepare(
        self, spec: RunSpec, poison: Sequence[PoisonSample]
    ) -> tuple[Pipeline, IngestionFacts]:
        from ragantine.index.opensearch import OpenSearchIndex

        ingestion_ids = ingestion_defenses(self.cfg, spec.defenses)
        names = index_names(self.cfg, ingestion_ids, poison_variant(spec, poison))
        backend = OpenSearchIndex.connect(
            self.settings.opensearch_url, self.cfg.index, self.cfg.embedding.dimension,
            name=names[0], quarantine_name=names[1],
        )  # fmt: skip
        manifest = self.manifest_path(names[0])
        if manifest.exists() and backend.client.indices.exists(index=names[0]):
            self.log(f"reusing index {names[0]}")
            facts = IngestionFacts.model_validate_json(manifest.read_text())
        else:
            self.log(f"building index {names[0]} (ingestion defenses: {ingestion_ids or 'none'})")
            backend.drop()
            backend.ensure()
            docs: list[RawDocument] = [*load_articles(self.cfg), *(p.document for p in poison)]
            report = ingest(
                docs, self.defenses.get(ingestion_ids), backend=backend, embedder=self.embedder,
                chunking=self.cfg.chunking, precomputed=load_embedding_lookup(corpus_paths(self.cfg)),
                on_batch=lambda r: self.log(f"  ingested {r.n_docs}/{len(docs)} docs"),
            )  # fmt: skip
            facts = IngestionFacts(doc_chunks=report.doc_chunks, decisions=report.decisions)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(facts.model_dump_json())
        self.defenses.prepare_d6(
            spec.defenses,
            backend,
            self.embedder,
            [d.title for d in load_articles(self.cfg)],
            names[0],
        )
        retriever = HybridRetriever(backend, self.embedder, self.reranker, self.cfg.retrieval)
        return self.defenses.pipeline(retriever, spec.defenses), facts


class MemoryWorkspace:
    def __init__(
        self,
        cfg: AppConfig,
        llm: LLMClient,
        docs: Sequence[RawDocument],
        embedder: Embedder,
        reranker: Reranker,
        defense_overrides: dict[str, Defense] | None = None,
        count: TokenCounter | None = None,
    ) -> None:
        self.cfg, self.llm, self.docs = cfg, llm, list(docs)
        self.embedder, self.reranker, self.count = embedder, reranker, count
        self.defenses = _DefensePool(cfg, llm, defense_overrides)
        self._indexes: dict[tuple[str, ...], tuple[MemoryIndex, IngestionFacts]] = {}

    def prepare(
        self, spec: RunSpec, poison: Sequence[PoisonSample]
    ) -> tuple[Pipeline, IngestionFacts]:
        ingestion_ids = ingestion_defenses(self.cfg, spec.defenses)
        key = (*ingestion_ids, poison_variant(spec, poison))
        if key not in self._indexes:
            backend = MemoryIndex(self.embedder.dimension)
            report = ingest(
                [*self.docs, *(p.document for p in poison)], self.defenses.get(ingestion_ids),
                backend=backend, embedder=self.embedder, chunking=self.cfg.chunking, count=self.count,
            )  # fmt: skip
            self._indexes[key] = (
                backend,
                IngestionFacts(doc_chunks=report.doc_chunks, decisions=report.decisions),
            )
        backend, facts = self._indexes[key]
        titles = [d.title for d in self.docs]
        self.defenses.prepare_d6(spec.defenses, backend, self.embedder, titles, "|".join(key))
        retriever = HybridRetriever(backend, self.embedder, self.reranker, self.cfg.retrieval)
        return self.defenses.pipeline(retriever, spec.defenses), facts


class AdapterWorkspace:
    """Benchmark an external retriever (FR-10, see ragantine.adapters).

    ``load_documents(docs)`` must REPLACE the external store's contents with ``docs`` and keep
    each document's ``doc_id``/``source_tier`` in its metadata. Ingestion defenses screen
    documents before they are handed over; a document with any quarantined chunk is withheld
    whole, because the external store's chunking is unknown.
    """

    def __init__(
        self,
        cfg: AppConfig,
        llm: LLMClient,
        retriever: Retriever,
        load_documents: Callable[[list[RawDocument]], None],
        corpus: Sequence[RawDocument],
    ) -> None:
        self.cfg, self.llm, self.retriever = cfg, llm, retriever
        self.load_documents, self.corpus = load_documents, list(corpus)
        self.defenses = _DefensePool(cfg, llm)
        self._loaded: str | None = None
        self._facts: IngestionFacts | None = None

    def prepare(
        self, spec: RunSpec, poison: Sequence[PoisonSample]
    ) -> tuple[Pipeline, IngestionFacts]:
        from ragantine.ingest.pipeline import prepare_document

        ingestion_ids = ingestion_defenses(self.cfg, spec.defenses)
        key = "|".join([*ingestion_ids, poison_variant(spec, poison)])
        if key != self._loaded or self._facts is None:
            active = self.defenses.get(ingestion_ids)
            keep: list[RawDocument] = []
            doc_chunks: dict[str, list[str]] = {}
            decisions = []
            for doc in [*self.corpus, *(p.document for p in poison)]:
                chunks, held, dec = prepare_document(doc, active, self.cfg.chunking)
                doc_chunks[doc.doc_id] = [c.chunk_id for c in chunks]
                decisions += dec
                if held:
                    decisions += [
                        d.model_copy(update={"target_id": doc.doc_id})
                        for d in dec
                        if d.action == "quarantine"
                    ]
                else:
                    keep.append(doc)
            self.load_documents(keep)
            self._loaded = key
            self._facts = IngestionFacts(doc_chunks=doc_chunks, decisions=decisions)
        return self.defenses.pipeline(self.retriever, spec.defenses), self._facts
