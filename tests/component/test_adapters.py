"""FR-10: benchmark an external retriever through adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bench.run import RunSpec, execute_run
from bench.workspace import AdapterWorkspace
from ragantine.adapters import AdapterRetriever, ExternalHit, to_retrieved
from ragantine.fakes import fake_models_config
from ragantine.schemas import RawDocument, SourceTier
from tests.fakes_bench import build_bench_data


class ToyStore:
    """An 'external' keyword store: what a user's own vector DB would be."""

    def __init__(self) -> None:
        self.docs: list[RawDocument] = []

    def load(self, docs: list[RawDocument]) -> None:
        self.docs = list(docs)

    def search(self, q: str) -> list[ExternalHit]:
        words = set(q.lower().replace("?", "").split())

        def score(d: RawDocument) -> float:
            title = set(d.title.lower().split())
            return 5 * len(words & title) + len(words & set(str(d.raw).lower().split())) / 50

        ranked = sorted(self.docs, key=score, reverse=True)[:3]
        return [ExternalHit(text=f"Title: {d.title}\n\n{d.raw}", doc_id=d.doc_id, title=d.title,
                            source_tier=d.source_tier) for d in ranked]  # fmt: skip


def test_to_retrieved_keeps_ragantine_doc_ids_and_ranks() -> None:
    hits = to_retrieved(
        [
            ExternalHit(text="a", doc_id="0123456789abcdef", score=0.5),
            ExternalHit(text="b", doc_id="ext-7"),
        ]
    )
    assert hits[0].chunk.chunk_id == "0123456789abcdef:0000" and hits[0].final_score == 0.5
    assert (
        hits[1].chunk.doc_id != "ext-7"
        and len(hits[1].chunk.doc_id) == 16
        and hits[1].final_score == 0.5
    )
    assert hits[1].chunk.source_tier is SourceTier.WEB  # unknown provenance is untrusted


def test_langchain_and_llamaindex_duck_types() -> None:
    lc = SimpleNamespace(
        invoke=lambda q: [
            SimpleNamespace(
                page_content="Eiffel 1889",
                metadata={"doc_id": "x1", "title": "Eiffel", "source_tier": "curated"},
            )
        ]
    )
    node = SimpleNamespace(get_content=lambda: "Baikal 1642 m", metadata={"title": "Baikal"})
    li = SimpleNamespace(retrieve=lambda q: [SimpleNamespace(node=node, score=0.8)])
    a = AdapterRetriever.from_langchain(lc).search("q")[0]
    b = AdapterRetriever.from_llamaindex(li).search("q")[0]
    assert (a.chunk.title, a.chunk.source_tier) == ("Eiffel", SourceTier.CURATED)
    assert (b.chunk.text, b.final_score) == ("Baikal 1642 m", 0.8)


def test_benchmark_runs_on_an_external_store(tmp_path: Path) -> None:
    cfg, llm, docs, questions, poison = build_bench_data(limit=12)
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"results_dir": tmp_path})})
    store = ToyStore()
    ws = AdapterWorkspace(cfg, llm, AdapterRetriever(store.search), store.load, docs)
    results: dict[str, Any] = {}
    for name, ids in (("d0", []), ("d1", ["d1"])):
        spec = RunSpec(
            run_id=f"adapter-{name}",
            defenses=ids,
            model_alias="fake",
            sets=["target", "clean"],
            split="test",
            seed=cfg.seed,
        )  # type: ignore[arg-type]
        results[name] = asyncio.run(
            execute_run(cfg, fake_models_config(), spec, questions, poison, ws, log=lambda _m: None)
        )[0]
    base, d1 = results["d0"], results["d1"]
    assert (
        base.asr.n == 6 and base.conditional_asr.n > 0
    )  # poison tracked through the adapter by doc_id
    assert d1.asr_by_attack["A5"].k <= base.asr_by_attack["A5"].k
    assert len(store.docs) < len(docs) + len(poison)  # D1 withheld hidden-text documents
