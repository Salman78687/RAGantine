"""Real OpenSearch round-trip (run on demand: `make up` then `pytest -m integration`)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pytest

from ragantine.fakes import HashingEmbedder, OverlapReranker
from ragantine.index.opensearch import OpenSearchIndex, index_body
from ragantine.ingest.pipeline import ingest
from ragantine.retrieve import HybridRetriever
from ragantine.schemas import DefenseDecision, QuarantineRecord
from tests.helpers import SAMPLE_DOCS, small_config, word_count

pytestmark = pytest.mark.integration
CFG = small_config()


@pytest.fixture
def index() -> Iterator[OpenSearchIndex]:
    url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    tag = uuid.uuid4().hex[:8]
    idx = OpenSearchIndex.connect(url, CFG.index, CFG.embedding.dimension,
                                  name=f"ragantine_test_{tag}", quarantine_name=f"ragantine_test_q_{tag}")  # fmt: skip
    if not idx.healthy():
        pytest.skip(f"OpenSearch not reachable at {url}")
    idx.ensure()
    yield idx
    idx.drop()


def test_mapping_has_no_ground_truth_fields() -> None:
    body = str(index_body(CFG.index, 768, quarantine=True))
    for field in ("is_poison", "attack", "canary", "target_false_answer"):
        assert f"'{field}'" not in body


def test_ingest_search_and_idempotency(index: OpenSearchIndex) -> None:
    emb = HashingEmbedder(CFG.embedding.dimension)
    report = ingest(
        SAMPLE_DOCS, [], backend=index, embedder=emb, chunking=CFG.chunking, count=word_count
    )
    assert index.count() == report.n_indexed > 0
    again = ingest(
        SAMPLE_DOCS, [], backend=index, embedder=emb, chunking=CFG.chunking, count=word_count
    )
    assert again.n_indexed == 0
    assert index.bm25("deepest lake Siberia", 3)[0].title == "Lake Baikal"
    assert index.knn(emb.embed_query("Eiffel Tower Paris"), 3)
    hits = HybridRetriever(index, emb, OverlapReranker(), CFG.retrieval).search(
        "How tall is Mount Everest?"
    )
    assert hits[0].chunk.title == "Mount Everest" and hits[0].bm25_rank is not None
    assert index.get(hits[0].chunk.chunk_id) == hits[0].chunk


def test_quarantine_round_trip(index: OpenSearchIndex) -> None:
    emb = HashingEmbedder(CFG.embedding.dimension)
    ingest(
        SAMPLE_DOCS[:1], [], backend=index, embedder=emb, chunking=CFG.chunking, count=word_count
    )
    chunk = index.bm25("Everest", 1)[0]
    decision = DefenseDecision(defense_id="d2", stage="ingestion", target_id=chunk.chunk_id, action="quarantine",
                               score=0.99, reason="injection_classifier: p=0.990")  # fmt: skip
    index.quarantine(
        QuarantineRecord(chunk=chunk, decisions=[decision], quarantined_at=datetime.now(UTC)),
        np.ones(CFG.embedding.dimension),
    )
    assert index.get(chunk.chunk_id) is None
    items, total = index.list_quarantine("quarantined", "injection_classifier", 10, 0)
    assert total == 1 and items[0].decisions[0].score == 0.99
    released = index.release(chunk.chunk_id, "reviewer")
    assert released is not None and released.released_by == "reviewer"
    assert index.get(chunk.chunk_id) is not None
    assert index.list_quarantine("released", None, 10, 0)[1] == 1
