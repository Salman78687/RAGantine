"""OpenSearch main and quarantine indexes: mappings, bulk indexing, BM25 and k-NN (SPEC §B4.2).

BM25 and k-NN live in the same index. The quarantine index uses the same mapping plus
``decisions`` (stored, not indexed), ``status`` and audit fields. Neither mapping has any
ground-truth field (see tests/unit/test_leakage.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from opensearchpy import OpenSearch, helpers

from ragantine.config import IndexConfig
from ragantine.schemas import Chunk, QuarantineRecord

BASE_PROPERTIES: dict[str, Any] = {
    "chunk_id": {"type": "keyword"},
    "doc_id": {"type": "keyword"},
    "title": {"type": "text"},
    "text": {"type": "text"},
    "ordinal": {"type": "integer"},
    "source_tier": {"type": "keyword"},
    "ingested_at": {"type": "date"},
    "n_tokens": {"type": "integer"},
    "content_hash": {"type": "keyword"},
}

QUARANTINE_EXTRA_PROPERTIES: dict[str, Any] = {
    "decisions": {"type": "object", "enabled": False},
    "reasons": {"type": "keyword"},  # decision reasons, indexed so the review queue can filter
    "status": {"type": "keyword"},
    "quarantined_at": {"type": "date"},
    "released_by": {"type": "keyword"},
}

_CHUNK_FIELDS = list(Chunk.model_fields)
BULK_CHUNK_SIZE = 500


def index_body(cfg: IndexConfig, dimension: int, quarantine: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        **BASE_PROPERTIES,
        "embedding": {
            "type": "knn_vector",
            "dimension": dimension,
            "method": {
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "cosinesimil",
                "parameters": {"m": cfg.hnsw.m, "ef_construction": cfg.hnsw.ef_construction},
            },
        },
    }
    if quarantine:
        properties.update(QUARANTINE_EXTRA_PROPERTIES)
    return {
        "settings": {
            "index": {"knn": True, "knn.algo_param.ef_search": cfg.hnsw.ef_search},
            "analysis": {"analyzer": {"default": {"type": "english"}}},
        },
        "mappings": {"properties": properties},
    }


def _chunk_from_source(src: dict[str, Any]) -> Chunk:
    return Chunk.model_validate({k: src[k] for k in _CHUNK_FIELDS})


class OpenSearchIndex:
    """``IndexBackend`` over a single-node OpenSearch (security plugin disabled for dev)."""

    def __init__(
        self,
        client: OpenSearch,
        cfg: IndexConfig,
        dimension: int,
        name: str | None = None,
        quarantine_name: str | None = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.dimension = dimension
        self.name = name or cfg.name
        self.quarantine_name = quarantine_name or cfg.quarantine_name

    @classmethod
    def connect(cls, url: str, cfg: IndexConfig, dimension: int, **names: str) -> OpenSearchIndex:
        client = OpenSearch(hosts=[url], timeout=60, http_compress=True)
        return cls(client, cfg, dimension, **names)

    # ------------------------------------------------------------------ lifecycle

    def ensure(self) -> None:
        for name, quarantine in ((self.name, False), (self.quarantine_name, True)):
            if not self.client.indices.exists(index=name):
                self.client.indices.create(
                    index=name, body=index_body(self.cfg, self.dimension, quarantine)
                )

    def drop(self) -> None:
        for name in (self.name, self.quarantine_name):
            if self.client.indices.exists(index=name):
                self.client.indices.delete(index=name)

    def refresh(self) -> None:
        self.client.indices.refresh(index=f"{self.name},{self.quarantine_name}")

    # ------------------------------------------------------------------ main index

    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        found: set[str] = set()
        unique = list(dict.fromkeys(hashes))
        for i in range(0, len(unique), 1000):
            batch = unique[i : i + 1000]
            resp = self.client.search(
                index=f"{self.name},{self.quarantine_name}",
                body={
                    "size": len(batch),
                    "_source": ["content_hash"],
                    "query": {"terms": {"content_hash": batch}},
                },
                params={"ignore_unavailable": "true"},
            )
            found |= {h["_source"]["content_hash"] for h in resp["hits"]["hits"]}
        return found

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        actions = (
            {
                "_index": self.name,
                "_id": c.chunk_id,
                "_source": {**c.model_dump(mode="json"), "embedding": v.tolist()},
            }
            for c, v in zip(chunks, vectors, strict=True)
        )
        ok, _ = helpers.bulk(self.client, actions, chunk_size=BULK_CHUNK_SIZE, refresh=True)
        return int(ok)

    def count(self) -> int:
        return int(self.client.count(index=self.name)["count"])

    def healthy(self) -> bool:
        try:
            status = self.client.cluster.health(request_timeout=2)["status"]
        except Exception:
            return False
        return status in ("green", "yellow")

    def _hits(self, body: dict[str, Any]) -> list[Chunk]:
        body = {**body, "_source": {"excludes": ["embedding"]}}
        resp = self.client.search(index=self.name, body=body)
        return [_chunk_from_source(h["_source"]) for h in resp["hits"]["hits"]]

    def bm25(self, query: str, k: int) -> list[Chunk]:
        return self._hits(
            {"size": k, "query": {"multi_match": {"query": query, "fields": ["title^2", "text"]}}}
        )

    def knn(self, vector: np.ndarray, k: int) -> list[Chunk]:
        return self._hits(
            {"size": k, "query": {"knn": {"embedding": {"vector": vector.tolist(), "k": k}}}}
        )

    def get(self, chunk_id: str) -> Chunk | None:
        resp = self.client.get(index=self.name, id=chunk_id, params={"ignore": 404})
        return _chunk_from_source(resp["_source"]) if resp.get("found") else None

    def sample(self, n: int, seed: int) -> list[Chunk]:
        return self._hits(
            {
                "size": n,
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "random_score": {"seed": seed, "field": "_seq_no"},
                    }
                },
            }
        )

    # ------------------------------------------------------------------ quarantine store

    def quarantine(self, record: QuarantineRecord, vector: np.ndarray) -> None:
        self.client.index(
            index=self.quarantine_name,
            id=record.chunk.chunk_id,
            body=self._quarantine_doc(record, vector),
            params={"refresh": "true"},
        )
        self.client.delete(
            index=self.name, id=record.chunk.chunk_id, params={"ignore": 404, "refresh": "true"}
        )

    @staticmethod
    def _quarantine_doc(record: QuarantineRecord, vector: np.ndarray) -> dict[str, Any]:
        return {
            **record.chunk.model_dump(mode="json"),
            "embedding": np.asarray(vector).tolist(),
            "decisions": [d.model_dump(mode="json") for d in record.decisions],
            "reasons": sorted({d.reason.split(":")[0] for d in record.decisions}),
            "status": record.status,
            "quarantined_at": record.quarantined_at.isoformat(),
            "released_by": record.released_by,
        }

    @staticmethod
    def _record(src: dict[str, Any]) -> QuarantineRecord:
        return QuarantineRecord.model_validate(
            {
                "chunk": _chunk_from_source(src),
                "decisions": src["decisions"],
                "quarantined_at": src["quarantined_at"],
                "status": src["status"],
                "released_by": src.get("released_by"),
            }
        )

    def list_quarantine(
        self, status: str | None, reason: str | None, limit: int, offset: int
    ) -> tuple[list[QuarantineRecord], int]:
        filters: list[dict[str, Any]] = []
        if status:
            filters.append({"term": {"status": status}})
        if reason:
            filters.append({"prefix": {"reasons": reason}})
        resp = self.client.search(
            index=self.quarantine_name,
            body={
                "from": offset,
                "size": limit,
                "track_total_hits": True,
                "sort": [{"quarantined_at": "asc"}, {"chunk_id": "asc"}],
                "_source": {"excludes": ["embedding"]},
                "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
            },
        )
        items = [self._record(h["_source"]) for h in resp["hits"]["hits"]]
        return items, int(resp["hits"]["total"]["value"])

    def get_quarantine(self, chunk_id: str) -> QuarantineRecord | None:
        resp = self.client.get(index=self.quarantine_name, id=chunk_id, params={"ignore": 404})
        return self._record(resp["_source"]) if resp.get("found") else None

    def release(self, chunk_id: str, released_by: str) -> QuarantineRecord | None:
        resp = self.client.get(index=self.quarantine_name, id=chunk_id, params={"ignore": 404})
        if not resp.get("found"):
            return None
        src = resp["_source"]
        record = self._record(src).model_copy(
            update={"status": "released", "released_by": released_by}
        )
        vector = np.asarray(src["embedding"], dtype=np.float32)
        self.add([record.chunk], vector[None, :])
        self.client.update(
            index=self.quarantine_name,
            id=chunk_id,
            body={
                "doc": {
                    "status": "released",
                    "released_by": released_by,
                    "released_at": datetime.now(UTC).isoformat(),
                }
            },
            params={"refresh": "true"},
        )
        return record
