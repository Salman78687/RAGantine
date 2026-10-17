"""In-process fallback index (rank_bm25 + numpy cosine) for the HF Spaces demo and tests (SPEC §B7.3).

Implements the same ``IndexBackend`` interface as the OpenSearch index, including the
quarantine store, so the query graph cannot tell the two apart.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ragantine.schemas import Chunk, QuarantineRecord

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class MemoryIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._chunks: list[Chunk] = []
        self._pos: dict[str, int] = {}
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._bm25: Any = None
        self._quarantine: dict[str, tuple[QuarantineRecord, np.ndarray]] = {}

    # ------------------------------------------------------------------ main index

    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        present = {c.content_hash for c in self._chunks}
        present |= {r.chunk.content_hash for r, _ in self._quarantine.values()}
        return present & set(hashes)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        base = len(self._vectors)
        new_rows: list[np.ndarray] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            row = np.asarray(vec, dtype=np.float32)
            i = self._pos.get(chunk.chunk_id)
            if i is None:  # new chunk
                self._pos[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                new_rows.append(row)
            elif i < base:  # update a stored chunk
                self._chunks[i] = chunk
                self._vectors[i] = row
            else:  # repeated within this batch: last write wins
                self._chunks[i] = chunk
                new_rows[i - base] = row
        if new_rows:
            self._vectors = np.vstack([self._vectors, np.stack(new_rows)])
        self._bm25 = None
        return len(chunks)

    def count(self) -> int:
        return len(self._chunks)

    def _bm25_model(self) -> Any:
        if self._bm25 is None and self._chunks:
            from rank_bm25 import BM25Okapi

            # title^2: title tokens counted twice, like the OpenSearch multi_match boost.
            corpus = [_tokens(c.title) * 2 + _tokens(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def bm25(self, query: str, k: int) -> list[Chunk]:
        model = self._bm25_model()
        q = _tokens(query)
        if model is None or not q:
            return []
        scores = np.asarray(model.get_scores(q))
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._chunks[i] for i in order if scores[i] > 0]

    def knn(self, vector: np.ndarray, k: int) -> list[Chunk]:
        if not self._chunks:
            return []
        v = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1) * (np.linalg.norm(v) or 1.0)
        sims = (self._vectors @ v) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-sims, kind="stable")[:k]
        return [self._chunks[i] for i in order]

    def get(self, chunk_id: str) -> Chunk | None:
        i = self._pos.get(chunk_id)
        return None if i is None else self._chunks[i]

    def vector(self, chunk_id: str) -> np.ndarray | None:
        i = self._pos.get(chunk_id)
        return None if i is None else self._vectors[i]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def sample(self, n: int, seed: int) -> list[Chunk]:
        def key(c: Chunk) -> str:
            return hashlib.sha1(f"{seed}:{c.chunk_id}".encode()).hexdigest()

        return sorted(self._chunks, key=key)[:n]

    def _remove(self, chunk_id: str) -> None:
        i = self._pos.pop(chunk_id, None)
        if i is None:
            return
        del self._chunks[i]
        self._vectors = np.delete(self._vectors, i, axis=0)
        self._pos = {c.chunk_id: j for j, c in enumerate(self._chunks)}
        self._bm25 = None

    # ------------------------------------------------------------------ quarantine store

    def quarantine(self, record: QuarantineRecord, vector: np.ndarray) -> None:
        self._remove(record.chunk.chunk_id)
        self._quarantine[record.chunk.chunk_id] = (record, np.asarray(vector, dtype=np.float32))

    def list_quarantine(
        self, status: str | None, reason: str | None, limit: int, offset: int
    ) -> tuple[list[QuarantineRecord], int]:
        items = [
            r
            for r, _ in sorted(self._quarantine.values(), key=lambda rv: rv[0].quarantined_at)
            if (status is None or r.status == status)
            and (reason is None or any(d.reason.startswith(reason) for d in r.decisions))
        ]
        return items[offset : offset + limit], len(items)

    def get_quarantine(self, chunk_id: str) -> QuarantineRecord | None:
        hit = self._quarantine.get(chunk_id)
        return None if hit is None else hit[0]

    def release(self, chunk_id: str, released_by: str) -> QuarantineRecord | None:
        hit = self._quarantine.get(chunk_id)
        if hit is None:
            return None
        record, vector = hit
        released = record.model_copy(update={"status": "released", "released_by": released_by})
        self._quarantine[chunk_id] = (released, vector)
        self.add([record.chunk], vector[None, :])
        return released
