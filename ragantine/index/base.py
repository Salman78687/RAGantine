"""Index backend interface shared by OpenSearch and the in-memory fallback (SPEC §B4.2, §B7.3).

The backend holds the main chunk index (BM25 + k-NN over the same documents) and the
quarantine store. No ground-truth field ever crosses this interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ragantine.schemas import Chunk, QuarantineRecord


class IndexBackend(Protocol):
    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        """content_hashes already present in the main or quarantine index (idempotent ingest)."""
        ...

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Index chunks with their embeddings; returns the number written."""
        ...

    def count(self) -> int: ...

    def bm25(self, query: str, k: int) -> list[Chunk]:
        """Top-k chunks by BM25 over ``title^2, text``, best first."""
        ...

    def knn(self, vector: np.ndarray, k: int) -> list[Chunk]:
        """Top-k chunks by cosine similarity, best first."""
        ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def sample(self, n: int, seed: int) -> list[Chunk]:
        """Deterministic sample of up to n chunks from the main index."""
        ...

    # ---- quarantine store
    def quarantine(self, record: QuarantineRecord, vector: np.ndarray) -> None: ...

    def list_quarantine(
        self, status: str | None, reason: str | None, limit: int, offset: int
    ) -> tuple[list[QuarantineRecord], int]: ...

    def get_quarantine(self, chunk_id: str) -> QuarantineRecord | None: ...

    def release(self, chunk_id: str, released_by: str) -> QuarantineRecord | None:
        """Move a quarantined chunk back into the main index and mark it released."""
        ...
