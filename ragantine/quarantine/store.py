"""Quarantine review service (SPEC §6, FR-7).

Flagged chunks are never silently deleted: they sit in the quarantine index with their
reasons and scores, excluded from answers, until a human releases them. Releases are
logged with ``released_by``.
"""

from __future__ import annotations

import logging

from ragantine.index.base import IndexBackend
from ragantine.schemas import QuarantineRecord

log = logging.getLogger("ragantine.quarantine")
MAX_PAGE = 200


class QuarantineService:
    def __init__(self, backend: IndexBackend) -> None:
        self.backend = backend

    def list(
        self, status: str | None = None, reason: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[QuarantineRecord], int]:
        if not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE}")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        return self.backend.list_quarantine(status, reason, limit, offset)

    def get(self, chunk_id: str) -> QuarantineRecord | None:
        return self.backend.get_quarantine(chunk_id)

    def release(self, chunk_id: str, released_by: str) -> QuarantineRecord | None:
        if not released_by.strip():
            raise ValueError("released_by is required")
        record = self.backend.release(chunk_id, released_by.strip())
        if record is not None:
            log.info(
                "released chunk %s from quarantine (released_by=%s)", chunk_id, released_by.strip()
            )
        return record
