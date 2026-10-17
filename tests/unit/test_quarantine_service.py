"""Quarantine service: list/filter/paginate/release with audit field (SPEC §6, FR-7)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from ragantine.quarantine import QuarantineService
from ragantine.retrieve.memory import MemoryIndex
from ragantine.schemas import DefenseDecision, QuarantineRecord
from tests.defense_helpers import chunk


def _service() -> QuarantineService:
    idx = MemoryIndex(4)
    for i, reason in enumerate(
        ["hidden_content: x", "injection_classifier: p=0.99", "hidden_content: y"]
    ):
        c = chunk(f"{i}", f"text {i}")
        d = DefenseDecision(
            defense_id="d1",
            stage="ingestion",
            target_id=c.chunk_id,
            action="quarantine",
            reason=reason,
        )
        idx.quarantine(
            QuarantineRecord(
                chunk=c, decisions=[d], quarantined_at=datetime(2026, 1, i + 1, tzinfo=UTC)
            ),
            np.ones(4),
        )
    return QuarantineService(idx)


def test_list_filters_and_paginates() -> None:
    svc = _service()
    items, total = svc.list(reason="hidden_content")
    assert total == 2 and all(i.decisions[0].reason.startswith("hidden_content") for i in items)
    page, total = svc.list(limit=1, offset=1)
    assert total == 3 and len(page) == 1
    with pytest.raises(ValueError):
        svc.list(limit=500)


def test_release_requires_and_records_reviewer() -> None:
    svc = _service()
    cid = svc.list()[0][0].chunk.chunk_id
    with pytest.raises(ValueError):
        svc.release(cid, "  ")
    rec = svc.release(cid, "alice")
    assert rec is not None and rec.status == "released" and rec.released_by == "alice"
    assert svc.list(status="released")[1] == 1
    assert svc.backend.get(cid) is not None
    assert svc.get("nope:0000") is None
