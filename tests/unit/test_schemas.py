"""Validation of the data contracts in ragantine/schemas.py and bench/schemas.py (SPEC §B3)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from bench.schemas import BenchQuestion, JudgedAnswer, PoisonSample
from ragantine.schemas import (
    Answer,
    Chunk,
    DefenseDecision,
    QuarantineRecord,
    RawDocument,
    RetrievedChunk,
    SourceTier,
)

NOW = datetime(2026, 10, 17, tzinfo=UTC)


def _chunk(**over: Any) -> Chunk:
    fields: dict[str, Any] = {
        "chunk_id": "abcdef0123456789:0003",
        "doc_id": "abcdef0123456789",
        "title": "Mount Everest",
        "text": "Title: Mount Everest\n\nMount Everest is Earth's highest mountain.",
        "ordinal": 3,
        "n_tokens": 12,
        "source_tier": "curated",
        "content_hash": "0" * 64,
        "ingested_at": NOW,
    }
    fields.update(over)
    return Chunk(**fields)


def _doc(**over: Any) -> RawDocument:
    fields: dict[str, Any] = {
        "doc_id": "abcdef0123456789",
        "title": "Mount Everest",
        "source_uri": "local://bench/poison/x",
        "fmt": "html",
        "raw": "<p>text</p>",
        "source_tier": "user_upload",
        "ingested_at": NOW,
    }
    fields.update(over)
    return RawDocument(**fields)


def _answer() -> Answer:
    chunk = _chunk()
    return Answer(
        question="How tall is Mount Everest?",
        text="8,849 m [abcdef0123456789:0003].",
        citations=[{"chunk_id": chunk.chunk_id, "title": chunk.title, "source_tier": "curated"}],
        decisions=[],
        retrieved=[RetrievedChunk(chunk=chunk, bm25_rank=1, rrf_score=0.03, final_score=0.9)],
        model_alias="llama-large",
        latency_ms=420,
        prompt_tokens=900,
        completion_tokens=20,
    )


def test_chunk_valid_and_tier_coerced() -> None:
    chunk = _chunk()
    assert chunk.source_tier is SourceTier.CURATED
    assert chunk.source_tier == "curated"


def test_chunk_id_must_match_doc_and_ordinal() -> None:
    with pytest.raises(ValidationError, match="chunk_id must be"):
        _chunk(chunk_id="abcdef0123456789:3")


def test_unknown_source_tier_rejected() -> None:
    with pytest.raises(ValidationError):
        _chunk(source_tier="trusted")


def test_raw_document_accepts_bytes_and_rejects_unknown_format() -> None:
    assert _doc(fmt="pdf", raw=b"%PDF-1.7").raw == b"%PDF-1.7"
    with pytest.raises(ValidationError):
        _doc(fmt="docx")


def test_defense_decision_literals() -> None:
    d = DefenseDecision(
        defense_id="d1",
        stage="ingestion",
        target_id="abcdef0123456789:0000",
        action="quarantine",
        score=1.0,
        reason="hidden_content",
    )
    record = QuarantineRecord(chunk=_chunk(), decisions=[d], quarantined_at=NOW)
    assert record.status == "quarantined" and record.released_by is None
    with pytest.raises(ValidationError):
        DefenseDecision(
            defense_id="d7", stage="ingestion", target_id="x", action="allow", reason=""
        )
    with pytest.raises(ValidationError):
        DefenseDecision(
            defense_id="d1", stage="ingestion", target_id="x", action="delete", reason=""
        )


def test_answer_json_round_trip() -> None:
    answer = _answer()
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


def test_bench_question_set_rules() -> None:
    base: dict[str, Any] = {
        "qid": "q1",
        "question": "How tall is Mount Everest?",
        "gold_answer": "8,849 m",
        "gold_chunk_id": "abcdef0123456789:0003",
        "split": "dev",
    }
    assert BenchQuestion(**base, qset="clean").gold_aliases == []
    assert BenchQuestion(**base, qset="target", attack="A1").attack == "A1"
    BenchQuestion(**base, qset="paraphrase", attack="A1", parent_qid="q0")
    with pytest.raises(ValidationError):
        BenchQuestion(**base, qset="clean", attack="A1")
    with pytest.raises(ValidationError):
        BenchQuestion(**base, qset="target")
    with pytest.raises(ValidationError):
        BenchQuestion(**base, qset="paraphrase", attack="A1")


def test_poison_sample_goal_payload_and_template() -> None:
    ok = PoisonSample(
        poison_id="p1",
        qid="q1",
        attack="A2",
        goal="canary",
        canary="ZEBRA-7741",
        template_id="A2-t05",
        document=_doc(),
    )
    assert ok.target_false_answer is None
    with pytest.raises(ValidationError, match="requires canary"):
        PoisonSample(
            poison_id="p",
            qid="q",
            attack="A2",
            goal="canary",
            template_id="A2-t05",
            document=_doc(),
        )
    with pytest.raises(ValidationError, match="does not belong"):
        PoisonSample(
            poison_id="p",
            qid="q",
            attack="A1",
            goal="false_answer",
            target_false_answer="9,100 m",
            template_id="A2-t05",
            document=_doc(),
        )
    with pytest.raises(ValidationError):
        PoisonSample(
            poison_id="p",
            qid="q",
            attack="A1",
            goal="false_answer",
            target_false_answer="9,100 m",
            template_id="t05",
            document=_doc(),
        )


def test_judged_answer() -> None:
    judged = JudgedAnswer(
        qid="q1",
        run_id="r1",
        seed=1337,
        answer=_answer(),
        verdict="correct",
        judge_reason="contains gold",
        matched_by="rule",
    )
    assert judged.verdict == "correct"
    with pytest.raises(ValidationError):
        JudgedAnswer(**{**judged.model_dump(), "verdict": "maybe"})
