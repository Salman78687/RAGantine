"""RAGAS-style clean-set quality with the lexical NLI stand-in."""

from bench.quality import compute_quality
from bench.schemas import BenchQuestion, JudgedAnswer
from ragantine.fakes import LexicalNLI
from ragantine.schemas import Answer
from tests.defense_helpers import chunk, hit

GOLD = chunk("g", "The Eiffel Tower was completed in 1889.", "curated", title="Eiffel Tower")
OTHER = chunk("o", "Lake Baikal is in Siberia.", "curated", title="Lake Baikal")


def _q(qid: str) -> BenchQuestion:
    return BenchQuestion(qid=qid, question="When was the Eiffel Tower completed?", gold_answer="1889",
                         gold_chunk_id=GOLD.chunk_id, qset="clean", split="test")  # fmt: skip


def _j(qid: str, text: str, hits: list) -> JudgedAnswer:  # type: ignore[type-arg]
    a = Answer(question="q", text=text, citations=[], decisions=[], retrieved=hits, model_alias="m",
               latency_ms=1, prompt_tokens=1, completion_tokens=1)  # fmt: skip
    return JudgedAnswer(
        qid=qid, run_id="r", seed=1, answer=a, verdict="correct", judge_reason="", matched_by="rule"
    )


def test_context_recall_and_faithfulness() -> None:
    qs = [_q("a"), _q("b"), _q("c")]
    judged = [
        _j("a", "It was completed in 1889.", [hit(GOLD, 1.0)]),  # faithful, gold retrieved
        _j("b", "It was completed in 1901.", [hit(OTHER, 1.0)]),  # unfaithful, gold missed
        _j("c", "I don't know.", [hit(GOLD, 1.0)]),  # refusal: excluded from faithfulness
    ]
    q = compute_quality(qs, judged, LexicalNLI(), 0.7)
    assert q.context_recall == 2 / 3 and q.faithfulness == 0.5 and q.n_answered == 2
