"""Query graph end-to-end with FakeLLM + in-memory index (SPEC §B4.5, §B8)."""

import asyncio

import pytest

from ragantine.fakes import FakeLLM
from ragantine.graph import arun_query, run_query, stream_query
from tests.helpers import SAMPLE_DOCS, memory_pipeline


def _cite_first_chunk(messages: list[dict[str, str]]) -> str:
    # the context lists chunks as "[<chunk_id>] Title: ...": cite the first one
    system = messages[0]["content"]
    cid = system.split("CONTEXT:\n[", 1)[1].split("]", 1)[0]
    return f"Mount Everest is 8,849 metres tall [{cid}]."


def test_run_query_returns_cited_answer() -> None:
    llm = FakeLLM(rules=[("How tall is Mount Everest", _cite_first_chunk)])
    pipeline, report = memory_pipeline(llm)
    assert report.n_indexed > 0 and report.n_quarantined == 0

    answer = run_query("How tall is Mount Everest?", pipeline)

    assert "8,849" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].title == "Mount Everest"
    assert answer.retrieved and answer.retrieved[0].chunk.title == "Mount Everest"
    assert len(answer.retrieved) <= pipeline.cfg.retrieval.final_k
    assert len(answer.candidate_ids) >= len(answer.retrieved)
    assert answer.decisions == [] and answer.draft == answer.text
    assert answer.prompt_tokens > 0 and answer.completion_tokens > 0
    assert answer.model_alias == "fake"


def test_prompt_is_the_baseline_template() -> None:
    llm = FakeLLM()
    pipeline, _ = memory_pipeline(llm)
    run_query("Where is Lake Baikal?", pipeline)
    system, user = llm.prompts[-1]
    assert system["role"] == "system" and user["role"] == "user"
    assert "cite chunk ids like [doc:0003]" in system["content"]
    assert "Lake Baikal" in system["content"]
    assert user["content"] == "Where is Lake Baikal?"


def test_ranks_and_scores_are_filled_for_tracing() -> None:
    pipeline, _ = memory_pipeline(FakeLLM())
    hits = pipeline.retriever.search("deepest lake Siberia")
    assert hits[0].chunk.title == "Lake Baikal"
    for h in hits:
        assert h.rrf_score > 0 and h.rerank_score is not None and h.final_score == h.rerank_score
        assert h.bm25_rank is not None or h.knn_rank is not None
    assert [h.final_score for h in hits] == sorted((h.final_score for h in hits), reverse=True)


def test_unknown_defense_rejected() -> None:
    pipeline, _ = memory_pipeline(FakeLLM())
    with pytest.raises(ValueError, match="unknown defense"):
        run_query("q", pipeline, defenses="d9")


async def test_stream_query_event_order() -> None:
    llm = FakeLLM(rules=[("Eiffel", "It was completed in 1889.")])
    pipeline, _ = memory_pipeline(llm)
    events = [e async for e in stream_query("When was the Eiffel Tower completed?", pipeline)]
    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert "".join(d["text"] for n, d in events if n == "token") == "It was completed in 1889."
    assert events[-1][1]["text"] == "It was completed in 1889."
    assert {"chunk_id", "title", "source_tier", "final_score"} <= set(events[0][1]["chunks"][0])


async def test_stream_query_reports_errors_as_events() -> None:
    def boom(_messages: list[dict[str, str]]) -> str:
        raise RuntimeError("model exploded")

    pipeline, _ = memory_pipeline(FakeLLM(default=boom))
    events = [e async for e in stream_query("anything", pipeline)]
    assert events[-1][0] == "error" and "model exploded" in events[-1][1]["message"]


async def test_concurrent_queries() -> None:
    pipeline, _ = memory_pipeline(FakeLLM(default="ok"))
    answers = await asyncio.gather(*(arun_query(f"question {i}", pipeline) for i in range(8)))
    assert [a.text for a in answers] == ["ok"] * 8


def test_corpus_docs_fixture_is_stable() -> None:
    assert [d.title for d in SAMPLE_DOCS] == ["Mount Everest", "Eiffel Tower", "Lake Baikal"]
