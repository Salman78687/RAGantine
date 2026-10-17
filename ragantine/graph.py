"""LangGraph query pipeline (SPEC §B4.5).

Nodes in order: retrieve -> filter_context (D2 retrieval-time, D3, D6) -> build_prompt
(D0 or D4) -> generate -> verify (D5) -> finalize. A node whose defenses are disabled is
a pass-through, so the graph shape never changes between configurations.

``run_query`` (sync), ``arun_query`` (async) and ``stream_query`` (async generator of
events) are the only public entry points used by the API, CLI and benchmark.
"""

from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Annotated, Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from ragantine.config import AppConfig, ModelsConfig, Settings
from ragantine.defenses.base import Defense, QueryContext, build_defenses, parse_defense_ids
from ragantine.generate.llm import LLMClient, LLMResult
from ragantine.generate.prompting import baseline_messages, parse_citations
from ragantine.index.base import IndexBackend
from ragantine.retrieve import HybridRetriever
from ragantine.retrieve.dense import Embedder
from ragantine.retrieve.reranker import Reranker
from ragantine.schemas import Answer, DefenseDecision, RetrievedChunk
from ragantine.tracing import NoopTracer, Tracer, make_tracer

Event = tuple[str, dict[str, Any]]


class QueryState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    model_alias: str
    enabled_defenses: list[str] = Field(default_factory=list)
    stream: bool = False
    started: float = 0.0
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    filtered: list[RetrievedChunk] = Field(default_factory=list)
    prompt: list[dict[str, str]] = Field(default_factory=list)
    draft: str = ""
    final_text: str = ""
    answer: Answer | None = None
    decisions: Annotated[list[DefenseDecision], operator.add] = Field(default_factory=list)
    llm_results: Annotated[list[LLMResult], operator.add] = Field(default_factory=list)


@dataclass
class Pipeline:
    """Everything a query needs. Build once, reuse for many queries."""

    cfg: AppConfig
    llm: LLMClient
    retriever: HybridRetriever
    _defenses: dict[str, Defense] = field(default_factory=dict)
    _graph: Any = None
    tracer: Tracer = field(default_factory=NoopTracer)

    @classmethod
    def from_parts(
        cls,
        cfg: AppConfig,
        llm: LLMClient,
        backend: IndexBackend,
        embedder: Embedder,
        reranker: Reranker,
    ) -> Pipeline:
        return cls(cfg, llm, HybridRetriever(backend, embedder, reranker, cfg.retrieval))

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        models: ModelsConfig,
        settings: Settings | None = None,
        index_name: str | None = None,
        quarantine_name: str | None = None,
        use_cache: bool = True,
    ) -> Pipeline:
        """Production wiring: OpenSearch + bge embedder/reranker + hosted LLMs."""
        from ragantine.index.opensearch import OpenSearchIndex
        from ragantine.retrieve.dense import SentenceTransformerEmbedder
        from ragantine.retrieve.reranker import CrossEncoderReranker

        settings = settings or Settings()
        backend = OpenSearchIndex.connect(
            settings.opensearch_url,
            cfg.index,
            cfg.embedding.dimension,
            name=index_name or cfg.index.name,
            quarantine_name=quarantine_name or cfg.index.quarantine_name,
        )
        llm = LLMClient(models, settings, cache_dir=cfg.paths.llm_cache_dir, use_cache=use_cache)
        pipeline = cls.from_parts(
            cfg,
            llm,
            backend,
            SentenceTransformerEmbedder(cfg.embedding),
            CrossEncoderReranker(cfg.retrieval.rerank_model),
        )
        pipeline.tracer = make_tracer(settings)
        return pipeline

    @property
    def backend(self) -> IndexBackend:
        return self.retriever.backend

    def defenses(self, ids: Sequence[str]) -> list[Defense]:
        """Instances for the enabled ids, in id order (built once and cached)."""
        missing = [d for d in ids if d not in self._defenses]
        for d in build_defenses(missing, self.cfg, self.llm):
            self._defenses[d.id] = d
        return [self._defenses[d] for d in sorted(ids)]

    def set_defense(self, defense: Defense) -> None:
        """Inject a pre-built defense instance (tests, demo snapshots)."""
        self._defenses[defense.id] = defense

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = _build_graph(self)
        return self._graph


# ---------------------------------------------------------------------- graph nodes


def _emit(event: str, data: dict[str, Any]) -> None:
    get_stream_writer()((event, data))


def _emit_decisions(decisions: list[DefenseDecision]) -> None:
    for d in decisions:
        _emit("defense", d.model_dump(mode="json"))


def build_messages(
    p: Pipeline, s: QueryState, hits: list[RetrievedChunk], note: str | None = None
) -> list[dict[str, str]]:
    messages = baseline_messages(s.question, hits)
    for d in p.defenses(s.enabled_defenses):
        messages = d.on_prompt(s.question, hits, messages)
    if note:
        messages = [*messages[:-1], {"role": "system", "content": note}, messages[-1]]
    return messages


def _build_graph(p: Pipeline) -> Any:
    async def retrieve(state: QueryState) -> dict[str, Any]:
        s = state
        hits = await asyncio.to_thread(p.retriever.search, s.question)
        _emit(
            "retrieval",
            {
                "chunks": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "title": h.chunk.title,
                        "source_tier": h.chunk.source_tier.value,
                        "final_score": h.final_score,
                    }
                    for h in hits[: p.cfg.retrieval.final_k]
                ]
            },
        )
        return {"retrieved": hits}

    async def filter_context(state: QueryState) -> dict[str, Any]:
        s = state
        hits = list(s.retrieved)
        decisions: list[DefenseDecision] = []
        for d in p.defenses(s.enabled_defenses):
            hits, dec = await asyncio.to_thread(d.on_retrieve, s.question, hits)
            _emit_decisions(dec)
            decisions += dec
        return {"filtered": hits[: p.cfg.retrieval.final_k], "decisions": decisions}

    async def build_prompt(state: QueryState) -> dict[str, Any]:
        s = state
        return {"prompt": build_messages(p, s, s.filtered)}

    async def generate(state: QueryState) -> dict[str, Any]:
        s = state
        if not s.stream:
            result = await p.llm.achat(s.model_alias, s.prompt)
            return {"draft": result.text, "llm_results": [result]}
        final: LLMResult | None = None
        async for item in p.llm.astream(s.model_alias, s.prompt):
            if isinstance(item, str):
                _emit("token", {"text": item})
            else:
                final = item
        assert final is not None
        return {"draft": final.text, "llm_results": [final]}

    async def verify(state: QueryState) -> dict[str, Any]:
        s = state
        verifiers = [
            d for d in p.defenses(s.enabled_defenses) if type(d).on_verify is not Defense.on_verify
        ]
        if not verifiers:
            return {"final_text": s.draft}

        def regenerate(hits: list[RetrievedChunk], note: str | None) -> LLMResult:
            result = p.llm.chat(s.model_alias, build_messages(p, s, hits, note))
            ctx.llm_results.append(result)  # counted in the answer's token totals
            return result

        ctx = QueryContext(model_alias=s.model_alias, regenerate=regenerate)
        text = s.draft
        decisions: list[DefenseDecision] = []
        for d in verifiers:
            text, dec = await asyncio.to_thread(d.on_verify, s.question, s.filtered, text, ctx)
            _emit_decisions(dec)
            decisions += dec
        return {"final_text": text, "decisions": decisions, "llm_results": ctx.llm_results}

    async def finalize(state: QueryState) -> dict[str, Any]:
        s = state
        answer = Answer(
            question=s.question,
            text=s.final_text,
            citations=parse_citations(s.final_text, s.filtered),
            decisions=s.decisions,
            retrieved=s.filtered,
            model_alias=s.model_alias,
            latency_ms=int((time.perf_counter() - s.started) * 1000),
            prompt_tokens=sum(r.prompt_tokens for r in s.llm_results),
            completion_tokens=sum(r.completion_tokens for r in s.llm_results),
            candidate_ids=[h.chunk.chunk_id for h in s.retrieved],
            draft=s.draft,
        )
        trace_id = await asyncio.to_thread(p.tracer.record, answer, s.enabled_defenses)
        return {"answer": answer.model_copy(update={"trace_id": trace_id})}

    g = StateGraph(QueryState)
    g.add_node("retrieve", retrieve)
    g.add_node("filter_context", filter_context)
    g.add_node("build_prompt", build_prompt)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    order = ["retrieve", "filter_context", "build_prompt", "generate", "verify", "finalize"]
    g.add_edge(START, order[0])
    for a, b in pairwise(order):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()


# ---------------------------------------------------------------------- entry points


def _initial_state(
    p: Pipeline,
    question: str,
    defenses: str | Sequence[str] | None,
    model_alias: str | None,
    stream: bool,
) -> QueryState:
    return QueryState(
        question=question,
        model_alias=model_alias or p.llm.models.defaults.generator,
        enabled_defenses=parse_defense_ids(
            defenses if defenses is not None else p.cfg.defenses.enabled
        ),
        stream=stream,
        started=time.perf_counter(),
    )


async def arun_query(
    question: str,
    pipeline: Pipeline,
    defenses: str | Sequence[str] | None = None,
    model_alias: str | None = None,
) -> Answer:
    state = _initial_state(pipeline, question, defenses, model_alias, stream=False)
    out = await pipeline.graph.ainvoke(state)
    answer: Answer = out["answer"]
    return answer


def run_query(
    question: str,
    pipeline: Pipeline,
    defenses: str | Sequence[str] | None = None,
    model_alias: str | None = None,
) -> Answer:
    """Synchronous entry point (do not call from inside a running event loop)."""
    return asyncio.run(arun_query(question, pipeline, defenses, model_alias))


async def stream_query(
    question: str,
    pipeline: Pipeline,
    defenses: str | Sequence[str] | None = None,
    model_alias: str | None = None,
) -> AsyncIterator[Event]:
    """Yield (event, data): retrieval, defense*, token*, then done (Answer) or error."""
    try:
        state = _initial_state(pipeline, question, defenses, model_alias, stream=True)
        answer: Answer | None = None
        async for mode, chunk in pipeline.graph.astream(state, stream_mode=["custom", "values"]):
            if mode == "custom":
                event, data = chunk
                yield event, data
            elif chunk.get("answer") is not None:
                answer = chunk["answer"]
        if answer is None:
            raise RuntimeError("query graph finished without an answer")
        yield "done", answer.model_dump(mode="json")
    except Exception as exc:  # surfaced to SSE clients as an error event
        yield "error", {"message": f"{type(exc).__name__}: {exc}"}


async def asafe_retrieve(
    question: str, pipeline: Pipeline, defenses: str | Sequence[str] | None = None
) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
    """Retrieval + retrieval-time defenses only (no LLM): the chunks a prompt would get."""
    hits = await asyncio.to_thread(pipeline.retriever.search, question)
    decisions: list[DefenseDecision] = []
    for d in pipeline.defenses(parse_defense_ids(defenses)):
        hits, dec = await asyncio.to_thread(d.on_retrieve, question, hits)
        decisions += dec
    return hits[: pipeline.cfg.retrieval.final_k], decisions
