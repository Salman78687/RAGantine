"""Per-request traces in Langfuse: retrieval, defense decisions, tokens, latency (SPEC §8).

``make_tracer`` returns a no-op tracer unless both Langfuse keys are set, so the pipeline
never needs a network connection or an account to run.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ragantine.config import Settings
from ragantine.schemas import Answer

log = logging.getLogger("ragantine.tracing")


class Tracer(Protocol):
    def record(self, answer: Answer, defenses: list[str]) -> str | None:
        """Send one trace for a finished answer; returns the trace id (None when disabled)."""
        ...


class NoopTracer:
    def record(self, answer: Answer, defenses: list[str]) -> str | None:
        return None


class LangfuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse

        self.client: Any = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    def record(self, answer: Answer, defenses: list[str]) -> str | None:
        try:
            root = self.client.start_observation(
                name="ragantine.query",
                as_type="chain",
                input={
                    "question": answer.question,
                    "defenses": defenses,
                    "model": answer.model_alias,
                },
                output={"text": answer.text, "citations": [c.chunk_id for c in answer.citations]},
                metadata={"latency_ms": answer.latency_ms, "candidates": len(answer.candidate_ids)},
            )
            root.start_observation(
                name="retrieve", as_type="retriever",
                output=[{"chunk_id": h.chunk.chunk_id, "tier": h.chunk.source_tier.value,
                         "final_score": h.final_score} for h in answer.retrieved],
            ).end()  # fmt: skip
            for d in answer.decisions:
                root.start_observation(
                    name=f"defense.{d.defense_id}",
                    as_type="guardrail",
                    output=d.model_dump(mode="json"),
                ).end()
            root.start_observation(
                name="generate", as_type="generation", model=answer.model_alias,
                output=answer.text, usage_details={"input": answer.prompt_tokens, "output": answer.completion_tokens},
            ).end()  # fmt: skip
            root.end()
            return str(root.trace_id)
        except Exception as exc:  # tracing must never break answering
            log.warning("langfuse trace failed: %s", type(exc).__name__)
            return None


def make_tracer(settings: Settings) -> Tracer:
    public = settings.api_key("LANGFUSE_PUBLIC_KEY")
    secret = settings.api_key("LANGFUSE_SECRET_KEY")
    if public and secret:
        return LangfuseTracer(public, secret, settings.langfuse_host)
    return NoopTracer()
