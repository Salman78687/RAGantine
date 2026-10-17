"""D3 source trust scoring (SPEC §B6).

``final_score = rerank_score_normalised x tier_weights[source_tier]`` (min-max over the
query's candidates). After re-sorting, the final top-``final_k`` holds at most
``max_low_tier_in_final`` chunks from ``web`` + ``user_upload``, filling from higher tiers
first, so low-trust chunks can never be the only evidence for an answer. Low-tier chunks
that would have been in the final set but were pushed out get a ``downweight`` decision.
"""

from __future__ import annotations

from typing import Any

from ragantine.defenses.base import Defense, register
from ragantine.generate.llm import LLMClient
from ragantine.schemas import DefenseDecision, RetrievedChunk, SourceTier

LOW_TIERS = frozenset({SourceTier.WEB, SourceTier.USER_UPLOAD})


@register
class SourceTrust(Defense):
    id = "d3"

    def __init__(self, cfg: dict[str, Any], llm: LLMClient | None = None) -> None:
        super().__init__(cfg, llm)
        self.weights = {SourceTier(k): float(v) for k, v in cfg["tier_weights"].items()}
        self.cap = int(cfg["max_low_tier_in_final"])
        self.final_k = int(cfg["final_k"])

    def on_retrieve(
        self, q: str, hits: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
        if not hits:
            return hits, []
        raw = [h.rerank_score if h.rerank_score is not None else h.final_score for h in hits]
        lo, hi = min(raw), max(raw)
        span = hi - lo
        rescored = [
            h.model_copy(
                update={
                    "final_score": ((r - lo) / span if span else 1.0)
                    * self.weights.get(h.chunk.source_tier, 0.0)
                }
            )
            for h, r in zip(hits, raw, strict=True)
        ]
        rescored.sort(key=lambda h: -h.final_score)

        final: list[RetrievedChunk] = []
        rest: list[RetrievedChunk] = []
        low = 0
        for h in rescored:
            is_low = h.chunk.source_tier in LOW_TIERS
            if len(final) < self.final_k and (not is_low or low < self.cap):
                final.append(h)
                low += is_low
            elif not is_low or len(final) >= self.final_k:
                rest.append(h)  # kept as backfill for later filters (D6) if still eligible
        kept = {h.chunk.chunk_id for h in final}
        would_have_been = sorted(hits, key=lambda h: -(h.rerank_score or h.final_score))[
            : self.final_k
        ]
        decisions = [
            self.decision(
                stage="retrieval",
                target_id=h.chunk.chunk_id,
                action="downweight",
                score=next(x.final_score for x in rescored if x.chunk.chunk_id == h.chunk.chunk_id),
                reason=f"low_trust_tier: {h.chunk.source_tier.value} chunk pushed out of the final set",
            )
            for h in would_have_been
            if h.chunk.source_tier in LOW_TIERS and h.chunk.chunk_id not in kept
        ]
        backfill = [h for h in rest if h.chunk.source_tier not in LOW_TIERS]
        return final + backfill, decisions
