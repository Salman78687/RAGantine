"""LLM call and cost estimates printed before any paid run (SPEC §B7.2)."""

from __future__ import annotations

from pydantic import BaseModel

from ragantine.config import ModelsConfig


class CallEstimate(BaseModel):
    purpose: str
    model_alias: str
    calls: int
    prompt_tokens_each: int
    completion_tokens_each: int


class CostEstimate(BaseModel):
    items: list[CallEstimate]
    total_calls: int
    total_usd: float | None  # None if a model has no price configured

    def render(self) -> str:
        lines = [f"Estimated LLM usage ({self.total_calls} calls):"]
        for i in self.items:
            lines.append(
                f"  {i.calls:6d} x {i.model_alias:14s} {i.purpose} "
                f"(~{i.prompt_tokens_each}+{i.completion_tokens_each} tokens each)"
            )
        usd = "unknown (no price set)" if self.total_usd is None else f"${self.total_usd:.2f}"
        lines.append(f"  estimated cost: {usd} (before cache hits)")
        return "\n".join(lines)


def estimate(models: ModelsConfig, items: list[CallEstimate]) -> CostEstimate:
    total: float | None = 0.0
    for i in items:
        cost = models.models[i.model_alias].cost_usd(
            i.calls * i.prompt_tokens_each, i.calls * i.completion_tokens_each
        )
        total = None if cost is None or total is None else total + cost
    return CostEstimate(items=items, total_calls=sum(i.calls for i in items), total_usd=total)
