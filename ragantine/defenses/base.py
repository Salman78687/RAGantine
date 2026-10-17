"""``Defense`` abstract base class, hook signatures and the id registry (SPEC §B6).

Every hook has a pass-through default; a defense overrides only the hooks it uses and
explains each block with a ``DefenseDecision``. Enabled defenses run in id order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from ragantine.config import DefensesConfig
from ragantine.generate.llm import LLMClient, LLMResult
from ragantine.schemas import Chunk, DefenseDecision, RawDocument, RetrievedChunk

Messages = list[dict[str, str]]


@dataclass
class QueryContext:
    """Per-query handles a generation-stage defense may need (e.g. D5 regenerating)."""

    model_alias: str
    regenerate: Callable[[list[RetrievedChunk], str | None], LLMResult]
    llm_results: list[LLMResult] = field(default_factory=list)


class Defense:
    id: ClassVar[str]

    def __init__(self, cfg: dict[str, Any], llm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.llm = llm

    # every hook has a pass-through default; override only what you use
    def on_load(self, doc: RawDocument, text: str) -> tuple[str, list[DefenseDecision]]:
        return text, []

    def on_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[DefenseDecision]]:
        return chunks, []

    def on_retrieve(
        self, q: str, hits: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
        return hits, []

    def on_prompt(self, q: str, hits: list[RetrievedChunk], messages: Messages) -> Messages:
        return messages

    def on_verify(
        self,
        q: str,
        hits: list[RetrievedChunk],
        draft: str,
        ctx: QueryContext | None = None,
    ) -> tuple[str, list[DefenseDecision]]:
        return draft, []

    def decision(self, **kwargs: Any) -> DefenseDecision:
        return DefenseDecision(defense_id=self.id, **kwargs)


REGISTRY: dict[str, type[Defense]] = {}
ALL_IDS = ("d1", "d2", "d3", "d4", "d5", "d6")
FULL_STACK = ("d1", "d2", "d3", "d4", "d5")  # the P0 stack; D6 is a P2 stretch defense
D = TypeVar("D", bound=type[Defense])


def register(cls: D) -> D:
    if cls.id in REGISTRY:
        raise ValueError(f"defense {cls.id!r} registered twice")
    REGISTRY[cls.id] = cls
    return cls


def parse_defense_ids(spec: str | Iterable[str] | None) -> list[str]:
    """'d1,d3' | ['d3','d1'] | 'none' | 'all' (= D1-D5 full stack) -> sorted ids."""
    if spec is None:
        return []
    items = [s.strip().lower() for s in (spec.split(",") if isinstance(spec, str) else spec)]
    items = [s for s in items if s and s != "none"]
    if "all" in items:
        items = [*items, *FULL_STACK]
        items.remove("all")
    unknown = [s for s in items if s not in ALL_IDS]
    if unknown:
        raise ValueError(f"unknown defense id(s): {', '.join(unknown)}")
    return sorted(set(items))


def build_defenses(
    ids: Iterable[str], cfg: DefensesConfig, llm: LLMClient | None = None
) -> list[Defense]:
    """Instantiate enabled defenses in id order."""
    from ragantine import defenses as _registered  # noqa: F401  (import registers all ids)

    out: list[Defense] = []
    for did in sorted(set(ids)):
        if did not in REGISTRY:
            raise ValueError(f"defense {did!r} is not implemented")
        sub = getattr(cfg, did)
        out.append(REGISTRY[did](sub.model_dump(mode="json"), llm))
    return out
