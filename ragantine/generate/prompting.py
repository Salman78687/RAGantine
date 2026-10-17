"""Prompt templates (stored as .txt under ``prompts/``) and citation parsing (SPEC §B4.4).

A template has a ``SYSTEM:`` part and a ``USER:`` part. Placeholders are substituted with
plain string replacement, never ``str.format``, because retrieved text may contain braces.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ragantine.schemas import Citation, RetrievedChunk

PROMPTS_DIR = Path(__file__).parent / "prompts"
_CITATION = re.compile(r"\[([0-9a-f]{16}:\d{4})\]")


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def split_template(template: str) -> tuple[str, str]:
    """Split a 'SYSTEM: ... USER: ...' template into (system, user) parts."""
    if not template.startswith("SYSTEM:") or "\nUSER:" not in template:
        raise ValueError("prompt template must start with 'SYSTEM:' and contain a 'USER:' line")
    system, user = template[len("SYSTEM:") :].split("\nUSER:", 1)
    return system.strip(), user.strip()


def format_context(hits: list[RetrievedChunk]) -> str:
    return "\n".join(f"[{h.chunk.chunk_id}] {h.chunk.text}" for h in hits)


def baseline_messages(question: str, hits: list[RetrievedChunk]) -> list[dict[str, str]]:
    """D0 prompt: grounded answer with chunk-id citations (prompts/baseline.txt)."""
    system, user = split_template(load_prompt("baseline"))
    return [
        {"role": "system", "content": fill(system, context=format_context(hits))},
        {"role": "user", "content": fill(user, question=question)},
    ]


def parse_citations(text: str, hits: list[RetrievedChunk]) -> list[Citation]:
    """Citations the answer actually makes, restricted to chunks that were in the prompt."""
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    seen: list[str] = []
    for cid in _CITATION.findall(text):
        if cid in by_id and cid not in seen:
            seen.append(cid)
    return [
        Citation(chunk_id=c, title=by_id[c].title, source_tier=by_id[c].source_tier) for c in seen
    ]
