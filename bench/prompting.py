"""Benchmark prompt templates (bench/prompts/*.txt) and tolerant JSON parsing of LLM output."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent / "prompts"
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def fill(template: str, **values: object) -> str:
    """Placeholder substitution without str.format (passages may contain braces)."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def user_message(name: str, **values: object) -> list[dict[str, str]]:
    return [{"role": "user", "content": fill(load_prompt(name), **values)}]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an LLM reply (tolerates code fences and preambles)."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError(f"no JSON object in model output: {text[:120]!r}") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data
