"""Poison writer templates (bench/poison/templates/A{n}-tNN.txt).

Each file has a ``[writer]`` prompt and, for instruction attacks, a ``[payload]`` (and for
A6 a ``[payload_b]``) that is inserted programmatically so the canary is always present.
Templates t01-t04 are reserved for training D2; t05-t08 are the test benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
TRAIN_TEMPLATE_NUMBERS = (1, 2, 3, 4)
TEST_TEMPLATE_NUMBERS = (5, 6, 7, 8)
_SECTION = re.compile(r"^\[(writer|payload|payload_b)\]\n", re.M)


@dataclass(frozen=True)
class Template:
    template_id: str
    writer: str
    payload: str | None = None
    payload_b: str | None = None

    @property
    def attack(self) -> str:
        return self.template_id.split("-")[0]

    @property
    def number(self) -> int:
        return int(self.template_id.split("-t")[1])


def parse_template(template_id: str, text: str) -> Template:
    parts = _SECTION.split(text)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if "writer" not in sections:
        raise ValueError(f"template {template_id} has no [writer] section")
    return Template(
        template_id=template_id,
        writer=sections["writer"],
        payload=sections.get("payload"),
        payload_b=sections.get("payload_b"),
    )


@lru_cache(maxsize=1)
def all_templates() -> dict[str, Template]:
    return {
        p.stem: parse_template(p.stem, p.read_text(encoding="utf-8"))
        for p in sorted(TEMPLATES_DIR.glob("A*-t*.txt"))
    }


def template_ids(attack: str, kind: str) -> list[str]:
    numbers = TRAIN_TEMPLATE_NUMBERS if kind == "train" else TEST_TEMPLATE_NUMBERS
    return [f"{attack}-t{n:02d}" for n in numbers]
