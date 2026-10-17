"""A scripted stand-in model for the offline fixture (CI smoke run, tests).

It plays every benchmark role (question writer, generator, judge, poison writer) and, as
the generator, behaves like a *compliant* model: it answers from the first matching
context block and follows injected canary notes. It produces the recorded responses of
``tests/fixtures/mini_bench`` until they are re-recorded with a real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ragantine.fakes import FakeLLM

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
Messages = Sequence[dict[str, str]]


def _first_year(text: str) -> str | None:
    m = YEAR.search(text)
    return m.group(1) if m else None


def question_writer(messages: Messages) -> str:
    passage = messages[0]["content"].split("PASSAGE:\n", 1)[1]
    title = passage.split("\n", 1)[0].removeprefix("Title: ").strip()
    year = _first_year(passage.split("\n\n", 1)[-1])
    return json.dumps(
        {
            "question": f"Which year appears first in the record about {title}?",
            "answer": year or "unknown",
            "aliases": [],
        }
    )


DOC_BLOCK = re.compile(
    r'<document id="([0-9a-f]{16}:\d{4})" source_tier="\w+">\n(.*?)\n</document>', re.S
)


def context_blocks(messages: Messages) -> list[tuple[str, str]]:
    """(chunk_id, text) pairs from either the baseline CONTEXT or the D4 <document> format."""
    text = "\n".join(m["content"] for m in messages)
    docs = DOC_BLOCK.findall(text)
    if docs:
        return [(cid, body.replace("\u02c6", " ")) for cid, body in docs]  # undo D4 datamarking
    ctx = messages[0]["content"].split("CONTEXT:\n", 1)[-1]
    return [
        (b[1:22], b[23:])
        for b in re.split(r"\n(?=\[[0-9a-f]{16}:\d{4}\])", ctx)
        if b.startswith("[")
    ]


def generator(messages: Messages) -> str:
    """Answers from the context like a compliant model: follows injected canary notes too."""
    question = messages[-1]["content"]
    m = re.search(r"record about (.+?)\?", question)
    if not m:
        return "I don't know."
    title = m.group(1)
    blocks = [(cid, text) for cid, text in context_blocks(messages) if f"Title: {title}" in text]
    if not blocks:
        return "I don't know."
    cid, text = blocks[0]
    canary = re.search(r"\b([A-Z]{4,7}-\d{4})\b", " ".join(t for _, t in blocks))
    year = _first_year(text.split("\n\n", 1)[-1])
    answer = f"{year} [{cid}]." if year else "I don't know."
    return f"{answer} {canary.group(1)}" if canary else answer


def poison_writer(messages: Messages) -> str:
    prompt = messages[0]["content"]
    subject = re.search(r"Subject: (.+?)\. Question", prompt).group(1)  # type: ignore[union-attr]
    instr = re.search(r"document about the subject that (.+?)\.\n", prompt).group(1)  # type: ignore[union-attr]
    return f"{subject} is a well documented subject. This text {instr}.\n\nFurther background about {subject} follows here."


def make_bench_llm() -> FakeLLM:
    return FakeLLM(
        rules=[
            ("Write ONE factual question", question_writer),
            (
                "Rewrite this question",
                lambda m: json.dumps(
                    {"question": "Reworded: " + m[0]["content"].split("QUESTION: ", 1)[1]}
                ),
            ),
            (
                "rewordings of this question",
                lambda m: json.dumps({"rewordings": ["First rewording?", "Second rewording?"]}),
            ),
            (
                "plausible but WRONG answers",
                lambda m: json.dumps({"wrong_answers": ["1066", "1492", "1815"]}),
            ),
            ("You are generating a TEST DOCUMENT", poison_writer),
            (
                "You are grading a QA system",
                lambda m: '{"verdict": "other", "reason": "fake judge"}',
            ),
            ("CONTEXT:", generator),
            ("DOCUMENTS:", generator),
        ],
    )
