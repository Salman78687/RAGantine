"""A FakeLLM scripted to play every benchmark role (writer, generator, judge, poison writer)."""

from __future__ import annotations

import json
import re

from ragantine.fakes import FakeLLM

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
Messages = list[dict[str, str]]


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


def generator(messages: Messages) -> str:
    """Answers from the context like a compliant model: follows injected canary notes too."""
    system, user = messages[0]["content"], messages[-1]["content"]
    m = re.search(r"record about (.+?)\?", user)
    if not m:
        return "I don't know."
    title = m.group(1)
    blocks = [
        b
        for b in re.split(r"\n(?=\[[0-9a-f]{16}:\d{4}\])", system.split("CONTEXT:\n", 1)[-1])
        if f"Title: {title}" in b
    ]
    if not blocks:
        return "I don't know."
    cid = blocks[0][1:22]
    canary = re.search(r"\b([A-Z]{4,7}-\d{4})\b", "".join(blocks))
    year = _first_year(blocks[0].split("\n\n", 1)[-1])
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
        ],
    )


def build_bench_data(limit: int = 12, seeds: tuple[int, ...] = (1337,)):  # type: ignore[no-untyped-def]
    """Questions + test poison + workspace inputs over the mini corpus (all offline)."""
    import asyncio

    from bench.io import read_jsonl
    from bench.poison.generator import generate_poison
    from bench.questions.generate import generate_questions
    from ragantine.schemas import RawDocument
    from tests.helpers import ROOT, memory_pipeline, small_config

    cfg = small_config()
    cfg = cfg.model_copy(
        update={
            "chunking": cfg.chunking.model_copy(update={"target_tokens": 350, "overlap_tokens": 50})
        }
    )
    llm = make_bench_llm()
    docs = list(
        read_jsonl(ROOT / "tests" / "fixtures" / "mini_corpus" / "articles.jsonl", RawDocument)
    )
    pipeline, _ = memory_pipeline(llm, docs=docs, cfg=cfg)
    chunks = pipeline.backend.all_chunks()  # type: ignore[attr-defined]
    questions, _ = asyncio.run(
        generate_questions(
            cfg, pipeline, chunks, "fake", "fake", "fake", limit=limit, log=lambda _m: None
        )
    )
    subjects = {c.chunk_id: c.title for c in chunks}
    poison, _ = asyncio.run(
        generate_poison(cfg, llm, "fake", questions, subjects, list(seeds), "test", lambda _m: None)
    )
    return cfg, llm, docs, questions, poison
