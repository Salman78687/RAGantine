"""A daily-quota stop must abort benchmark steps, never be counted as a rejection or error."""

import asyncio
from collections.abc import Sequence

import pytest

from bench.questions.generate import generate_questions
from ragantine.fakes import FakeLLM
from ragantine.generate.llm import QuotaExhaustedError
from tests.helpers import memory_pipeline


def _quota(_messages: Sequence[dict[str, str]]) -> str:
    raise QuotaExhaustedError("daily quota reached")


def test_question_generation_aborts_on_daily_quota() -> None:
    pipeline, _ = memory_pipeline(FakeLLM(default=_quota))
    chunks = pipeline.backend.all_chunks()  # type: ignore[attr-defined]
    with pytest.raises(QuotaExhaustedError):
        asyncio.run(
            generate_questions(
                pipeline.cfg, pipeline, chunks, "fake", "fake", "fake", limit=6, log=lambda _m: None
            )
        )
