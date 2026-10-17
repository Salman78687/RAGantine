"""Test helpers built on the scripted fixture model (bench/fixture_llm.py)."""

from __future__ import annotations

from bench.fixture_llm import make_bench_llm

__all__ = ["build_bench_data", "make_bench_llm"]


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
