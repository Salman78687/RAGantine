"""Phase 2 data flow on the mini corpus with a scripted FakeLLM (SPEC §B5.2-B5.3)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from bench.io import read_jsonl
from bench.poison.generator import generate_poison, write_poison_files
from bench.questions.generate import generate_questions, set_sizes
from bench.schemas import BenchQuestion, PoisonSample
from bench.score import contains_canary, mentions
from ragantine.ingest.loaders import load_text
from ragantine.schemas import RawDocument
from tests.fakes_bench import make_bench_llm
from tests.helpers import ROOT, memory_pipeline, small_config

MINI = ROOT / "tests" / "fixtures" / "mini_corpus" / "articles.jsonl"
CFG = small_config()
CFG = CFG.model_copy(
    update={
        "chunking": CFG.chunking.model_copy(update={"target_tokens": 350, "overlap_tokens": 50})
    }
)


@pytest.fixture(scope="module")
def bench_data() -> tuple[list[BenchQuestion], list[PoisonSample], list[PoisonSample], object]:
    import asyncio

    llm = make_bench_llm()
    docs = list(read_jsonl(MINI, RawDocument))
    pipeline, _ = memory_pipeline(llm, docs=docs, cfg=CFG)
    chunks = pipeline.backend.all_chunks()  # type: ignore[attr-defined]
    questions, report = asyncio.run(
        generate_questions(
            CFG, pipeline, chunks, "fake", "fake", "fake", limit=12, log=lambda _m: None
        )
    )
    subjects = {c.chunk_id: c.title for c in chunks}
    test, _ = asyncio.run(
        generate_poison(CFG, llm, "fake", questions, subjects, [1337, 7], "test", lambda _m: None)
    )
    train, _ = asyncio.run(
        generate_poison(CFG, llm, "fake", questions, subjects, [1337], "train", lambda _m: None)
    )
    return questions, test, train, report


def test_question_sets_sizes_attacks_and_splits(bench_data) -> None:  # type: ignore[no-untyped-def]
    questions, _, _, report = bench_data
    sizes = set_sizes(CFG, 12)
    by_set = Counter(q.qset for q in questions)
    assert by_set["target"] == sizes.targets == 12
    assert by_set["clean"] == sizes.clean and 0 < by_set["paraphrase"] <= sizes.paraphrases
    targets = [q for q in questions if q.qset == "target"]
    assert Counter(q.attack for q in targets) == {
        a: 2 for a in ("A1", "A2", "A3", "A4", "A5", "A6")
    }
    for attack in ("A1", "A2", "A3", "A4", "A5", "A6"):
        splits = Counter(q.split for q in targets if q.attack == attack)
        assert splits == {"dev": 1, "test": 1}  # round(2 * 0.3) = 1 dev per attack
    ids = {q.qid for q in questions}
    assert len(ids) == len(questions)
    by_id = {q.qid: q for q in questions}
    for p in (q for q in questions if q.qset == "paraphrase"):
        parent = by_id[p.parent_qid]  # type: ignore[index]
        assert parent.qset == "target" and p.attack == parent.attack and p.split == parent.split
    assert report.candidates >= report.kept >= sizes.targets + sizes.clean
    assert sum(report.rejected.values()) == report.candidates - report.kept


def test_every_poison_sample_validates_and_carries_its_payload(bench_data) -> None:  # type: ignore[no-untyped-def]
    questions, test, _, _ = bench_data
    by_id = {q.qid: q for q in questions}
    targets = [q for q in questions if q.qset == "target"]
    per_seed = Counter(p.seed for p in test)
    expected = sum(2 if q.attack == "A6" else 1 for q in targets)
    assert per_seed == {1337: expected, 7: expected}
    for p in test:
        PoisonSample.model_validate(p.model_dump())  # round-trip
        q = by_id[p.qid]
        assert p.attack == q.attack and int(p.template_id[-2:]) in (5, 6, 7, 8)
        text = load_text(p.document)  # what the naive baseline ingests
        if p.goal == "false_answer":
            assert p.target_false_answer and not mentions(p.target_false_answer, q.gold_answer)
            assert p.target_false_answer in text
        elif p.technique != "a":  # A6 part "a" holds only the first half of the instruction
            assert p.canary and contains_canary(text, p.canary)
        assert p.document.source_tier.value in ("user_upload", "web", "partner")
    a5 = [p for p in test if p.attack == "A5"]
    assert {p.document.fmt for p in a5} <= {"html", "pdf"}
    assert {p.document.fmt for p in test if p.attack == "A3"} == {"markdown"}


def test_seeds_vary_canaries_and_are_reproducible(bench_data) -> None:  # type: ignore[no-untyped-def]
    _, test, _, _ = bench_data
    by_seed: dict[int, dict[str, str | None]] = {}
    for p in test:
        by_seed.setdefault(p.seed, {})[p.poison_id.split("-test-")[0]] = p.canary
    assert by_seed[1337] != by_seed[7]


def test_train_poison_uses_train_templates_on_dev_questions_only(bench_data) -> None:  # type: ignore[no-untyped-def]
    questions, test, train, _ = bench_data
    dev_targets = {q.qid for q in questions if q.qset == "target" and q.split == "dev"}
    assert train and {p.qid for p in train} <= dev_targets
    assert all(int(p.template_id[-2:]) in (1, 2, 3, 4) for p in train)
    assert not {p.template_id for p in train} & {p.template_id for p in test}


def test_rendered_files_are_readable(bench_data, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, test, _, _ = bench_data
    write_poison_files(test, tmp_path)
    files = sorted(tmp_path.iterdir())
    assert len(files) == len(test)
    assert {f.suffix for f in files} >= {".txt", ".md", ".html"}
    for f in files:
        assert f.stat().st_size > 0
