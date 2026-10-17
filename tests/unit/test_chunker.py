"""Chunker: sentence packing, overlap, title prefix, min size (SPEC §B4.1)."""

import hashlib
from itertools import pairwise

from ragantine.config import ChunkingConfig
from ragantine.ingest.chunker import chunk_document, chunk_text, split_sentences
from ragantine.schemas import SourceTier
from tests.helpers import NOW, word_count

CFG = ChunkingConfig(target_tokens=20, overlap_tokens=6, min_tokens=3, tokenizer="cl100k_base")


def _sentences(n: int, words: int = 5) -> list[str]:
    return [" ".join([f"S{i}w{j}" for j in range(words - 1)] + [f"end{i}."]) for i in range(n)]


def test_split_sentences_keeps_abbreviations_and_numbers_together() -> None:
    para = 'It rose 8,849 m. Then it fell. "Quoted start." 1953 was the year.'
    assert split_sentences(para) == [
        "It rose 8,849 m.",
        "Then it fell.",
        '"Quoted start."',
        "1953 was the year.",
    ]


def test_chunks_never_split_sentences_and_respect_budget() -> None:
    sents = _sentences(12)
    bodies = chunk_text(" ".join(sents), CFG, word_count)
    for body in bodies:
        assert word_count(body) <= CFG.target_tokens
    pieces = [s for b in bodies for s in split_sentences(b)]
    assert set(pieces) == set(sents)  # every sentence survives intact


def test_overlap_repeats_trailing_sentences() -> None:
    bodies = chunk_text(" ".join(_sentences(12)), CFG, word_count)
    assert len(bodies) > 1
    for prev, nxt in pairwise(bodies):
        last = split_sentences(prev)[-1]
        assert split_sentences(nxt)[0] == last  # 5-word sentence fits in a 6-token overlap


def test_no_overlap_when_disabled() -> None:
    cfg = CFG.model_copy(update={"overlap_tokens": 0})
    bodies = chunk_text(" ".join(_sentences(12)), cfg, word_count)
    pieces = [s for b in bodies for s in split_sentences(b)]
    assert len(pieces) == 12


def test_paragraph_breaks_are_preserved_inside_chunks() -> None:
    bodies = chunk_text("One two three.\n\nFour five six.", CFG, word_count)
    assert bodies == ["One two three.\n\nFour five six."]


def test_overlong_run_on_is_cut_at_word_boundaries() -> None:
    text = " ".join(f"w{i}" for i in range(50))  # no sentence ends at all
    bodies = chunk_text(text, CFG.model_copy(update={"overlap_tokens": 0}), word_count)
    assert all(word_count(b) <= CFG.target_tokens for b in bodies)
    assert " ".join(bodies).split() == text.split()


def test_chunk_document_prefix_ids_hashes_and_min_size() -> None:
    text = " ".join(_sentences(8)) + "\n\nTiny."
    chunks = chunk_document("abcdef0123456789", "Topic", text, SourceTier.WEB, NOW, CFG, word_count)
    assert chunks, "expected at least one chunk"
    for i, c in enumerate(chunks):
        assert c.text.startswith("Title: Topic\n\n")
        assert c.chunk_id == f"abcdef0123456789:{i:04d}" and c.ordinal == i
        assert c.content_hash == hashlib.sha256(c.text.encode()).hexdigest()
        assert c.n_tokens == word_count(c.text)
        assert c.source_tier is SourceTier.WEB
    assert all(word_count(c.text) <= CFG.target_tokens for c in chunks)


def test_chunks_below_min_tokens_are_dropped() -> None:
    cfg = CFG.model_copy(update={"min_tokens": 4})
    assert (
        chunk_document("abcdef0123456789", "T", "Too short.", SourceTier.WEB, NOW, cfg, word_count)
        == []
    )
