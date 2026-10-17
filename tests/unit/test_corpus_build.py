"""Corpus build: filters, deterministic sampling, tier assignment (SPEC §B5.1)."""

from collections import Counter
from pathlib import Path

import numpy as np

from bench.corpus.build import (
    article_hash,
    assign_tier,
    build_corpus,
    corpus_paths,
    keep_article,
    load_embedding_lookup,
    sample_articles,
    snapshot_date,
)
from bench.io import read_jsonl
from ragantine.fakes import HashingEmbedder
from ragantine.schemas import Chunk, SourceTier
from tests.helpers import small_config, word_count

CFG = small_config()


def _rows(n: int) -> list[dict[str, object]]:
    body = "Sentence about the topic with facts and figures. " * 60
    rows: list[dict[str, object]] = [
        {
            "id": str(i),
            "url": f"https://en.wikipedia.org/wiki?curid={i}",
            "title": f"Article {i}",
            "text": body,
        }
        for i in range(n)
    ]
    rows += [
        {"id": "x1", "url": "u1", "title": "List of rivers", "text": body},
        {"id": "x2", "url": "u2", "title": "Mercury (disambiguation)", "text": body},
        {"id": "x3", "url": "u3", "title": "Mercury", "text": "Mercury may refer to: " + body},
        {"id": "x4", "url": "u4", "title": "Stub", "text": "Too short."},
    ]
    return rows


def test_filters() -> None:
    c = CFG.corpus
    long = "x" * c.min_chars
    assert keep_article("Paris", long, c)
    assert not keep_article("List of rivers", long, c)
    assert not keep_article("Index of physics articles", long, c)
    assert not keep_article("Mercury (disambiguation)", long, c)
    assert not keep_article("Mercury", "Mercury may refer to: " + long, c)
    assert not keep_article("Paris", "short", c)
    assert not keep_article("Paris", "x" * (c.max_chars + 1), c)


def test_sampling_is_deterministic_and_independent_of_order() -> None:
    rows = _rows(50)
    a, scanned, eligible = sample_articles(rows, CFG.corpus, seed=1337, n_articles=10)
    b, _, _ = sample_articles(list(reversed(rows)), CFG.corpus, seed=1337, n_articles=10)
    c, _, _ = sample_articles(rows, CFG.corpus, seed=7, n_articles=10)
    assert [d.doc_id for d in a] == [d.doc_id for d in b]
    assert [d.doc_id for d in a] != [d.doc_id for d in c]
    assert scanned == 54 and eligible == 50 and len(a) == 10
    assert all(d.ingested_at == snapshot_date(CFG.corpus.hf_config) for d in a)


def test_scan_limit_bounds_the_stream() -> None:
    cfg = CFG.corpus.model_copy(update={"scan_limit": 20})
    _, scanned, _ = sample_articles(_rows(50), cfg, seed=1, n_articles=5)
    assert scanned == 20


def test_tier_assignment_follows_mix() -> None:
    mix = CFG.corpus.tier_mix
    counts = Counter(assign_tier(article_hash(1337, str(i)), mix) for i in range(20000))
    for tier, share in mix.items():
        assert abs(counts[tier] / 20000 - share) < 0.02
    assert assign_tier("0" * 40, {SourceTier.WEB: 1.0}) is SourceTier.WEB


def test_build_corpus_writes_aligned_outputs(tmp_path: Path) -> None:
    cfg = CFG.model_copy(update={"paths": CFG.paths.model_copy(update={"data_dir": tmp_path})})
    emb = HashingEmbedder(cfg.embedding.dimension)
    report = build_corpus(
        cfg, emb, rows=_rows(30), n_articles=5, count=word_count, log=lambda _m: None
    )
    paths = corpus_paths(cfg)
    chunks = list(read_jsonl(paths["chunks"], Chunk))
    vectors = np.load(paths["embeddings"])
    assert report.articles == 5 and report.chunks == len(chunks) == len(vectors)
    assert paths["ids"].read_text().split() == [c.chunk_id for c in chunks]
    lookup = load_embedding_lookup(paths)
    assert np.allclose(lookup[chunks[0].content_hash], emb.embed_passages([chunks[0].text])[0])
