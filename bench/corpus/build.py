"""Clean corpus build (SPEC §B5.1).

Streams ``wikimedia/wikipedia`` and keeps articles with ``min_chars <= len <= max_chars``,
skipping lists, disambiguation and "Index of" pages. The deterministic sample is the
``n_articles`` articles with the smallest ``sha1(seed + id)`` among the first
``scan_limit`` streamed rows (a bounded download instead of all 6M+ articles). Each
article's ``source_tier`` comes from the low bits of the same hash using ``tier_mix``.

Outputs under ``data/corpus/``: articles.jsonl (RawDocument), chunks.jsonl (Chunk),
embeddings.npy (float32, row i = chunk i) and embeddings.ids.txt (chunk_id per row).
"""

from __future__ import annotations

import hashlib
import heapq
import re
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from bench.io import read_jsonl, write_jsonl
from ragantine.config import AppConfig, CorpusConfig
from ragantine.ingest.chunker import TokenCounter, chunk_document
from ragantine.retrieve.dense import Embedder
from ragantine.schemas import Chunk, RawDocument, SourceTier

TIER_ORDER = (SourceTier.CURATED, SourceTier.PARTNER, SourceTier.WEB, SourceTier.USER_UPLOAD)
_SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Outline of)\b|\(disambiguation\)$", re.I)
_DISAMBIG_TEXT = re.compile(r"\bmay (also )?refer to:", re.I)


class CorpusReport(BaseModel):
    scanned: int
    eligible: int
    articles: int
    chunks: int
    tiers: dict[str, int]


def article_hash(seed: int, article_id: str) -> str:
    return hashlib.sha1(f"{seed}{article_id}".encode()).hexdigest()


def assign_tier(hash_hex: str, tier_mix: dict[SourceTier, float]) -> SourceTier:
    """Tier from the hash's low 32 bits (the sort uses the high bits, so they stay uniform)."""
    u = int(hash_hex[-8:], 16) / 2**32
    acc = 0.0
    for tier in TIER_ORDER:
        acc += tier_mix.get(tier, 0.0)
        if u < acc:
            return tier
    return TIER_ORDER[0]


def keep_article(title: str, text: str, cfg: CorpusConfig) -> bool:
    if not cfg.min_chars <= len(text) <= cfg.max_chars:
        return False
    if _SKIP_TITLE.search(title):
        return False
    return not _DISAMBIG_TEXT.search(text[:500])


def snapshot_date(hf_config: str) -> datetime:
    """'20231101.en' -> 2023-11-01T00:00Z, used as ingested_at for clean articles."""
    try:
        return datetime.strptime(hf_config.split(".")[0], "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)


def stream_wikipedia(cfg: CorpusConfig) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset, cfg.hf_config, split="train", streaming=True)
    yield from ds


def sample_articles(
    rows: Iterable[dict[str, Any]],
    cfg: CorpusConfig,
    seed: int,
    n_articles: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[RawDocument], int, int]:
    """Deterministic sample; returns (documents sorted by hash, scanned, eligible)."""
    heap: list[tuple[str, dict[str, Any]]] = []  # max-heap via negated ordering
    scanned = eligible = 0
    for row in rows:
        if scanned >= cfg.scan_limit:
            break
        scanned += 1
        title, text = str(row["title"]), str(row["text"])
        if not keep_article(title, text, cfg):
            continue
        eligible += 1
        h = article_hash(seed, str(row["id"]))
        item = (
            _neg(h),
            {"id": row["id"], "url": row.get("url", ""), "title": title, "text": text, "hash": h},
        )
        if len(heap) < n_articles:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
        if on_progress and scanned % 5000 == 0:
            on_progress(scanned, eligible)
    ingested_at = snapshot_date(cfg.hf_config)
    docs = []
    for _, row in sorted(heap, key=lambda it: it[1]["hash"]):
        uri = row["url"] or f"wikipedia:{row['id']}"
        docs.append(
            RawDocument(
                doc_id=hashlib.sha1(uri.encode()).hexdigest()[:16],
                title=row["title"],
                source_uri=uri,
                fmt="text",
                raw=row["text"],
                source_tier=assign_tier(row["hash"], cfg.tier_mix),
                ingested_at=ingested_at,
            )
        )
    return docs, scanned, eligible


def _neg(hex_hash: str) -> str:
    """Order-reversing key so heapq (a min-heap) keeps the n smallest hashes."""
    return "".join(f"{15 - int(c, 16):x}" for c in hex_hash)


def corpus_paths(cfg: AppConfig) -> dict[str, Path]:
    d = cfg.paths.corpus_dir
    return {
        "articles": d / "articles.jsonl",
        "chunks": d / "chunks.jsonl",
        "embeddings": d / "embeddings.npy",
        "ids": d / "embeddings.ids.txt",
    }


def build_corpus(
    cfg: AppConfig,
    embedder: Embedder,
    rows: Iterable[dict[str, Any]] | None = None,
    n_articles: int | None = None,
    count: TokenCounter | None = None,
    log: Callable[[str], None] = print,
) -> CorpusReport:
    n = n_articles or cfg.corpus.n_articles
    paths = corpus_paths(cfg)
    docs, scanned, eligible = sample_articles(
        rows if rows is not None else stream_wikipedia(cfg.corpus),
        cfg.corpus,
        cfg.seed,
        n,
        on_progress=lambda s, e: log(f"  scanned {s:,} rows, {e:,} eligible"),
    )
    if len(docs) < n:
        log(
            f"warning: only {len(docs)} eligible articles in the first {scanned:,} rows (wanted {n})"
        )
    write_jsonl(paths["articles"], docs)
    log(f"wrote {len(docs)} articles -> {paths['articles']}")

    chunks: list[Chunk] = []
    for doc in docs:
        assert isinstance(doc.raw, str)
        chunks += chunk_document(
            doc.doc_id, doc.title, doc.raw, doc.source_tier, doc.ingested_at, cfg.chunking, count
        )
    write_jsonl(paths["chunks"], chunks)
    log(f"wrote {len(chunks)} chunks -> {paths['chunks']}")

    vectors = _embed_with_cache(chunks, embedder, paths, log)
    np.save(paths["embeddings"], vectors)
    paths["ids"].write_text("".join(c.chunk_id + "\n" for c in chunks), encoding="utf-8")
    log(f"wrote embeddings {vectors.shape} -> {paths['embeddings']}")

    tiers: dict[str, int] = {}
    for d in docs:
        tiers[d.source_tier.value] = tiers.get(d.source_tier.value, 0) + 1
    return CorpusReport(
        scanned=scanned, eligible=eligible, articles=len(docs), chunks=len(chunks), tiers=tiers
    )


def _embed_with_cache(
    chunks: list[Chunk], embedder: Embedder, paths: dict[str, Path], log: Callable[[str], None]
) -> np.ndarray:
    """Reuse vectors from a previous build for unchanged chunks (keyed by content_hash)."""
    previous = load_embedding_lookup(paths)
    out = np.zeros((len(chunks), embedder.dimension), dtype=np.float32)
    todo = []
    for i, c in enumerate(chunks):
        if (v := previous.get(c.content_hash)) is not None and len(v) == embedder.dimension:
            out[i] = v
        else:
            todo.append(i)
    if todo:
        log(f"embedding {len(todo)} chunks ({len(chunks) - len(todo)} reused) ...")
        embed: Any = embedder.embed_passages
        texts = [chunks[i].text for i in todo]
        try:
            out[todo] = embed(texts, show_progress=True)
        except TypeError:
            out[todo] = embed(texts)
    return out


def load_embedding_lookup(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """content_hash -> vector from data/corpus (empty if not built yet or out of sync)."""
    if not (paths["embeddings"].exists() and paths["chunks"].exists()):
        return {}
    vectors = np.load(paths["embeddings"])
    chunks = list(read_jsonl(paths["chunks"], Chunk))
    if len(chunks) != len(vectors):
        return {}
    return {c.content_hash: vectors[i] for i, c in enumerate(chunks)}


def load_articles(cfg: AppConfig) -> list[RawDocument]:
    return list(read_jsonl(corpus_paths(cfg)["articles"], RawDocument))
