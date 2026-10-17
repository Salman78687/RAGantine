"""Paragraph/sentence chunker (SPEC §B4.1).

Splits on paragraphs, then sentences, and packs whole sentences up to ``target_tokens``
(cl100k) with about ``overlap_tokens`` of trailing sentences repeated at the start of the
next chunk. Every chunk is prefixed with ``Title: <title>\\n\\n``; chunks whose body is
shorter than ``min_tokens`` are dropped. A sentence is never split, except a single
"sentence" longer than the target (e.g. a table row run-on), which is cut at word
boundaries so it still fits the embedding model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from ragantine.config import ChunkingConfig
from ragantine.schemas import Chunk, SourceTier

TokenCounter = Callable[[str], int]

# Sentence end: . ! ? (optionally closed by a quote/bracket) followed by space + capital/digit/quote.
_SENT_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")
_PARA = re.compile(r"\n\s*\n|\n")


@lru_cache(maxsize=4)
def tiktoken_counter(encoding: str) -> TokenCounter:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class _Unit:
    text: str
    para: int
    tokens: int


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(paragraph) if s.strip()]


def _units(text: str, count: TokenCounter, max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for p, para in enumerate(x for x in _PARA.split(text) if x.strip()):
        for sent in split_sentences(para):
            n = count(sent)
            if n <= max_tokens:
                units.append(_Unit(sent, p, n))
                continue
            words, buf = sent.split(), list[str]()
            for w in words:
                if buf and count(" ".join([*buf, w])) > max_tokens:
                    piece = " ".join(buf)
                    units.append(_Unit(piece, p, count(piece)))
                    buf = []
                buf.append(w)
            if buf:
                piece = " ".join(buf)
                units.append(_Unit(piece, p, count(piece)))
    return units


def _join(units: list[_Unit]) -> str:
    out: list[str] = []
    for i, u in enumerate(units):
        if i and units[i - 1].para != u.para:
            out.append("\n\n")
        elif i:
            out.append(" ")
        out.append(u.text)
    return "".join(out)


def chunk_text(
    text: str, cfg: ChunkingConfig, count: TokenCounter, title_tokens: int = 0
) -> list[str]:
    """Pack text into chunk bodies (without the title prefix)."""
    budget = max(cfg.target_tokens - title_tokens, 1)
    units = _units(text, count, budget)
    bodies: list[str] = []
    current: list[_Unit] = []
    size = 0
    fresh = 0  # units in `current` that are not overlap carried from the previous chunk
    for u in units:
        if current and size + u.tokens > budget:
            if fresh:
                bodies.append(_join(current))
            carry: list[_Unit] = []
            carried = 0
            for prev in reversed(current):
                if (
                    carried + prev.tokens > cfg.overlap_tokens
                    or carried + prev.tokens + u.tokens > budget
                ):
                    break
                carry.insert(0, prev)
                carried += prev.tokens
            current, size, fresh = carry, carried, 0
        current.append(u)
        size += u.tokens
        fresh += 1
    if current and fresh:
        bodies.append(_join(current))
    return bodies


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    source_tier: SourceTier,
    ingested_at: datetime,
    cfg: ChunkingConfig,
    count: TokenCounter | None = None,
) -> list[Chunk]:
    count = count or tiktoken_counter(cfg.tokenizer)
    prefix = f"Title: {title}\n\n"
    chunks: list[Chunk] = []
    for body in chunk_text(text, cfg, count, title_tokens=count(prefix)):
        if count(body) < cfg.min_tokens:
            continue
        full = prefix + body
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{ordinal:04d}",
                doc_id=doc_id,
                title=title,
                text=full,
                ordinal=ordinal,
                n_tokens=count(full),
                source_tier=source_tier,
                content_hash=hashlib.sha256(full.encode()).hexdigest(),
                ingested_at=ingested_at,
            )
        )
    return chunks
