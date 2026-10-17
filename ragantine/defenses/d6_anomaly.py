"""D6 retrieval anomaly detector (SPEC §B6, P2).

Offline (``build_hubs``): run ``probe_queries`` generic questions built from random article
titles ("What is X?", "History of X") and count how often each chunk lands in the fused
top ``fused_k``. Hub score = z-score of that count across chunks with count > 0; chunks
above ``hub_z_threshold`` form the hub list. At query time hub chunks are filtered unless
they are ``curated``. Rebuild after each ingestion batch.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragantine.defenses.base import Defense, register
from ragantine.index.base import IndexBackend
from ragantine.retrieve.dense import Embedder
from ragantine.retrieve.rrf import rrf_fuse
from ragantine.schemas import DefenseDecision, RetrievedChunk, SourceTier

PROBE_TEMPLATES = ("What is {t}?", "History of {t}", "Tell me about {t}.", "Where is {t}?")


def probe_queries(titles: Sequence[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    unique = sorted(set(titles))
    return (
        [rng.choice(PROBE_TEMPLATES).format(t=rng.choice(unique)) for _ in range(n)]
        if unique
        else []
    )


def build_hubs(
    backend: IndexBackend,
    embedder: Embedder,
    titles: Sequence[str],
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """chunk_id -> hub z-score for chunks above the threshold."""
    counts: Counter[str] = Counter()
    k = int(cfg["fused_k"])
    for q in probe_queries(titles, int(cfg["probe_queries"]), seed):
        bm = [c.chunk_id for c in backend.bm25(q, k)]
        kn = [c.chunk_id for c in backend.knn(embedder.embed_query(q), k)]
        counts.update(cid for cid, _ in rrf_fuse([bm, kn], rrf_k=60, top_k=k))
    values = list(counts.values())
    if len(values) < 2:
        return {}
    mu, sd = statistics.fmean(values), statistics.pstdev(values)
    if sd == 0:
        return {}
    z = {cid: (n - mu) / sd for cid, n in counts.items()}
    return {cid: round(s, 3) for cid, s in z.items() if s > float(cfg["hub_z_threshold"])}


def hub_file(cfg: dict[str, Any], index_name: str) -> Path:
    return Path(str(cfg["data_dir"])) / "d6_hubs" / f"{index_name}.json"


@register
class AnomalyDetector(Defense):
    id = "d6"

    def __init__(self, cfg: dict[str, Any], llm: Any = None) -> None:
        super().__init__(cfg, llm)
        self.hubs: dict[str, float] = {}

    def load(self, path: Path) -> None:
        self.hubs = json.loads(path.read_text()) if path.exists() else {}

    def on_retrieve(
        self, q: str, hits: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
        keep, decisions = [], []
        for h in hits:
            z = self.hubs.get(h.chunk.chunk_id)
            if z is not None and h.chunk.source_tier is not SourceTier.CURATED:
                decisions.append(
                    self.decision(
                        stage="retrieval",
                        target_id=h.chunk.chunk_id,
                        action="filter",
                        score=z,
                        reason=f"retrieval_hub: z={z:.2f} across probe queries",
                    )
                )
            else:
                keep.append(h)
        return keep, decisions
