"""D2 training data and threshold selection (SPEC §B6, Phase 5).

* ``build_training_set``: positives are chunks of poison generated ONLY from training
  templates t01-t04 on dev-split questions; negatives are random clean chunks (5x the
  positives) plus hard negatives (clean chunks with imperative / instruction-like wording).
* ``build_dev_set``: dev-split poison from the TEST templates t05-t08 plus clean chunks,
  used only to pick the threshold.
* ``pick_threshold``: highest recall whose ingestion false-quarantine rate is <= 2%.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence

from pydantic import BaseModel

from bench.schemas import BenchQuestion, PoisonSample
from ragantine.config import AppConfig
from ragantine.ingest.chunker import TokenCounter
from ragantine.ingest.pipeline import prepare_document
from ragantine.schemas import Chunk

NEGATIVE_RATIO = 5
HARD_NEGATIVE_RATIO = 1
MAX_FALSE_QUARANTINE = 0.02
VAL_FRACTION = 0.15
HARD_NEGATIVE = re.compile(
    r"\b(must|should|note:|instructions?|how to|step \d|always|never|do not|make sure|ensure|follow)\b",
    re.I,
)


class D2Example(BaseModel):
    text: str
    label: int  # 1 = injected/poison, 0 = clean
    source: str  # poison | clean | hard_negative
    split: str  # train | val | dev
    attack: str | None = None
    template_id: str | None = None


def body(chunk: Chunk) -> str:
    """D2 scores chunk bodies (the shared 'Title:' prefix carries no signal)."""
    return chunk.text.split("\n\n", 1)[-1]


def poison_chunks(
    samples: Sequence[PoisonSample], cfg: AppConfig, count: TokenCounter | None = None
) -> list[tuple[PoisonSample, Chunk]]:
    """Chunk poison documents exactly as the naive baseline would ingest them."""
    out = []
    for p in samples:
        chunks, _, _ = prepare_document(p.document, [], cfg.chunking, count)
        out += [(p, c) for c in chunks]
    return out


def _val(key: str, seed: int) -> bool:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 2**32 < VAL_FRACTION


def build_training_set(
    cfg: AppConfig,
    train_poison: Sequence[PoisonSample],
    clean_chunks: Sequence[Chunk],
    count: TokenCounter | None = None,
) -> list[D2Example]:
    bad = [p for p in train_poison if int(p.template_id[-2:]) > 4]
    if bad:
        raise ValueError(f"training poison must use templates t01-t04, got {bad[0].template_id}")
    positives = [
        D2Example(text=body(c), label=1, source="poison", split="val" if _val(p.qid, cfg.seed) else "train",
                  attack=p.attack, template_id=p.template_id)
        for p, c in poison_chunks(train_poison, cfg, count)
    ]  # fmt: skip
    rng = random.Random(cfg.seed)
    pool = sorted(clean_chunks, key=lambda c: c.chunk_id)
    hard = [c for c in pool if HARD_NEGATIVE.search(body(c))]
    n_neg = min(len(pool), NEGATIVE_RATIO * len(positives))
    n_hard = min(len(hard), HARD_NEGATIVE_RATIO * len(positives))
    negatives = rng.sample(pool, n_neg)
    hard_pick = rng.sample(hard, n_hard)
    seen = {c.chunk_id for c in negatives}
    examples = list(positives)
    for c, source in [
        *((c, "clean") for c in negatives),
        *((c, "hard_negative") for c in hard_pick if c.chunk_id not in seen),
    ]:
        examples.append(
            D2Example(
                text=body(c),
                label=0,
                source=source,
                split="val" if _val(c.doc_id, cfg.seed) else "train",
            )
        )
    rng.shuffle(examples)
    return examples


def build_dev_set(
    cfg: AppConfig,
    poison: Sequence[PoisonSample],
    questions: Sequence[BenchQuestion],
    clean_chunks: Sequence[Chunk],
    count: TokenCounter | None = None,
) -> list[D2Example]:
    dev_targets = {q.qid for q in questions if q.qset == "target" and q.split == "dev"}
    dev_poison = [p for p in poison if p.qid in dev_targets and p.seed == cfg.seed]
    examples = [
        D2Example(
            text=body(c),
            label=1,
            source="poison",
            split="dev",
            attack=p.attack,
            template_id=p.template_id,
        )
        for p, c in poison_chunks(dev_poison, cfg, count)
    ]
    examples += [
        D2Example(text=body(c), label=0, source="clean", split="dev") for c in clean_chunks
    ]
    return examples


class ThresholdChoice(BaseModel):
    threshold: float
    recall: float
    false_quarantine: float
    n_poison: int
    n_clean: int
    recall_by_attack: dict[str, float]


def pick_threshold(
    scores: Sequence[float], examples: Sequence[D2Example], max_fq: float = MAX_FALSE_QUARANTINE
) -> ThresholdChoice:
    """Lowest threshold (= highest recall) whose false-quarantine rate on clean chunks is <= max_fq."""
    pos = [(s, e) for s, e in zip(scores, examples, strict=True) if e.label == 1]
    neg = sorted((s for s, e in zip(scores, examples, strict=True) if e.label == 0), reverse=True)
    if not pos or not neg:
        raise ValueError("need both poison and clean examples to pick a threshold")
    allowed_fp = int(max_fq * len(neg))
    # any threshold strictly above the (allowed_fp+1)-th highest clean score keeps FP <= allowed
    floor = neg[allowed_fp] if allowed_fp < len(neg) else -1.0
    threshold = min(1.0, float(floor) + 1e-6)
    fq = sum(s >= threshold for s in neg) / len(neg)
    by_attack: dict[str, list[bool]] = {}
    for s, e in pos:
        by_attack.setdefault(e.attack or "?", []).append(s >= threshold)
    return ThresholdChoice(
        threshold=round(threshold, 6),
        recall=sum(s >= threshold for s, _ in pos) / len(pos),
        false_quarantine=fq,
        n_poison=len(pos),
        n_clean=len(neg),
        recall_by_attack={a: sum(v) / len(v) for a, v in sorted(by_attack.items())},
    )


def write_threshold(config_path: str, threshold: float) -> None:
    """Update defenses.d2.threshold in a config YAML in place, keeping comments."""
    from pathlib import Path

    path = Path(config_path)
    text = path.read_text()
    new, n = re.subn(
        r"(d2: \{[^}]*?threshold: )[0-9.]+", rf"\g<1>{threshold}", text, count=1, flags=re.S
    )
    if n != 1:
        raise ValueError(f"could not find defenses.d2.threshold in {path}")
    path.write_text(new)
