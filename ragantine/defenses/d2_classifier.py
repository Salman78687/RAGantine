"""D2 injection classifier (SPEC §B6).

Scores each chunk for instruction-like content. At ingestion (``on_chunks``) chunks
scoring at or above ``threshold`` are quarantined; at retrieval (``on_retrieve``) the
top ``fused_k`` are re-scored and filtered, covering anything ingested before D2 existed.

The scorer is pluggable: ``HFInjectionScorer`` wraps a Hugging Face sequence classifier
(the fine-tuned ``models/d2-injection`` or a zero-shot open model such as
``protectai/deberta-v3-base-prompt-injection-v2``). The positive label is read from the
model config, never hard-coded. Scores are cached per content_hash.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ragantine.defenses.base import Defense, register
from ragantine.local_models import serialized
from ragantine.schemas import Chunk, DefenseDecision, RetrievedChunk

INJECTION_LABEL_HINTS = ("injection", "unsafe", "malicious", "jailbreak", "label_1", "1")
MAX_LENGTH = 512
BATCH_SIZE = 16


class InjectionScorer(Protocol):
    def score(self, texts: Sequence[str]) -> list[float]:
        """Probability that each text contains an injected instruction."""
        ...


class HFInjectionScorer:
    def __init__(self, model: str, device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None
        self._tok: Any = None
        self.positive_index: int | None = None

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self.positive_index = positive_label_index(id2label)

    @serialized
    def score(self, texts: Sequence[str]) -> list[float]:
        import torch

        if self._model is None:
            self._load()
        out: list[float] = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self._tok(
                list(texts[i : i + BATCH_SIZE]), truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            ).to(self.device)  # fmt: skip
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits, dim=-1)
            out += [float(p) for p in probs[:, self.positive_index]]
        return out


def positive_label_index(id2label: dict[int, str]) -> int:
    """Index of the 'injection' class, from the model's own label names."""
    if len(id2label) == 1:
        return 0
    names = {idx: name.lower() for idx, name in id2label.items()}
    for hint in INJECTION_LABEL_HINTS:  # exact names first ...
        for idx, name in sorted(names.items()):
            if name == hint:
                return idx
    for idx, name in sorted(names.items()):  # ... then anything mentioning injection
        if "inject" in name:
            return idx
    raise ValueError(f"cannot tell which label means injection: {id2label}")


def resolve_model(cfg: dict[str, Any]) -> str:
    """Fine-tuned local weights if present, else the configured zero-shot model."""
    variant = str(cfg.get("variant", "finetuned"))
    local = Path(str(cfg["model_path"]))
    if variant == "finetuned":
        if (local / "config.json").exists():
            return str(local)
        raise FileNotFoundError(
            f"D2 fine-tuned weights not found at {local}; train them with notebooks/train_d2.ipynb "
            "or set defenses.d2.variant=zeroshot"
        )
    return str(cfg["zeroshot_model"])


@register
class InjectionClassifier(Defense):
    id = "d2"

    def __init__(self, cfg: dict[str, Any], llm: Any = None) -> None:
        super().__init__(cfg, llm)
        self.threshold = float(cfg["threshold"])
        self.apply_at = set(cfg.get("apply_at", ["ingestion", "retrieval"]))
        self._scorer: InjectionScorer | None = None
        self._cache: dict[str, float] = {}

    @property
    def scorer(self) -> InjectionScorer:
        if self._scorer is None:
            self._scorer = HFInjectionScorer(resolve_model(self.cfg))
        return self._scorer

    @scorer.setter
    def scorer(self, value: InjectionScorer) -> None:
        self._scorer = value
        self._cache.clear()

    def scores(self, chunks: Sequence[Chunk]) -> list[float]:
        todo = [c for c in chunks if c.content_hash not in self._cache]
        if todo:
            # score the body only: the shared "Title: ..." prefix carries no instructions
            bodies = [c.text.split("\n\n", 1)[-1] for c in todo]
            for c, s in zip(todo, self.scorer.score(bodies), strict=True):
                self._cache[c.content_hash] = s
        return [self._cache[c.content_hash] for c in chunks]

    def _reason(self, score: float) -> str:
        return f"injection_classifier: p={score:.3f} >= {self.threshold}"

    def on_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[DefenseDecision]]:
        if "ingestion" not in self.apply_at or not chunks:
            return chunks, []
        keep, decisions = [], []
        for c, s in zip(chunks, self.scores(chunks), strict=True):
            if s >= self.threshold:
                decisions.append(
                    self.decision(
                        stage="ingestion",
                        target_id=c.chunk_id,
                        action="quarantine",
                        score=s,
                        reason=self._reason(s),
                    )
                )
            else:
                keep.append(c)
        return keep, decisions

    def on_retrieve(
        self, q: str, hits: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
        if "retrieval" not in self.apply_at or not hits:
            return hits, []
        keep, decisions = [], []
        for h, s in zip(hits, self.scores([h.chunk for h in hits]), strict=True):
            if s >= self.threshold:
                decisions.append(
                    self.decision(
                        stage="retrieval",
                        target_id=h.chunk.chunk_id,
                        action="filter",
                        score=s,
                        reason=self._reason(s),
                    )
                )
            else:
                keep.append(h)
        return keep, decisions
