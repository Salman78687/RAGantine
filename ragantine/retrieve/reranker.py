"""Cross-encoder reranking of fused candidates (SPEC §B4.3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """bge-reranker-base cross-encoder on CPU; higher score = more relevant."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], convert_to_numpy=True)
        return [float(s) for s in scores]
