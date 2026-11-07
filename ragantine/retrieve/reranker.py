"""Cross-encoder reranking of fused candidates (SPEC §B4.3)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ragantine.local_models import serialized


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """bge-reranker-base cross-encoder on CPU; higher score = more relevant."""

    def __init__(
        self, model_name: str, device: str = "cpu", cache_dir: Path | str | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        # Scores are a pure function of (model, query, passage): caching them on disk means the
        # same question over the same clean passages is scored once across runs (CPU-bound).
        self._cache: Any = None
        if cache_dir is not None:
            import diskcache

            self._cache = diskcache.Cache(str(cache_dir))

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    @serialized
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if self._cache is None:
            return self._predict(query, texts)
        keys = [self._key(query, t) for t in texts]
        found: dict[str, float] = {}
        for k in dict.fromkeys(keys):
            if (v := self._cache.get(k)) is not None:
                found[k] = float(v)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            new = self._predict(query, [texts[i] for i in todo])
            for i, score in zip(todo, new, strict=True):
                found[keys[i]] = score
                self._cache.set(keys[i], score)
        return [found[k] for k in keys]

    def _predict(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts], convert_to_numpy=True)
        return [float(s) for s in scores]

    def _key(self, query: str, text: str) -> str:
        blob = "\x1f".join((self.model_name, query, text))
        return hashlib.sha256(blob.encode()).hexdigest()
