"""Dense embeddings with bge (query-prefixed), used for k-NN retrieval (SPEC §B4.3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from ragantine.config import EmbeddingConfig
from ragantine.local_models import serialized


class Embedder(Protocol):
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """bge-base-en-v1.5 on CPU via sentence-transformers; vectors are L2-normalised."""

    def __init__(self, cfg: EmbeddingConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.dimension = cfg.dimension
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model, device=self.device)
        return self._model

    @serialized
    def embed_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    @serialized
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [self.cfg.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vec[0], dtype=np.float32)
