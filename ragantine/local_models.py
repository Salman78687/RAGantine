"""Memory safety for the local CPU models (embedder, reranker, D2 classifier, D5 NLI).

The query graph and the benchmark run many questions concurrently in worker threads. Without
a guard, each thread runs its own model inference at the same time (and first calls can load
duplicate copies of a model), which pushed a laptop run past 8 GB of RAM. Every local-model
call goes through one process-wide lock, so at most one inference is in memory at a time; LLM
calls (network) still run concurrently. Torch's thread count is capped so the desktop stays
responsive (``RAGANTINE_TORCH_THREADS``, default 4).
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

MODEL_LOCK = threading.RLock()
DEFAULT_TORCH_THREADS = 4
_configured = False


def configure_torch_threads() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("RAGANTINE_TORCH_THREADS", DEFAULT_TORCH_THREADS)))
    except ImportError:  # pragma: no cover - torch is a dependency, but keep fakes importable
        pass


def serialized(fn: Callable[P, R]) -> Callable[P, R]:
    """Run a local-model call under the process-wide model lock."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with MODEL_LOCK:
            configure_torch_threads()
            return fn(*args, **kwargs)

    return wrapper
