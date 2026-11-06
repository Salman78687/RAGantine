"""Memory safety for the local CPU models (embedder, reranker, D2 classifier, D5 NLI).

The query graph and the benchmark run many questions concurrently in worker threads. Without
a guard, each thread runs its own model inference at the same time (and first calls can load
duplicate copies of a model), which pushed a laptop run past 8 GB of RAM. Every local-model
call goes through one process-wide lock, so at most one inference is in memory at a time; LLM
calls (network) still run concurrently. Torch's thread count is capped so the desktop stays
responsive (``RAGANTINE_TORCH_THREADS``, default 4).
"""

from __future__ import annotations

import asyncio
import functools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


# Local-model work gets its own small thread pool. asyncio's default executor is also where
# DNS lookups for outgoing HTTPS run (getaddrinfo); when many queued model calls occupied it,
# LLM connections could not resolve their host and timed out.
DEFAULT_MODEL_WORKERS = 2
_executor: ThreadPoolExecutor | None = None


def model_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = int(os.environ.get("RAGANTINE_MODEL_WORKERS", DEFAULT_MODEL_WORKERS))
        _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ragantine-model")
    return _executor


async def run_model(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run blocking retrieval/defense/model work off the event loop, in the model pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(model_executor(), functools.partial(fn, *args, **kwargs))
