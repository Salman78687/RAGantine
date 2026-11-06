"""Local model calls are serialised: concurrent threads never overlap inside a model."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ragantine.local_models import serialized


def test_concurrent_calls_never_overlap() -> None:
    active, peak = 0, 0
    guard = threading.Lock()

    @serialized
    def infer(x: int) -> int:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return x * 2

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(infer, range(16))) == [x * 2 for x in range(16)]
    assert peak == 1


def test_serialized_is_reentrant() -> None:
    @serialized
    def outer() -> int:
        return inner() + 1

    @serialized
    def inner() -> int:
        return 1

    assert outer() == 2


async def test_dns_lookups_are_not_starved_by_queued_model_work() -> None:
    """Regression (2026-11-06): queued model calls filled asyncio's default executor, so
    getaddrinfo for LLM connections waited behind them and the requests timed out."""
    import asyncio

    from ragantine.local_models import run_model

    release = threading.Event()
    blocked = [asyncio.ensure_future(run_model(release.wait, 5)) for _ in range(50)]
    try:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        await asyncio.wait_for(loop.getaddrinfo("localhost", 443), timeout=2)
        assert time.perf_counter() - start < 2
    finally:
        release.set()
        await asyncio.gather(*blocked)
