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
