"""Client-side per-minute token pacing."""

from ragantine.generate.ratelimit import TokenRateLimiter, estimate_tokens


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_requests_within_budget_go_immediately_and_overflow_waits_for_the_window() -> None:
    clock = FakeClock()
    lim = TokenRateLimiter(1000, clock=clock, sleep=clock.sleep)
    await lim.acquire(400)
    clock.now = 10.0
    await lim.acquire(500)
    assert clock.slept == []  # 900 <= 1000
    await lim.acquire(300)  # would be 1200: wait until the first charge (t=0) leaves the window
    assert clock.now == 60.0 and clock.slept == [50.0]


async def test_settling_to_actual_usage_frees_budget() -> None:
    clock = FakeClock()
    lim = TokenRateLimiter(1000, clock=clock, sleep=clock.sleep)
    charge = await lim.acquire(900)  # estimate includes max_tokens
    TokenRateLimiter.settle(charge, 300)  # actually used far less
    await lim.acquire(600)
    assert clock.slept == []


async def test_oversized_request_still_proceeds_alone() -> None:
    clock = FakeClock()
    lim = TokenRateLimiter(1000, clock=clock, sleep=clock.sleep)
    await lim.acquire(5000)
    assert clock.slept == []


def test_estimate_counts_prompt_and_completion_budget() -> None:
    msgs = [{"role": "user", "content": "x" * 300}]
    assert estimate_tokens(msgs, 600) == 100 + 8 + 600
