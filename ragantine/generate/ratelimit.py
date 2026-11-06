"""Client-side token pacing per model (e.g. Groq's tokens-per-minute limit).

Groq charges each request its prompt plus ``max_completion_tokens`` against a per-minute
budget. Reacting to 429s alone made concurrent requests collide and exhaust their retries,
so requests now wait *before* sending until the last 60 s of charges leave room. After the
response, the charge is corrected to the tokens actually used.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

Message = dict[str, str]
CHARS_PER_TOKEN = 3  # conservative: overestimating only slows us a little
PER_MESSAGE_OVERHEAD = 8
WINDOW_S = 60.0


def estimate_tokens(messages: Sequence[Message], max_completion_tokens: int) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // CHARS_PER_TOKEN + PER_MESSAGE_OVERHEAD * len(messages) + max_completion_tokens


@dataclass
class _Charge:
    at: float
    tokens: int


@dataclass
class TokenRateLimiter:
    tokens_per_minute: int
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    window_s: float = WINDOW_S
    _charges: deque[_Charge] = field(default_factory=deque)

    def _used(self, now: float) -> int:
        while self._charges and now - self._charges[0].at >= self.window_s:
            self._charges.popleft()
        return sum(c.tokens for c in self._charges)

    async def acquire(self, tokens: int) -> _Charge:
        """Wait until ``tokens`` fit in the window, then charge them. Oversized requests go alone."""
        while True:
            now = self.clock()
            used = self._used(now)
            if used + tokens <= self.tokens_per_minute or not self._charges:
                charge = _Charge(at=now, tokens=tokens)
                self._charges.append(charge)
                return charge
            await self.sleep(max(0.05, self._charges[0].at + self.window_s - now))

    @staticmethod
    def settle(charge: _Charge, actual_tokens: int) -> None:
        """Replace the estimate with what the request really used."""
        charge.tokens = actual_tokens
