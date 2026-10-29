"""Transient-failure retries, measured separately from success.

The important rule here is not the backoff curve, it is the bookkeeping: the
latency a report shows is the latency of the *final* attempt, and the time spent
retrying is carried alongside it as `retry_ms` plus a count. A candidate that
needs three retries per call is slow and unreliable, and it must not be able to
hide behind a healthy-looking p95.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from slmeval.providers.base import ProviderError, RetryPolicy

T = TypeVar("T")


@dataclass(slots=True)
class Attempted(Generic[T]):
    """A result plus what it took to get there."""

    value: T
    retries: int
    retry_ms: float


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Exponential backoff with jitter.

    Jitter matters more than it looks: without it, a 90-execution matrix that
    hits one rate limit will have every worker retry in lockstep and hit it
    again together.
    """
    raw: float = min(policy.base_delay_s * float(2**attempt), policy.max_delay_s)
    spread: float = raw * policy.jitter
    delay: float = raw + rng.uniform(-spread, spread)
    return max(0.0, delay)


async def with_retries(
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> Attempted[T]:
    """Run `call`, retrying only transient provider failures.

    A 400 is never retried: it means our request or the user's config is wrong,
    and retrying it three times only makes that slower to discover.
    """
    rng = rng or random.Random()
    nap = sleep or asyncio.sleep
    retries = 0
    retry_ms = 0.0
    last: ProviderError | None = None

    for attempt in range(policy.attempts):
        try:
            value = await call()
        except ProviderError as exc:
            if not exc.transient or attempt == policy.attempts - 1:
                raise
            last = exc
            delay = backoff_delay(attempt, policy, rng)
            retries += 1
            retry_ms += delay * 1000.0
            await nap(delay)
            continue
        return Attempted(value=value, retries=retries, retry_ms=retry_ms)

    # Unreachable: the loop either returns or re-raises on the last attempt.
    raise last if last else ProviderError("retry loop exhausted with no error recorded")


__all__ = ["Attempted", "backoff_delay", "with_retries"]
