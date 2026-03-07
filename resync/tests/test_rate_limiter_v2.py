from __future__ import annotations

from collections import deque

import pytest

from resync.core.security.rate_limiter_v2 import SlidingWindowLimiter


@pytest.mark.asyncio
async def test_cleanup_loop_eviction_removes_stale_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = SlidingWindowLimiter()
    limiter._cleanup_interval_seconds = 0
    limiter._stale_after_seconds = 10
    limiter._events = {
        "stale": deque([1.0]),
        "fresh": deque([100.0]),
        "empty": deque(),
    }

    sleep_calls = {"count": 0}

    async def _sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] > 1:
            raise asyncio.CancelledError

    import asyncio

    monkeypatch.setattr("resync.core.security.rate_limiter_v2.time.monotonic", lambda: 20.0)
    monkeypatch.setattr(asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await limiter._cleanup_loop()

    assert "stale" not in limiter._events
    assert "empty" not in limiter._events
    assert "fresh" in limiter._events
