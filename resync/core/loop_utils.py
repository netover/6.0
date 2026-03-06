from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Dict, Any


@dataclass
class Backoff:
    base_seconds: float = 1.0
    max_seconds: float = 30.0
    factor: float = 2.0
    jitter: float = 0.2

    _current: float = 0.0

    def __post_init__(self) -> None:
        self._current = float(self.base_seconds)

    def reset(self) -> None:
        self._current = float(self.base_seconds)

    def next_delay(self) -> float:
        delay = self._current
        # Apply jitter (+/- jitter*delay)
        if self.jitter:
            spread = delay * float(self.jitter)
            delay = max(0.0, delay + random.uniform(-spread, spread))  # noqa: S311 - jitter is non-crypto by design
        # Increase for next time
        self._current = min(float(self.max_seconds), self._current * float(self.factor))
        return delay


# Best-effort, in-process loop stats (useful for spotting stuck loops).
_LOOP_STATS: Dict[str, Dict[str, Any]] = {}


def get_loop_stats() -> Dict[str, Dict[str, Any]]:
    # return a shallow copy to avoid accidental mutation by callers
    return {k: dict(v) for k, v in _LOOP_STATS.items()}


def _record_loop_stat(name: str, *, duration_s: float, ok: bool, delay_s: float = 0.0) -> None:
    s = _LOOP_STATS.setdefault(
        name,
        {
            "iterations": 0,
            "ok": 0,
            "errors": 0,
            "last_duration_s": None,
            "avg_duration_s": None,
            "last_backoff_s": 0.0,
            "last_error": None,
            "last_updated_ts": None,
        },
    )
    s["iterations"] += 1
    s["ok"] += 1 if ok else 0
    s["errors"] += 0 if ok else 1
    s["last_duration_s"] = float(duration_s)
    # Exponential moving average for stability
    prev = s["avg_duration_s"]
    if prev is None:
        s["avg_duration_s"] = float(duration_s)
    else:
        alpha = 0.15
        s["avg_duration_s"] = float(prev) * (1.0 - alpha) + float(duration_s) * alpha
    s["last_backoff_s"] = float(delay_s)
    s["last_updated_ts"] = time.time()


async def run_resilient_loop(
    name: str,
    step: Callable[[], Awaitable[None]],
    *,
    logger=None,
    backoff: Optional[Backoff] = None,
    step_timeout_seconds: Optional[float] = 60.0,
) -> None:
    """Run an async loop forever with cancellation-safety, step timeouts and exponential backoff.

    - `step()` is one iteration.
    - `step_timeout_seconds` protects against stuck awaits; set to None to disable.
    """
    bo = backoff or Backoff()
    while True:
        started = time.perf_counter()
        try:
            if step_timeout_seconds is None:
                await step()
            else:
                await asyncio.wait_for(step(), timeout=float(step_timeout_seconds))
            bo.reset()
            _record_loop_stat(name, duration_s=time.perf_counter() - started, ok=True, delay_s=0.0)
        except asyncio.CancelledError:
            # Never swallow cancellations (shutdown / disconnect / reload)
            raise
        except asyncio.TimeoutError as e:
            # Treat as operational error (retry with backoff)
            duration = time.perf_counter() - started
            st = _LOOP_STATS.setdefault(name, {})
            if isinstance(st, dict):
                st["last_error"] = "timeout"
            if logger is not None:
                try:
                    logger.warning("Resilient loop '%s' step timed out after %ss", name, step_timeout_seconds)
                except Exception:
                    # logging must never break loop progression
                    _ = None
            delay = bo.next_delay()
            _record_loop_stat(name, duration_s=duration, ok=False, delay_s=delay)
            await asyncio.sleep(delay)
        except Exception as e:
            duration = time.perf_counter() - started
            st = _LOOP_STATS.setdefault(name, {})
            if isinstance(st, dict):
                st["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if logger is not None:
                try:
                    logger.error("Resilient loop '%s' step failed: %s", name, e, exc_info=True)
                except Exception:
                    # logging must never break the loop
                    _ = None
            delay = bo.next_delay()
            _record_loop_stat(name, duration_s=duration, ok=False, delay_s=delay)
            await asyncio.sleep(delay)
