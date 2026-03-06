# pylint: disable=unused-argument
"""Async utilities for consistent timeouts and error classification.

This module centralizes:
- asyncio.wait_for wrappers with consistent error messages
- classification of HTTP/provider errors (timeout vs 429 vs 5xx vs network)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Optional, Tuple


@dataclass(frozen=True)
class ClassifiedError:
    reason: str  # timeout|rate_limit|provider_5xx|network|auth|bad_request|other
    status_code: int | None = None
    provider: str | None = None


async def with_timeout(coro: Awaitable[Any], timeout_s: float, *, op: str = "operation") -> Any:
    """Run coroutine with timeout, raising asyncio.TimeoutError on expiry."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise asyncio.TimeoutError(f"{op} timed out after {timeout_s:.2f}s") from exc


def _extract_status_code(exc: BaseException) -> int | None:
    # Common patterns across httpx, requests-like, and LiteLLM/OpenAI compatible errors
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def classify_exception(exc: BaseException) -> ClassifiedError:
    """Best-effort classification for external/provider exceptions."""
    status = _extract_status_code(exc)
    msg = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError) or "timed out" in msg or "timeout" in msg:
        return ClassifiedError(reason="timeout", status_code=status)

    if status == 429 or "429" in msg or ("rate" in msg and "limit" in msg):
        return ClassifiedError(reason="rate_limit", status_code=status)

    if status in (401, 403) or "unauthorized" in msg or "forbidden" in msg or "auth" in msg:
        return ClassifiedError(reason="auth", status_code=status)

    if status is not None and 400 <= status < 500:
        return ClassifiedError(reason="bad_request", status_code=status)

    if status is not None and 500 <= status < 600:
        return ClassifiedError(reason="provider_5xx", status_code=status)

    # Network-ish heuristics
    if any(x in msg for x in ("connection reset", "connection refused", "dns", "name or service not known", "temporarily unavailable", "network", "socket")):
        return ClassifiedError(reason="network", status_code=status)

    return ClassifiedError(reason="other", status_code=status)
