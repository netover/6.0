# ruff: noqa: E501
# pylint: disable=too-few-public-methods
"""
Enterprise-grade rate limiting for FastAPI / Starlette.

Why not depend on external libraries?
- In VM deployments with small concurrency (~25 users), a simple in-memory
  sliding-window limiter is reliable, low-dependency, and easy to audit.
- Each Gunicorn worker maintains its own limiter (acceptable for this scale).
- If you later deploy behind a load balancer / multiple nodes, you can switch
  to a Redis-backed limiter without changing the call sites.

Design goals:
- Fail-closed in production ONLY if explicitly enabled (RATE_LIMIT_ENABLED=true).
- Different defaults for auth vs general API.
- Exempt healthcheck endpoints by default.
- Minimal overhead, safe under asyncio (uses an asyncio.Lock).

References:
- OWASP API Security recommends throttling/rate limiting as a standard control.
  https://owasp.org/API-Security/
"""

from __future__ import annotations

import asyncio
import functools
import os
import re
import time
from collections import deque
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# -----------------------------
# Configuration (env-driven)
# -----------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# Defaults: tuned for small internal network usage.
AUTH_LIMIT_REQUESTS = _env_int("RATE_LIMIT_AUTH_REQUESTS", 10)
AUTH_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_AUTH_WINDOW_SECONDS", 60)

API_LIMIT_REQUESTS = _env_int("RATE_LIMIT_API_REQUESTS", 120)
API_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_API_WINDOW_SECONDS", 60)

WS_CONNECT_LIMIT_REQUESTS = _env_int("RATE_LIMIT_WS_CONNECT_REQUESTS", 20)
WS_CONNECT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WS_CONNECT_WINDOW_SECONDS", 60)

# Comma-separated path prefixes that bypass rate limiting.
# Health endpoints should be cheap and always reachable by monitoring.
BYPASS_PREFIXES = tuple(
    p.strip()
    for p in os.getenv(
        "RATE_LIMIT_BYPASS_PREFIXES",
        "/health,/healthz,/liveness,/readiness,/metrics",
    ).split(",")
    if p.strip()
)

# -----------------------------
# In-memory sliding window
# -----------------------------

@dataclass(frozen=True, slots=True)
class Limit:
    requests: int
    window_seconds: int

class SlidingWindowLimiter:
    """
    Sliding-window limiter using a deque of timestamps per key.

    Key suggestions:
      - HTTP: f"ip:{client_ip}:{bucket}"
      - WS:   f"ws:{client_ip}"
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str, limit: Limit) -> tuple[bool, int]:
        """
        Returns: (allowed, retry_after_seconds)
        """
        now = time.monotonic()
        cutoff = now - float(limit.window_seconds)

        async with self._lock:
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q

            # evict old
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) < limit.requests:
                q.append(now)
                return True, 0

            # compute retry-after
            oldest = q[0]
            retry_after = int(max(1, (oldest + limit.window_seconds) - now))
            return False, retry_after

# Module-global limiter (per worker process)
_LIMITER = SlidingWindowLimiter()


def _client_ip(request: Request) -> str:
    """Extract client IP from Starlette Request (for decorators)."""
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _client_ip_from_scope(scope: Scope) -> str:
    """Extract client IP from ASGI scope (for middleware)."""
    client = scope.get("client")
    if client and client[0]:
        return client[0]
    return "unknown"


def _choose_http_limit(path: str) -> Limit:
    if path.startswith("/api/v1/auth") or path.startswith("/auth"):
        return Limit(AUTH_LIMIT_REQUESTS, AUTH_LIMIT_WINDOW_SECONDS)
    return Limit(API_LIMIT_REQUESTS, API_LIMIT_WINDOW_SECONDS)


class RateLimitMiddleware:
    """Pure ASGI rate limiting middleware - no body buffering."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if BYPASS_PREFIXES and path.startswith(BYPASS_PREFIXES):
            await self.app(scope, receive, send)
            return

        client_ip = _client_ip_from_scope(scope)
        limit = _choose_http_limit(path)
        bucket = "auth" if limit.requests == AUTH_LIMIT_REQUESTS and limit.window_seconds == AUTH_LIMIT_WINDOW_SECONDS else "api"
        key = f"ip:{client_ip}:{bucket}"

        allowed, retry_after = await _LIMITER.allow(key, limit)
        if not allowed:
            headers = [
                (b"retry-after", str(retry_after).encode()),
                (b"x-ratelimit-limit", str(limit.requests).encode()),
                (b"x-ratelimit-window", str(limit.window_seconds).encode()),
            ]
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after_seconds": retry_after},
                headers=dict(headers),
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

async def ws_allow_connect(client_ip: str) -> tuple[bool, int]:
    """Rate limit for websocket connect attempts."""
    if not RATE_LIMIT_ENABLED:
        return True, 0
    key = f"ws:{client_ip}"
    return await _LIMITER.allow(key, Limit(WS_CONNECT_LIMIT_REQUESTS, WS_CONNECT_WINDOW_SECONDS))

def setup_rate_limiting(app) -> None:
    """
    Called from app_factory during startup.

    We keep the API compatible with previous versions that used slowapi.
    """
    app.add_middleware(RateLimitMiddleware)


# =============================================================================
# rate_limit decorator — wraps endpoints with per-route sliding-window limiting
# Usage: @rate_limit("30/minute")
# =============================================================================

def rate_limit(limit_str: str):
    """Decorator: apply a sliding-window rate limit to an endpoint.

    Args:
        limit_str: Limit spec string, e.g. ``"30/minute"``, ``"10/second"``.

    Usage::

        @router.post("/login")
        @rate_limit("10/minute")
        async def login(request: Request): ...
    """
    _window_map = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

    def _parse(s: str) -> Limit:
        m = re.match(r"(\d+)\s*/\s*(\w+)", s.strip())
        if not m:
            raise ValueError(f"Invalid rate limit spec: {s!r}. Use 'N/unit' (e.g. '30/minute')")
        count = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")  # "minutes" -> "minute"
        if unit not in _window_map:
            raise ValueError(f"Unknown time unit {unit!r}. Use: second, minute, hour, day")
        return Limit(requests=count, window_seconds=_window_map[unit])

    lim = _parse(limit_str)
    _limiter = SlidingWindowLimiter()

    def decorator(func):
        if not RATE_LIMIT_ENABLED:
            return func

        @functools.wraps(func)
        async def wrapper(request: "Request", *args, **kwargs):
            ip = _client_ip(request)
            key = f"rl:{func.__name__}:{ip}"
            allowed, retry_after = await _limiter.allow(key, lim)
            if not allowed:
                from starlette.responses import JSONResponse
                return JSONResponse(
                    {"detail": "Too Many Requests", "retry_after": retry_after},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def rate_limit_auth(func):
    """Decorator: apply the auth-specific rate limit to an endpoint.

    Uses ``AUTH_LIMIT_REQUESTS / AUTH_LIMIT_WINDOW_SECONDS`` from config.
    """
    if not RATE_LIMIT_ENABLED:
        return func

    _limiter = SlidingWindowLimiter()
    _lim = Limit(requests=AUTH_LIMIT_REQUESTS, window_seconds=AUTH_LIMIT_WINDOW_SECONDS)

    @functools.wraps(func)
    async def wrapper(request: "Request", *args, **kwargs):
        ip = _client_ip(request)
        key = f"rl_auth:{ip}"
        allowed, retry_after = await _limiter.allow(key, _lim)
        if not allowed:
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"detail": "Too Many Requests", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return await func(request, *args, **kwargs)

    return wrapper
