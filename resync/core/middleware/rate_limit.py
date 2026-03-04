from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from resync.core.valkey_init import get_redis_client
from resync.settings import get_settings

_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
  redis.call('EXPIRE', key, ttl)
end
if current > limit then
  return 0
end
return 1
"""

@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int


# Trusted proxy IPs - configure via environment
_TRUSTED_PROXIES: list[str] = []


def _init_trusted_proxies() -> list[str]:
    """Initialize trusted proxy list from environment."""
    import os
    proxies = os.getenv("TRUSTED_PROXY_IPS", "")
    return [p.strip() for p in proxies.split(",") if p.strip()] if proxies else []


def _client_ip(request: Request) -> str:
    """Extract client IP with proxy trust validation.
    
    Security fix: Only trust X-Forwarded-For from known proxy IPs.
    This prevents attackers from spoofing their IP via custom headers.
    """
    global _TRUSTED_PROXIES
    if not _TRUSTED_PROXIES:
        _TRUSTED_PROXIES = _init_trusted_proxies()
    
    # Check direct client first (always trusted)
    if request.client:
        client_ip = request.client.host
        if client_ip and (_TRUSTED_PROXIES is None or client_ip not in _TRUSTED_PROXIES):
            # Direct client - use it
            return client_ip
    
    # Only trust X-Forwarded-For if from trusted proxy
    client_host = request.client.host if request.client else None
    if _TRUSTED_PROXIES and client_host in _TRUSTED_PROXIES:
        hdr = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if hdr:
            return hdr
    
    # Fall back to direct client IP (untrusted proxy header ignored)
    if request.client:
        return request.client.host or "unknown"
    return "unknown"

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global Redis-backed rate limiter with per-path rules.

    Disabled if Redis is not configured. Fail-closed in production when Redis is unavailable.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._redis = None
        self._settings = get_settings()
        self._env = getattr(self._settings, "environment", "").lower()
        self._default = RateLimitRule(limit=int(getattr(self._settings, "rate_limit_default_per_minute", 300)), window_seconds=60)
        self._rules: dict[str, RateLimitRule] = {
            "/api/v1/feedback/submit": RateLimitRule(limit=int(getattr(self._settings, "rate_limit_feedback_per_minute", 30)), window_seconds=60),
        }

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path.startswith("/health"):
            return await call_next(request)

        try:
            redis = self._redis or get_redis_client()
            self._redis = redis
            rule = self._rules.get(request.url.path, self._default)
            ip = _client_ip(request)
            key = f"rate:{request.url.path}:{ip}"
            ok = await redis.eval(_LUA, 1, key, str(rule.limit), str(rule.window_seconds))
            if int(ok) != 1:
                resp = Response("Too Many Requests", status_code=429)
                resp.headers["Retry-After"] = str(rule.window_seconds)
                resp.headers["X-RateLimit-Limit"] = str(rule.limit)
                resp.headers["X-RateLimit-Reset"] = str(rule.window_seconds)
                # Remaining is unknown without additional Redis call; set to 0 for clarity.
                resp.headers["X-RateLimit-Remaining"] = "0"
                return resp
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._env in {"prod", "production"}:
                return Response("Rate limiter unavailable", status_code=503)
        return await call_next(request)
