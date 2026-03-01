from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from resync.core.redis_init import get_redis_client
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

def _client_ip(request: Request) -> str:
    hdr = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if hdr:
        return hdr
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
