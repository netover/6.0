"""Logging CORS middleware.

This middleware wraps Starlette's built-in :class:`~starlette.middleware.cors.CORSMiddleware`
to preserve correct CORS behavior (especially preflight / OPTIONS handling) while adding:

- Security monitoring logs for disallowed origins
- Lightweight counters/metrics (in-memory) for observability

Why this exists:
- A common anti-pattern is implementing CORS as BaseHTTPMiddleware and manually
  adding a few headers. That often breaks preflight semantics because OPTIONS
  requests are not short-circuited correctly.
- Starlette's CORSMiddleware has correct behavior; we delegate to it and only
  add logging/metrics around the decision.

Design constraints:
- Must be ASGI-native (not BaseHTTPMiddleware) to avoid body buffering and
  performance regressions in async stacks.
- Compatible with FastAPI's `app.add_middleware(LoggingCORSMiddleware, ...)`.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from resync.core.context import get_correlation_id
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CORSMetrics:
    """Thread/task-safe CORS metrics using a lock."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _total_requests: int = 0
    _preflight_requests: int = 0
    _allowed_origins: int = 0
    _denied_origins: int = 0

    def inc_total(self) -> None:
        with self._lock:
            self._total_requests += 1

    def inc_preflight(self) -> None:
        with self._lock:
            self._preflight_requests += 1

    def inc_allowed(self) -> None:
        with self._lock:
            self._allowed_origins += 1

    def inc_denied(self) -> None:
        with self._lock:
            self._denied_origins += 1

    @property
    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "preflight_requests": self._preflight_requests,
                "allowed_origins": self._allowed_origins,
                "denied_origins": self._denied_origins,
            }

class LoggingCORSMiddleware:
    """CORS middleware with security logging, delegating to CORSMiddleware."""

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list[str] | None = None,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        allow_credentials: bool = False,
        expose_headers: list[str] | None = None,
        max_age: int = 86400,
        allow_origin_regex: str | None = None,
        log_violations: bool = True,
    ) -> None:
        self._allow_origins = allow_origins or []
        self._allow_origin_regex = allow_origin_regex
        self._allow_all = "*" in self._allow_origins
        self._log_violations = bool(log_violations)
        self.metrics = CORSMetrics()

        # Build the canonical CORSMiddleware app
        self._cors_app = CORSMiddleware(
            app,
            allow_origins=self._allow_origins,
            allow_methods=allow_methods or ["*"],
            allow_headers=allow_headers or ["*"],
            allow_credentials=allow_credentials,
            expose_headers=expose_headers or [],
            max_age=max_age,
            allow_origin_regex=allow_origin_regex,
        )

    def _is_origin_allowed(self, origin: str) -> bool:
        if self._allow_all:
            return True
        if origin in self._allow_origins:
            return True
        if self._allow_origin_regex:
            try:
                return re.match(self._allow_origin_regex, origin) is not None
            except re.error:
                # If regex is invalid, fail closed and log once per request
                logger.warning(
                    "cors_invalid_origin_regex", pattern=self._allow_origin_regex
                )
                return False
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._cors_app(scope, receive, send)
            return

        self.metrics.inc_total()

        headers = Headers(scope=scope)
        origin = headers.get("origin")

        is_preflight = (
            scope.get("method") == "OPTIONS"
            and headers.get("access-control-request-method") is not None
        )
        if is_preflight:
            self.metrics.inc_preflight()

        if origin:
            allowed = self._is_origin_allowed(origin)
            if allowed:
                self.metrics.inc_allowed()
            else:
                self.metrics.inc_denied()
                if self._log_violations:
                    logger.warning(
                        "cors_violation",
                        origin=origin,
                        method=scope.get("method"),
                        path=scope.get("path"),
                        correlation_id=get_correlation_id(),
                        preflight=is_preflight,
                        user_agent=headers.get("user-agent"),
                        referer=headers.get("referer"),
                    )

        # Delegate to CORSMiddleware for correct behavior
        started = time.perf_counter()
        try:
            await self._cors_app(scope, receive, send)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            # Keep this as debug to avoid high-cardinality logs
            logger.debug(
                "cors_request_processed",
                ms=round(duration_ms, 2),
                preflight=is_preflight,
                correlation_id=get_correlation_id(),
            )
