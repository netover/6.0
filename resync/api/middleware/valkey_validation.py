"""Valkey validation middleware.

Intercepts requests and validates Valkey availability based on the endpoint tier.

Tiers:
- READ_ONLY: always allows (does not require Valkey)
- BEST_EFFORT: allows but annotates response headers for degraded mode
- CRITICAL: returns 503 if Valkey is unavailable

This module is the canonical import location:
    from resync.api.middleware.valkey_validation import ValkeyValidationMiddleware
"""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

_valkey_strategy = None


def get_valkey_strategy():
    """Lazy load Valkey strategy to avoid circular imports."""
    global _valkey_strategy
    if _valkey_strategy is None:
        from resync.core.valkey_strategy import get_valkey_strategy as _get_strategy

        _valkey_strategy = _get_strategy()
    return _valkey_strategy


class ValkeyValidationMiddleware(BaseHTTPMiddleware):
    """Validate Valkey availability per endpoint tier."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._strategy = None

    @property
    def strategy(self):
        if self._strategy is None:
            self._strategy = get_valkey_strategy()
        return self._strategy

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip validation for static files and docs
        path = request.url.path
        if path.startswith(("/static", "/docs", "/redoc", "/openapi.json")):
            return await call_next(request)

        start_time = time.time()
        method = request.method

        from resync.core.valkey_strategy import ValkeyTier

        tier = self.strategy.get_tier(method, path)
        valkey_available = getattr(request.app.state, "valkey_available", True)

        logger.debug(
            "valkey_validation_check",
            method=method,
            path=path,
            tier=tier.value,
            valkey_available=valkey_available,
        )

        # READ_ONLY: Always allow
        if tier == ValkeyTier.READ_ONLY:
            response = await call_next(request)
            response.headers["X-Valkey-Status"] = "available" if valkey_available else "unavailable"
            return response

        # CRITICAL: Fail fast if Valkey down
        if tier == ValkeyTier.CRITICAL and not valkey_available:
            critical_config = self.strategy.get_critical_config(method, path)
            reason = (
                critical_config.get("reason", "Valkey required for this operation")
                if critical_config
                else "Valkey required"
            )
            retry_after = critical_config.get("retry_after", 60) if critical_config else 60

            logger.warning(
                "valkey_critical_endpoint_blocked",
                method=method,
                path=path,
                tier="critical",
            )

            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service Temporarily Unavailable",
                    "reason": reason,
                    "tier": "critical",
                    "endpoint": f"{method} {path}",
                    "retry_after": retry_after,
                    "message": "Valkey is required for this operation. Please try again later.",
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-Valkey-Status": "unavailable",
                },
            )

        # BEST_EFFORT: Degrade gracefully
        if tier == ValkeyTier.BEST_EFFORT and not valkey_available:
            degraded_config = self.strategy.get_degraded_config(method, path)
            request.state.degraded_mode = True
            request.state.degraded_config = degraded_config

            logger.info(
                "valkey_degraded_mode_request",
                method=method,
                path=path,
                tier="best_effort",
                degraded_behavior=degraded_config.get("behavior") if degraded_config else None,
            )

        response = await call_next(request)

        response.headers["X-Valkey-Status"] = "available" if valkey_available else "unavailable"

        if getattr(request.state, "degraded_mode", False):
            response.headers["X-Degraded-Mode"] = "true"
            config = getattr(request.state, "degraded_config", None)
            if config:
                if warning := config.get("warning"):
                    response.headers["X-Degraded-Reason"] = warning
                if cost_impact := config.get("cost_impact"):
                    response.headers["X-Cost-Impact"] = str(cost_impact)

        response.headers["X-Processing-Time"] = f"{(time.time() - start_time):.3f}s"
        return response


class ValkeyHealthMiddleware(BaseHTTPMiddleware):
    """Attach Valkey availability as a response header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        valkey_available = getattr(request.app.state, "valkey_available", True)
        response.headers["X-Valkey-Status"] = "available" if valkey_available else "unavailable"
        return response


__all__ = ["ValkeyValidationMiddleware", "ValkeyHealthMiddleware"]
