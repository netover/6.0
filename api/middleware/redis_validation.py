"""
Redis Validation Middleware

Intercepta todas as requests e valida disponibilidade de Redis
baseado no tier do endpoint.

Comportamento por Tier:
- READ_ONLY: Sempre permite (nunca precisa Redis)
- BEST_EFFORT: Permite mas adiciona header de degradação
- CRITICAL: Retorna 503 se Redis indisponível

v5.4.2: Unifica modo degradado Redis com documentação de tiers.
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

# Lazy import to avoid circular dependencies
_redis_strategy = None


def get_redis_strategy():
    """Lazy load Redis strategy to avoid import issues."""
    global _redis_strategy
    if _redis_strategy is None:
        from resync.core.redis_strategy import get_redis_strategy as _get_strategy

        _redis_strategy = _get_strategy()
    return _redis_strategy


class RedisValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware que valida disponibilidade de Redis por tier de endpoint.

    Adiciona headers:
    - X-Redis-Status: available|unavailable
    - X-Degraded-Mode: true (se degradado)
    - X-Degraded-Reason: mensagem explicativa
    - X-Cost-Impact: impacto de custo (se aplicável)

    Comportamento:
    - READ_ONLY: Sempre permite
    - BEST_EFFORT: Permite com headers de aviso
    - CRITICAL: Retorna 503 se Redis down
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._strategy = None

    @property
    def strategy(self):
        """Lazy load strategy."""
        if self._strategy is None:
            self._strategy = get_redis_strategy()
        return self._strategy

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Valida Redis e processa request."""

        # Skip validation for static files and docs
        path = request.url.path
        if path.startswith(("/static", "/docs", "/redoc", "/openapi.json")):
            return await call_next(request)

        start_time = time.time()

        # Get endpoint info
        method = request.method

        # Import here to avoid issues
        from resync.core.redis_strategy import RedisTier

        tier = self.strategy.get_tier(method, path)

        # Check Redis availability from app state
        redis_available = getattr(request.app.state, "redis_available", True)

        # Log validation
        logger.debug(
            "redis_validation_check",
            method=method,
            path=path,
            tier=tier.value,
            redis_available=redis_available,
        )

        # TIER 1: READ_ONLY - Always allow
        if tier == RedisTier.READ_ONLY:
            response = await call_next(request)
            response.headers["X-Redis-Status"] = "available" if redis_available else "unavailable"
            return response

        # TIER 3: CRITICAL - Fail fast if Redis down
        if tier == RedisTier.CRITICAL and not redis_available:
            logger.warning(
                "redis_critical_endpoint_blocked",
                method=method,
                path=path,
                tier="critical",
            )

            # Get detailed config
            critical_config = self.strategy.get_critical_config(method, path)
            reason = (
                critical_config.get("reason", "Redis required for this operation")
                if critical_config
                else "Redis required"
            )
            retry_after = critical_config.get("retry_after", 60) if critical_config else 60

            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service Temporarily Unavailable",
                    "reason": reason,
                    "tier": "critical",
                    "endpoint": f"{method} {path}",
                    "retry_after": retry_after,
                    "message": "Redis is required for this operation. Please try again later.",
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-Redis-Status": "unavailable",
                },
            )

        # TIER 2: BEST_EFFORT - Degrade gracefully
        if tier == RedisTier.BEST_EFFORT and not redis_available:
            # Get degradation config
            degraded_config = self.strategy.get_degraded_config(method, path)

            logger.info(
                "redis_degraded_mode_request",
                method=method,
                path=path,
                tier="best_effort",
                degraded_behavior=degraded_config.get("behavior") if degraded_config else None,
            )

            # Store degradation info in request state
            request.state.degraded_mode = True
            request.state.degraded_config = degraded_config

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(
                "request_processing_error",
                method=method,
                path=path,
                error=str(e),
            )
            raise

        # Add standard headers
        response.headers["X-Redis-Status"] = "available" if redis_available else "unavailable"

        # Add degradation headers if applicable
        if getattr(request.state, "degraded_mode", False):
            response.headers["X-Degraded-Mode"] = "true"

            config = getattr(request.state, "degraded_config", None)
            if config:
                if warning := config.get("warning"):
                    response.headers["X-Degraded-Reason"] = warning
                if cost_impact := config.get("cost_impact"):
                    response.headers["X-Cost-Impact"] = str(cost_impact)

        # Add processing time
        duration = time.time() - start_time
        response.headers["X-Processing-Time"] = f"{duration:.3f}s"

        return response


class RedisHealthMiddleware(BaseHTTPMiddleware):
    """
    Middleware simplificado que apenas adiciona status do Redis aos headers.

    Usar quando não precisar de validação por tier, apenas informação.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Adiciona header de status Redis."""
        response = await call_next(request)

        redis_available = getattr(request.app.state, "redis_available", True)
        response.headers["X-Redis-Status"] = "available" if redis_available else "unavailable"

        return response
