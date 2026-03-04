"""
Valkey Validation Middleware

Intercepta todas as requests e valida disponibilidade de Valkey
baseado no tier do endpoint.

Comportamento por Tier:
- READ_ONLY: Sempre permite (nunca precisa Valkey)
- BEST_EFFORT: Permite mas adiciona header de degradação
- CRITICAL: Retorna 503 se Valkey indisponível

v6.3: Migrado de Redis para Valkey.
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
_valkey_strategy = None

def get_valkey_strategy():
    """Lazy load Valkey strategy to avoid import issues."""
    global _valkey_strategy
    if _valkey_strategy is None:
        from resync.core.valkey_strategy import get_valkey_strategy as _get_strategy

        _valkey_strategy = _get_strategy()
    return _valkey_strategy

class ValkeyValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware que valida disponibilidade de Valkey por tier de endpoint.

    Adiciona headers:
    - X-Valkey-Status: available|unavailable (X-Redis-Status para compatibilidade)
    - X-Degraded-Mode: true (se degradado)
    - X-Degraded-Reason: mensagem explicativa
    - X-Cost-Impact: impacto de custo (se aplicável)

    Comportamento:
    - READ_ONLY: Sempre permite
    - BEST_EFFORT: Permite com headers de aviso
    - CRITICAL: Retorna 503 se Valkey down
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._strategy = None

    @property
    def strategy(self):
        """Lazy load strategy."""
        if self._strategy is None:
            self._strategy = get_valkey_strategy()
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
        from resync.core.valkey_strategy import ValkeyTier

        tier = self.strategy.get_tier(method, path)

        # Check Valkey availability from app state
        valkey_available = getattr(request.app.state, "valkey_available", True)

        # Log validation
        logger.debug(
            "valkey_validation_check",
            method=method,
            path=path,
            tier=tier.value,
            valkey_available=valkey_available,
        )

        # TIER 1: READ_ONLY - Always allow
        if tier == ValkeyTier.READ_ONLY:
            response = await call_next(request)
            response.headers["X-Valkey-Status"] = (
                "available" if valkey_available else "unavailable"
            )
            # Legacy header for backward compatibility
            response.headers["X-Redis-Status"] = response.headers["X-Valkey-Status"]
            return response

        # TIER 3: CRITICAL - Fail fast if Valkey down
        if tier == ValkeyTier.CRITICAL and not valkey_available:
            logger.warning(
                "valkey_critical_endpoint_blocked",
                method=method,
                path=path,
                tier="critical",
            )

            # Get detailed config
            critical_config = self.strategy.get_critical_config(method, path)
            reason = (
                critical_config.get("reason", "Valkey required for this operation")
                if critical_config
                else "Valkey required"
            )
            retry_after = (
                critical_config.get("retry_after", 60) if critical_config else 60
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
                    "X-Redis-Status": "unavailable",
                },
            )

        # TIER 2: BEST_EFFORT - Degrade gracefully
        if tier == ValkeyTier.BEST_EFFORT and not valkey_available:
            # Get degradation config
            degraded_config = self.strategy.get_degraded_config(method, path)

            logger.info(
                "valkey_degraded_mode_request",
                method=method,
                path=path,
                tier="best_effort",
                degraded_behavior=degraded_config.get("behavior")
                if degraded_config
                else None,
            )

            # Store degradation info in request state
            request.state.degraded_mode = True
            request.state.degraded_config = degraded_config

        # Process request
        try:
            response = await call_next(request)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "request_processing_error",
                method=method,
                path=path,
                error=str(e),
            )
            raise

        # Add standard headers
        response.headers["X-Valkey-Status"] = (
            "available" if valkey_available else "unavailable"
        )
        response.headers["X-Redis-Status"] = response.headers["X-Valkey-Status"]

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

class ValkeyHealthMiddleware(BaseHTTPMiddleware):
    """
    Middleware simplificado que apenas adiciona status do Valkey aos headers.

    Usar quando não precisar de validação por tier, apenas informação.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Adiciona header de status Valkey."""
        response = await call_next(request)

        valkey_available = getattr(request.app.state, "valkey_available", True)
        response.headers["X-Valkey-Status"] = (
            "available" if valkey_available else "unavailable"
        )
        response.headers["X-Redis-Status"] = response.headers["X-Valkey-Status"]

        return response


# Legacy aliases for backward compatibility
RedisValidationMiddleware = ValkeyValidationMiddleware
RedisHealthMiddleware = ValkeyHealthMiddleware
