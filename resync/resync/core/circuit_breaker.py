"""Backward-compatible circuit breaker shim.

Historically, Resync shipped multiple independent circuit breaker
implementations (this module, ``resync.core.resilience``, and a registry).
That duplication is risky: importing the "wrong" CircuitBreaker can silently
change semantics, metrics, and state handling.

This module now **consolidates** on the full-featured implementation in
``resync.core.resilience`` while preserving the legacy constructor signature
used by older code.

New code should import from ``resync.core.resilience`` directly.
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar

from resync.core.resilience import CircuitBreaker as _ResilienceCircuitBreaker
from resync.core.resilience import CircuitBreakerConfig

T = TypeVar("T")


class CircuitBreaker(_ResilienceCircuitBreaker):
    """Compatibility wrapper around :class:`resync.core.resilience.CircuitBreaker`.

    Legacy callers used ``CircuitBreaker(failure_threshold=..., recovery_timeout=..., ...)``.
    The resilience implementation expects a :class:`CircuitBreakerConfig`.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        name: str = "default",
        expected_exception: type[Exception] = Exception,
        *,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        warnings.warn(
            "resync.core.circuit_breaker.CircuitBreaker is deprecated; "
            "use resync.core.resilience.CircuitBreaker instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if config is None:
            config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
            )
        super().__init__(config)


# ---------------------------------------------------------------------------
# Backward-compatible convenience singletons
# ---------------------------------------------------------------------------


adaptive_tws_api_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    name="adaptive_tws_api",
)


adaptive_llm_api_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120,
    name="adaptive_llm_api",
)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "adaptive_tws_api_breaker",
    "adaptive_llm_api_breaker",
]
