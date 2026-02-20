"""
Circuit Breaker Registry

v5.4.2: Centralized circuit breaker management for all critical paths.

Features:
- Pre-configured circuit breakers for all services
- Automatic registration
- Metrics and monitoring
- Health reporting
- Decorator helpers

Usage:
    from resync.core.circuit_breaker_registry import (
        get_circuit_breaker,
        circuit_protected,
        CircuitBreakers,
    )

    # Use pre-configured breaker
    cb = get_circuit_breaker(CircuitBreakers.TWS_API)
    result = await cb.call(my_async_func)

    # Use decorator
    @circuit_protected(CircuitBreakers.LLM_API)
    async def call_llm(prompt: str):
        ...

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

import structlog

from resync.core.exceptions import CircuitBreakerError
from resync.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# CIRCUIT BREAKER DEFINITIONS
# =============================================================================


class CircuitBreakers(str, Enum):
    """
    Pre-defined circuit breakers for all critical paths.

    Each breaker has optimized settings for its use case.
    """

    # External APIs
    TWS_API = "tws_api"  # TWS REST API calls
    LLM_API = "llm_api"  # LLM provider calls
    LLM_OPENAI = "llm_openai"  # OpenAI specifically
    LLM_ANTHROPIC = "llm_anthropic"  # Anthropic specifically
    LLM_OPENROUTER = "llm_openrouter"  # OpenRouter specifically

    # Internal Services
    REDIS = "redis"  # Redis operations
    DATABASE = "database"  # Database queries
    RAG_RETRIEVAL = "rag_retrieval"  # RAG retrieval operations
    RAG_EMBEDDING = "rag_embedding"  # Embedding generation

    # Integration Points
    SIEM = "siem"  # SIEM integration
    TEAMS = "teams"  # Teams notifications
    WEBHOOK = "webhook"  # Webhook calls

    # Background Tasks
    HEALTH_CHECK = "health_check"  # Health check operations
    BACKGROUND_SYNC = "background_sync"  # Background sync tasks


@dataclass
class CircuitBreakerSpec:
    """Specification for a circuit breaker."""

    name: CircuitBreakers
    failure_threshold: int
    recovery_timeout: int
    description: str
    critical: bool = False  # If True, failures should alert


# Pre-configured specs for all circuit breakers
CIRCUIT_BREAKER_SPECS: dict[CircuitBreakers, CircuitBreakerSpec] = {
    # TWS API - moderate tolerance, medium recovery
    CircuitBreakers.TWS_API: CircuitBreakerSpec(
        name=CircuitBreakers.TWS_API,
        failure_threshold=5,
        recovery_timeout=60,
        description="TWS REST API calls",
        critical=True,
    ),
    # LLM APIs - higher tolerance (rate limits common), longer recovery
    CircuitBreakers.LLM_API: CircuitBreakerSpec(
        name=CircuitBreakers.LLM_API,
        failure_threshold=10,
        recovery_timeout=120,
        description="Generic LLM API calls",
        critical=False,
    ),
    CircuitBreakers.LLM_OPENAI: CircuitBreakerSpec(
        name=CircuitBreakers.LLM_OPENAI,
        failure_threshold=5,
        recovery_timeout=60,
        description="OpenAI API calls",
        critical=False,
    ),
    CircuitBreakers.LLM_ANTHROPIC: CircuitBreakerSpec(
        name=CircuitBreakers.LLM_ANTHROPIC,
        failure_threshold=5,
        recovery_timeout=60,
        description="Anthropic API calls",
        critical=False,
    ),
    CircuitBreakers.LLM_OPENROUTER: CircuitBreakerSpec(
        name=CircuitBreakers.LLM_OPENROUTER,
        failure_threshold=5,
        recovery_timeout=60,
        description="OpenRouter API calls",
        critical=False,
    ),
    # Redis - low tolerance (should be reliable), fast recovery
    CircuitBreakers.REDIS: CircuitBreakerSpec(
        name=CircuitBreakers.REDIS,
        failure_threshold=3,
        recovery_timeout=30,
        description="Redis operations",
        critical=True,
    ),
    # Database - low tolerance, fast recovery
    CircuitBreakers.DATABASE: CircuitBreakerSpec(
        name=CircuitBreakers.DATABASE,
        failure_threshold=3,
        recovery_timeout=30,
        description="Database queries",
        critical=True,
    ),
    # RAG - moderate tolerance, medium recovery
    CircuitBreakers.RAG_RETRIEVAL: CircuitBreakerSpec(
        name=CircuitBreakers.RAG_RETRIEVAL,
        failure_threshold=5,
        recovery_timeout=60,
        description="RAG retrieval operations",
        critical=False,
    ),
    CircuitBreakers.RAG_EMBEDDING: CircuitBreakerSpec(
        name=CircuitBreakers.RAG_EMBEDDING,
        failure_threshold=5,
        recovery_timeout=60,
        description="Embedding generation",
        critical=False,
    ),
    # Integrations - higher tolerance, longer recovery
    CircuitBreakers.SIEM: CircuitBreakerSpec(
        name=CircuitBreakers.SIEM,
        failure_threshold=10,
        recovery_timeout=300,
        description="SIEM integration",
        critical=False,
    ),
    CircuitBreakers.TEAMS: CircuitBreakerSpec(
        name=CircuitBreakers.TEAMS,
        failure_threshold=5,
        recovery_timeout=120,
        description="Teams notifications",
        critical=False,
    ),
    CircuitBreakers.WEBHOOK: CircuitBreakerSpec(
        name=CircuitBreakers.WEBHOOK,
        failure_threshold=5,
        recovery_timeout=120,
        description="Webhook calls",
        critical=False,
    ),
    # Background - very tolerant, slow recovery
    CircuitBreakers.HEALTH_CHECK: CircuitBreakerSpec(
        name=CircuitBreakers.HEALTH_CHECK,
        failure_threshold=10,
        recovery_timeout=60,
        description="Health check operations",
        critical=False,
    ),
    CircuitBreakers.BACKGROUND_SYNC: CircuitBreakerSpec(
        name=CircuitBreakers.BACKGROUND_SYNC,
        failure_threshold=10,
        recovery_timeout=300,
        description="Background sync tasks",
        critical=False,
    ),
}


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Centralized registry for all circuit breakers.

    Provides:
    - Lazy initialization
    - Thread-safe access
    - Metrics collection
    - Health reporting
    """

    _instance: "CircuitBreakerRegistry | None" = None

    # Async lock used only for singleton instantiation.
    #
    # IMPORTANT: do not create asyncio primitives at import time. In deployment
    # modes such as gunicorn --preload, module import can happen before the
    # worker event loop exists, causing "Future attached to a different loop"
    # errors when awaiting.
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Return a lock bound to the *current* running event loop."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._specs = CIRCUIT_BREAKER_SPECS
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> CircuitBreakerRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._get_lock():
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._initialize()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> CircuitBreakerRegistry:
        """Get singleton instance (sync version)."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize all circuit breakers."""
        if self._initialized:
            return

        for cb_type, spec in self._specs.items():
            config = CircuitBreakerConfig(
                failure_threshold=spec.failure_threshold,
                recovery_timeout=spec.recovery_timeout,
                name=cb_type.value,
            )
            self._breakers[cb_type.value] = CircuitBreaker(config)

        self._initialized = True
        logger.info(
            "circuit_breaker_registry_initialized",
            breaker_count=len(self._breakers),
        )

    def get(self, cb_type: CircuitBreakers) -> CircuitBreaker:
        """Get a circuit breaker by type."""
        if cb_type.value not in self._breakers:
            raise KeyError(f"Unknown circuit breaker: {cb_type}")
        return self._breakers[cb_type.value]

    def get_by_name(self, name: str) -> CircuitBreaker:
        """Get a circuit breaker by name."""
        if name not in self._breakers:
            raise KeyError(f"Unknown circuit breaker: {name}")
        return self._breakers[name]

    def register_custom(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
    ) -> CircuitBreaker:
        """Register a custom circuit breaker."""
        if name in self._breakers:
            return self._breakers[name]

        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name,
        )
        self._breakers[name] = CircuitBreaker(config)
        return self._breakers[name]

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: {
                "state": cb.state.value,
                "failures": cb.metrics.consecutive_failures,
                "total_calls": cb.metrics.total_calls,
                "success_rate": (
                    cb.metrics.successful_calls / cb.metrics.total_calls
                    if cb.metrics.total_calls > 0
                    else 1.0
                ),
            }
            for name, cb in self._breakers.items()
        }

    def get_open_breakers(self) -> list[str]:
        """Get list of open circuit breakers."""
        return [
            name
            for name, cb in self._breakers.items()
            if cb.state == CircuitBreakerState.OPEN
        ]

    def get_health_report(self) -> dict[str, Any]:
        """Get health report for all circuit breakers."""
        open_breakers = self.get_open_breakers()
        critical_open = [
            name
            for name in open_breakers
            if name in self._specs and self._specs[CircuitBreakers(name)].critical
        ]

        return {
            "healthy": len(open_breakers) == 0,
            "total_breakers": len(self._breakers),
            "open_breakers": len(open_breakers),
            "critical_open": len(critical_open),
            "open_list": open_breakers,
            "critical_open_list": critical_open,
            "details": self.get_all_status(),
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for cb in self._breakers.values():
            cb.state = CircuitBreakerState.CLOSED
            cb.metrics.consecutive_failures = 0
        logger.info("all_circuit_breakers_reset")

    def reset(self, cb_type: CircuitBreakers) -> None:
        """Reset a specific circuit breaker."""
        cb = self.get(cb_type)
        cb.state = CircuitBreakerState.CLOSED
        cb.metrics.consecutive_failures = 0
        logger.info("circuit_breaker_reset", name=cb_type.value)

    def get_breaker(self, cb_type: CircuitBreakers) -> CircuitBreaker:
        """Alias for get() - used by admin routes."""
        return self.get(cb_type)

    def get_config(self, cb_type: CircuitBreakers) -> dict[str, Any]:
        """Get configuration for a circuit breaker."""
        spec = self._specs.get(cb_type)
        if not spec:
            return {}
        return {
            "name": spec.name.value,
            "threshold": spec.failure_threshold,
            "recovery_timeout": spec.recovery_timeout,
            "description": spec.description,
            "critical": spec.critical,
        }

    def get_metrics(self, cb_type: CircuitBreakers) -> dict[str, Any]:
        """Get metrics for a circuit breaker."""
        cb = self.get(cb_type)
        return {
            "failure_count": cb.metrics.consecutive_failures,
            "success_count": cb.metrics.successful_calls,
            "total_calls": cb.metrics.total_calls,
            "last_failure": getattr(cb.metrics, "last_failure_time", None),
            "last_success": getattr(cb.metrics, "last_success_time", None),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_circuit_breaker(cb_type: CircuitBreakers) -> CircuitBreaker:
    """
    Get a circuit breaker by type.

    Args:
        cb_type: Type of circuit breaker

    Returns:
        CircuitBreaker instance
    """
    registry = CircuitBreakerRegistry.get_instance_sync()
    return registry.get(cb_type)


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the circuit breaker registry."""
    return CircuitBreakerRegistry.get_instance_sync()


def get_all_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    """Get status of all circuit breakers."""
    registry = CircuitBreakerRegistry.get_instance_sync()
    return registry.get_all_status()


def get_circuit_breaker_health() -> dict[str, Any]:
    """Get health report for circuit breakers."""
    registry = CircuitBreakerRegistry.get_instance_sync()
    return registry.get_health_report()


def get_registry() -> CircuitBreakerRegistry:
    """
    Get the circuit breaker registry instance.

    Alias for get_circuit_breaker_registry for admin routes.
    """
    return CircuitBreakerRegistry.get_instance_sync()


# =============================================================================
# DECORATORS
# =============================================================================


def circuit_protected(
    cb_type: CircuitBreakers,
    fallback: Callable[..., Any] | None = None,
):
    """
    Decorator to protect a function with a circuit breaker.

    Args:
        cb_type: Type of circuit breaker to use
        fallback: Optional fallback function if circuit is open

    Example:
        @circuit_protected(CircuitBreakers.TWS_API)
        async def get_tws_status():
            return await tws_client.get_status()

        @circuit_protected(CircuitBreakers.LLM_API, fallback=lambda: "Default response")
        async def call_llm(prompt):
            return await llm.complete(prompt)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cb = get_circuit_breaker(cb_type)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await cb.call(func, *args, **kwargs)
            except CircuitBreakerError:
                if fallback is not None:
                    logger.warning(
                        "circuit_breaker_fallback_used",
                        function=func.__name__,
                        circuit=cb_type.value,
                    )
                    if inspect.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    fallback_result = fallback(*args, **kwargs)
                    if inspect.isawaitable(fallback_result):
                        return await fallback_result
                    return fallback_result
                raise

        # Expose circuit breaker for inspection
        wrapper.circuit_breaker = cb
        wrapper.circuit_type = cb_type
        return wrapper

    return decorator


def multi_circuit_protected(
    primary: CircuitBreakers,
    fallback_circuits: list[CircuitBreakers],
):
    """
    Decorator for functions that should try multiple circuits.

    Tries primary circuit first, then fallback circuits in order.

    Args:
        primary: Primary circuit breaker
        fallback_circuits: List of fallback circuit breakers

    Example:
        @multi_circuit_protected(
            primary=CircuitBreakers.LLM_OPENAI,
            fallback_circuits=[CircuitBreakers.LLM_ANTHROPIC, CircuitBreakers.LLM_OPENROUTER]
        )
        async def call_llm(prompt):
            return await llm.complete(prompt)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        all_circuits = [primary] + fallback_circuits

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None

            for cb_type in all_circuits:
                cb = get_circuit_breaker(cb_type)

                # Skip if circuit is open
                if cb.state == CircuitBreakerState.OPEN:
                    logger.debug(
                        "skipping_open_circuit",
                        circuit=cb_type.value,
                    )
                    continue

                try:
                    return await cb.call(func, *args, **kwargs)
                except CircuitBreakerError as e:
                    last_error = e
                    logger.warning(
                        "circuit_breaker_tripped_trying_next",
                        circuit=cb_type.value,
                        function=func.__name__,
                    )
                    continue
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "circuit_call_failed_trying_next",
                        circuit=cb_type.value,
                        error=str(e),
                    )
                    continue

            # All circuits failed
            raise CircuitBreakerError(
                f"All circuits failed for {func.__name__}: {last_error}"
            )

        return wrapper

    return decorator


# =============================================================================
# ASYNC CONTEXT MANAGER
# =============================================================================


class CircuitBreakerContext:
    """
    Async context manager for circuit breaker operations.

    Example:
        async with CircuitBreakerContext(CircuitBreakers.TWS_API) as cb:
            result = await cb.call(my_func)
    """

    def __init__(self, cb_type: CircuitBreakers):
        self.cb_type = cb_type
        self.cb: CircuitBreaker | None = None

    async def __aenter__(self) -> CircuitBreaker:
        self.cb = get_circuit_breaker(self.cb_type)
        return self.cb

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Don't suppress exceptions
        return False
