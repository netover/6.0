"""Pre-built circuit-breaker singletons for common Resync subsystems.

These singletons were historically created in the deprecated
``resync.core.circuit_breaker`` module.  They now live here so that
consumers can import them without the compatibility wrapper overhead.

Usage::

    from resync.core.resilience_singletons import adaptive_tws_api_breaker
"""



from resync.core.resilience import CircuitBreaker, CircuitBreakerConfig

# --- TWS API circuit breaker (tolerant, 60 s recovery) ---
adaptive_tws_api_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="adaptive_tws_api",
        failure_threshold=5,
        recovery_timeout=60,
    )
)

# --- LLM API circuit breaker (strict, 120 s recovery) ---
adaptive_llm_api_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="adaptive_llm_api",
        failure_threshold=3,
        recovery_timeout=120,
    )
)

__all__ = [
    "adaptive_tws_api_breaker",
    "adaptive_llm_api_breaker",
]
