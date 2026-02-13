"""
Runtime Metrics Module.

Provides runtime metrics collection and tracking using internal metrics system.
This module creates all the standard metrics counters, gauges and histograms
used throughout the application.
"""

import logging
import time
import uuid
from typing import Any

from resync.core.metrics_internal import (
    create_counter,
    create_gauge,
    create_histogram,
)

logger = logging.getLogger(__name__)


class RuntimeMetricsCollector:
    """
    Runtime metrics collector with all standard application metrics.

    Provides counters, gauges, and histograms for tracking:
    - API requests and responses
    - Cache operations
    - TWS operations
    - Health checks
    - Agent operations
    """

    def __init__(self):
        """Initialize all metrics."""
        # API Metrics
        self.api_requests_total = create_counter("api_requests_total", "Total API requests")
        self.api_requests_success = create_counter(
            "api_requests_success", "Successful API requests"
        )
        self.api_requests_failed = create_counter("api_requests_failed", "Failed API requests")
        self.api_errors_total = create_counter("api_errors_total", "Total API errors")
        self.api_response_time = create_histogram("api_response_time", "API response time")
        self.api_request_duration_histogram = create_histogram(
            "api_request_duration", "API request duration"
        )

        # Cache Metrics
        self.cache_hits = create_counter("cache_hits", "Cache hits")
        self.cache_misses = create_counter("cache_misses", "Cache misses")
        self.cache_evictions = create_counter("cache_evictions", "Cache evictions")
        self.cache_cleanup_cycles = create_counter("cache_cleanup_cycles", "Cache cleanup cycles")
        self.cache_avg_latency = create_gauge("cache_avg_latency", "Average cache latency")
        self.cache_size = create_gauge("cache_size", "Current cache size")

        # TWS Metrics
        self.tws_requests_total = create_counter("tws_requests_total", "Total TWS requests")
        self.tws_requests_success = create_counter(
            "tws_requests_success", "Successful TWS requests"
        )
        self.tws_requests_failed = create_counter("tws_requests_failed", "Failed TWS requests")
        self.tws_status_requests_failed = create_counter(
            "tws_status_requests_failed", "Failed TWS status requests"
        )
        self.tws_response_time = create_histogram("tws_response_time", "TWS response time")

        # Health Check Metrics
        self.health_checks_total = create_counter("health_checks_total", "Total health checks")
        self.health_checks_success = create_counter(
            "health_checks_success", "Successful health checks"
        )
        self.health_checks_failed = create_counter("health_checks_failed", "Failed health checks")
        self.health_check_duration = create_histogram(
            "health_check_duration", "Health check duration"
        )
        self.health_check_with_auto_enable = create_counter(
            "health_check_with_auto_enable", "Health checks with auto-enable"
        )

        # Agent Metrics
        self.agent_requests_total = create_counter("agent_requests_total", "Total agent requests")
        self.agent_mock_fallbacks = create_counter("agent_mock_fallbacks", "Agent mock fallbacks")
        self.agent_response_time = create_histogram("agent_response_time", "Agent response time")

        # Connection Pool Metrics
        self.pool_connections_active = create_gauge(
            "pool_connections_active", "Active pool connections"
        )
        self.pool_connections_idle = create_gauge("pool_connections_idle", "Idle pool connections")

        # Correlation tracking
        self._correlations: dict[str, dict[str, Any]] = {}

        logger.info("RuntimeMetricsCollector initialized")

    def create_correlation_id(self, operation: str, **kwargs) -> str:
        """Create a correlation ID for tracking an operation."""
        correlation_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self._correlations[correlation_id] = {
            "operation": operation,
            "start_time": time.time(),
            **kwargs,
        }
        return correlation_id

    def close_correlation_id(self, correlation_id: str, error: bool = False) -> float:
        """Close a correlation ID and return duration in ms."""
        if correlation_id in self._correlations:
            data = self._correlations.pop(correlation_id)
            return (time.time() - data["start_time"]) * 1000
        return 0.0

    def record_health_check(
        self,
        component: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a health check result."""
        self.health_checks_total.inc()
        if status in ("healthy", "ok", "success"):
            self.health_checks_success.inc()
        else:
            self.health_checks_failed.inc()
        logger.debug("Health check for %s: %s", component, status)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record an API request."""
        self.api_requests_total.inc()
        if 200 <= status_code < 400:
            self.api_requests_success.inc()
        else:
            self.api_requests_failed.inc()
        self.api_response_time.observe(duration_ms / 1000)

    def record_cache_operation(
        self,
        hit: bool,
        latency_ms: float = 0,
    ) -> None:
        """Record a cache operation."""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()

    def get_stats(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "api": {
                "requests_total": self.api_requests_total.get(),
                "requests_success": self.api_requests_success.get(),
                "requests_failed": self.api_requests_failed.get(),
            },
            "cache": {
                "hits": self.cache_hits.get(),
                "misses": self.cache_misses.get(),
                "evictions": self.cache_evictions.get(),
            },
            "tws": {
                "requests_total": self.tws_requests_total.get(),
                "requests_success": self.tws_requests_success.get(),
                "requests_failed": self.tws_requests_failed.get(),
            },
            "health": {
                "checks_total": self.health_checks_total.get(),
                "checks_success": self.health_checks_success.get(),
                "checks_failed": self.health_checks_failed.get(),
            },
        }


# Global instance
_instance: RuntimeMetricsCollector | None = None


def get_runtime_metrics() -> RuntimeMetricsCollector:
    """Get the global RuntimeMetricsCollector instance."""
    global _instance
    if _instance is None:
        _instance = RuntimeMetricsCollector()
    return _instance


# Convenience reference
runtime_metrics = get_runtime_metrics()
