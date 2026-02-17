"""
Runtime Metrics Module.

Provides runtime metrics collection and tracking using internal metrics system.
This module creates all the standard metrics counters, gauges and histograms
used throughout the application.
"""

import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
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

        # Router Cache Metrics (Intent Cache)
        self.router_cache_hits = create_counter("router_cache_hits", "Router cache hits")
        self.router_cache_misses = create_counter("router_cache_misses", "Router cache misses")
        self.router_cache_sets = create_counter("router_cache_sets", "Router cache sets")

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

        # RAG Metrics
        self.rag_bm25_index_loads = create_counter("rag_bm25_index_loads", "BM25 index load operations")
        self.rag_bm25_index_saves = create_counter("rag_bm25_index_saves", "BM25 index save operations")
        self.rag_bm25_queries_total = create_counter("rag_bm25_queries_total", "Total BM25 queries")
        self.rag_bm25_queries_success = create_counter("rag_bm25_queries_success", "Successful BM25 queries")
        self.rag_bm25_queries_failed = create_counter("rag_bm25_queries_failed", "Failed BM25 queries")
        self.rag_vector_queries_total = create_counter("rag_vector_queries_total", "Total vector search queries")
        self.rag_vector_queries_success = create_counter("rag_vector_queries_success", "Successful vector queries")
        self.rag_hybrid_queries_total = create_counter("rag_hybrid_queries_total", "Total hybrid queries")
        self.rag_hybrid_queries_success = create_counter("rag_hybrid_queries_success", "Successful hybrid queries")
        self.rag_chat_turns_stored = create_counter("rag_chat_turns_stored", "Chat turns stored")
        self.rag_chat_searches = create_counter("rag_chat_searches", "Chat memory searches")
        
        # RAG Gauges
        self.rag_bm25_index_documents = create_gauge("rag_bm25_index_documents", "Number of documents in BM25 index")
        self.rag_bm25_index_terms = create_gauge("rag_bm25_index_terms", "Number of terms in BM25 index")
        self.rag_bm25_index_size_bytes = create_gauge("rag_bm25_index_size_bytes", "BM25 index size in bytes")
        self.rag_cache_size = create_gauge("rag_cache_size", "RAG cache size")
        self.rag_cache_hits = create_gauge("rag_cache_hits", "RAG cache hits")
        self.rag_cache_misses = create_gauge("rag_cache_misses", "RAG cache misses")
        
        # RAG Histograms
        self.rag_query_duration = create_histogram("rag_query_duration", "RAG query duration in seconds")
        self.rag_index_build_duration = create_histogram("rag_index_build_duration", "BM25 index build duration")

        # Routing Monitoring (Phase 2)
        self.routing_decisions_total = create_counter(
            "routing_decisions_total", "Total routing decisions", labels=["mode", "intent"]
        )
        self.routing_errors_total = create_counter("routing_errors_total", "Total routing errors")
        self.routing_duration_seconds = create_histogram(
            "routing_duration_seconds", "Routing duration in seconds"
        )
        self.recent_decisions = deque(maxlen=50)

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

    def record_routing_decision(
        self,
        mode: str,
        intent: str,
        confidence: float,
        latency_ms: float,
        trace_id: str | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        """Record a routing decision (Phase 2)."""
        self.routing_decisions_total.labels(mode=mode, intent=intent).inc()
        self.routing_duration_seconds.observe(latency_ms / 1000)

        if error:
            self.routing_errors_total.inc()

        self.recent_decisions.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "intent": intent,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "trace_id": trace_id,
                "error": error,
                "message": message[:100] if message else None,  # Truncate for safety
            }
        )

    def get_stats(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        cache_hits = self.cache_hits.get()
        cache_misses = self.cache_misses.get()
        cache_total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0.0
        
        return {
            "api": {
                "requests_total": self.api_requests_total.get(),
                "requests_success": self.api_requests_success.get(),
                "requests_failed": self.api_requests_failed.get(),
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "evictions": self.cache_evictions.get(),
                "hit_rate": cache_hit_rate,
            },
            "router_cache": {
                "hits": self.router_cache_hits.get(),
                "misses": self.router_cache_misses.get(),
                "sets": self.router_cache_sets.get(),
                "hit_rate": (self.router_cache_hits.get() / (self.router_cache_hits.get() + self.router_cache_misses.get()) * 100) if (self.router_cache_hits.get() + self.router_cache_misses.get()) > 0 else 0.0,
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
            "rag": {
                "bm25": {
                    "index_loads": self.rag_bm25_index_loads.get(),
                    "index_saves": self.rag_bm25_index_saves.get(),
                    "queries_total": self.rag_bm25_queries_total.get(),
                    "queries_success": self.rag_bm25_queries_success.get(),
                    "queries_failed": self.rag_bm25_queries_failed.get(),
                    "index_documents": self.rag_bm25_index_documents.get(),
                    "index_terms": self.rag_bm25_index_terms.get(),
                    "index_size_bytes": self.rag_bm25_index_size_bytes.get(),
                },
                "vector": {
                    "queries_total": self.rag_vector_queries_total.get(),
                    "queries_success": self.rag_vector_queries_success.get(),
                },
                "hybrid": {
                    "queries_total": self.rag_hybrid_queries_total.get(),
                    "queries_success": self.rag_hybrid_queries_success.get(),
                },
                "chat": {
                    "turns_stored": self.rag_chat_turns_stored.get(),
                    "searches": self.rag_chat_searches.get(),
                },
                "cache": {
                    "hits": self.rag_cache_hits.get(),
                    "misses": self.rag_cache_misses.get(),
                },
            },
            "routing": {
                "total_decisions": self.routing_decisions_total.get_sum(),
                "errors": self.routing_errors_total.get(),
                "avg_latency_ms": (
                    self.routing_duration_seconds.get_percentile(50) * 1000
                    if self.routing_duration_seconds.get_percentile(50)
                    else 0
                ),
                "recent_decisions": list(self.recent_decisions),
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
