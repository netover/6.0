"""
Advanced Performance Tracking

This module provides comprehensive performance tracking capabilities including
cache statistics, response time monitoring, and performance metrics collection.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: float
    response_time_ms: float
    cpu_percent: float
    memory_percent: float
    cache_hits: int
    cache_misses: int
    active_connections: int
    error_count: int
    throughput: float


@dataclass
class CacheStatistics:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, hit: bool) -> None:
        """Update cache statistics."""
        if hit:
            self.hits += 1
        else:
            self.misses += 1

        self.total_requests = self.hits + self.misses
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0
        self.last_updated = datetime.now()


class PerformanceTracker:
    """Advanced performance tracking system."""

    def __init__(self, max_history_size: int = 1000, history_retention_hours: int = 24):
        self.max_history_size = max_history_size
        self.history_retention_hours = history_retention_hours

        # Performance history
        self.performance_history: deque = deque(maxlen=max_history_size)

        # Cache statistics
        self.cache_stats = CacheStatistics()

        # Component-specific metrics
        self.component_metrics: dict[str, CacheStatistics] = defaultdict(CacheStatistics)

        # Response time tracking
        self.response_times: deque = deque(maxlen=1000)

        # Error tracking
        self.error_counts: dict[str, int] = defaultdict(int)

        # Throughput tracking
        self.request_counts: deque = deque(maxlen=1000)

        # Locks for thread safety
        self._lock = asyncio.Lock()

    async def record_cache_access(self, component: str, hit: bool) -> None:
        """Record a cache access for statistics."""
        async with self._lock:
            self.cache_stats.update(hit)
            self.component_metrics[component].update(hit)

    async def record_performance_metrics(
        self,
        response_time_ms: float,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        active_connections: int = 0,
        error_count: int = 0,
    ) -> None:
        """Record comprehensive performance metrics."""
        async with self._lock:
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                response_time_ms=response_time_ms,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                cache_hits=self.cache_stats.hits,
                cache_misses=self.cache_stats.misses,
                active_connections=active_connections,
                error_count=error_count,
                throughput=self._calculate_throughput(),
            )

            self.performance_history.append(metrics)
            self.response_times.append(response_time_ms)

            # Cleanup old metrics
            await self._cleanup_old_metrics()

    async def record_error(self, component: str, error_type: str) -> None:
        """Record an error for tracking."""
        async with self._lock:
            error_key = f"{component}:{error_type}"
            self.error_counts[error_key] += 1

    async def record_request(self) -> None:
        """Record a request for throughput calculation."""
        async with self._lock:
            self.request_counts.append(time.time())

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get current cache statistics."""
        return {
            "overall": {
                "hits": self.cache_stats.hits,
                "misses": self.cache_stats.misses,
                "total_requests": self.cache_stats.total_requests,
                "hit_rate": round(self.cache_stats.hit_rate, 4),
                "last_updated": self.cache_stats.last_updated.isoformat(),
            },
            "by_component": {
                name: {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "total_requests": stats.total_requests,
                    "hit_rate": round(stats.hit_rate, 4),
                    "last_updated": stats.last_updated.isoformat(),
                }
                for name, stats in self.component_metrics.items()
            },
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        recent_metrics = list(self.performance_history)[-100:]  # Last 100 entries

        response_times = [m.response_time_ms for m in recent_metrics]
        cpu_usage = [m.cpu_percent for m in recent_metrics]
        memory_usage = [m.memory_percent for m in recent_metrics]

        return {
            "response_time": {
                "current_ms": response_times[-1] if response_times else 0,
                "average_ms": round(sum(response_times) / len(response_times), 2),
                "min_ms": min(response_times) if response_times else 0,
                "max_ms": max(response_times) if response_times else 0,
                "p95_ms": round(self._percentile(response_times, 95), 2),
                "p99_ms": round(self._percentile(response_times, 99), 2),
            },
            "resource_usage": {
                "cpu_percent": {
                    "current": cpu_usage[-1] if cpu_usage else 0,
                    "average": (round(sum(cpu_usage) / len(cpu_usage), 2) if cpu_usage else 0),
                    "max": max(cpu_usage) if cpu_usage else 0,
                },
                "memory_percent": {
                    "current": memory_usage[-1] if memory_usage else 0,
                    "average": (
                        round(sum(memory_usage) / len(memory_usage), 2) if memory_usage else 0
                    ),
                    "max": max(memory_usage) if memory_usage else 0,
                },
            },
            "cache_performance": {
                "hit_rate": round(self.cache_stats.hit_rate, 4),
                "total_hits": self.cache_stats.hits,
                "total_misses": self.cache_stats.misses,
            },
            "throughput": {
                "current_rps": self._calculate_throughput(),
                "average_rps": self._calculate_average_throughput(),
            },
            "errors": dict(self.error_counts),
            "sample_size": len(recent_metrics),
            "time_range": {
                "start": (
                    datetime.fromtimestamp(recent_metrics[0].timestamp).isoformat()
                    if recent_metrics
                    else None
                ),
                "end": (
                    datetime.fromtimestamp(recent_metrics[-1].timestamp).isoformat()
                    if recent_metrics
                    else None
                ),
            },
        }

    def get_health_score(self) -> float:
        """Calculate overall health score based on performance metrics."""
        try:
            if not self.performance_history:
                return 0.0

            recent_metrics = list(self.performance_history)[-50:]  # Last 50 entries

            # Calculate weighted health score
            response_time_score = self._calculate_response_time_score(recent_metrics)
            cache_score = self.cache_stats.hit_rate
            error_score = self._calculate_error_score()
            resource_score = self._calculate_resource_score(recent_metrics)

            # Weighted average
            health_score = (
                response_time_score * 0.3
                + cache_score * 0.25
                + error_score * 0.25
                + resource_score * 0.2
            )

            return round(min(1.0, max(0.0, health_score)), 4)

        except Exception as e:
            logger.warning("failed_to_calculate_health_score", error=str(e))
            return 0.0

    def _calculate_response_time_score(self, metrics: list[PerformanceMetrics]) -> float:
        """Calculate score based on response times (lower is better)."""
        if not metrics:
            return 0.0

        avg_response_time = sum(m.response_time_ms for m in metrics) / len(metrics)

        # Score degrades as response time increases
        # Perfect score for < 100ms, degrades linearly to 0 at 1000ms
        if avg_response_time < 100:
            return 1.0
        if avg_response_time > 1000:
            return 0.0
        return 1.0 - ((avg_response_time - 100) / 900)

    def _calculate_error_score(self) -> float:
        """Calculate score based on error rates (lower errors is better)."""
        total_errors = sum(self.error_counts.values())
        total_requests = self.cache_stats.total_requests

        if total_requests == 0:
            return 1.0

        error_rate = total_errors / total_requests

        # Perfect score for < 1% errors, degrades linearly to 0 at 10% errors
        if error_rate < 0.01:
            return 1.0
        if error_rate > 0.1:
            return 0.0
        return 1.0 - ((error_rate - 0.01) / 0.09)

    def _calculate_resource_score(self, metrics: list[PerformanceMetrics]) -> float:
        """Calculate score based on resource usage."""
        if not metrics:
            return 0.0

        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_percent for m in metrics) / len(metrics)

        # Score based on resource utilization (lower is better)
        cpu_score = max(0.0, 1.0 - (avg_cpu / 100))
        memory_score = max(0.0, 1.0 - (avg_memory / 100))

        return (cpu_score + memory_score) / 2

    def _calculate_throughput(self) -> float:
        """Calculate current requests per second."""
        if len(self.request_counts) < 2:
            return 0.0

        # Calculate rate based on recent requests
        recent_requests = list(self.request_counts)[-20:]
        if len(recent_requests) < 2:
            return 0.0

        time_span = recent_requests[-1] - recent_requests[0]
        if time_span == 0:
            return 0.0

        return len(recent_requests) / time_span

    def _calculate_average_throughput(self) -> float:
        """Calculate average throughput over history."""
        if len(self.request_counts) < 2:
            return 0.0

        total_time = self.request_counts[-1] - self.request_counts[0]
        if total_time == 0:
            return 0.0

        return len(self.request_counts) / total_time

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        lower_index = int(index)
        upper_index = lower_index + 1
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old performance metrics."""
        try:
            cutoff_time = time.time() - (self.history_retention_hours * 3600)

            # Remove old performance history entries
            while self.performance_history:
                if self.performance_history[0].timestamp < cutoff_time:
                    self.performance_history.popleft()
                else:
                    break

            # Remove old request counts
            while self.request_counts:
                if self.request_counts[0] < cutoff_time:
                    self.request_counts.popleft()
                else:
                    break

            # Remove old response times
            while self.response_times:
                # We don't have timestamps for response times, so use a simple size limit
                if len(self.response_times) > 500:
                    self.response_times.popleft()
                else:
                    break

        except Exception as e:
            logger.warning("failed_to_cleanup_metrics", error=str(e))

    def get_detailed_report(self) -> dict[str, Any]:
        """Get detailed performance report."""
        return {
            "summary": self.get_performance_summary(),
            "cache_statistics": self.get_cache_statistics(),
            "health_score": self.get_health_score(),
            "configuration": {
                "max_history_size": self.max_history_size,
                "history_retention_hours": self.history_retention_hours,
            },
            "metadata": {
                "report_generated_at": datetime.now().isoformat(),
                "total_metrics_collected": len(self.performance_history),
            },
        }

    async def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = self.get_detailed_report()

        if format.lower() == "json":
            import json

            return json.dumps(data, indent=2, default=str)
        return str(data)
