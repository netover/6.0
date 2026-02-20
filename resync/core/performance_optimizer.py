"""
Performance Optimization Module for Phase 2 Enhancements.

This module provides advanced performance optimization features including:
- Cache performance monitoring and auto-tuning
- Connection pool optimization and monitoring
- Resource management utilities
- Performance metrics collection and analysis
"""

import asyncio
import inspect
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from resync.core.metrics import runtime_metrics

logger = logging.getLogger(__name__)


@dataclass
class CachePerformanceMetrics:
    """Comprehensive cache performance metrics."""

    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    average_access_time_ms: float = 0.0
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_sets: int = 0
    current_size: int = 0
    memory_usage_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_efficiency_score(self) -> float:
        """
        Calculate overall cache efficiency score (0-100).

        Higher scores indicate better cache performance.
        Factors: hit rate (60%), low eviction rate (20%), memory efficiency (20%)
        """
        hit_score = self.hit_rate * 60
        eviction_score = max(0, (1 - self.eviction_rate) * 20)

        # Memory efficiency: prefer caches that use memory effectively
        memory_score = (
            20
            if self.memory_usage_mb < 50
            else max(0, 20 - (self.memory_usage_mb - 50) / 5)
        )

        return hit_score + eviction_score + memory_score


@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics."""

    pool_name: str
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    waiting_requests: int = 0
    average_wait_time_ms: float = 0.0
    peak_connections: int = 0
    connection_errors: int = 0
    pool_exhaustions: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_utilization(self) -> float:
        """Calculate pool utilization percentage."""
        if self.total_connections == 0:
            return 0.0
        return (self.active_connections / self.total_connections) * 100

    def calculate_efficiency_score(self) -> float:
        """
        Calculate pool efficiency score (0-100).

        Factors: utilization (40%), low wait time (30%), low errors (30%)
        """
        utilization = self.calculate_utilization()
        utilization_score = min(utilization, 80) / 80 * 40  # Optimal is 60-80%

        wait_time_score = max(0, 30 - (self.average_wait_time_ms / 10))

        total_requests = self.pool_hits + self.pool_misses
        error_rate = (
            self.connection_errors / total_requests if total_requests > 0 else 0
        )
        error_score = max(0, (1 - error_rate) * 30)

        return utilization_score + wait_time_score + error_score


class CachePerformanceMonitor:
    """
    Monitor and optimize cache performance in real-time.

    Features:
    - Real-time hit rate tracking
    - Automatic TTL adjustment based on access patterns
    - Memory usage monitoring
    - Performance recommendations
    """

    def __init__(self, cache_name: str = "default"):
        self.cache_name = cache_name
        self.metrics_history: deque = deque(maxlen=1000)
        self.access_times: deque = deque(maxlen=100)
        self._lock = asyncio.Lock()

    async def record_access(self, hit: bool, access_time_ms: float) -> None:
        """Record a cache access for performance tracking."""
        async with self._lock:
            self.access_times.append(access_time_ms)

            # Update runtime metrics
            if hit:
                runtime_metrics.cache_hits.increment()
            else:
                runtime_metrics.cache_misses.increment()

    async def get_current_metrics(self) -> CachePerformanceMetrics:
        """Get current cache performance metrics."""
        async with self._lock:
            total_hits = runtime_metrics.cache_hits.value
            total_misses = runtime_metrics.cache_misses.value
            total_requests = total_hits + total_misses

            hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            miss_rate = total_misses / total_requests if total_requests > 0 else 0.0

            avg_access_time = (
                sum(self.access_times) / len(self.access_times)
                if self.access_times
                else 0.0
            )

            metrics = CachePerformanceMetrics(
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                average_access_time_ms=avg_access_time,
                total_hits=total_hits,
                total_misses=total_misses,
                total_evictions=runtime_metrics.cache_evictions.value,
                total_sets=runtime_metrics.cache_sets.value,
                current_size=runtime_metrics.cache_size.value,
            )

            self.metrics_history.append(metrics)
            return metrics

    async def get_optimization_recommendations(self) -> list[str]:
        """
        Analyze cache performance and provide optimization recommendations.

        Returns:
            List of actionable recommendations
        """
        metrics = await self.get_current_metrics()
        recommendations = []

        # Check hit rate
        if metrics.hit_rate < 0.5:
            recommendations.append(
                "Low hit rate ({metrics.hit_rate:.1%}). Consider increasing TTL or cache size."
            )

        # Check eviction rate
        if metrics.total_sets > 0:
            eviction_rate = metrics.total_evictions / metrics.total_sets
            if eviction_rate > 0.3:
                recommendations.append(
                    "High eviction rate ({eviction_rate:.1%}). Consider increasing cache size."
                )

        # Check memory usage
        if metrics.memory_usage_mb > 80:
            recommendations.append(
                "High memory usage ({metrics.memory_usage_mb:.1f}MB). Consider reducing cache size or TTL."
            )

        # Check access time
        if metrics.average_access_time_ms > 10:
            recommendations.append(
                "Slow cache access ({metrics.average_access_time_ms:.2f}ms). Consider reducing shard contention."
            )

        if not recommendations:
            recommendations.append("Cache performance is optimal.")

        return recommendations


class ConnectionPoolOptimizer:
    """
    Optimize connection pool configurations based on runtime metrics.

    Features:
    - Auto-tuning of pool sizes
    - Connection timeout optimization
    - Pool exhaustion detection
    - Performance recommendations
    """

    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        self.metrics_history: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        self.last_optimization = datetime.now(timezone.utc)

    async def record_connection_acquisition(
        self, wait_time_ms: float, success: bool
    ) -> None:
        """Record a connection acquisition attempt."""
        async with self._lock:
            if success:
                runtime_metrics.record_histogram(
                    f"connection_pool.{self.pool_name}.acquire_time",
                    wait_time_ms / 1000,  # Convert to seconds
                    {"pool_name": self.pool_name},
                )

    async def get_current_metrics(
        self, pool_stats: dict[str, Any]
    ) -> ConnectionPoolMetrics:
        """Get current connection pool metrics."""
        async with self._lock:
            metrics = ConnectionPoolMetrics(
                pool_name=self.pool_name,
                active_connections=pool_stats.get("active_connections", 0),
                idle_connections=pool_stats.get("idle_connections", 0),
                total_connections=pool_stats.get("total_connections", 0),
                waiting_requests=pool_stats.get("waiting_connections", 0),
                average_wait_time_ms=pool_stats.get("average_wait_time", 0.0) * 1000,
                peak_connections=pool_stats.get("peak_connections", 0),
                connection_errors=pool_stats.get("connection_errors", 0),
                pool_exhaustions=pool_stats.get("pool_exhaustions", 0),
                pool_hits=pool_stats.get("pool_hits", 0),
                pool_misses=pool_stats.get("pool_misses", 0),
            )

            self.metrics_history.append(metrics)
            return metrics

    def suggest_pool_size(
        self, current_metrics: ConnectionPoolMetrics
    ) -> tuple[int, int]:
        """
        Suggest optimal min and max pool sizes based on metrics.

        Returns:
            Tuple of (suggested_min_size, suggested_max_size)
        """
        utilization = current_metrics.calculate_utilization()

        # If utilization is consistently high, increase pool size
        if utilization > 80:
            suggested_max = int(current_metrics.total_connections * 1.5)
            suggested_min = int(current_metrics.total_connections * 0.5)
        # If utilization is low, decrease pool size
        elif utilization < 30:
            suggested_max = max(10, int(current_metrics.total_connections * 0.7))
            suggested_min = max(5, int(current_metrics.total_connections * 0.3))
        else:
            # Current size is good
            suggested_max = current_metrics.total_connections
            suggested_min = max(5, int(current_metrics.total_connections * 0.3))

        return suggested_min, suggested_max

    async def get_optimization_recommendations(
        self, current_metrics: ConnectionPoolMetrics
    ) -> list[str]:
        """
        Analyze pool performance and provide optimization recommendations.

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        utilization = current_metrics.calculate_utilization()

        # Check utilization
        if utilization > 90:
            recommendations.append(
                "Very high pool utilization ({utilization:.1f}%). "
                "Consider increasing max pool size to prevent exhaustion."
            )
        elif utilization < 20:
            recommendations.append(
                "Low pool utilization ({utilization:.1f}%). "
                "Consider decreasing min pool size to save resources."
            )

        # Check wait times
        if current_metrics.average_wait_time_ms > 100:
            recommendations.append(
                "High connection wait time ({current_metrics.average_wait_time_ms:.1f}ms). "
                "Consider increasing pool size or optimizing queries."
            )

        # Check errors
        total_requests = current_metrics.pool_hits + current_metrics.pool_misses
        if total_requests > 0:
            error_rate = current_metrics.connection_errors / total_requests
            if error_rate > 0.05:
                recommendations.append(
                    "High connection error rate ({error_rate:.1%}). "
                    "Check database health and network connectivity."
                )

        # Check pool exhaustions
        if current_metrics.pool_exhaustions > 0:
            recommendations.append(
                f"Pool exhaustion detected ({current_metrics.pool_exhaustions} times). "
                f"Increase max pool size immediately."
            )

        # Suggest optimal sizes
        suggested_min, suggested_max = self.suggest_pool_size(current_metrics)
        if suggested_max != current_metrics.total_connections:
            recommendations.append(
                f"Suggested pool size: min={suggested_min}, max={suggested_max}"
            )

        if not recommendations:
            recommendations.append("Connection pool performance is optimal.")

        return recommendations


class ResourceManager:
    """
    Centralized resource management for deterministic cleanup.

    Features:
    - Automatic resource tracking
    - Deterministic cleanup on context exit
    - Resource leak detection
    - Resource usage monitoring
    """

    def __init__(self):
        self.active_resources: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.resource_creation_times: dict[str, datetime] = {}

    async def register_resource(self, resource_id: str, resource: Any) -> None:
        """Register a resource for tracking."""
        async with self._lock:
            self.active_resources[resource_id] = resource
            self.resource_creation_times[resource_id] = datetime.now(timezone.utc)

            logger.debug("Registered resource: %s", resource_id)

    async def unregister_resource(self, resource_id: str) -> None:
        """Unregister a resource after cleanup."""
        async with self._lock:
            if resource_id in self.active_resources:
                del self.active_resources[resource_id]

                # Calculate resource lifetime
                if resource_id in self.resource_creation_times:
                    lifetime = (
                        datetime.now(timezone.utc)
                        - self.resource_creation_times[resource_id]
                    )
                    del self.resource_creation_times[resource_id]

                    logger.debug(
                        f"Unregistered resource: {resource_id}, "
                        f"lifetime: {lifetime.total_seconds():.2f}s"
                    )

    async def cleanup_all(self) -> None:
        """Cleanup all registered resources."""
        async with self._lock:
            for resource_id, resource in list(self.active_resources.items()):
                try:
                    # Try to close the resource if it has a close method
                    if hasattr(resource, "close"):
                        if inspect.iscoroutinefunction(resource.close):
                            await resource.close()
                        else:
                           cleanup_result = await asyncio.to_thread(cleanup_fn, resource)
                        if inspect.isawaitable(cleanup_result):
                            await cleanup_result
                except Exception as e:
                    logger.error("Error cleaning up resource %s: %s", resource_id, e)
                finally:
                    if resource_id in self.active_resources:
                        del self.active_resources[resource_id]
                    self.resource_creation_times.pop(resource_id, None)

    async def detect_leaks(self, max_lifetime_seconds: int = 3600) -> list[str]:
        """
        Detect potential resource leaks.

        Args:
            max_lifetime_seconds: Maximum expected resource lifetime

        Returns:
            List of resource IDs that may be leaking
        """
        async with self._lock:
            leaks = []
            now = datetime.now(timezone.utc)

            for resource_id, creation_time in self.resource_creation_times.items():
                lifetime = (now - creation_time).total_seconds()
                if lifetime > max_lifetime_seconds:
                    leaks.append(resource_id)
                    logger.warning(
                        f"Potential resource leak detected: {resource_id}, "
                        "lifetime: {lifetime:.2f}s"
                    )

            return leaks

    def get_resource_count(self) -> int:
        """Get the number of active resources."""
        return len(self.active_resources)


# Global resource manager instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


class PerformanceOptimizationService:
    """
    Centralized service for performance optimization.

    Coordinates cache monitoring, connection pool optimization,
    and resource management.
    """

    def __init__(self):
        self.cache_monitors: dict[str, CachePerformanceMonitor] = {}
        self.pool_optimizers: dict[str, ConnectionPoolOptimizer] = {}
        self.resource_manager = get_resource_manager()
        self._lock = asyncio.Lock()

    async def register_cache(self, cache_name: str) -> CachePerformanceMonitor:
        """Register a cache for monitoring."""
        async with self._lock:
            if cache_name not in self.cache_monitors:
                self.cache_monitors[cache_name] = CachePerformanceMonitor(cache_name)
            return self.cache_monitors[cache_name]

    async def register_pool(self, pool_name: str) -> ConnectionPoolOptimizer:
        """Register a connection pool for optimization."""
        async with self._lock:
            if pool_name not in self.pool_optimizers:
                self.pool_optimizers[pool_name] = ConnectionPoolOptimizer(pool_name)
            return self.pool_optimizers[pool_name]

    async def get_system_performance_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive system performance report.

        Returns:
            Dictionary containing performance metrics and recommendations
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "caches": {},
            "connection_pools": {},
            "resources": {
                "active_count": self.resource_manager.get_resource_count(),
                "potential_leaks": await self.resource_manager.detect_leaks(),
            },
            "overall_health": "healthy",
        }

        # Collect cache metrics
        for cache_name, monitor in self.cache_monitors.items():
            metrics = await monitor.get_current_metrics()
            recommendations = await monitor.get_optimization_recommendations()

            report["caches"][cache_name] = {
                "metrics": {
                    "hit_rate": f"{metrics.hit_rate:.1%}",
                    "miss_rate": f"{metrics.miss_rate:.1%}",
                    "efficiency_score": "{metrics.calculate_efficiency_score():.1f}/100",
                    "current_size": metrics.current_size,
                    "memory_usage_mb": f"{metrics.memory_usage_mb:.2f}",
                },
                "recommendations": recommendations,
            }

        # Determine overall health
        cache_issues = sum(
            1
            for cache_data in report["caches"].values()
            if len(cache_data["recommendations"]) > 1
        )

        if cache_issues > 0 or len(report["resources"]["potential_leaks"]) > 0:
            report["overall_health"] = "degraded"

        return report


# Global performance optimization service
_performance_service: PerformanceOptimizationService | None = None


def get_performance_service() -> PerformanceOptimizationService:
    """Get the global performance optimization service."""
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceOptimizationService()
    return _performance_service
