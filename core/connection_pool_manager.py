"""
Advanced Connection Pool Manager with Smart Pooling Integration.

This module provides an enhanced connection pool management system that integrates
with the smart pooling system for automatic scaling and circuit breaker protection.
It combines traditional pool management with intelligent load-based scaling.
"""

import statistics
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from resync.core.pools.base_pool import (
    ConnectionPool,
    ConnectionPoolConfig,
    ConnectionPoolStats,
)
from resync.core.pools.db_pool import DatabaseConnectionPool
from resync.core.pools.http_pool import HTTPConnectionPool
from resync.core.pools.pool_manager import (
    ConnectionPoolManager,
    get_connection_pool_manager,
)
from resync.core.pools.redis_pool import RedisConnectionPool
from resync.core.smart_pooling import SmartConnectionPool, SmartPoolConfig
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadMetrics:
    """Real-time load metrics for connection pools."""

    # Request rate metrics (requests per second)
    request_rate: float = 0.0
    avg_request_rate: float = 0.0

    # Latency metrics (milliseconds)
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Connection utilization
    connection_utilization: float = 0.0  # 0.0 to 1.0

    # Error rate
    error_rate: float = 0.0

    # Queue depth (waiting requests)
    queue_depth: int = 0

    # Historical data (sliding window)
    latency_history: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    request_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    error_history: deque[bool] = field(default_factory=lambda: deque(maxlen=1000))

    # Timestamps
    last_updated: float = 0.0
    measurement_period: float = 60.0  # 1 minute

    def update_request_rate(self, requests_per_second: float) -> None:
        """Update request rate with sliding window average."""
        self.request_history.append(requests_per_second)
        self.request_rate = requests_per_second
        self.avg_request_rate = (
            statistics.mean(self.request_history) if self.request_history else 0.0
        )

    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics."""
        self.latency_history.append(latency_ms)

        if len(self.latency_history) >= 10:  # Minimum samples for percentiles
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            self.avg_latency = statistics.mean(sorted_latencies)
            self.p95_latency = sorted_latencies[int(0.95 * n)]
            self.p99_latency = sorted_latencies[int(0.99 * n)] if n > 2 else sorted_latencies[-1]

    def update_errors(self, is_error: bool) -> None:
        """Update error rate."""
        self.error_history.append(is_error)
        if len(self.error_history) >= 10:
            self.error_rate = sum(self.error_history) / len(self.error_history)

    def get_load_score(self) -> float:
        """Calculate comprehensive load score (0.0 to 1.0)."""
        # Weighted combination of metrics
        weights = {
            "latency": 0.3,
            "utilization": 0.3,
            "error_rate": 0.2,
            "queue_depth": 0.2,
        }

        # Normalize latency score (higher latency = higher score)
        latency_score = min(1.0, self.p95_latency / 5000.0)  # 5s threshold

        # Utilization score
        utilization_score = self.connection_utilization

        # Error score
        error_score = min(1.0, self.error_rate * 5.0)  # 20% error rate = 1.0

        # Queue score (exponential scaling)
        queue_score = min(1.0, self.queue_depth / 50.0)  # 50 waiting requests = 1.0

        return (
            weights["latency"] * latency_score
            + weights["utilization"] * utilization_score
            + weights["error_rate"] * error_score
            + weights["queue_depth"] * queue_score
        )


@dataclass
class AutoScalingConfig:
    """Configuration for automatic pool scaling."""

    # Base scaling thresholds
    scale_up_threshold: float = 0.7  # Scale up when load > 70%
    scale_down_threshold: float = 0.3  # Scale down when load < 30%

    # Scaling factors
    scale_up_factor: float = 1.5  # Multiply connections by 1.5 when scaling up
    scale_down_factor: float = 0.7  # Multiply connections by 0.7 when scaling down

    # Connection limits
    min_connections: int = 5
    max_connections: int = 100

    # Cooldown periods (seconds)
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes

    # Predictive scaling
    enable_predictive_scaling: bool = True
    prediction_window: int = 5  # Look ahead 5 minutes

    # Gradual scaling
    gradual_scaling: bool = True
    scaling_step_size: int = 5  # Add/remove 5 connections at a time


class AutoScalingManager:
    """Intelligent auto-scaling manager for connection pools."""

    def __init__(self, config: AutoScalingConfig | None = None):
        self.config = config or AutoScalingConfig()
        self.load_metrics = LoadMetrics()
        self.current_connections = 10  # Default starting point

        # Scaling state
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        self.scaling_in_progress = False

        # Predictive scaling data
        self.load_trend = deque(maxlen=10)
        self.predicted_load = 0.0

        # Threading for background monitoring
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()
        self._monitoring_active = False

    def start_monitoring(self) -> None:
        """Start background load monitoring."""
        if self._monitoring_active:
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self._monitoring_active = True
        logger.info("auto_scaling_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop background load monitoring."""
        if not self._monitoring_active:
            return

        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self._monitoring_active = False
        logger.info("auto_scaling_monitoring_stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Update load metrics
                self._update_load_metrics()

                # Check for scaling decisions
                self._evaluate_scaling_decision()

                # Update predictions
                self._update_predictions()

            except Exception as e:
                logger.error("auto_scaling_monitoring_error", error=str(e))

            # Sleep for measurement period
            self._stop_monitoring.wait(self.load_metrics.measurement_period)

    def _update_load_metrics(self) -> None:
        """Update current load metrics."""
        # This would integrate with actual pool metrics
        # For now, simulate based on current state
        current_time = time.time()
        self.load_metrics.last_updated = current_time

        # Simulate load calculation (would come from actual pool stats)
        # In real implementation, this would query pool statistics
        simulated_load = self._calculate_simulated_load()
        self.load_metrics.connection_utilization = simulated_load

    def _calculate_simulated_load(self) -> float:
        """Calculate simulated load for demonstration."""
        # This is a placeholder - real implementation would use actual metrics
        base_load = 0.5  # 50% baseline
        variation = 0.2 * (time.time() % 60) / 60.0  # Some variation
        return min(1.0, base_load + variation)

    def _evaluate_scaling_decision(self) -> None:
        """Evaluate if scaling is needed."""
        current_time = time.time()
        load_score = self.load_metrics.get_load_score()

        # Check scale up conditions
        if (
            load_score > self.config.scale_up_threshold
            and current_time - self.last_scale_up_time > self.config.scale_up_cooldown
        ):
            # NOTE: _scale_up is synchronous; calling asyncio.run() here can
            # raise "asyncio.run() cannot be called from a running event loop"
            # if this code is ever executed in an async thread/context.
            self._scale_up()
            self.last_scale_up_time = current_time

        # Check scale down conditions
        elif (
            load_score < self.config.scale_down_threshold
            and current_time - self.last_scale_down_time > self.config.scale_down_cooldown
        ):
            self._scale_down()
            self.last_scale_down_time = current_time

    def _scale_up(self) -> None:
        """Scale up connection pool."""
        if self.scaling_in_progress:
            return

        try:
            self.scaling_in_progress = True

            if self.config.gradual_scaling:
                new_connections = min(
                    self.config.max_connections,
                    self.current_connections + self.config.scaling_step_size,
                )
            else:
                new_connections = min(
                    self.config.max_connections,
                    int(self.current_connections * self.config.scale_up_factor),
                )

            if new_connections > self.current_connections:
                self._apply_scaling(new_connections, "up")

        finally:
            self.scaling_in_progress = False

    def _scale_down(self) -> None:
        """Scale down connection pool."""
        if self.scaling_in_progress:
            return

        try:
            self.scaling_in_progress = True

            if self.config.gradual_scaling:
                new_connections = max(
                    self.config.min_connections,
                    self.current_connections - self.config.scaling_step_size,
                )
            else:
                new_connections = max(
                    self.config.min_connections,
                    int(self.current_connections * self.config.scale_down_factor),
                )

            if new_connections < self.current_connections:
                self._apply_scaling(new_connections, "down")

        finally:
            self.scaling_in_progress = False

    def _apply_scaling(self, new_connection_count: int, direction: str) -> None:
        """Apply scaling changes to the actual pool."""
        # This would integrate with the actual connection pool
        # For now, just log and update state
        old_count = self.current_connections
        self.current_connections = new_connection_count

        logger.info(
            "connection_pool_scaled",
            direction=direction,
            old_connections=old_count,
            new_connections=new_connection_count,
            load_score=self.load_metrics.get_load_score(),
        )

    def _update_predictions(self) -> None:
        """Update load predictions for proactive scaling."""
        if not self.config.enable_predictive_scaling:
            return

        # Simple trend analysis
        current_load = self.load_metrics.get_load_score()
        self.load_trend.append(current_load)

        if len(self.load_trend) >= 3:
            # Linear trend prediction
            trend = statistics.mean(self.load_trend) - self.load_trend[0]
            self.predicted_load = max(0.0, min(1.0, current_load + trend))

    def get_scaling_metrics(self) -> dict[str, Any]:
        """Get comprehensive scaling metrics."""
        return {
            "current_connections": self.current_connections,
            "load_score": self.load_metrics.get_load_score(),
            "predicted_load": self.predicted_load,
            "last_scale_up": self.last_scale_up_time,
            "last_scale_down": self.last_scale_down_time,
            "scaling_in_progress": self.scaling_in_progress,
            "config": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "scale_up_threshold": self.config.scale_up_threshold,
                "scale_down_threshold": self.config.scale_down_threshold,
            },
        }


class AdvancedConnectionPoolManager:
    """
    Advanced connection pool manager with smart pooling integration.

    Provides intelligent pool management with:
    - Auto-scaling based on load metrics
    - Circuit breaker protection
    - Health monitoring and recovery
    - Performance optimization
    """

    def __init__(self):
        """Initialize advanced connection pool manager."""
        self.traditional_manager: ConnectionPoolManager | None = None
        self.smart_pool: SmartConnectionPool | None = None
        self.auto_scaler: AutoScalingManager | None = None
        self._initialized = False
        self._performance_metrics: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the advanced connection pool manager."""
        if self._initialized:
            return

        # Initialize traditional pool manager
        self.traditional_manager = await get_connection_pool_manager()

        # Initialize smart pooling
        smart_config = SmartPoolConfig(
            min_connections=5,
            max_connections=20,
            connection_timeout=10.0,
            latency_threshold_ms=3000.0,  # 3 seconds for circuit breaker
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
        )

        self.smart_pool = SmartConnectionPool(smart_config)
        await self.smart_pool.start()

        # Initialize auto-scaling manager
        self.auto_scaler = AutoScalingManager()
        self.auto_scaler.start_monitoring()

        self._initialized = True
        logger.info("advanced_connection_pool_manager_initialized")

    async def shutdown(self) -> None:
        """Shutdown the advanced connection pool manager."""
        if not self._initialized:
            return

        # Shutdown auto-scaling monitoring
        if self.auto_scaler:
            self.auto_scaler.stop_monitoring()

        # Shutdown smart pooling
        if self.smart_pool:
            await self.smart_pool.stop()

        # Shutdown traditional pools
        if self.traditional_manager:
            await self.traditional_manager.close_all()

        self._initialized = False
        logger.info("advanced_connection_pool_manager_shutdown")

    @asynccontextmanager
    async def get_connection(self, pool_type: str, connection_id: str = "default"):
        """
        Get a connection with intelligent pool selection.

        Args:
            pool_type: Type of connection pool (db, redis, http)
            connection_id: Connection identifier

        Returns:
            Connection object with circuit breaker protection
        """
        if not self._initialized:
            await self.initialize()

        if self.smart_pool and pool_type in ["db", "redis", "http"]:
            # Use smart pooling for critical connections
            async with self.smart_pool.get_connection(f"{pool_type}_{connection_id}") as conn:
                yield conn
        else:
            # Fall back to traditional pooling
            if self.traditional_manager:
                pool = await self.traditional_manager.get_pool(pool_type)
                if pool:
                    connection = await pool.acquire()
                    try:
                        yield connection
                    finally:
                        await pool.release(connection)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "traditional_pools": {},
            "smart_pool": {},
        }

        # Traditional pool metrics
        if self.traditional_manager:
            for pool_name, pool in self.traditional_manager.pools.items():
                metrics["traditional_pools"][pool_name] = pool.get_stats()

        # Smart pool metrics
        if self.smart_pool:
            metrics["smart_pool"] = self.smart_pool.get_pool_stats()

        # Auto-scaling metrics
        if self.auto_scaler:
            metrics["auto_scaling"] = self.auto_scaler.get_scaling_metrics()

        return metrics

    async def force_health_check(self) -> dict[str, Any]:
        """Force health check across all pools."""
        results = {
            "smart_pool": {},
            "traditional_pools": {},
        }

        if self.smart_pool:
            results["smart_pool"] = await self.smart_pool.force_health_check()

        if self.traditional_manager:
            for pool_name, pool in self.traditional_manager.pools.items():
                try:
                    pool_stats = pool.get_stats()
                    results["traditional_pools"][pool_name] = {
                        "healthy": pool_stats.total_connections > 0,
                        "total_connections": pool_stats.total_connections,
                    }
                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    results["traditional_pools"][pool_name] = {
                        "healthy": False,
                        "error": str(e),
                    }

        return results


# Global advanced pool manager instance
_advanced_pool_manager: AdvancedConnectionPoolManager | None = None


def get_advanced_connection_pool_manager() -> AdvancedConnectionPoolManager:
    """Get the global advanced connection pool manager instance."""
    global _advanced_pool_manager
    if _advanced_pool_manager is None:
        _advanced_pool_manager = AdvancedConnectionPoolManager()
    return _advanced_pool_manager


async def initialize_advanced_pooling() -> None:
    """Initialize the advanced connection pooling system."""
    manager = get_advanced_connection_pool_manager()
    await manager.initialize()
    logger.info("advanced_connection_pooling_initialized")


async def shutdown_advanced_pooling() -> None:
    """Shutdown the advanced connection pooling system."""
    global _advanced_pool_manager

    if _advanced_pool_manager:
        await _advanced_pool_manager.shutdown()
        _advanced_pool_manager = None

    logger.info("advanced_connection_pooling_shutdown")


async def shutdown_connection_pool_manager() -> None:
    """Shutdown the connection pool manager (backward compatibility)."""
    await shutdown_advanced_pooling()


# Re-export the main classes and functions for backward compatibility
__all__ = [
    "ConnectionPool",
    "ConnectionPoolConfig",
    "ConnectionPoolStats",
    "DatabaseConnectionPool",
    "RedisConnectionPool",
    "HTTPConnectionPool",
    "ConnectionPoolManager",
    "SmartConnectionPool",
    "SmartPoolConfig",
    "AdvancedConnectionPoolManager",
    "get_connection_pool_manager",
    "get_advanced_connection_pool_manager",
    "shutdown_connection_pool_manager",
    "initialize_advanced_pooling",
    "shutdown_advanced_pooling",
]
