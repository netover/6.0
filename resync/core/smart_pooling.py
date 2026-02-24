# pylint: disable=all
"""
Smart Connection Pooling System.

This module provides intelligent connection pooling with:
- Circuit breaker integration
- Health monitoring and recovery
- Performance metrics collection
- Connection lifecycle management
"""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from resync.core.resilience_singletons import adaptive_tws_api_breaker
from resync.core.structured_logger import get_logger
from resync.core.task_tracker import create_tracked_task

logger = get_logger(__name__)


@dataclass
class SmartPoolConfig:
    """Configuration for smart connection pooling."""

    # Connection limits
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 10.0

    # Circuit breaker settings
    latency_threshold_ms: float = 3000.0  # 3 seconds

    # Health check settings
    health_check_interval: float = 30.0  # 30 seconds
    health_check_timeout: float = 5.0  # 5 seconds
    max_health_check_failures: int = 3

    # Scaling thresholds
    scale_up_threshold: float = 0.8  # 80% utilization
    scale_down_threshold: float = 0.3  # 30% utilization

    # Connection lifecycle
    max_connection_age: float = 3600.0  # 1 hour
    max_connection_uses: int = 100  # Reuse up to 100 times
    idle_timeout: float = 300.0  # 5 minutes


@dataclass
class ConnectionStats:
    """Statistics for individual connections."""

    connection_id: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    total_latency: float = 0.0
    error_count: int = 0
    health_check_failures: int = 0
    is_healthy: bool = True

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency / max(1, self.use_count)

    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used

    def should_retire(self, config: SmartPoolConfig) -> bool:
        """Check if connection should be retired."""
        return (
            self.age > config.max_connection_age
            or self.use_count >= config.max_connection_uses
            or self.idle_time > config.idle_timeout
            or self.health_check_failures >= config.max_health_check_failures
            or not self.is_healthy
        )


@dataclass
class PoolMetrics:
    """Comprehensive pool performance metrics."""

    # Connection counts
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    unhealthy_connections: int = 0

    # Request metrics
    total_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0

    # Performance metrics
    avg_request_latency: float = 0.0
    p95_latency: float = 0.0
    error_rate: float = 0.0

    # Health metrics
    health_check_success_rate: float = 1.0
    connection_turnover_rate: float = 0.0

    # Circuit breaker metrics
    circuit_breaker_trips: int = 0
    circuit_breaker_recoveries: int = 0

    def get_utilization(self) -> float:
        """Get pool utilization ratio."""
        return self.active_connections / max(1, self.total_connections)

    def get_queue_ratio(self) -> float:
        """Get queue pressure ratio."""
        return self.queued_requests / max(1, self.total_requests)


class SmartConnectionPool:
    """
    Intelligent connection pool with circuit breaker integration.

    Features:
    - Automatic connection lifecycle management
    - Circuit breaker protection
    - Health monitoring and recovery
    - Performance metrics collection
    - Adaptive scaling signals
    """

    def __init__(self, config: SmartPoolConfig | None = None):
        self.config = config or SmartPoolConfig()

        # Connection storage
        self._connections: dict[str, Any] = {}  # connection_id -> connection
        self._connection_stats: dict[str, ConnectionStats] = {}
        self._idle_connections: deque[str] = deque()
        self._active_connections: set[str] = set()

        # Request queue (lazy-initialized to avoid event loop issues at import time)
        self._request_queue: asyncio.Queue[Any] | None = None
        self._waiting_requests: set[asyncio.Future[Any]] = set()

        # Metrics and monitoring
        self.metrics = PoolMetrics()
        self._latency_history: deque[float] = deque(maxlen=1000)

        # Health monitoring
        self._health_monitor_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Synchronization
        self._lock: asyncio.Lock = asyncio.Lock()
        self._connection_counter = 0

    @property
    def request_queue(self) -> asyncio.Queue[Any]:
        """Lazy initialization of request queue."""
        if self._request_queue is None:
            self._request_queue = asyncio.Queue()
        return self._request_queue

    async def start(self) -> None:
        """Start the smart connection pool."""
        if self._running:
            return

        self._running = True

        # Initialize minimum connections
        await self._initialize_connections()

        # Start health monitoring
        self._health_monitor_task = await create_tracked_task(
            self._health_monitor_loop(), name="health_monitor_loop"
        )

        logger.info(
            "smart_connection_pool_started",
            min_connections=self.config.min_connections,
            max_connections=self.config.max_connections,
        )

    async def stop(self) -> None:
        """Stop the smart connection pool."""
        if not self._running:
            return

        self._running = False

        # Stop health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_monitor_task

        # Close all connections
        await self._close_all_connections()

        logger.info("smart_connection_pool_stopped")

    @asynccontextmanager
    async def get_connection(self, connection_key: str) -> AsyncGenerator[Any, None]:
        """
        Get a connection with circuit breaker protection.

        Args:
            connection_key: Identifier for the connection type/request

        Yields:
            Connection object with automatic lifecycle management
        """
        # Check circuit breaker first
        if not await adaptive_tws_api_breaker.can_execute():
            raise RuntimeError("Circuit breaker is open")

        start_time = time.time()

        try:
            # Get connection from pool
            connection_id = await self._acquire_connection()
            connection = self._connections[connection_id]

            # Update stats
            self._connection_stats[connection_id].last_used = time.time()
            self._connection_stats[connection_id].use_count += 1

            # Update metrics
            async with self._lock:
                self.metrics.active_requests += 1
                self.metrics.total_requests += 1

            yield connection

            # Record successful usage
            latency = (time.time() - start_time) * 1000  # Convert to ms
            await self._record_success(connection_id, latency)

        except Exception as e:
            # Record failure
            latency = (time.time() - start_time) * 1000
            await self._record_failure(str(e), latency)
            raise

        finally:
            # Update metrics
            async with self._lock:
                self.metrics.active_requests -= 1

    async def _acquire_connection(self) -> str:
        """Acquire a connection from the pool."""
        async with self._lock:
            # Try to get idle connection first
            if self._idle_connections:
                connection_id = self._idle_connections.popleft()
                if self._is_connection_valid(connection_id):
                    self._active_connections.add(connection_id)
                    self.metrics.idle_connections -= 1
                    return connection_id

            # Create new connection if under limit
            if len(self._connections) < self.config.max_connections:
                return self._create_connection()

            # Wait for available connection
            return await self._wait_for_connection()

    def _create_connection(self) -> str:
        """Create a new connection."""
        connection_id = f"conn_{self._connection_counter}"
        self._connection_counter += 1

        try:
            # In real implementation this would create
            # actual database/redis/http connections
            # For now, simulate connection creation
            connection = f"mock_connection_{connection_id}"

            # Store connection and stats
            self._connections[connection_id] = connection
            self._connection_stats[connection_id] = ConnectionStats(
                connection_id=connection_id
            )

            # Add to active set
            self._active_connections.add(connection_id)

            # Update metrics
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1

            logger.debug("connection_created", connection_id=connection_id)
            return connection_id

        except Exception as e:
            logger.error("connection_creation_failed", error=str(e))
            raise

    async def _wait_for_connection(self) -> str:
        """Wait for an available connection."""
        # Create a future for this request
        future = asyncio.Future()
        self._waiting_requests.add(future)

        try:
            # Wait with timeout
            return await asyncio.wait_for(
                future, timeout=self.config.connection_timeout
            )

        except asyncio.TimeoutError:
            # Remove from waiting requests
            self._waiting_requests.discard(future)
            self.metrics.queued_requests += 1
            raise RuntimeError("Connection timeout") from None

        finally:
            self._waiting_requests.discard(future)

    def _is_connection_valid(self, connection_id: str) -> bool:
        """Check if a connection is still valid."""
        if connection_id not in self._connection_stats:
            return False

        stats = self._connection_stats[connection_id]
        return not stats.should_retire(self.config)

    async def _record_success(self, connection_id: str, latency_ms: float) -> None:
        """Record successful connection usage."""
        if connection_id in self._connection_stats:
            stats = self._connection_stats[connection_id]
            stats.total_latency += latency_ms

        # Update latency history
        self._latency_history.append(latency_ms)

        # Update pool metrics
        if len(self._latency_history) >= 10:
            sorted_latencies = sorted(self._latency_history)
            self.metrics.avg_request_latency = sum(sorted_latencies) / len(
                sorted_latencies
            )
            self.metrics.p95_latency = sorted_latencies[
                int(0.95 * len(sorted_latencies))
            ]

        # Return connection to idle pool
        await self._release_connection(connection_id)

    async def _record_failure(self, error: str, latency_ms: float) -> None:
        """Record connection failure."""
        async with self._lock:
            self.metrics.error_rate = (
                (self.metrics.error_rate * (self.metrics.total_requests - 1)) + 1
            ) / self.metrics.total_requests

        logger.warning("connection_request_failed", error=error, latency=latency_ms)

    async def _release_connection(self, connection_id: str) -> None:
        """Release connection back to idle pool."""
        async with self._lock:
            if connection_id in self._active_connections:
                self._active_connections.remove(connection_id)
                self.metrics.active_connections -= 1

                # Check if connection should be retired
                if not self._is_connection_valid(connection_id):
                    self._retire_connection(connection_id)
                else:
                    self._idle_connections.append(connection_id)
                    self.metrics.idle_connections += 1

                    # Wake up waiting requests
                    if self._waiting_requests:
                        future = next(iter(self._waiting_requests))
                        future.set_result(connection_id)
                        self._waiting_requests.remove(future)

    def _retire_connection(self, connection_id: str) -> None:
        """Retire and remove a connection."""
        if connection_id in self._connections:
            # Close connection (in real implementation)
            del self._connections[connection_id]
            del self._connection_stats[connection_id]

            self.metrics.total_connections -= 1
            self.metrics.connection_turnover_rate += 1

            logger.debug("connection_retired", connection_id=connection_id)

    async def _initialize_connections(self) -> None:
        """Initialize minimum number of connections."""
        for _ in range(self.config.min_connections):
            try:
                self._create_connection()
                # Immediately mark as idle
                connection_id = f"conn_{self._connection_counter - 1}"
                await self._release_connection(connection_id)
            except Exception as e:
                logger.error("initial_connection_creation_failed", error=str(e))

    async def _close_all_connections(self) -> None:
        """Close all connections."""
        connection_ids = list(self._connections.keys())
        for connection_id in connection_ids:
            self._retire_connection(connection_id)

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_monitor_error", error=str(e))
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on connections."""
        connection_ids = list(self._connections.keys())
        healthy_checks = 0

        for connection_id in connection_ids:
            try:
                is_healthy = await self._health_check_connection(connection_id)
                self._connection_stats[connection_id].is_healthy = is_healthy

                if is_healthy:
                    healthy_checks += 1
                    self._connection_stats[connection_id].health_check_failures = 0
                else:
                    self._connection_stats[connection_id].health_check_failures += 1

            except Exception as e:
                logger.warning(
                    "health_check_failed", connection_id=connection_id, error=str(e)
                )
                self._connection_stats[connection_id].health_check_failures += 1

        # Update health metrics
        total_checks = len(connection_ids)
        if total_checks > 0:
            self.metrics.health_check_success_rate = healthy_checks / total_checks
            self.metrics.unhealthy_connections = total_checks - healthy_checks

    async def _health_check_connection(self, connection_id: str) -> bool:
        """Perform health check on a single connection."""
        # In real implementation, this would ping the actual connection
        # For now, simulate health check
        await asyncio.sleep(0.01)  # Simulate network latency

        # Simulate occasional failures (1% failure rate)
        import random

        return random.random() > 0.01

    def get_pool_stats(self) -> dict[str, Any]:
        """Get comprehensive pool statistics."""
        return {
            "connections": {
                "total": self.metrics.total_connections,
                "active": self.metrics.active_connections,
                "idle": self.metrics.idle_connections,
                "unhealthy": self.metrics.unhealthy_connections,
            },
            "requests": {
                "total": self.metrics.total_requests,
                "active": self.metrics.active_requests,
                "queued": self.metrics.queued_requests,
            },
            "performance": {
                "avg_latency": self.metrics.avg_request_latency,
                "p95_latency": self.metrics.p95_latency,
                "error_rate": self.metrics.error_rate,
                "utilization": self.metrics.get_utilization(),
            },
            "health": {
                "success_rate": self.metrics.health_check_success_rate,
                "turnover_rate": self.metrics.connection_turnover_rate,
            },
            "scaling_signals": {
                "should_scale_up": self.metrics.get_utilization()
                > self.config.scale_up_threshold,
                "should_scale_down": self.metrics.get_utilization()
                < self.config.scale_down_threshold,
                "queue_pressure": self.metrics.get_queue_ratio(),
            },
            "config": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "scale_up_threshold": self.config.scale_up_threshold,
                "scale_down_threshold": self.config.scale_down_threshold,
            },
        }

    async def force_health_check(self) -> dict[str, Any]:
        """Force immediate health check and return results."""
        await self._perform_health_checks()

        return {
            "timestamp": time.time(),
            "healthy_connections": self.metrics.total_connections
            - self.metrics.unhealthy_connections,
            "unhealthy_connections": self.metrics.unhealthy_connections,
            "health_success_rate": self.metrics.health_check_success_rate,
            "connection_details": {
                conn_id: {
                    "healthy": stats.is_healthy,
                    "age": stats.age,
                    "use_count": stats.use_count,
                    "failures": stats.health_check_failures,
                }
                for conn_id, stats in self._connection_stats.items()
            },
        }
