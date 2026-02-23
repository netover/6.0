# pylint: disable=all
# mypy: no-rerun
"""
Connection pool base classes and configuration for the Resync project.
Separated to follow Single Responsibility Principle.
"""

import asyncio
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Type Definitions ---
T = TypeVar("T")


@dataclass(frozen=True)
class ConnectionPoolStats:
    """Statistics for connection pool monitoring."""

    pool_name: str
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    waiting_connections: int = 0
    connection_errors: int = 0
    connection_creations: int = 0
    connection_closures: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    pool_exhaustions: int = 0
    acquisition_attempts: int = (
        0  # Nova métrica para contagem de tentativas de aquisição
    )
    session_acquisitions: int = (
        0  # Nova métrica para contagem de sessões adquiridas com sucesso
    )
    last_health_check: datetime | None = None
    average_wait_time: float = 0.0
    peak_connections: int = 0


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""

    pool_name: str
    min_size: int = 5
    max_size: int = 20
    idle_timeout: int = 300  # 5 minutes
    connection_timeout: int = 30  # 30 seconds
    health_check_interval: int = 60  # 1 minute
    max_lifetime: int = 1800  # 30 minutes


class ConnectionPool(ABC, Generic[T]):
    """Abstract base class for connection pools."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._initialized = False
        self._shutdown = False
        self.stats = ConnectionPoolStats(pool_name=config.pool_name)
        self._lock = asyncio.Lock()  # For thread-safe operations
        self._stats_lock = asyncio.Lock()  # For thread-safe stats operations
        self._wait_times: deque[float] = deque(
            maxlen=1000
        )  # Track connection acquisition times for metrics

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized or self._shutdown:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                self._setup_pool()
                self._initialized = True
                logger.info("Initialized %s connection pool", self.config.pool_name)
            except Exception as e:
                logger.error(
                    "Failed to initialize %s connection pool: %s",
                    self.config.pool_name,
                    e,
                )
                raise

    def _setup_pool(self) -> None:
        """Setup the connection pool - to be implemented by subclasses."""

    @asynccontextmanager
    @abstractmethod
    def get_connection(self) -> AsyncIterator[T]:
        """Get a connection from the pool."""

    async def close(self) -> None:
        """Close the connection pool."""
        if not self._initialized or self._shutdown:
            return

        async with self._lock:
            if self._shutdown:
                return

            try:
                self._close_pool()
                self._shutdown = True
                logger.info("Closed %s connection pool", self.config.pool_name)
            except Exception as e:
                logger.error(
                    "Error closing %s connection pool: %s", self.config.pool_name, e
                )
                raise

    def _close_pool(self) -> None:
        """Close the connection pool - to be implemented by subclasses."""

    async def update_wait_time(self, wait_time: float) -> None:
        """Update wait time statistics."""
        async with self._stats_lock:
            self._wait_times.append(wait_time)
            # Calculate average
            if self._wait_times:
                average = sum(self._wait_times) / len(self._wait_times)
                # Update stats with new average (immutable, so create new instance)
                self.stats = dataclasses.replace(self.stats, average_wait_time=average)

    async def increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """Increment a statistic in a thread-safe manner."""
        async with self._stats_lock:
            current_value = getattr(self.stats, stat_name, 0)
            new_value = current_value + amount
            # Create a new immutable stats object with the updated value
            updated_stats = dataclasses.replace(self.stats, **{stat_name: new_value})
            self.stats = updated_stats

    def get_stats_copy(self) -> dict:
        """Return a mutable copy of the current stats for safe access in tests or external use."""
        return dataclasses.asdict(self.stats)

    async def health_check(self) -> bool:
        """Perform a health check on the pool."""
        if not self._initialized or self._shutdown:
            return False

        try:
            # Try to get and use a connection briefly
            async with self.get_connection():
                # The actual health check depends on the connection type
                # This is a basic check that just tries to acquire a connection
                pass
            return True
        except Exception as e:
            logger.warning("Health check failed for %s: %s", self.config.pool_name, e)
            return False
