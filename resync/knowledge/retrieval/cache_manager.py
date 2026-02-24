"""
Knowledge Graph Cache Manager with TTL.

Provides automatic cache invalidation and reload for the in-memory
NetworkX graph to ensure data freshness.

Features:
- Configurable TTL (Time-To-Live)
- Background refresh task
- Manual invalidation
- Cache statistics

Usage:
    from resync.knowledge.retrieval.cache_manager import (
        KGCacheManager,
        get_cache_manager,
        start_cache_refresh_task
    )

    # Start background refresh (typically at app startup)
    await start_cache_refresh_task(ttl_seconds=300)

    # Manual invalidation
    cache = get_cache_manager()
    await cache.invalidate()
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from resync.core.structured_logger import get_logger
from resync.core.task_tracker import track_task

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    last_load: datetime | None = None
    last_invalidation: datetime | None = None
    load_count: int = 0
    invalidation_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    avg_load_time_ms: float = 0.0
    _load_times: list = field(default_factory=list)

    def record_load(self, duration_ms: float):
        """Record a cache load operation."""
        self.load_count += 1
        self.last_load = datetime.now(timezone.utc)
        self._load_times.append(duration_ms)
        # Keep last 100 for rolling average
        if len(self._load_times) > 100:
            self._load_times = self._load_times[-100:]
        self.avg_load_time_ms = sum(self._load_times) / len(self._load_times)

    def record_invalidation(self):
        """Record a cache invalidation."""
        self.invalidation_count += 1
        self.last_invalidation = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "last_load": self.last_load.isoformat() if self.last_load else None,
            "last_invalidation": self.last_invalidation.isoformat()
            if self.last_invalidation
            else None,
            "load_count": self.load_count,
            "invalidation_count": self.invalidation_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "avg_load_time_ms": round(self.avg_load_time_ms, 2),
        }


class KGCacheManager:
    """
    Manages cache lifecycle for the Knowledge Graph.

    Provides TTL-based invalidation and background refresh.
    """

    _instance: Optional["KGCacheManager"] = None

    def __new__(cls) -> "KGCacheManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize cache manager."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._ttl_seconds: int = 300  # 5 minutes default
        self._last_refresh: datetime | None = None
        self._refresh_task: asyncio.Task | None = None
        # P0 fix: Initialize lock eagerly to prevent race condition
        self._lock: asyncio.Lock = asyncio.Lock()
        self._stats = CacheStats()
        self._on_refresh_callbacks: list[Callable[[], Awaitable[None]]] = []

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_ttl(self, seconds: int) -> None:
        """
        Set cache TTL in seconds.

        Args:
            seconds: TTL duration (minimum 60 seconds)
        """
        self._ttl_seconds = max(60, seconds)
        logger.info("cache_ttl_updated", ttl_seconds=self._ttl_seconds)

    def get_ttl(self) -> int:
        """Get current TTL in seconds."""
        return self._ttl_seconds

    def register_refresh_callback(
        self, callback: Callable[[], Awaitable[None]]
    ) -> None:
        """
        Register a callback to be called on cache refresh.

        The Knowledge Graph's reload() method should be registered here.

        Args:
            callback: Async function to call on refresh
        """
        self._on_refresh_callbacks.append(callback)
        logger.debug("refresh_callback_registered", callback=callback.__name__)

    # =========================================================================
    # CACHE OPERATIONS
    # =========================================================================

    def is_stale(self) -> bool:
        """
        Check if cache needs refresh.

        Returns:
            True if TTL exceeded or never loaded
        """
        if self._last_refresh is None:
            return True

        age = datetime.now(timezone.utc) - self._last_refresh
        return age.total_seconds() > self._ttl_seconds

    def time_until_stale(self) -> timedelta:
        """
        Get time remaining until cache becomes stale.

        Returns:
            Timedelta until stale (negative if already stale)
        """
        if self._last_refresh is None:
            return timedelta(seconds=0)

        age = datetime.now(timezone.utc) - self._last_refresh
        return timedelta(seconds=self._ttl_seconds) - age

    async def refresh(self, force: bool = False) -> bool:
        """
        Refresh the cache if stale or forced.

        Args:
            force: Force refresh even if not stale

        Returns:
            True if refresh occurred
        """
        if not force and not self.is_stale():
            self._stats.hit_count += 1
            return False

        async with self._lock:
            # Double-check after acquiring lock
            if not force and not self.is_stale():
                self._stats.hit_count += 1
                return False

            self._stats.miss_count += 1

            start_time = datetime.now(timezone.utc)

            try:
                # Call all registered refresh callbacks
                for callback in self._on_refresh_callbacks:
                    await callback()

                duration_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                self._stats.record_load(duration_ms)
                self._last_refresh = datetime.now(timezone.utc)

                logger.info(
                    "cache_refreshed",
                    duration_ms=round(duration_ms, 2),
                    callbacks_executed=len(self._on_refresh_callbacks),
                )

                return True

            except Exception as e:
                logger.error("cache_refresh_failed", error=str(e))
                raise

    async def invalidate(self) -> None:
        """
        Invalidate the cache, forcing next access to refresh.
        """
        async with self._lock:
            self._last_refresh = None
            self._stats.record_invalidation()
            logger.info("cache_invalidated")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self._stats.to_dict()
        stats["ttl_seconds"] = self._ttl_seconds
        stats["is_stale"] = self.is_stale()
        stats["time_until_stale_seconds"] = max(
            0, self.time_until_stale().total_seconds()
        )
        return stats

    # =========================================================================
    # BACKGROUND REFRESH TASK
    # =========================================================================

    def start_background_refresh(self) -> None:
        """
        Start background task that refreshes cache periodically.
        """
        if self._refresh_task is not None and not self._refresh_task.done():
            logger.warning("background_refresh_already_running")
            return

        self._refresh_task = track_task(
            self._background_refresh_loop(), name="background_refresh_loop"
        )
        logger.info("background_refresh_started", interval_seconds=self._ttl_seconds)

    async def stop_background_refresh(self) -> None:
        """Stop background refresh task."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None
            logger.info("background_refresh_stopped")

    async def _background_refresh_loop(self) -> None:
        """Background loop that refreshes cache periodically."""
        while True:
            try:
                # Wait for TTL duration
                await asyncio.sleep(self._ttl_seconds)

                # Refresh cache
                await self.refresh(force=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("background_refresh_error", error=str(e))
                # Continue loop even on error
                await asyncio.sleep(60)  # Wait a bit before retrying


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_cache_manager: KGCacheManager | None = None


def get_cache_manager() -> KGCacheManager:
    """Get or create the singleton cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = KGCacheManager()
    return _cache_manager


async def start_cache_refresh_task(
    ttl_seconds: int = 300, auto_register_kg: bool = True
) -> KGCacheManager:
    """
    Start the cache refresh background task.

    Args:
        ttl_seconds: Cache TTL in seconds (default 5 minutes)
        auto_register_kg: Automatically register KG reload callback

    Returns:
        The cache manager instance
    """
    cache = get_cache_manager()
    cache.set_ttl(ttl_seconds)

    if auto_register_kg:
        # Import here to avoid circular import
        from resync.knowledge.retrieval.graph import get_knowledge_graph

        kg = await get_knowledge_graph()
        cache.register_refresh_callback(kg.reload)

    cache.start_background_refresh()

    return cache


async def stop_cache_refresh_task() -> None:
    """Stop the cache refresh background task."""
    cache = get_cache_manager()
    await cache.stop_background_refresh()
