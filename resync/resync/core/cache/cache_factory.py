"""
Cache Factory Module

This module provides a factory class for creating different types of cache instances
in the resync application. It centralizes cache creation logic and allows for
easy extension with new cache implementations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resync.core.cache.base_cache import BaseCache

if TYPE_CHECKING:
    from resync.core.cache.cache_with_stampede_protection import CacheConfig


@dataclass
class SimpleCacheConfig:
    """Simple cache configuration."""

    ttl_seconds: float = 300.0
    max_entries: int = 10000
    enable_stats: bool = True


@dataclass
class CacheEntry:
    """Cache entry with timestamp and TTL."""

    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0

    def is_expired(self) -> bool:
        return time.time() > self.timestamp + self.ttl


class MemoryCache(BaseCache):
    """In-memory cache implementation."""

    def __init__(self, config: SimpleCacheConfig | None = None):
        self.config = config or SimpleCacheConfig()
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.data

    async def set(self, key: str, value: Any, ttl: float | None = None) -> bool:
        async with self._lock:
            # Evict if at capacity
            if len(self._store) >= self.config.max_entries:
                self._evict_oldest()

            self._store[key] = CacheEntry(data=value, ttl=ttl or self.config.ttl_seconds)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if not self._store:
            return
        oldest_key = min(self._store.keys(), key=lambda k: self._store[k].timestamp)
        del self._store[oldest_key]

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._store),
        }


class EnhancedCache(MemoryCache):
    """Enhanced cache with stampede protection."""

    def __init__(self, config: SimpleCacheConfig | None = None):
        super().__init__(config)
        self._computing: dict[str, asyncio.Event] = {}

    async def get_or_compute(self, key: str, compute_fn: callable, ttl: float | None = None) -> Any:
        """Get value or compute if missing (with stampede protection)."""
        # Check cache first
        value = await self.get(key)
        if value is not None:
            return value

        # Check if already computing
        if key in self._computing:
            await self._computing[key].wait()
            return await self.get(key)

        # Start computing
        self._computing[key] = asyncio.Event()
        try:
            value = await compute_fn() if asyncio.iscoroutinefunction(compute_fn) else compute_fn()
            await self.set(key, value, ttl)
            return value
        finally:
            self._computing[key].set()
            del self._computing[key]


class HybridCache(BaseCache):
    """Hybrid cache combining memory and persistent storage."""

    def __init__(
        self, memory_config: SimpleCacheConfig | None = None, persistent_store: dict | None = None
    ):
        self._memory = MemoryCache(memory_config)
        self._persistent = persistent_store or {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        # Try memory first
        value = await self._memory.get(key)
        if value is not None:
            return value

        # Try persistent
        async with self._lock:
            if key in self._persistent:
                value = self._persistent[key]
                # Promote to memory
                await self._memory.set(key, value)
                return value

        return None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> bool:
        # Set in both stores
        await self._memory.set(key, value, ttl)
        async with self._lock:
            self._persistent[key] = value
        return True

    async def delete(self, key: str) -> bool:
        await self._memory.delete(key)
        async with self._lock:
            if key in self._persistent:
                del self._persistent[key]
                return True
        return False

    async def clear(self) -> None:
        await self._memory.clear()
        async with self._lock:
            self._persistent.clear()


class CacheFactory:
    """
    Factory class for creating cache instances.

    This class provides methods to create different types of cache implementations
    based on configuration parameters. It serves as a central point for cache
    creation and allows for easy extension with new cache types.
    """

    @staticmethod
    def create_enhanced_cache(config: "CacheConfig" = None) -> BaseCache:
        """
        Create an enhanced cache instance with advanced features.

        Args:
            config: Cache configuration containing settings for TTL,
                   stampede protection, and other advanced features

        Returns:
            BaseCache: An enhanced cache implementation instance
        """
        simple_config = (
            SimpleCacheConfig(
                ttl_seconds=getattr(config, "ttl_seconds", 300),
                max_entries=getattr(config, "max_entries", 10000),
            )
            if config
            else SimpleCacheConfig()
        )

        return EnhancedCache(simple_config)

    @staticmethod
    def create_memory_cache(config: "CacheConfig" = None) -> BaseCache:
        """
        Create a memory-based cache instance.

        Args:
            config: Cache configuration containing settings for TTL and other parameters

        Returns:
            BaseCache: A memory-based cache implementation instance
        """
        simple_config = (
            SimpleCacheConfig(
                ttl_seconds=getattr(config, "ttl_seconds", 300),
                max_entries=getattr(config, "max_entries", 10000),
            )
            if config
            else SimpleCacheConfig()
        )

        return MemoryCache(simple_config)

    @staticmethod
    def create_hybrid_cache(config: "CacheConfig" = None) -> BaseCache:
        """
        Create a hybrid cache instance combining multiple storage backends.

        Args:
            config: Cache configuration containing settings for the hybrid cache

        Returns:
            BaseCache: A hybrid cache implementation instance
        """
        memory_config = (
            SimpleCacheConfig(
                ttl_seconds=getattr(config, "ttl_seconds", 300),
                max_entries=getattr(config, "max_entries", 5000),
            )
            if config
            else SimpleCacheConfig()
        )

        return HybridCache(memory_config)
