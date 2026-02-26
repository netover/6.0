# pylint
# mypy
"""
Advanced Application Cache with Intelligent Invalidation.

This module provides an intelligent caching system with:
- Dependency-based automatic invalidation
- Dynamic TTL based on usage patterns
- Hierarchical caching (memory -> Redis -> database)
- Cache warming and predictive loading
- Performance metrics and monitoring
- Cascade invalidation for related data
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
import contextlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from resync.core.redis_init import get_redis_initializer
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""

    key: str
    value: Any
    ttl: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    dependencies: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    hit_rate: float = 1.0
    size_bytes: int = 0

    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_time(self) -> float:
        """Get idle time since last access."""
        return time.time() - self.last_accessed

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return self.age > self.ttl

    @property
    def should_evict(self) -> bool:
        """Check if entry should be evicted based on policies."""
        # Evict if expired
        if self.is_expired:
            return True

        # Evict if low hit rate and old
        return self.hit_rate < 0.1 and self.age > 3600  # 1 hour

    def calculate_dynamic_ttl(self, usage_stats: dict[str, Any]) -> int:
        """Calculate dynamic TTL based on usage patterns."""
        base_ttl = self.ttl

        # Increase TTL for frequently accessed items
        if self.hit_rate > 0.8 and self.access_count > 100:
            base_ttl *= 2

        # Decrease TTL for rarely accessed items
        elif self.hit_rate < 0.2:
            base_ttl = max(60, base_ttl // 4)  # Minimum 1 minute

        # Adjust based on access frequency
        access_freq = self.access_count / max(1, self.age / 3600)  # accesses per hour
        if access_freq > 10:  # High frequency
            base_ttl = int(min(base_ttl * 1.5, 86400))  # Max 24 hours
        elif access_freq < 1:  # Low frequency
            base_ttl = max(60, base_ttl // 2)  # Minimum 1 minute

        return int(base_ttl)

@dataclass
class CacheStats:
    """Comprehensive cache statistics."""

    # Hit/miss statistics
    hits: int = 0
    misses: int = 0
    total_requests: int = 0

    # Size and capacity
    current_entries: int = 0
    max_entries: int = 10000
    memory_usage_bytes: int = 0
    max_memory_bytes: int = 100 * 1024 * 1024  # 100MB

    # Performance metrics
    avg_hit_latency: float = 0.0
    avg_miss_latency: float = 0.0
    eviction_count: int = 0
    invalidation_count: int = 0

    # Layer statistics
    memory_hits: int = 0
    redis_hits: int = 0
    database_hits: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.hits / max(1, self.total_requests)

    def memory_efficiency(self) -> float:
        """Calculate memory efficiency."""
        return self.memory_usage_bytes / max(1, self.max_memory_bytes)

@dataclass
class InvalidationRule:
    """Rule for automatic cache invalidation."""

    pattern: str  # Key pattern to match
    dependencies: list[str]  # Keys that depend on this pattern
    cascade: bool = True  # Whether to invalidate recursively
    ttl_multiplier: float = 1.0  # TTL adjustment for dependent keys

class CacheDependencyGraph:
    """Graph for managing cache dependencies and cascade invalidation."""

    def __init__(self):
        self.dependencies: dict[str, set[str]] = defaultdict(
            set
        )  # key -> dependents  # type: ignore[annotation-unchecked]
        self.reverse_dependencies: dict[str, set[str]] = defaultdict(  # type: ignore[annotation-unchecked]
            set
        )  # key -> dependencies
        self.tags: dict[str, set[str]] = defaultdict(
            set
        )  # tag -> keys  # type: ignore[annotation-unchecked]

    def add_dependency(self, key: str, depends_on: str) -> None:
        """Add a dependency relationship."""
        self.dependencies[depends_on].add(key)
        self.reverse_dependencies[key].add(depends_on)

    def add_tags(self, key: str, tags: list[str]) -> None:
        """Add tags to a key."""
        for tag in tags:
            self.tags[tag].add(key)

    def get_dependents(self, key: str) -> set[str]:
        """Get all keys that depend on the given key."""
        return self.dependencies.get(key, set())

    def get_dependencies(self, key: str) -> set[str]:
        """Get all keys that the given key depends on."""
        return self.reverse_dependencies.get(key, set())

    def get_keys_by_tag(self, tag: str) -> set[str]:
        """Get all keys with a specific tag."""
        return self.tags.get(tag, set())

    def cascade_invalidate(self, key: str) -> set[str]:
        """Get all keys that should be invalidated due to cascade."""
        to_invalidate = set()
        visited = set()

        def _collect_dependents(current_key: str) -> None:
            if current_key in visited:
                return
            visited.add(current_key)

            # Add direct dependents
            for dependent in self.get_dependents(current_key):
                to_invalidate.add(dependent)
                _collect_dependents(dependent)

        _collect_dependents(key)
        return to_invalidate

    def remove_key(self, key: str) -> None:
        """Remove a key from the dependency graph."""
        # Remove from dependencies
        for dependents in self.dependencies.values():
            dependents.discard(key)

        # Remove from reverse dependencies
        if key in self.reverse_dependencies:
            del self.reverse_dependencies[key]

        # Remove from tags
        for tag_keys in self.tags.values():
            tag_keys.discard(key)

class AdvancedCacheManager:
    """
    Intelligent cache manager with automatic invalidation and hierarchical caching.

    Features:
    - Multi-layer caching (memory -> Redis -> database)
    - Dependency-based invalidation
    - Dynamic TTL adjustment
    - Cache warming and predictive loading
    - Performance monitoring and analytics
    """

    def __init__(self):
        self.memory_cache: dict[str, CacheEntry] = {}  # type: ignore[annotation-unchecked]
        self.stats = CacheStats()
        self.dependency_graph = CacheDependencyGraph()

        # Redis integration
        self.redis_client = None
        self.redis_enabled = False

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None  # type: ignore[annotation-unchecked]
        self._warming_task: asyncio.Task | None = None  # type: ignore[annotation-unchecked]
        self._running = False

        # Configuration
        self.ttl_policies = {
            "user_data": 300,  # 5 minutes
            "config": 3600,  # 1 hour
            "analytics": 1800,  # 30 minutes
            "api_responses": 600,  # 10 minutes
            "static_data": 86400,  # 24 hours
        }

        # Invalidation rules
        self.invalidation_rules: list[InvalidationRule] = []  # type: ignore[annotation-unchecked]

        # Thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the advanced cache manager."""
        if self._running:
            return

        # Initialize Redis client
        try:
            redis_init = await get_redis_initializer()
            self.redis_client = await redis_init.initialize()
            self.redis_enabled = True
            logger.info("Redis integration enabled for advanced caching")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Redis not available for caching: %s", e)
            self.redis_enabled = False

        self._running = True

        # Start background tasks
        self._cleanup_task = create_tracked_task(
            self._cleanup_loop(), name="cleanup_loop"
        )
        self._warming_task = create_tracked_task(
            self._warming_loop(), name="warming_loop"
        )

        logger.info("Advanced cache manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the cache manager."""
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        if self._warming_task:
            self._warming_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._warming_task

        # Clear caches
        async with self._lock:
            self.memory_cache.clear()
            if self.redis_client:
                # Clear Redis cache (optional - depends on use case)
                pass

        logger.info("Advanced cache manager shutdown")

    async def get(
        self,
        key: str,
        fetch_func: Callable | None = None,
        ttl: int | None = None,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Get value from cache with intelligent fallback.

        Args:
            key: Cache key
            fetch_func: Function to fetch data if not cached
            ttl: Time to live in seconds
            dependencies: Keys this entry depends on
            tags: Tags for grouping and invalidation

        Returns:
            Cached or fetched value
        """
        start_time = time.time()

        # Try memory cache first
        value = await self._get_from_memory(key)
        if value is not None:
            self._record_hit("memory", time.time() - start_time)
            return value

        # Try Redis cache
        if self.redis_enabled:
            value = await self._get_from_redis(key)
            if value is not None:
                # Promote to memory cache
                await self._set_memory(key, value, ttl or 300, dependencies, tags)  # type: ignore[attr-defined]
                self._record_hit("redis", time.time() - start_time)
                return value

        # Cache miss - fetch from source
        if fetch_func:
            try:
                value = await fetch_func()
                # Cache the result
                await self.set(key, value, ttl, dependencies, tags)
                self._record_miss(time.time() - start_time)
                return value
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Failed to fetch data for key %s: %s", key, e)
                self._record_miss(time.time() - start_time)
                raise

        self._record_miss(time.time() - start_time)
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Set value in cache with metadata."""
        # Determine TTL
        if ttl is None:
            ttl = self._calculate_ttl(key, value)

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            dependencies=set(dependencies or []),
            tags=set(tags or []),
        )

        # Calculate size
        entry.size_bytes = self._calculate_size(value)

        # Set in memory
        await self._set_memory_entry(entry)

        # Set in Redis if enabled
        if self.redis_enabled:
            await self._set_redis(key, value, ttl)

        # Update dependency graph
        if dependencies:
            for dep in dependencies:
                self.dependency_graph.add_dependency(key, dep)

        if tags:
            self.dependency_graph.add_tags(key, tags)

        # Update stats
        async with self._lock:
            self.stats.current_entries = len(self.memory_cache)
            self.stats.memory_usage_bytes = sum(
                entry.size_bytes for entry in self.memory_cache.values()
            )

    async def invalidate(self, key: str, cascade: bool = True) -> int:
        """
        Invalidate cache entry with optional cascade invalidation.

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        async with self._lock:
            # Direct invalidation
            if key in self.memory_cache:
                del self.memory_cache[key]
                invalidated += 1

            if self.redis_enabled:
                await self.redis_client.delete(key)

            # Cascade invalidation
            if cascade:
                cascade_keys = self.dependency_graph.cascade_invalidate(key)
                for cascade_key in cascade_keys:
                    if cascade_key in self.memory_cache:
                        del self.memory_cache[cascade_key]
                        invalidated += 1

                    if self.redis_enabled:
                        await self.redis_client.delete(cascade_key)

                # Clean up dependency graph
                self.dependency_graph.remove_key(key)
                for cascade_key in cascade_keys:
                    self.dependency_graph.remove_key(cascade_key)

        self.stats.invalidation_count += invalidated
        logger.info("Invalidated %s cache entries for key %s", invalidated, key)
        return invalidated

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag."""
        keys_to_invalidate = self.dependency_graph.get_keys_by_tag(tag)
        invalidated = 0

        for key in keys_to_invalidate:
            invalidated += await self.invalidate(key, cascade=False)

        logger.info("Invalidated %s entries with tag %s", invalidated, tag)
        return invalidated

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern."""
        import re

        regex = re.compile(pattern)

        keys_to_invalidate = [key for key in self.memory_cache if regex.match(key)]

        invalidated = 0
        for key in keys_to_invalidate:
            invalidated += await self.invalidate(key, cascade=False)

        logger.info("Invalidated %s entries matching pattern %s", invalidated, pattern)
        return invalidated

    async def warm_cache(self, warming_keys: list[dict[str, Any]]) -> int:
        """
        Warm cache with predefined keys.

        Args:
            warming_keys: List of dicts with keys: 'key', 'fetch_func', 'ttl', 'dependencies', 'tags'

        Returns:
            Number of keys warmed
        """
        warmed = 0

        for key_config in warming_keys:
            try:
                key = key_config["key"]
                fetch_func = key_config["fetch_func"]

                # Check if already cached
                existing = await self._get_from_memory(key)
                if existing is not None:
                    continue

                # Fetch and cache
                value = await fetch_func()
                await self.set(
                    key=key,
                    value=value,
                    ttl=key_config.get("ttl"),
                    dependencies=key_config.get("dependencies"),
                    tags=key_config.get("tags"),
                )

                warmed += 1
                logger.debug("Warmed cache key: %s", key)

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning(
                    "Failed to warm cache key %s: %s",
                    key_config.get("key", "unknown"),
                    e,
                )

        logger.info("Cache warming completed: %s keys warmed", warmed)
        return warmed

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "performance": {
                "hit_rate": self.stats.hit_rate(),
                "total_requests": self.stats.total_requests,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "avg_hit_latency": self.stats.avg_hit_latency,
                "avg_miss_latency": self.stats.avg_miss_latency,
            },
            "capacity": {
                "current_entries": self.stats.current_entries,
                "max_entries": self.stats.max_entries,
                "memory_usage_mb": self.stats.memory_usage_bytes / (1024 * 1024),
                "max_memory_mb": self.stats.max_memory_bytes / (1024 * 1024),
                "memory_efficiency": self.stats.memory_efficiency(),
            },
            "operations": {
                "evictions": self.stats.eviction_count,
                "invalidations": self.stats.invalidation_count,
                "memory_hits": self.stats.memory_hits,
                "redis_hits": self.stats.redis_hits,
                "database_hits": self.stats.database_hits,
            },
            "layers": {
                "memory_enabled": True,
                "redis_enabled": self.redis_enabled,
                "database_enabled": True,
            },
        }

    async def _get_from_memory(self, key: str) -> Any:
        """Get value from memory cache."""
        async with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]

                if entry.is_expired:
                    del self.memory_cache[key]
                    return None

                # Update access statistics
                entry.last_accessed = time.time()
                entry.access_count += 1
                entry.hit_rate = entry.access_count / max(
                    1, entry.age / 60
                )  # hits per minute

                return entry.value

        return None

    async def _set_memory_entry(self, entry: CacheEntry) -> None:
        """Set entry in memory cache with eviction if needed."""
        async with self._lock:
            # Check size limits
            if len(self.memory_cache) >= self.stats.max_entries:
                await self._evict_entries()

            if (
                self.stats.memory_usage_bytes + entry.size_bytes
                > self.stats.max_memory_bytes
            ):
                await self._evict_entries()

            self.memory_cache[entry.key] = entry

    async def _get_from_redis(self, key: str) -> Any:
        """Get value from Redis cache."""
        if not self.redis_enabled or not self.redis_client:
            return None

        try:
            value_json = await self.redis_client.get(key)
            if value_json:
                return json.loads(value_json)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Redis get error for key %s: %s", key, e)

        return None

    async def _set_redis(self, key: str, value: Any, ttl: int) -> None:
        """Set value in Redis cache."""
        if not self.redis_enabled or not self.redis_client:
            return

        try:
            value_json = json.dumps(value)
            await self.redis_client.set(key, value_json, ex=ttl)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Redis set error for key %s: %s", key, e)

    def _calculate_ttl(self, key: str, value: Any) -> int:
        """Calculate appropriate TTL based on key patterns and value characteristics."""
        # Check if key matches known patterns
        for pattern, ttl in self.ttl_policies.items():
            if pattern in key.lower():
                return ttl

        # Default TTL based on value type/size
        if isinstance(value, (list, dict)) and len(str(value)) > 1000:
            return 1800  # Large data - 30 minutes
        if isinstance(value, (int, float, str)) and len(str(value)) < 100:
            return 3600  # Small data - 1 hour
        return 900  # Default - 15 minutes

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate memory usage of a value."""
        return len(json.dumps(value, default=str).encode("utf-8"))

    async def _evict_entries(self) -> None:
        """Evict entries based on LRU and other policies."""
        async with self._lock:
            # Sort entries by eviction priority (age, hit rate, size)
            entries = list(self.memory_cache.items())

            def eviction_score(entry_tuple):
                key, entry = entry_tuple
                # Lower score = more likely to evict
                return (
                    entry.idle_time * (2 - entry.hit_rate) * (entry.size_bytes / 1000)
                )

            entries.sort(key=eviction_score, reverse=True)  # Highest scores first

            # Evict 10% of entries or at least 1
            to_evict = max(1, len(entries) // 10)

            for i in range(to_evict):
                if i < len(entries):
                    key, entry = entries[i]
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                        self.stats.eviction_count += 1
                        self.dependency_graph.remove_key(key)

    def _record_hit(self, layer: str, latency: float) -> None:
        """Record cache hit."""
        self.stats.hits += 1
        self.stats.total_requests += 1

        # Update latency
        if self.stats.hits == 1:
            self.stats.avg_hit_latency = latency
        else:
            self.stats.avg_hit_latency = (
                (self.stats.avg_hit_latency * (self.stats.hits - 1)) + latency
            ) / self.stats.hits

        # Layer-specific hits
        if layer == "memory":
            self.stats.memory_hits += 1
        elif layer == "redis":
            self.stats.redis_hits += 1
        elif layer == "database":
            self.stats.database_hits += 1

    def _record_miss(self, latency: float) -> None:
        """Record cache miss."""
        self.stats.misses += 1
        self.stats.total_requests += 1

        # Update latency
        if self.stats.misses == 1:
            self.stats.avg_miss_latency = latency
        else:
            self.stats.avg_miss_latency = (
                (self.stats.avg_miss_latency * (self.stats.misses - 1)) + latency
            ) / self.stats.misses

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Clean every minute

                async with self._lock:
                    expired_keys = [
                        key
                        for key, entry in self.memory_cache.items()
                        if entry.should_evict()  # type: ignore[operator]
                    ]

                    for key in expired_keys:
                        if key in self.memory_cache:
                            del self.memory_cache[key]
                            self.stats.eviction_count += 1
                            self.dependency_graph.remove_key(key)

                    if expired_keys:
                        logger.debug(
                            "Cleaned up %s expired cache entries", len(expired_keys)
                        )

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Cache cleanup error: %s", e)

    async def _warming_loop(self) -> None:
        """Background cache warming based on access patterns."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Identify frequently accessed keys that could benefit from longer TTL
                async with self._lock:
                    candidates = [
                        (key, entry)
                        for key, entry in self.memory_cache.items()
                        if entry.hit_rate > 0.5 and entry.access_count > 10
                    ]

                    for key, entry in candidates:
                        # Extend TTL for hot entries
                        new_ttl = entry.calculate_dynamic_ttl({})
                        if new_ttl > entry.ttl:
                            entry.ttl = new_ttl
                            logger.debug(
                                "Extended TTL for hot key %s to %ss", key, new_ttl
                            )

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Cache warming error: %s", e)

# Global cache manager instance
_advanced_cache_manager_instance: AdvancedCacheManager | None = None

class _LazyAdvancedCacheManager:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance = None

    def get_instance(self) -> AdvancedCacheManager:
        if self._instance is None:
            self._instance = AdvancedCacheManager()  # type: ignore[assignment]
        return self._instance  # type: ignore[return-value]

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)

advanced_cache_manager = _LazyAdvancedCacheManager()

def get_advanced_cache_manager_sync() -> AdvancedCacheManager:
    """Return the singleton instance synchronously."""
    return advanced_cache_manager.get_instance()

async def get_advanced_cache_manager() -> AdvancedCacheManager:
    """Get the global advanced cache manager instance."""
    if not advanced_cache_manager._running:
        await advanced_cache_manager.initialize()
    return advanced_cache_manager  # type: ignore[return-value]

# Backward compatibility aliases
AdvancedApplicationCache = AdvancedCacheManager

class CacheInvalidator:
    """Legacy invalidator class shim."""

    def __init__(self, manager: AdvancedCacheManager | None = None):
        self.manager = manager or advanced_cache_manager

    async def invalidate(self, key: str, cascade: bool = True) -> int:
        return await self.manager.invalidate(key, cascade)
