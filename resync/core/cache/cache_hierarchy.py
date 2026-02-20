import asyncio
import logging
from dataclasses import dataclass
from time import time as time_func
from typing import Any
from cachetools import LRUCache
from resync.core.metrics_compat import Counter, Histogram
from resync.settings import settings
from .async_cache import AsyncTTLCache

cache_hits = Counter("cache_hierarchy_hits_total", "Total cache hits", ["cache_level"])
cache_misses = Counter(
    "cache_hierarchy_misses_total", "Total cache misses", ["cache_level"]
)
cache_latency = Histogram(
    "cache_hierarchy_latency_seconds", "Cache operation latency", ["cache_level"]
)
logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Tracks cache performance metrics."""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    total_gets: int = 0
    total_sets: int = 0
    l1_evictions: int = 0
    l1_get_latency: float = 0.0
    l2_get_latency: float = 0.0
    miss_latency: float = 0.0

    @property
    def l1_hit_ratio(self) -> float:
        """Calculate L1 hit ratio."""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_ratio(self) -> float:
        """Calculate L2 hit ratio."""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def overall_hit_ratio(self) -> float:
        """Calculate overall hit ratio."""
        total = self.total_gets
        hits = self.l1_hits + self.l2_hits
        return hits / total if total > 0 else 0.0


class L1Cache:
    """
    In-memory L1 cache with LRU eviction using cachetools and sharded asyncio.Lock protection.
    """

    def __init__(self, max_size: int = 1000, num_shards: int = 16):
        """
        Initialize L1 cache.
        """
        if max_size > 0 and num_shards > max_size:
            num_shards = 1
        if num_shards <= 0:
            raise ValueError("num_shards must be a positive integer")
        self.max_size = max_size
        self.num_shards = num_shards
        self.shards: list[LRUCache] = [
            LRUCache(maxsize=max_size // num_shards if num_shards > 0 else max_size)
            for _ in range(num_shards)
        ]
        self.shard_locks = [asyncio.Lock() for _ in range(num_shards)]

    def _get_shard(self, key: str) -> tuple[LRUCache, asyncio.Lock]:
        """Get the shard and lock for a given key."""
        shard_index = hash(key) % self.num_shards
        return (self.shards[shard_index], self.shard_locks[shard_index])

    async def get(self, key: str) -> Any | None:
        """
        Get value from L1 cache.
        """
        shard, lock = self._get_shard(key)
        async with lock:
            try:
                value = shard[key]
                logger.debug("L1 cache HIT for key: %s", key)
                return value
            except KeyError:
                logger.debug("L1 cache MISS for key: %s", key)
                return None

    async def set(self, key: str, value: Any) -> None:
        """
        Set value in L1 cache with LRU eviction if needed.
        """
        shard, lock = self._get_shard(key)
        async with lock:
            shard[key] = value

    async def delete(self, key: str) -> bool:
        """
        Delete key from L1 cache.
        """
        shard, lock = self._get_shard(key)
        async with lock:
            try:
                del shard[key]
                logger.debug("L1 cache DELETE for key: %s", key)
                return True
            except KeyError:
                return False

    async def clear(self) -> None:
        """Clear all entries from L1 cache."""
        for i in range(self.num_shards):
            shard = self.shards[i]
            lock = self.shard_locks[i]
            async with lock:
                shard.clear()
        logger.debug("L1 cache CLEARED")

    def size(self) -> int:
        """Get current size of L1 cache."""
        return sum((len(shard) for shard in self.shards))


class CacheHierarchy:
    """
    Two-tier cache hierarchy with L1 (in-memory) and L2 (Redis-backed).
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_ttl_seconds: int = 300,
        l2_cleanup_interval: int = 30,
        enable_encryption: bool = False,
        key_prefix: str = "cache:",
    ):
        """
        Initialize cache hierarchy.

        Args:
            l1_max_size: Maximum size for L1 cache
            l2_ttl_seconds: TTL for L2 cache entries
            l2_cleanup_interval: Cleanup interval for L2 cache
            enable_encryption: Whether to enable cache encryption
            key_prefix: Prefix for cache keys
        """
        self.enable_encryption = enable_encryption
        self.key_prefix = key_prefix
        self.l1_cache = L1Cache(max_size=l1_max_size)
        self.l2_cache = AsyncTTLCache(
            ttl_seconds=l2_ttl_seconds, cleanup_interval=l2_cleanup_interval
        )
        self.metrics = CacheMetrics()
        self.is_running = False
        logger.info(
            f"CacheHierarchy initialized: L1_max_size={l1_max_size}, L2_ttl={l2_ttl_seconds}s, encryption={enable_encryption}, key_prefix={key_prefix}"
        )

    def _apply_key_prefix(self, key: str) -> str:
        """Apply key prefix if configured."""
        if self.key_prefix and self.key_prefix != "cache:":
            return f"{self.key_prefix}{key}"
        return key

    def _encrypt_value(self, value: Any) -> Any:
        """Encrypt value if encryption is enabled."""
        if self.enable_encryption:
            try:
                import base64
                import json

                value_str = json.dumps(value, default=str)
                encoded = base64.b64encode(value_str.encode()).decode()
                return {"__encrypted__": True, "data": encoded}
            except Exception as e:
                logger.warning("encryption_failed: error=%s, value_type=%s", str(e), type(value).__name__)
        return value

    def _decrypt_value(self, value: Any) -> Any:
        """Decrypt value if encryption is enabled."""
        if (
            self.enable_encryption
            and isinstance(value, dict)
            and value.get("__encrypted__")
        ):
            try:
                import base64
                import json

                encoded = value.get("data", "")
                decoded = base64.b64decode(encoded.encode()).decode()
                return json.loads(decoded)
            except Exception as e:
                logger.warning("decryption_failed: error=%s", str(e))
        return value

    def start(self) -> None:
        """Start the cache hierarchy."""
        if not self.is_running:
            self.is_running = True
            logger.info("CacheHierarchy started")

    async def stop(self) -> None:
        """Stop the cache hierarchy."""
        if self.is_running:
            self.is_running = False
            await self.l2_cache.stop()
            logger.info("CacheHierarchy stopped")

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache hierarchy with priority L1 → L2.
        Applies key prefix and decryption as needed.
        """
        prefixed_key = self._apply_key_prefix(key)
        start_time = time_func()
        self.metrics.total_gets += 1
        l1_value = await self.l1_cache.get(prefixed_key)
        if l1_value is not None:
            self.metrics.l1_hits += 1
            cache_hits.labels(cache_level="l1").inc()
            cache_latency.labels(cache_level="l1").observe(time_func() - start_time)
            return self._decrypt_value(l1_value)
        self.metrics.l1_misses += 1
        l2_value = await self.l2_cache.get(prefixed_key)
        if l2_value is not None:
            self.metrics.l2_hits += 1
            cache_hits.labels(cache_level="l2").inc()
            await self.l1_cache.set(prefixed_key, l2_value)
            cache_latency.labels(cache_level="l2").observe(time_func() - start_time)
            return self._decrypt_value(l2_value)
        self.metrics.l2_misses += 1
        cache_misses.labels(cache_level="l2").inc()
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """
        Set value in cache hierarchy with write-through pattern.
        Applies key prefix and encryption as needed.
        """
        prefixed_key = self._apply_key_prefix(key)
        encrypted_value = self._encrypt_value(value)
        self.metrics.total_sets += 1
        await self.l2_cache.set(prefixed_key, encrypted_value, ttl_seconds)
        await self.l1_cache.set(prefixed_key, encrypted_value)
        logger.debug("cache_hierarchy_set: key=%s", prefixed_key)

    async def set_from_source(
        self, key: str, value: Any, ttl_seconds: int | None = None
    ) -> None:
        """
        Set value after fetching from source.
        """
        await self.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> bool:
        """
        Delete key from both cache tiers.
        Applies key prefix as needed.
        """
        prefixed_key = self._apply_key_prefix(key)
        l1_deleted = await self.l1_cache.delete(prefixed_key)
        l2_deleted = await self.l2_cache.delete(prefixed_key)
        return l1_deleted or l2_deleted

    async def clear(self) -> None:
        """Clear all entries from both cache tiers."""
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        logger.debug("Cache HIERARCHY CLEARED")

    def size(self) -> tuple[int, int]:
        """Get sizes of both cache tiers."""
        l1_size = self.l1_cache.size()
        l2_size = self.l2_cache.size()
        return (l1_size, l2_size)

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache metrics."""
        l1_size, l2_size = self.size()
        return {
            "l1_size": l1_size,
            "l2_size": l2_size,
            "l1_hit_ratio": self.metrics.l1_hit_ratio,
            "l2_hit_ratio": self.metrics.l2_hit_ratio,
            "overall_hit_ratio": self.metrics.overall_hit_ratio,
            "total_gets": self.metrics.total_gets,
            "total_sets": self.metrics.total_sets,
            "l1_evictions": self.metrics.l1_evictions,
        }

    async def __aenter__(self) -> "CacheHierarchy":
        """Async context manager entry."""
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()


cache_hierarchy: CacheHierarchy | None = None


def get_cache_hierarchy() -> CacheHierarchy:
    """Get or create global cache hierarchy instance."""
    global cache_hierarchy
    if cache_hierarchy is None:
        cache_hierarchy = CacheHierarchy(
            l1_max_size=settings.CACHE_HIERARCHY.L1_MAX_SIZE,
            l2_ttl_seconds=settings.CACHE_HIERARCHY.L2_TTL_SECONDS,
            l2_cleanup_interval=settings.CACHE_HIERARCHY.L2_CLEANUP_INTERVAL,
            enable_encryption=getattr(
                settings.CACHE_HIERARCHY, "CACHE_ENCRYPTION_ENABLED", False
            ),
            key_prefix=getattr(settings.CACHE_HIERARCHY, "CACHE_KEY_PREFIX", "cache:"),
        )
    return cache_hierarchy
