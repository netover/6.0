"""
Memory management functionality for the async cache system.

This module provides the CacheMemoryManager class that handles memory bounds checking
and intelligent eviction strategies for the async cache implementation.
"""

from __future__ import annotations
import logging
import sys
import time
from typing import Any

try:
    from resync.core.metrics import runtime_metrics
except Exception as _e:

    class _DummyRuntimeMetrics:
        """_ dummy runtime metrics."""

        def __getattr__(self, name):
            class _Metric:
                """_ metric."""

                def increment(self, *args, **kwargs):
                    """No‑op increment for missing metrics."""

                def observe(self, *args, **kwargs):
                    """No‑op observe for missing metrics."""

                def set(self, *args, **kwargs):
                    """No‑op set for missing metrics."""

                @property
                def value(self):
                    """Returns a default value of 0 for missing metrics."""
                    return 0

            return _Metric()

    runtime_metrics = _DummyRuntimeMetrics()
logger = logging.getLogger(__name__)


class CacheEntry:
    """
    Represents a single entry in the cache with separate creation and access timestamps.

    Many caches previously conflated the creation timestamp with the last access time,
    leading to confusion when implementing TTL (based on age) and LRU (based on
    recent access). This implementation stores both ``created_at`` and
    ``last_accessed_at``. The ``timestamp`` attribute remains as an alias for
    ``last_accessed_at`` for backwards compatibility with existing code.
    """

    def __init__(self, data: Any, timestamp: float, ttl: float):
        self.data = data
        self.created_at: float = timestamp
        self.last_accessed_at: float = timestamp
        self.ttl = ttl
        self.timestamp: float = self.last_accessed_at
        self.value = data

    def is_expired(self, current_time: float | None = None) -> bool:
        """
        Check if the cache entry has expired based on its creation time and TTL.

        Args:
            current_time: Current time to check against. If None, uses time.time().

        Returns:
            True if the entry has expired, False otherwise.
        """
        if current_time is None:
            current_time = time.time()
        return current_time > self.created_at + self.ttl

    def refresh_access(self) -> None:
        """
        Refresh the last access timestamp for the cache entry.

        This method is called when the entry is accessed to update
        the last access time for LRU eviction calculations. The ``timestamp``
        attribute is updated to reflect the latest access time for backwards
        compatibility.
        """
        now = time.time()
        self.last_accessed_at = now
        self.timestamp = now


class CacheMemoryManager:
    """
    Manages memory bounds and eviction strategies for the async cache.

    This class provides centralized memory management functionality including:
    - Memory usage estimation and bounds checking
    - LRU-based eviction strategies
    - Memory-aware cache sizing decisions

    The memory manager works with sharded cache data structures to provide
    efficient memory management across multiple cache shards.
    """

    def __init__(
        self,
        max_entries: int = 100000,
        max_memory_mb: int = 100,
        paranoia_mode: bool = False,
        enable_weak_refs: bool = False,
    ):
        """
        Initialize the cache memory manager.

        Args:
            max_entries: Maximum number of entries allowed in cache
            max_memory_mb: Maximum memory usage in MB
            paranoia_mode: Enable paranoid operational mode with lower bounds
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.paranoia_mode = paranoia_mode
        if self.paranoia_mode:
            self.max_entries = min(self.max_entries, 10000)
            self.max_memory_mb = min(self.max_memory_mb, 10)

    def check_memory_bounds(
        self, shards: list[dict[str, CacheEntry]], current_size: int
    ) -> bool:
        """
        Check if cache size and memory usage are within safe bounds.

        Args:
            shards: List of cache shards to analyze
            current_size: Current number of entries in cache

        Returns:
            True if within bounds, False if too large or memory usage exceeded
        """
        if not self._check_item_count_bounds(current_size):
            return False
        return self._check_memory_usage_bounds(shards, current_size)

    def _check_item_count_bounds(self, current_size: int) -> bool:
        """Check if item count is within safe bounds."""
        max_safe_size = self.max_entries
        if current_size > max_safe_size:
            correlation_id = runtime_metrics.create_correlation_id({
                "component": "cache_memory_manager",
                "operation": "bounds_check",
                "current_size": current_size,
                "max_safe_size": max_safe_size,
            })
            logger.warning(
                "Cache size %s exceeds safe bounds %s",
                current_size, max_safe_size,
            )
            return False
        return True

    def _check_memory_usage_bounds(
        self, shards: list[dict[str, CacheEntry]], current_size: int
    ) -> bool:
        """
        Check if memory usage is within safe bounds using sampling.

        Args:
            shards: List of cache shards to sample from
            current_size: Current number of entries in cache

        Returns:
            True if within bounds, False if memory usage exceeded
        """
        try:
            estimated_memory_mb: float = 0.0
            sample_size = min(100, current_size)
            sample_count = 0
            sample_memory = 0
            for shard in shards:
                for key, entry in shard.items():
                    if sample_count >= sample_size:
                        break
                    sample_memory += sys.getsizeof(key)
                    sample_memory += sys.getsizeof(entry.data)
                    sample_memory += sys.getsizeof(entry.created_at)
                    sample_memory += sys.getsizeof(entry.last_accessed_at)
                    sample_memory += sys.getsizeof(entry.ttl)
                    sample_count += 1
                if sample_count >= sample_size:
                    break
            if sample_count > 0:
                avg_memory_per_item: float = sample_memory / sample_count
                estimated_memory_mb = avg_memory_per_item * current_size / (1024 * 1024)
            else:
                estimated_memory_mb = float(current_size) * 0.5
            memory_threshold = float(self.max_memory_mb) * 0.8
            if estimated_memory_mb > memory_threshold:
                correlation_id = runtime_metrics.create_correlation_id({
                    "component": "cache_memory_manager",
                    "operation": "memory_bounds_approaching",
                    "estimated_mb": estimated_memory_mb,
                    "current_size": current_size,
                    "sample_count": sample_count,
                    "avg_memory_per_item": avg_memory_per_item if sample_count > 0 else 0.0,
                    "max_memory_mb": self.max_memory_mb,
                    "threshold_reached": "80%",
                })
                logger.warning(
                    "Cache memory usage %.1fMB approaching limit of %sMB",
                    estimated_memory_mb, self.max_memory_mb,
                )
                if estimated_memory_mb > self.max_memory_mb:
                    correlation_id = runtime_metrics.create_correlation_id({
                        "component": "cache_memory_manager",
                        "operation": "memory_bounds_exceeded",
                        "estimated_mb": estimated_memory_mb,
                        "current_size": current_size,
                        "sample_count": sample_count,
                        "avg_memory_per_item": avg_memory_per_item if sample_count > 0 else 0,
                        "max_memory_mb": self.max_memory_mb,
                    })
                    logger.warning(
                        "Estimated cache memory usage %.1fMB exceeds %sMB limit",
                        estimated_memory_mb, self.max_memory_mb,
                    )
                    return False
        except Exception as e:
            correlation_id = runtime_metrics.create_correlation_id({
                "component": "cache_memory_manager",
                "operation": "memory_bounds_check_error",
                "error": str(e),
            })
            logger.warning(
                "Failed to estimate memory usage: %s, proceeding with basic size check",
                str(e),
            )
            max_safe_size = self.max_entries
            if current_size > max_safe_size:
                return False
        return True

    async def evict_to_fit(
        self,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[Any],
        required_bytes: int,
        exclude_key: str | None = None,
    ) -> int:
        """
        Evict entries asynchronously to make room for new data requiring the specified number of bytes.

        Evictions will continue until either the estimated number of bytes freed
        meets or exceeds ``required_bytes`` **or** the memory bounds check
        reports that the cache is within safe limits. A maximum of ``2 *
        len(shards)`` evictions is performed to prevent infinite loops.

        Args:
            shards: List of cache shards
            shard_locks: List of locks for each shard
            required_bytes: Number of bytes required for the incoming entry
            exclude_key: Key to exclude from eviction (the key just inserted)

        Returns:
            Approximate number of bytes freed via evictions.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "cache_memory_manager",
                "operation": "evict_to_fit",
                "required_bytes": required_bytes,
                "exclude_key": exclude_key,
            }
        )
        try:
            bytes_freed = 0
            eviction_count = 0
            max_evictions = len(shards) * 2
            while (
                bytes_freed < required_bytes
                or not self._check_memory_usage_bounds(
                    shards, sum((len(s) for s in shards))
                )
            ) and eviction_count < max_evictions:
                lru_key = None
                lru_shard_idx: int | None = None
                if exclude_key:
                    for i, shard in enumerate(shards):
                        if exclude_key in shard:
                            lru_key = self._get_lru_key(shard, exclude_key)
                            lru_shard_idx = i
                            break
                if lru_key is None:
                    for i, shard in enumerate(shards):
                        candidate_key = self._get_lru_key(shard, exclude_key)
                        if candidate_key:
                            lru_key = candidate_key
                            lru_shard_idx = i
                            break
                if not lru_key or lru_shard_idx is None:
                    break
                shard = shards[lru_shard_idx]
                lock = shard_locks[lru_shard_idx]

                async def do_eviction(
                    *, lock=lock, lru_key=lru_key, shard=shard
                ) -> bool:
                    nonlocal bytes_freed
                    async with lock:
                        if lru_key in shard:
                            entry = shard[lru_key]
                            bytes_freed += sys.getsizeof(lru_key)
                            bytes_freed += sys.getsizeof(entry.data)
                            bytes_freed += sys.getsizeof(entry.created_at)
                            bytes_freed += sys.getsizeof(entry.last_accessed_at)
                            bytes_freed += sys.getsizeof(entry.ttl)
                            del shard[lru_key]
                            runtime_metrics.cache_evictions.increment()
                            return True
                    return False

                evicted = await do_eviction()
                if evicted:
                    eviction_count += 1
                    log_with_correlation(
                        logging.DEBUG,
                        f"LRU eviction freed key: {lru_key}",
                        correlation_id,
                    )
                else:
                    break
            log_with_correlation(
                logging.DEBUG,
                f"Eviction completed: freed {bytes_freed} bytes via {eviction_count} evictions",
                correlation_id,
            )
            return bytes_freed
        except Exception as e:
            logger.error("Error during eviction: %s", str(e), exc_info=True)
            log_with_correlation(logging.ERROR, f"Error during eviction: {e}", correlation_id)
            return 0
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _get_lru_key(
        self, shard: dict[str, CacheEntry], exclude_key: str | None = None
    ) -> str | None:
        """
        Get the least recently used key in a shard, excluding specified key.

        Args:
            shard: Cache shard to search
            exclude_key: Key to exclude from consideration

        Returns:
            LRU key or None if shard is empty or only contains exclude_key
        """
        if not shard:
            return None
        lru_key = None
        lru_timestamp = float("inf")
        for key, entry in shard.items():
            if exclude_key and key == exclude_key:
                continue
            if entry.timestamp < lru_timestamp:
                lru_timestamp = entry.timestamp
                lru_key = key
        return lru_key

    def estimate_cache_memory_usage(self, shards: list[dict[str, CacheEntry]]) -> float:
        """
        Estimate current memory usage of cache in MB.

        Args:
            shards: List of cache shards to analyze

        Returns:
            Estimated memory usage in MB
        """
        try:
            total_memory = 0
            total_entries = 0
            for shard in shards:
                for key, entry in shard.items():
                    total_memory += sys.getsizeof(key)
                    total_memory += sys.getsizeof(entry.data)
                    total_memory += sys.getsizeof(entry.created_at)
                    total_memory += sys.getsizeof(entry.last_accessed_at)
                    total_memory += sys.getsizeof(entry.ttl)
                    total_entries += 1
            if total_entries == 0:
                return 0.0
            return total_memory / (1024 * 1024)
        except Exception as e:
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("Failed to estimate cache memory usage: %s", e)
            return 0.0

    def get_memory_info(self, shards: list[dict[str, CacheEntry]]) -> dict[str, Any]:
        """
        Get comprehensive memory information for the cache.

        Args:
            shards: List of cache shards to analyze

        Returns:
            Dictionary with memory usage information
        """
        current_size = sum((len(shard) for shard in shards))
        estimated_memory_mb = self.estimate_cache_memory_usage(shards)
        return {
            "current_size": current_size,
            "estimated_memory_mb": estimated_memory_mb,
            "max_entries": self.max_entries,
            "max_memory_mb": self.max_memory_mb,
            "paranoia_mode": self.paranoia_mode,
            "within_bounds": self.check_memory_bounds(shards, current_size),
            "memory_utilization_percent": estimated_memory_mb / self.max_memory_mb * 100
            if self.max_memory_mb > 0
            else 0,
        }


try:
    from resync.core.metrics import log_with_correlation
except Exception as e:
    logger.error("Failed to import log_with_correlation: %s", str(e), exc_info=True)

    def log_with_correlation(
        level: int, message: str, correlation_id: str | None = None, **kwargs: Any
    ) -> None:
        """Log a message at the given level without correlation context."""
        logger.log(level, message, **kwargs)
