"""
Cache Metrics Mixin.

Provides metrics and monitoring functionality for cache implementations.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CacheMetricsMixin:
    """
    Mixin providing metrics collection for cache.

    Requires base class to have:
    - self.shards: List of cache shards
    - self._hits: int
    - self._misses: int
    """

    _hits: int = 0
    _misses: int = 0
    _evictions: int = 0
    _errors: int = 0
    _start_time: float = 0

    def _init_metrics(self):
        """Initialize metrics tracking."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._errors = 0
        self._start_time = time.time()

    def _record_hit(self):
        """Record a cache hit."""
        self._hits += 1

    def _record_miss(self):
        """Record a cache miss."""
        self._misses += 1

    def _record_eviction(self, count: int = 1):
        """Record cache evictions."""
        self._evictions += count

    def _record_error(self):
        """Record an error."""
        self._errors += 1

    def get_detailed_metrics(self) -> dict[str, Any]:
        """
        Get detailed cache metrics.

        Returns:
            Dict containing cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(total_requests, 1)

        # Count entries across shards
        total_entries = 0
        total_memory_estimate = 0

        if hasattr(self, "shards"):
            for shard in self.shards:
                total_entries += len(shard)
                # Rough memory estimate
                for entry in shard.values():
                    if hasattr(entry, "data"):
                        total_memory_estimate += len(str(entry.data))

        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "total_entries": total_entries,
            "total_requests": total_requests,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "errors": self._errors,
            "memory_estimate_bytes": total_memory_estimate,
            "uptime_seconds": round(uptime, 2),
            "requests_per_second": round(total_requests / max(uptime, 1), 2),
        }

    def reset_metrics(self):
        """Reset all metrics counters."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._errors = 0
        self._start_time = time.time()
