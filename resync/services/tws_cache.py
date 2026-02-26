# pylint
"""
TWS API Cache - Near Real-Time Strategy v5.9.3

Implements TTL-differentiated caching for TWS API calls:
- Job status: 10s (near real-time)
- Logs/output: 30s (semi-live)
- Static structure: 1h (rarely changes)
- Graph: 5min (dependency structure)

Features:
- Request coalescing (prevents API overload)
- Transparency via _fetched_at timestamp
- age_seconds calculation for UI feedback

Usage:
    from resync.services.tws_cache import tws_cache, CacheCategory

    @tws_cache(CacheCategory.JOB_STATUS)
    async def get_job_status(job_id: str) -> dict:
        return await tws_client.get_current_plan_job(job_id)
"""

import asyncio
import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any

import structlog

from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)

class CacheCategory(Enum):
    """Cache categories with different TTLs."""

    JOB_STATUS = "job_status"  # 10 seconds
    JOB_LOGS = "job_logs"  # 30 seconds
    STATIC_STRUCTURE = "static"  # 1 hour
    GRAPH = "graph"  # 5 minutes
    DEFAULT = "default"  # 60 seconds

# Default TTLs per category (can be overridden by settings)
DEFAULT_TTLS: dict[CacheCategory, int] = {
    CacheCategory.JOB_STATUS: 10,
    CacheCategory.JOB_LOGS: 30,
    CacheCategory.STATIC_STRUCTURE: 3600,
    CacheCategory.GRAPH: 300,
    CacheCategory.DEFAULT: 60,
}

# Stale-While-Revalidate windows (seconds after expiry to serve stale + refresh)
# If data is expired but within stale window, return stale data
# and refresh in background.
DEFAULT_STALE_WINDOWS: dict[CacheCategory, int] = {
    CacheCategory.JOB_STATUS: 30,  # Serve stale for 30s after 10s TTL expires
    CacheCategory.JOB_LOGS: 60,  # Serve stale for 60s after 30s TTL expires
    CacheCategory.STATIC_STRUCTURE: 7200,  # Serve stale for 2h after 1h TTL
    CacheCategory.GRAPH: 600,  # Serve stale for 10min after 5min TTL
    CacheCategory.DEFAULT: 120,  # Serve stale for 2min after 1min TTL
}

@dataclass
class CacheEntry:
    """Cache entry with metadata and SWR support."""

    value: Any
    fetched_at: datetime
    category: CacheCategory
    ttl: int
    stale_window: int = 0  # How long to serve stale after expiry

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired (past TTL)."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > self.ttl

    @property
    def is_stale(self) -> bool:
        """Check if entry is stale (expired but within stale window)."""
        if not self.is_expired:
            return False
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age <= (self.ttl + self.stale_window)

    @property
    def should_refresh(self) -> bool:
        """Check if entry should trigger background refresh.

        Entry is expired but still stale-servable.
        """
        return self.is_expired and self.is_stale

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds()

@dataclass
class CacheStats:
    """Cache statistics with SWR metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stale_hits: int = 0  # Served stale data (within stale window)
    refreshes: int = 0  # Background refreshes triggered
    refresh_failures: int = 0  # Background refresh failures

    @property
    def hit_rate(self) -> float:
        """Traditional hit rate (fresh hits / total requests)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def effective_hit_rate(self) -> float:
        """Effective hit rate including stale hits (user didn't wait)."""
        total = self.hits + self.stale_hits + self.misses
        effective_hits = self.hits + self.stale_hits
        return effective_hits / total if total > 0 else 0.0

class TWSAPICache:
    """
    In-memory async cache with TTL differentiation.

    Designed for TWS API caching with:
    - Different TTLs per data category
    - _fetched_at injection for transparency
    - Request coalescing via locks
    - Cache statistics
    """

    _instance: "TWSAPICache | None" = None

    def __new__(cls) -> "TWSAPICache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._cache: dict[str, CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Reference counts for per-key locks.
        #
        # We keep a counter of how many tasks are either holding *or waiting for*
        # a given lock. This prevents a subtle race where the lock is deleted
        # while other coroutines are still waiting for it (which can lead to a
        # second lock being created and duplicate upstream fetches).
        self._lock_refcounts: dict[str, int] = {}
        self._stats = CacheStats()
        self._ttls = DEFAULT_TTLS.copy()
        self._stale_windows = DEFAULT_STALE_WINDOWS.copy()
        self._initialized = True

        # Metrics
        from resync.core.metrics_compat import Counter

        self._metric_stale_hits = Counter(
            "tws_cache_stale_hits_total",
            "Total number of stale cache hits (SWR)",
            ["category"],
        )
        self._metric_refresh_failures = Counter(
            "tws_cache_refresh_failures_total",
            "Total number of background refresh failures",
            ["category"],
        )

        logger.info(
            "tws_api_cache_initialized",
            ttls=self._ttls,
            stale_windows=self._stale_windows,
        )

    def configure_ttls(
        self,
        job_status: int | None = None,
        job_logs: int | None = None,
        static_structure: int | None = None,
        graph: int | None = None,
    ) -> None:
        """Configure TTLs from settings."""
        if job_status is not None:
            self._ttls[CacheCategory.JOB_STATUS] = job_status
        if job_logs is not None:
            self._ttls[CacheCategory.JOB_LOGS] = job_logs
        if static_structure is not None:
            self._ttls[CacheCategory.STATIC_STRUCTURE] = static_structure
        if graph is not None:
            self._ttls[CacheCategory.GRAPH] = graph

        logger.info("tws_cache_ttls_configured", ttls=self._ttls)

    def _get_ttl(self, category: CacheCategory) -> int:
        """Get TTL for category."""
        return self._ttls.get(category, self._ttls[CacheCategory.DEFAULT])

    def _make_key(self, prefix: str, *args: Any, **kwargs: Any) -> str:
        """Create cache key from function arguments."""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def get(
        self,
        key: str,
        category: CacheCategory = CacheCategory.DEFAULT,
    ) -> tuple[Any, bool, float, bool] | None:
        """
        Get value from cache with SWR support.

        Returns:
            Tuple of (value, is_cached, age_seconds, is_stale)
            or None if not found/expired beyond stale window
        """
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        # If fresh (not expired), return normally
        if not entry.is_expired:
            self._stats.hits += 1
            return entry.value, True, entry.age_seconds, False

        # If stale (expired but within stale window), return stale
        if entry.is_stale:
            self._stats.stale_hits += 1
            self._metric_stale_hits.labels(category=entry.category.value).inc()
            return entry.value, True, entry.age_seconds, True

        # If expired beyond stale window, remove and return None
        self._stats.misses += 1
        self._stats.evictions += 1
        del self._cache[key]
        return None

    def set(
        self,
        key: str,
        value: Any,
        category: CacheCategory = CacheCategory.DEFAULT,
    ) -> None:
        """Set value in cache with metadata injection and SWR support."""
        # Inject _fetched_at for transparency
        if isinstance(value, dict):
            value = copy.deepcopy(value)
            value["_fetched_at"] = datetime.now(timezone.utc).isoformat()

        ttl = self._get_ttl(category)
        stale_window = self._stale_windows.get(
            category, self._stale_windows[CacheCategory.DEFAULT]
        )

        self._cache[key] = CacheEntry(
            value=value,
            fetched_at=datetime.now(timezone.utc),
            category=category,
            ttl=ttl,
            stale_window=stale_window,
        )

    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable,
        category: CacheCategory = CacheCategory.DEFAULT,
    ) -> tuple[Any, bool, float]:
        """
        Get from cache or fetch and cache with SWR support.

        Stale-While-Revalidate behavior:
        - If fresh: return immediately
        - If stale (expired but within stale window):
          return stale data + trigger background refresh
        - If missing/expired beyond stale: wait for fetch

        Uses locking for request coalescing - if multiple requests come in
        for the same key while a fetch is in progress, they all wait for
        the same result instead of making duplicate API calls.

        Returns:
            Tuple of (value, is_cached, age_seconds)
        """
        # Check cache first (fast path)
        result = self.get(key, category)
        if result is not None:
            value, is_cached, age_seconds, is_stale = result

            # If stale, trigger background refresh (fire-and-forget)
            if is_stale:
                create_tracked_task(
                    self._background_refresh(key, fetch_func, category)
                )

            return value, is_cached, age_seconds

        # Get or create lock for this key.
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
            self._lock_refcounts[key] = 0

        # Track the number of coroutines that are interested in this lock (both
        # waiters and the current holder). This avoids deleting the lock while
        # other tasks are still queued on it.
        self._lock_refcounts[key] += 1
        lock = self._locks[key]

        try:
            async with lock:
                # Double-check after acquiring lock (another request might have fetched)
                result = self.get(key, category)
                if result is not None:
                    value, is_cached, age_seconds, is_stale = result
                    # No background refresh here;
                    # the first request already triggered it.
                    return value, is_cached, age_seconds

                # Fetch fresh data
                value = await fetch_func()
                self.set(key, value, category)
                return value, False, 0.0
        finally:
            # Decrement refcount and cleanup only when no other coroutine is
            # waiting/holding this lock.
            self._lock_refcounts[key] = self._lock_refcounts.get(key, 1) - 1
            if self._lock_refcounts[key] <= 0:
                self._locks.pop(key, None)
                self._lock_refcounts.pop(key, None)

    async def _background_refresh(
        self,
        key: str,
        fetch_func: Callable[[], Any],
        category: CacheCategory,
    ) -> None:
        """
        Background refresh for SWR - runs asynchronously without blocking the caller.

        Uses locking to prevent duplicate refreshes.
        """
        # Create a refresh lock key to prevent duplicate background refreshes
        refresh_lock_key = f"_refresh_{key}"

        if refresh_lock_key not in self._locks:
            self._locks[refresh_lock_key] = asyncio.Lock()
            self._lock_refcounts[refresh_lock_key] = 0

        self._lock_refcounts[refresh_lock_key] += 1
        lock = self._locks[refresh_lock_key]

        try:
            # Try to acquire lock without blocking
            if lock.locked():
                # Another refresh is already in progress, skip
                return

            async with lock:
                try:
                    # Fetch fresh data
                    value = await fetch_func()
                    self.set(key, value, category)
                    self._stats.refreshes += 1

                    logger.debug(
                        "tws_cache.background_refresh_success",
                        key=key[:50],  # Truncate for logging
                        category=category.value,
                    )
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    self._stats.refresh_failures += 1
                    self._metric_refresh_failures.labels(category=category.value).inc()
                    logger.warning(
                        "tws_cache.background_refresh_failed",
                        key=key[:50],
                        category=category.value,
                        error=str(e),
                    )
        finally:
            # Cleanup refresh lock
            self._lock_refcounts[refresh_lock_key] = (
                self._lock_refcounts.get(refresh_lock_key, 1) - 1
            )
            if self._lock_refcounts[refresh_lock_key] <= 0:
                self._locks.pop(refresh_lock_key, None)
                self._lock_refcounts.pop(refresh_lock_key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._locks.clear()
        self._lock_refcounts.clear()
        logger.info("tws_cache_cleared", entries_cleared=count)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics including SWR metrics."""
        # Count entries by category
        category_counts: dict[str, int] = {}
        for entry in self._cache.values():
            cat = entry.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_entries": len(self._cache),
            "entries_by_category": category_counts,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "stale_hits": self._stats.stale_hits,
            "refreshes": self._stats.refreshes,
            "refresh_failures": self._stats.refresh_failures,
            "hit_rate": round(self._stats.hit_rate, 3),
            "effective_hit_rate": round(self._stats.effective_hit_rate, 3),
            "ttls": {k.value: v for k, v in self._ttls.items()},
            "stale_windows": {k.value: v for k, v in self._stale_windows.items()},
        }

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_tws_cache: TWSAPICache | None = None

def get_tws_cache() -> TWSAPICache:
    """Get singleton cache instance."""
    global _tws_cache
    if _tws_cache is None:
        _tws_cache = TWSAPICache()
    return _tws_cache

# =============================================================================
# DECORATOR
# =============================================================================

def tws_cache(
    category: CacheCategory = CacheCategory.DEFAULT,
    key_prefix: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for caching TWS API calls.

    Usage:
        @tws_cache(CacheCategory.JOB_STATUS)
        async def get_job_status(job_id: str) -> dict:
            return await client.get(f"/plan/job/{job_id}")

    The decorated function will:
    - Return cached value if available and not expired
    - Inject _fetched_at timestamp for transparency
    - Use request coalescing for concurrent calls
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        prefix = key_prefix or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_tws_cache()
            key = cache._make_key(prefix, *args, **kwargs)

            async def fetch() -> Any:
                return await func(*args, **kwargs)

            value, is_cached, age = await cache.get_or_fetch(key, fetch, category)
            return value

        return wrapper

    return decorator

# =============================================================================
# RESPONSE WRAPPER
# =============================================================================

def enrich_response_with_cache_meta(
    data: Any,
    is_cached: bool = False,
    age_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Wrap response with cache metadata for API endpoints.

    Usage in FastAPI route:
        @router.get("/job/{job_id}")
        async def get_job(job_id: str):
            data, is_cached, age = await cache.get_or_fetch(...)
            return enrich_response_with_cache_meta(data, is_cached, age)

    Returns:
        {
            "data": <original data>,
            "meta": {
                "cached": true/false,
                "age_seconds": 4.2,
                "fetched_at": "2024-12-16T10:00:00Z"
            }
        }
    """
    fetched_at = None
    if isinstance(data, dict):
        fetched_at = data.get("_fetched_at")

    return {
        "data": data,
        "meta": {
            "cached": is_cached,
            "age_seconds": round(age_seconds, 1),
            "fetched_at": fetched_at,
            "freshness": "live"
            if age_seconds < 2
            else "recent"
            if age_seconds < 10
            else "cached",
        },
    }
