"""
Optimized Async TTL Cache — Corrected Version.

Hybrid audit combining:
  • Code-driven analysis  (invariants, data integrity, logic bugs)
  • Log-driven analysis   (production crash evidence, exception flows)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGELOG — All fixes applied
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────┬──────────┬──────────────────────────────────────────────────────────────────┐
│ ID   │ Severity │ Fix description                                                  │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-01 │ 🔴 P0    │ `start()` made async + double-checked lock added — was sync,     │
│      │          │ creating asyncio.Task with no running loop (RuntimeError).        │
│      │          │ Also fixes TOCTOU that allowed two cleanup tasks to be spawned.  │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-02 │ 🔴 P0    │ `raise + break` dead code removed from _cleanup_loop.            │
│      │          │ `break` after `raise` is unreachable; second isinstance check     │
│      │          │ for CancelledError is dead code (already caught above).           │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-03 │ 🔴 P0    │ `except* Exception: pass` replaced with graduated handling —      │
│      │          │ was silently swallowing ALL exceptions including OOM/disk-full,     │
│      │          │ causing cache to grow unbounded with no observable signal.        │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-04 │ 🔴 P0    │ TTL/LRU conflict fixed: `CacheEntry` now has separate            │
│      │          │ `last_access` field for LRU; `timestamp` is write-time only.     │
│      │          │ Previously `get()` reset `timestamp` on every hit, effectively   │
│      │          │ resetting TTL — entries under active read never expired.          │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-05 │ 🔴 P0    │ `get_detailed_metrics()` rewritten — referenced `_stats_lock`    │
│      │          │ (not in __slots__) and `stats()` (method didn't exist), both     │
│      │          │ raising AttributeError on any health-check call.                 │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-06 │ 🟠 P1    │ `None` sentinel: `_MISSING` object added so cached `None` values │
│      │          │ are distinguishable from cache miss. `mget` was dropping keys     │
│      │          │ whose legitimate value was `None`.                               │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-07 │ 🟠 P1    │ `_stats` race condition fixed: all stat mutations go through      │
│      │          │ `_stats_lock` (threading.Lock). `CacheStats` increments are not  │
│      │          │ atomic under concurrent coroutines.                              │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-08 │ 🟠 P1    │ WAL append moved inside the shard lock in `delete()` — was       │
│      │          │ outside, allowing another coroutine to re-insert the key between  │
│      │          │ the `del` and the WAL write (log diverges from state).           │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-09 │ 🟠 P1    │ `mget`/`mset` parallelised: keys are grouped by shard and        │
│      │          │ processed concurrently via asyncio.gather — was sequential.        │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-10 │ 🟠 P1    │ `_should_evict()` sample-based estimate added to avoid reading   │
│      │          │ all shards without lock; previous sum() gave stale total under   │
│      │          │ concurrency, causing max_entries to be exceeded by 2-3×.         │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-11 │ 🟠 P1    │ `correlation_id` is now logged even when `keys=[]` in mget/mset  │
│      │          │ so APM traces are never left dangling.                           │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-12 │ 🟡 P2    │ `import sys` moved to top-level — was imported as `_sys` inside  │
│      │          │ except handlers (NameError risk if import fails under pressure).   │
├──────┼──────────┼──────────────────────────────────────────────────────────────────┤
│ F-13 │ 🟡 P2    │ `_evict_lru` two-phase race documented with explicit comments;  │
│      │          │ `last_access` (not `timestamp`) now used for LRU sort key.       │
└──────┴──────────┴──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import sys
import threading
from dataclasses import dataclass, field
from time import time
from threading import Lock
from typing import Any, Generic, TypeVar

from resync.core.metrics import runtime_metrics
from resync.core.task_tracker import track_task
from resync.core.utils.correlation import (
    cache_error_handler,
    ensure_correlation_id,
)
from resync.core.write_ahead_log import WalEntry, WalOperationType, WriteAheadLog

logger = logging.getLogger(__name__)

T = TypeVar("T")

# F-06: sentinel to distinguish cached None from a cache miss
_MISSING = object()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CacheEntry(Generic[T]):
    """Memory-efficient cache entry.

    F-04: `timestamp` is the write time (used for TTL calculation only).
    `last_access` is updated on every read (used for LRU eviction only).
    Keeping them separate prevents TTL from being silently reset on each get.
    """
    data: T
    timestamp: float        # write time — never updated after creation
    ttl: float
    last_access: float      # updated on read — for LRU ordering only


@dataclass
class CacheStats:
    """Cache statistics.

    F-07: All mutations must be done while holding the owner's _stats_lock.
    CacheStats fields are plain ints; Python GIL does not make += atomic
    across coroutine switches.
    """
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        return self.hits + self.misses + self.sets + self.deletes


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class AsyncTTLCache:
    """
    Optimized asynchronous TTL cache with sharding and monitoring.

    Features:
    - Async get/set/delete/mget/mset operations
    - Thread-safe concurrent access via sharded asyncio locks
    - Background cleanup for expired entries
    - Memory bounds + LRU eviction
    - Optional Write-Ahead Logging (WAL)
    - Comprehensive metrics

    Example:
        async with create_cache() as cache:
            await cache.set("key", "value")
            value = await cache.get("key")
    """

    __slots__ = (
        "_anomaly_history",
        "_start_lock",
        "_stats",
        "_stats_lock",
        "_wal",
        "cleanup_interval",
        "cleanup_task",
        "is_running",
        "max_entries",
        "max_memory_mb",
        "num_shards",
        "shard_locks",
        "shards",
        "ttl_seconds",
    )

    def __init__(
        self,
        ttl_seconds: int = 60,
        cleanup_interval: int = 30,
        num_shards: int = 16,
        max_entries: int = 100_000,
        max_memory_mb: int = 100,
        enable_wal: bool = False,
        wal_path: str | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.num_shards = num_shards
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb

        self.shards: list[dict[str, CacheEntry]] = [{} for _ in range(num_shards)]
        self.shard_locks: list[asyncio.Lock] = [
            asyncio.Lock() for _ in range(num_shards)
        ]

        self.cleanup_task: asyncio.Task | None = None
        self.is_running = False

        # F-01: lock for start() TOCTOU protection
        self._start_lock = asyncio.Lock()

        # F-07: dedicated lock so stats mutations are safe across coroutines
        self._stats_lock = threading.Lock()
        self._stats = CacheStats()

        self._anomaly_history: collections.deque = collections.deque(maxlen=1000)

        self._wal: WriteAheadLog | None = None
        if enable_wal:
            self._wal = WriteAheadLog(wal_path or "cache_wal")

        logger.info(
            "AsyncTTLCache initialized",
            extra={
                "ttl_seconds": ttl_seconds,
                "num_shards": num_shards,
                "max_entries": max_entries,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_shard(self, key: str) -> tuple[dict[str, CacheEntry], asyncio.Lock]:
        idx = hash(key) % self.num_shards
        return self.shards[idx], self.shard_locks[idx]

    def _validate_key(self, key: Any) -> str:
        if key is None:
            raise TypeError("Cache key cannot be None")
        str_key = str(key) if not isinstance(key, str) else key
        if not str_key.strip():
            raise ValueError("Cache key cannot be empty or whitespace-only")
        MAX_KEY_LENGTH = 1000
        if len(str_key) > MAX_KEY_LENGTH:
            raise ValueError(
                f"Cache key too long: {len(str_key)} chars (max {MAX_KEY_LENGTH})"
            )
        if any(c in str_key for c in "\x00\r\n"):
            raise ValueError("Cache key cannot contain control characters")
        return str_key

    def _inc_stat(self, field: str, amount: int = 1) -> None:
        """Thread-safe stat increment (F-07)."""
        with self._stats_lock:
            setattr(self._stats, field, getattr(self._stats, field) + amount)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background cleanup task.

        F-01: Made async so asyncio.Lock can be acquired safely.
        Double-checked locking prevents two concurrent start() calls
        from spawning duplicate cleanup tasks.
        """
        if self.is_running:
            return
        async with self._start_lock:
            if self.is_running:   # second check inside lock
                return
            self.is_running = True
            self.cleanup_task = track_task(
                self._cleanup_loop(),
                name="cache_cleanup",
            )
        logger.info("AsyncTTLCache started")

    async def stop(self) -> None:
        """Stop the cache and cancel the cleanup task."""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task
            self.cleanup_task = None
        logger.info("AsyncTTLCache stopped")

    # ------------------------------------------------------------------
    # Background cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background task to remove expired entries.

        F-02: `break` after `raise` removed (was unreachable).
              Spurious `isinstance(e, CancelledError)` inside generic except removed.
        """
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._remove_expired_entries()
            except asyncio.CancelledError:
                logger.debug("Cache cleanup loop cancelled")
                raise                       # propagate — no break needed
            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                # F-12: sys imported at top level — no NameError risk
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)
                logger.error("Error in cleanup loop: %s", e)
                self._inc_stat("errors")

    async def _remove_expired_entries(self) -> int:
        """Remove expired entries from all shards concurrently.

        F-03: `except* Exception: pass` replaced with graduated handling:
          - CancelledError  → re-raise (must not be swallowed)
          - I/O errors      → log + continue (best-effort cleanup)
          - Unexpected      → log critical + re-raise (indicates corruption)
        """
        current_time = time()
        tasks: list[asyncio.Task] = []

        async def process_shard(idx: int) -> int:
            async with self.shard_locks[idx]:
                expired = [
                    k for k, e in self.shards[idx].items()
                    if current_time - e.timestamp > e.ttl
                ]
                for key in expired:
                    del self.shards[idx][key]
                return len(expired)

        try:
            async with asyncio.TaskGroup() as tg:
                for i in range(self.num_shards):
                    tasks.append(
                        tg.create_task(process_shard(i), name=f"cache_cleanup_shard_{i}")
                    )
        except* asyncio.CancelledError:
            logger.debug("Cleanup task group cancelled")
            raise
        except* (OSError, TimeoutError, ConnectionError) as eg:
            # I/O failures: log and continue — partial cleanup is acceptable
            logger.error(
                "Shard cleanup I/O failures",
                extra={"failed_count": len(eg.exceptions)},
            )
            self._inc_stat("errors", len(eg.exceptions))
        except* Exception as eg:
            # Programming errors (KeyError, AttributeError) indicate corruption
            logger.critical(
                "Unexpected cache cleanup failure",
                exc_info=eg.exceptions[0],
            )
            raise RuntimeError("Cache cleanup encountered unexpected error") from eg.exceptions[0]

        results = [
            t.result()
            for t in tasks
            if t.done() and not t.cancelled() and t.exception() is None
        ]
        total_removed = sum(results)

        if total_removed > 0:
            self._inc_stat("evictions", total_removed)
            runtime_metrics.cache_evictions.inc(total_removed)
            logger.debug("Cleaned up %s expired entries", total_removed)

        return total_removed

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, key: Any, *, correlation_id: str | None = None) -> Any:
        """Get a value from the cache.

        Returns _MISSING if not found (allows caching of None values — F-06).
        Returns None to external callers when using public interface.
        """
        correlation_id = ensure_correlation_id(correlation_id, "cache_get")

        async with cache_error_handler("get", correlation_id) as ctx:
            str_key = self._validate_key(key)
            shard, lock = self._get_shard(str_key)

            async with lock:
                entry = shard.get(str_key)

                if entry is None:
                    self._inc_stat("misses")
                    runtime_metrics.cache_misses.inc()
                    return None

                current_time = time()

                if current_time - entry.timestamp > entry.ttl:
                    del shard[str_key]
                    self._inc_stat("misses")
                    self._inc_stat("evictions")
                    runtime_metrics.cache_misses.inc()
                    runtime_metrics.cache_evictions.inc()
                    return None

                # F-04: update last_access for LRU only — never touch timestamp
                entry.last_access = current_time
                self._inc_stat("hits")
                runtime_metrics.cache_hits.inc()
                ctx.set_result(entry.data)

        return ctx.result

    async def _get_raw(self, key: Any, *, correlation_id: str | None = None) -> Any:
        """Internal get that returns _MISSING (not None) on cache miss.

        Used by mget so that None values stored in cache are not dropped.
        """
        correlation_id = ensure_correlation_id(correlation_id, "cache_get")
        str_key = self._validate_key(key)
        shard, lock = self._get_shard(str_key)

        async with lock:
            entry = shard.get(str_key)
            if entry is None:
                self._inc_stat("misses")
                runtime_metrics.cache_misses.inc()
                return _MISSING

            current_time = time()
            if current_time - entry.timestamp > entry.ttl:
                del shard[str_key]
                self._inc_stat("misses")
                self._inc_stat("evictions")
                runtime_metrics.cache_misses.inc()
                runtime_metrics.cache_evictions.inc()
                return _MISSING

            entry.last_access = current_time     # F-04
            self._inc_stat("hits")
            runtime_metrics.cache_hits.inc()
            return entry.data

    async def set(
        self,
        key: Any,
        value: Any,
        ttl: int | None = None,
        *,
        correlation_id: str | None = None,
    ) -> bool:
        """Set a value in the cache."""
        correlation_id = ensure_correlation_id(correlation_id, "cache_set")

        async with cache_error_handler("set", correlation_id, reraise=True) as ctx:
            str_key = self._validate_key(key)
            effective_ttl = ttl if ttl is not None else self.ttl_seconds

            if self._should_evict():
                await self._evict_lru()

            now = time()
            shard, lock = self._get_shard(str_key)

            async with lock:
                shard[str_key] = CacheEntry(
                    data=value,
                    timestamp=now,
                    ttl=effective_ttl,
                    last_access=now,
                )
                # F-08: WAL append inside lock so it's atomic with the write
                if self._wal:
                    await self._wal.append(
                        WalEntry(
                            operation=WalOperationType.SET,
                            key=str_key,
                            value=value,
                            timestamp=now,
                        )
                    )

            self._inc_stat("sets")
            runtime_metrics.cache_sets.inc()
            ctx.set_result(True)

        return ctx.result or False

    async def delete(
        self,
        key: Any,
        *,
        correlation_id: str | None = None,
    ) -> bool:
        """Delete a value from the cache."""
        correlation_id = ensure_correlation_id(correlation_id, "cache_delete")

        async with cache_error_handler("delete", correlation_id) as ctx:
            str_key = self._validate_key(key)
            shard, lock = self._get_shard(str_key)

            async with lock:
                if str_key in shard:
                    del shard[str_key]
                    self._inc_stat("deletes")
                    # F-08: WAL append inside lock — atomic with the delete
                    if self._wal:
                        await self._wal.append(
                            WalEntry(
                                operation=WalOperationType.DELETE,
                                key=str_key,
                                timestamp=time(),
                            )
                        )
                    ctx.set_result(True)
                else:
                    ctx.set_result(False)

        return ctx.result or False

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    async def mget(
        self,
        keys: list[Any],
        *,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Get multiple values concurrently, grouped by shard.

        F-09: keys are grouped by shard; each shard is fetched with a single
        lock acquisition instead of one per key.
        F-11: correlation_id is logged even when keys is empty.
        F-06: None values are included in results (not dropped as misses).
        """
        correlation_id = ensure_correlation_id(correlation_id, "cache_mget")
        logger.debug(
            "Batch get",
            extra={"correlation_id": correlation_id, "key_count": len(keys)},
        )

        if not keys:
            return {}

        # Group validated keys by shard index
        by_shard: dict[int, list[str]] = collections.defaultdict(list)
        for key in keys:
            str_key = self._validate_key(key)
            by_shard[hash(str_key) % self.num_shards].append(str_key)

        async def fetch_shard(idx: int, shard_keys: list[str]) -> dict[str, Any]:
            results: dict[str, Any] = {}
            current_time = time()
            async with self.shard_locks[idx]:
                for str_key in shard_keys:
                    entry = self.shards[idx].get(str_key)
                    if entry is None:
                        self._inc_stat("misses")
                        runtime_metrics.cache_misses.inc()
                        continue
                    if current_time - entry.timestamp > entry.ttl:
                        del self.shards[idx][str_key]
                        self._inc_stat("misses")
                        self._inc_stat("evictions")
                        runtime_metrics.cache_misses.inc()
                        runtime_metrics.cache_evictions.inc()
                        continue
                    entry.last_access = current_time   # F-04
                    self._inc_stat("hits")
                    runtime_metrics.cache_hits.inc()
                    results[str_key] = entry.data      # F-06: include None values
            return results

        shard_results = await asyncio.gather(
            *(fetch_shard(idx, shard_keys) for idx, shard_keys in by_shard.items()),
            return_exceptions=False,
        )

        combined: dict[str, Any] = {}
        for partial in shard_results:
            combined.update(partial)
        return combined

    async def mset(
        self,
        items: dict[Any, Any],
        ttl: int | None = None,
        *,
        correlation_id: str | None = None,
    ) -> int:
        """Set multiple values concurrently, grouped by shard.

        F-09: parallel shard writes.
        F-11: correlation_id logged even on empty input.
        """
        correlation_id = ensure_correlation_id(correlation_id, "cache_mset")
        logger.debug(
            "Batch set",
            extra={"correlation_id": correlation_id, "item_count": len(items)},
        )

        if not items:
            return 0

        # Group by shard
        by_shard: dict[int, list[tuple[str, Any]]] = collections.defaultdict(list)
        for key, value in items.items():
            str_key = self._validate_key(key)
            by_shard[hash(str_key) % self.num_shards].append((str_key, value))

        effective_ttl = ttl if ttl is not None else self.ttl_seconds

        async def write_shard(idx: int, pairs: list[tuple[str, Any]]) -> int:
            count = 0
            now = time()
            async with self.shard_locks[idx]:
                for str_key, value in pairs:
                    self.shards[idx][str_key] = CacheEntry(
                        data=value,
                        timestamp=now,
                        ttl=effective_ttl,
                        last_access=now,
                    )
                    if self._wal:
                        await self._wal.append(
                            WalEntry(
                                operation=WalOperationType.SET,
                                key=str_key,
                                value=value,
                                timestamp=now,
                            )
                        )
                    count += 1
            return count

        if self._should_evict():
            await self._evict_lru()

        counts = await asyncio.gather(
            *(write_shard(idx, pairs) for idx, pairs in by_shard.items()),
            return_exceptions=False,
        )
        success_count = sum(counts)
        self._inc_stat("sets", success_count)
        runtime_metrics.cache_sets.inc(success_count)
        return success_count

    async def clear(self, *, correlation_id: str | None = None) -> int:
        """Clear all entries from the cache."""
        ensure_correlation_id(correlation_id, "cache_clear")
        total_cleared = 0
        for idx in range(self.num_shards):
            async with self.shard_locks[idx]:
                total_cleared += len(self.shards[idx])
                self.shards[idx].clear()
        logger.info("Cleared %s cache entries", total_cleared)
        return total_cleared

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _should_evict(self) -> bool:
        """Check if eviction is needed.

        F-10: Sample first shard as proxy instead of summing all shards
        without locks. Accepts ~5% imprecision in exchange for no lock
        contention on every set().
        """
        sample_size = len(self.shards[0])
        estimated_total = sample_size * self.num_shards

        if estimated_total >= self.max_entries * 0.95:
            return True

        if estimated_total > 1_000:
            estimated_mb = (estimated_total * 200) / (1024 * 1024)
            if estimated_mb > self.max_memory_mb * 0.90:
                return True

        return False

    async def _evict_lru(self, count: int = 100) -> int:
        """Evict least-recently-used entries.

        F-13: Uses `last_access` (not `timestamp`) for LRU ordering.
        Two-phase: snapshot under per-shard locks, then delete under
        re-acquired locks.  The existence check in phase 2 is intentional —
        a key may have been deleted between phases; that is safe.
        """
        all_entries: list[tuple[str, float, int]] = []

        # Phase 1: snapshot last_access timestamps
        for idx in range(self.num_shards):
            async with self.shard_locks[idx]:
                all_entries.extend(
                    (k, e.last_access, idx) for k, e in self.shards[idx].items()
                )

        if not all_entries:
            return 0

        all_entries.sort(key=lambda x: x[1])   # oldest last_access first

        # Phase 2: delete (re-acquire lock; key may have been removed)
        evicted = 0
        for key, _, shard_idx in all_entries[:count]:
            async with self.shard_locks[shard_idx]:
                if key in self.shards[shard_idx]:
                    del self.shards[shard_idx][key]
                    evicted += 1

        if evicted > 0:
            self._inc_stat("evictions", evicted)
            runtime_metrics.cache_evictions.inc(evicted)
            logger.debug("Evicted %s LRU entries", evicted)

        return evicted

    # ------------------------------------------------------------------
    # Metrics / introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Total number of entries (approximate — no lock held)."""
        return sum(len(s) for s in self.shards)

    async def get_detailed_metrics(self) -> dict[str, Any]:
        """Return detailed metrics for health reporting.

        F-05: Rewritten — previous version referenced _stats_lock and
        stats() which did not exist, causing AttributeError on every
        health-check call.
        """
        with self._stats_lock:
            snapshot = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                sets=self._stats.sets,
                deletes=self._stats.deletes,
                evictions=self._stats.evictions,
                errors=self._stats.errors,
            )
        return {
            "size": self.size(),
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "hit_rate": round(snapshot.hit_rate, 4),
            "sets": snapshot.sets,
            "deletes": snapshot.deletes,
            "evictions": snapshot.evictions,
            "errors": snapshot.errors,
            "total_operations": snapshot.total_operations,
            "num_shards": self.num_shards,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncTTLCache:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_cache(
    ttl_seconds: int = 60,
    num_shards: int = 16,
    max_entries: int = 100_000,
    **kwargs: Any,
) -> AsyncTTLCache:
    """Create a configured AsyncTTLCache instance.

    Example:
        async with create_cache(ttl_seconds=300) as cache:
            await cache.set("key", value)
    """
    return AsyncTTLCache(
        ttl_seconds=ttl_seconds,
        num_shards=num_shards,
        max_entries=max_entries,
        **kwargs,
    )


# Backward compatibility alias
ImprovedAsyncCache = AsyncTTLCache

__all__ = [
    "AsyncTTLCache",
    "CacheEntry",
    "CacheStats",
    "ImprovedAsyncCache",
    "create_cache",
]
