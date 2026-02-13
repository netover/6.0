from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
from dataclasses import dataclass
from time import time
from typing import Any

from resync.core.exceptions import CacheError
from resync.core.metrics import log_with_correlation, runtime_metrics
from resync.core.write_ahead_log import WalEntry, WalOperationType, WriteAheadLog

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single entry in the cache with timestamp and TTL."""

    data: Any
    timestamp: float
    ttl: float


class AsyncTTLCache:
    """
    A truly asynchronous TTL cache that eliminates blocking I/O with comprehensive monitoring.

    Features:
    - Async get() and set() methods for non-blocking operations
    - Thread-safe concurrent access using sharded asyncio.Lock
    - Background cleanup task for expired entries
    - Time-based eviction using asyncio.sleep()
    - Memory bounds checking with intelligent sampling
    - Comprehensive metrics collection and health monitoring
    - Transaction support with rollback capability
    - Snapshot and restore functionality for persistence
    - LRU eviction when cache bounds are exceeded
    - Production-grade error handling and logging

    The cache uses sharding to distribute entries across multiple locked segments,
    reducing contention under high concurrency. Each shard has its own asyncio.Lock
    to ensure thread-safe access while maximizing parallelism.
    """

    def __init__(
        self,
        ttl_seconds: int = 60,
        cleanup_interval: int = 30,
        num_shards: int = 16,
        enable_wal: bool = False,
        wal_path: str | None = None,
        max_entries: int = 100000,
        max_memory_mb: int = 100,
        paranoia_mode: bool = False,
    ):
        """
        Orquestra a inicialização do cache assíncrono.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            cleanup_interval: How often to run background cleanup in seconds
            num_shards: Number of shards for lock
            enable_wal: Whether to enable Write-Ahead Logging for persistence
            wal_path: Path for WAL files (default: cache_wal in current directory)
            max_entries: Maximum number of entries in cache
            max_memory_mb: Maximum memory usage in MB
            paranoia_mode: Enable paranoid operational mode with lower bounds
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "async_cache",
                "operation": "init",
                "ttl_seconds": ttl_seconds,
                "cleanup_interval": cleanup_interval,
                "num_shards": num_shards,
                "enable_wal": enable_wal,
                "max_entries": max_entries,
                "max_memory_mb": max_memory_mb,
                "paranoia_mode": paranoia_mode,
            }
        )

        try:
            # Define num_shards imediatamente (será sobrescrito por _load_configuration se necessário)
            self.num_shards = num_shards

            # Initialize anomaly history with deque for O(1) operations
            self.anomaly_history = collections.deque()  # More efficient
            # Pre-allocate with reasonable max size to avoid frequent reallocations
            self.anomaly_history = collections.deque(maxlen=1000)
            # Initialize cache configuration efficiently
            self.shards: list[dict[str, CacheEntry]] = [{} for _ in range(self.num_shards)]
            # v5.9.5: Fixed bug - was annotation (:) instead of assignment (=)
            self.shard_locks = [asyncio.Lock() for _ in range(self.num_shards)]
            self.cleanup_task: asyncio.Task[None] | None = None
            self.is_running = False
            self._background_cleanup_started = False

        except Exception as e:
            runtime_metrics.record_health_check("async_cache", "init_failed", {"error": str(e)})
            log_with_correlation(
                logging.CRITICAL,
                f"Failed to initialize AsyncTTLCache: {e}",
                correlation_id,
            )
            raise
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _load_configuration(
        self,
        ttl_seconds: int,
        cleanup_interval: int,
        num_shards: int,
        enable_wal: bool,
        wal_path: str | None,
        max_entries: int,
        max_memory_mb: int,
        paranoia_mode: bool,
    ) -> None:
        """
        Carrega configurações do cache com fallback para settings.
        """
        # Create correlation_id for logging
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "load_configuration"}
        )

        try:
            # Try to load from settings module
            from resync.settings import settings

            self.ttl_seconds = (
                ttl_seconds
                if ttl_seconds != 60
                else getattr(settings, "ASYNC_CACHE_TTL", ttl_seconds)
            )
            self.cleanup_interval = (
                cleanup_interval
                if cleanup_interval != 30
                else getattr(settings, "ASYNC_CACHE_CLEANUP_INTERVAL", cleanup_interval)
            )
            self.num_shards = (
                num_shards
                if num_shards != 16
                else getattr(settings, "ASYNC_CACHE_NUM_SHARDS", num_shards)
            )
            self.enable_wal = (
                enable_wal
                if enable_wal
                else getattr(settings, "ASYNC_CACHE_ENABLE_WAL", enable_wal)
            )
            self.wal_path = (
                wal_path
                if wal_path is not None
                else getattr(settings, "ASYNC_CACHE_WAL_PATH", wal_path)
            )
            self.max_entries = (
                max_entries
                if max_entries != 100000
                else getattr(settings, "ASYNC_CACHE_MAX_ENTRIES", max_entries)
            )
            self.max_memory_mb = (
                max_memory_mb
                if max_memory_mb != 100
                else getattr(settings, "ASYNC_CACHE_MAX_MEMORY_MB", max_memory_mb)
            )
            self.paranoia_mode = (
                paranoia_mode
                if paranoia_mode
                else getattr(settings, "ASYNC_CACHE_PARANOIA_MODE", paranoia_mode)
            )

            # Apply paranoia mode restrictions
            self._apply_paranoia_mode_restrictions()

            log_with_correlation(
                logging.DEBUG,
                "Loaded cache config from settings module",
                correlation_id,
            )
        except ImportError:
            # Handle case where settings module is not available
            self.ttl_seconds = ttl_seconds
            self.cleanup_interval = cleanup_interval
            self.num_shards = num_shards
            self.enable_wal = enable_wal
            self.wal_path = wal_path
            self.max_entries = max_entries
            self.max_memory_mb = max_memory_mb
            self.paranoia_mode = paranoia_mode
            log_with_correlation(
                logging.WARNING,
                "Settings module not available, using provided values or defaults",
                correlation_id,
            )
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _apply_paranoia_mode_restrictions(self) -> None:
        """
        Aplica restrições do modo paranoia.
        """
        if self.paranoia_mode:
            self.max_entries = min(self.max_entries, 10000)  # Max 10K entries
            self.max_memory_mb = min(self.max_memory_mb, 10)  # Max 10MB

    def _initialize_cache_structure(self) -> None:
        """
        Inicializa a estrutura do cache (shards e locks).
        """
        self.shards: list[dict[str, CacheEntry]] = [{} for _ in range(self.num_shards)]
        self.shard_locks = [asyncio.Lock() for _ in range(self.num_shards)]
        self.cleanup_task: asyncio.Task[None] | None = None
        self.is_running = False
        self._background_cleanup_started = False

    def _setup_write_ahead_log(self, enable_wal: bool, wal_path: str | None) -> None:
        """
        Configura o Write-Ahead Log se habilitado.
        """
        if enable_wal:
            # Create correlation_id for logging
            correlation_id = runtime_metrics.create_correlation_id(
                {"component": "async_cache", "operation": "setup_wal"}
            )

            try:
                wal_path_to_use = wal_path or "./cache_wal"
                self.wal = WriteAheadLog(wal_path_to_use)
                log_with_correlation(
                    logging.INFO,
                    f"WAL enabled for cache, path: {wal_path_to_use}",
                    correlation_id,
                )
                # Schedule WAL replay if needed
                self._schedule_wal_replay_if_needed()
            finally:
                runtime_metrics.close_correlation_id(correlation_id)

    def _schedule_wal_replay_if_needed(self) -> None:
        """
        Agenda o replay do WAL se necessário.
        """
        if self.enable_wal and self.wal:
            try:
                if asyncio.get_event_loop().is_running():
                    asyncio.create_task(self._replay_wal_on_startup())
                else:
                    self._needs_wal_replay_on_first_use = True
            except RuntimeError:
                self._needs_wal_replay_on_first_use = True

    def _setup_background_services(self) -> None:
        """
        Configura serviços background.
        """
        # Note: Cleanup task will be started on first use to avoid blocking during import

    def _validate_initialization(self, correlation_id: str) -> None:
        """
        Validação final da inicialização e registro de métricas.
        """
        runtime_metrics.record_health_check(
            "async_cache",
            "initialized",
            {
                "ttl_seconds": self.ttl_seconds,
                "cleanup_interval": self.cleanup_interval,
                "num_shards": self.num_shards,
                "enable_wal": self.enable_wal,
            },
        )
        log_with_correlation(
            logging.INFO,
            "AsyncTTLCache initialized successfully",
            correlation_id,
        )

    async def _replay_wal_on_startup(self) -> int:
        """
        Replay the WAL log on cache startup to restore state.

        Returns:
            Number of operations replayed
        """
        if not self.enable_wal or not self.wal:
            return 0

        # Replay the WAL to rebuild cache state
        return await self.wal.replay_log(self)

    def _get_shard(self, key: str) -> tuple[dict[str, CacheEntry], asyncio.Lock]:
        """Get the shard and lock for a given key with bounds checking."""
        # BOUNDS CHECKING - Prevent hash overflow/underflow
        try:
            key_hash = hash(key)
            # Handle special case for hash collision more deterministically
            if key_hash == 0:
                key_hash = sum(ord(c) for c in str(key)) + len(str(key))

            # Use modulo to ensure consistent distribution
            shard_index = abs(key_hash) % self.num_shards

            # Double-check bounds (defense in depth) with deterministic fallback
            if not (0 <= shard_index < self.num_shards):
                # Use a deterministic fallback based on key content
                # rather than always index 0 to ensure consistency
                shard_index = (len(key) + (ord(key[0]) if key else 0)) % self.num_shards

        except (OverflowError, ValueError) as e:
            # Hash computation failed - use deterministic fallback
            # Use a more robust approach that considers more of the key content
            key_sum = sum(
                ord(c) for c in str(key)[:20]
            )  # Use first 20 chars for better distribution
            shard_index = key_sum % self.num_shards
            logger.warning(
                f"Hash computation failed for key {repr(key)}: {e}, using fallback shard {shard_index}"
            )

        return self.shards[shard_index], self.shard_locks[shard_index]

    def _get_lru_key(self, shard: dict[str, CacheEntry]) -> str:
        """
        Get the least recently used key in a shard.
        This is used for LRU eviction when cache bounds are exceeded.
        """
        if not shard:
            return None

        # Find the entry with the oldest timestamp
        return min(shard.keys(), key=lambda k: shard[k].timestamp)

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if not self.is_running:
            try:
                # Only start the task if there's a running event loop
                asyncio.get_running_loop()
                self.is_running = True
                self.cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
            except RuntimeError as exc:
                # No event loop running, skip starting the task
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

    async def _cleanup_expired_entries(self) -> None:
        """Background task to cleanup expired entries."""
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "cleanup_task"}
        )

        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._remove_expired_entries()
                runtime_metrics.cache_cleanup_cycles.increment()
                runtime_metrics.cache_size.set(self.size())

            except asyncio.CancelledError:
                log_with_correlation(
                    logging.DEBUG,
                    "AsyncTTLCache cleanup task cancelled",
                    correlation_id,
                )
                break
            except MemoryError as e:
                log_with_correlation(
                    logging.ERROR,
                    f"Memory error in AsyncTTLCache cleanup task: {e}",
                    correlation_id,
                )
                runtime_metrics.record_health_check(
                    "async_cache", "memory_error", {"error": str(e)}
                )
                raise CacheError("Memory error during cache cleanup") from e
            except KeyError as e:
                log_with_correlation(
                    logging.ERROR,
                    f"Key error in AsyncTTLCache cleanup task: {e}",
                    correlation_id,
                )
                runtime_metrics.record_health_check("async_cache", "key_error", {"error": str(e)})
                raise CacheError(f"Key error during cache cleanup: {e}") from e
            except RuntimeError as e:  # pragma: no cover
                log_with_correlation(
                    logging.ERROR,
                    f"Runtime error in AsyncTTLCache cleanup task: {e}",
                    correlation_id,
                )
                runtime_metrics.record_health_check(
                    "async_cache", "runtime_error", {"error": str(e)}
                )
                raise CacheError(f"Runtime error during cache cleanup: {e}") from e
            except Exception as e:  # pragma: no cover
                log_with_correlation(
                    logging.CRITICAL,
                    f"Unexpected error in AsyncTTLCache cleanup task: {e}",
                    correlation_id,
                    exc_info=True,
                )
                runtime_metrics.record_health_check(
                    "async_cache", "critical_error", {"error": str(e)}
                )
                # Depending on the desired behavior, we might want to stop the loop
                # or just log and continue. For now, we re-raise to make the failure visible.
                raise CacheError("Unexpected critical error during cache cleanup") from e

        runtime_metrics.close_correlation_id(correlation_id)

    async def _remove_expired_entries(self) -> None:
        """Remove expired entries from cache using parallel processing.

        This method efficiently removes all expired cache entries across all shards
        by processing each shard concurrently. It maintains thread safety by
        acquiring each shard's lock before modifying its contents.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "remove_expired"}
        )

        current_time = time()
        total_removed = 0

        # Function to process a single shard
        async def process_shard(i):
            """Process a single shard to remove expired entries.

            Args:
                i: Index of the shard to process

            Returns:
                Number of entries removed from this shard
            """
            shard = self.shards[i]
            lock = self.shard_locks[i]
            async with lock:
                expired_keys = [
                    key
                    for key, entry in shard.items()
                    if current_time - entry.timestamp > entry.ttl
                ]
                for key in expired_keys:
                    del shard[key]
                    log_with_correlation(
                        logging.DEBUG,
                        f"Removed expired cache entry: {key}",
                        correlation_id,
                    )
                return len(expired_keys)

        # Process all shards concurrently
        shard_indices = list(range(self.num_shards))
        tasks = [process_shard(i) for i in shard_indices]
        results = await asyncio.gather(*tasks)

        total_removed = sum(results)

        if total_removed > 0:
            runtime_metrics.cache_evictions.increment(total_removed)
            log_with_correlation(
                logging.DEBUG,
                f"Cleaned up {total_removed} expired cache entries",
                correlation_id,
            )

        runtime_metrics.close_correlation_id(correlation_id)

    async def get(self, key: Any) -> Any | None:
        """
        Asynchronously retrieve an item from the cache with input validation.

        This method performs comprehensive validation of the input key, checks
        for cache entry expiration, and appropriately updates cache metrics
        including hit/miss rates and performance indicators.

        Args:
            key: Cache key to retrieve (will be validated/normalized)

        Returns:
            Cached value if exists and not expired, None otherwise

        Raises:
            ValueError: If key validation fails
            TypeError: If key is invalid
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "get", "key": repr(key)}
        )

        # Ensure cleanup task is running
        self._start_cleanup_task()

        # Perform WAL replay if needed on first use
        if hasattr(self, "_needs_wal_replay_on_first_use") and self._needs_wal_replay_on_first_use:
            # Remove the flag to prevent replay on every operation
            self._needs_wal_replay_on_first_use = False
            # Perform the WAL replay
            replayed_ops = await self._replay_wal_on_startup()
            logger.info("replayed_operations_from_WAL_on_first_use", replayed_ops=replayed_ops)

        try:
            # Validate and normalize key
            if key is None:
                raise TypeError("Cache key cannot be None")

            # Apply same validation as set operation
            if not isinstance(key, (str, int, float, bool)):
                try:
                    str_key = str(key)
                    if len(str_key) > 1000:
                        raise ValueError(
                            f"Cache key too long: {len(str_key)} characters (max 1000)"
                        )
                    if "\x00" in str_key:
                        raise ValueError("Cache key cannot contain null bytes")
                    key = str_key
                except Exception as _e:
                    raise TypeError(f"Cache key must be convertible to string: {type(key)}") from _e
            else:
                key = str(key)

            # Additional validations
            if len(key) == 0:
                raise ValueError("Cache key cannot be empty")
            if len(key) > 1000:
                raise ValueError(f"Cache key too long: {len(key)} characters (max 1000)")
            if "\x00" in key or "\r" in key or "\n" in key:
                raise ValueError("Cache key cannot contain control characters")

            shard, lock = self._get_shard(key)
            async with lock:
                entry = shard.get(key)
                if entry:
                    current_time = time()
                    if current_time - entry.timestamp <= entry.ttl:
                        runtime_metrics.cache_hits.increment()
                        # Update hit rate dynamically
                        total_requests = (
                            runtime_metrics.cache_hits.value + runtime_metrics.cache_misses.value
                        )
                        if total_requests > 0:
                            hit_rate = runtime_metrics.cache_hits.value / total_requests
                            runtime_metrics.record_health_check(
                                "async_cache",
                                "performance",
                                {
                                    "hit_rate": hit_rate,
                                    "total_requests": total_requests,
                                },
                            )
                        entry.timestamp = current_time  # Update timestamp for LRU
                        log_with_correlation(
                            logging.DEBUG,
                            f"Cache HIT for key: {repr(key)}",
                            correlation_id,
                        )
                        return entry.data
                    # Entry expired, remove it
                    del shard[key]
                    runtime_metrics.cache_evictions.increment()
                    # Update eviction rate
                    total_evictions = runtime_metrics.cache_evictions.value
                    total_sets = runtime_metrics.cache_sets.value
                    if total_sets > 0:
                        eviction_rate = total_evictions / total_sets
                        runtime_metrics.record_health_check(
                            "async_cache",
                            "eviction_rate",
                            {
                                "eviction_rate": eviction_rate,
                                "total_evictions": total_evictions,
                            },
                        )
                    log_with_correlation(
                        logging.DEBUG,
                        f"Cache EXPIRED for key: {repr(key)}",
                        correlation_id,
                    )

                runtime_metrics.cache_misses.increment()
                # Update miss rate
                total_requests = (
                    runtime_metrics.cache_hits.value + runtime_metrics.cache_misses.value
                )
                if total_requests > 0:
                    miss_rate = runtime_metrics.cache_misses.value / total_requests
                    runtime_metrics.record_health_check(
                        "async_cache",
                        "miss_rate",
                        {"miss_rate": miss_rate, "total_requests": total_requests},
                    )
                log_with_correlation(
                    logging.DEBUG, f"Cache MISS for key: {repr(key)}", correlation_id
                )
                return None

        except (ValueError, TypeError) as e:
            log_with_correlation(
                logging.WARNING,
                f"Cache GET validation failed for key {repr(key)}: {e}",
                correlation_id,
            )
            raise
        except Exception as e:
            log_with_correlation(
                logging.ERROR,
                f"Cache GET failed for key {repr(key)}: {e}",
                correlation_id,
            )
            raise
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """
        Asynchronously add an item to the cache with comprehensive input validation.

        Args:
            key: Cache key (must be valid string/hashable)
            value: Value to cache (validated for serializability)
            ttl_seconds: Optional TTL override for this specific entry

        Raises:
            ValueError: If key or value validation fails
            TypeError: If key is not hashable or value is invalid
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "async_cache",
                "operation": "set",
                "key": repr(key),
                "ttl_seconds": ttl_seconds,
            }
        )

        # Ensure cleanup task is running
        self._start_cleanup_task()

        # Perform WAL replay if needed on first use
        if hasattr(self, "_needs_wal_replay_on_first_use") and self._needs_wal_replay_on_first_use:
            # Remove the flag to prevent replay on every operation
            self._needs_wal_replay_on_first_use = False
            # Perform the WAL replay
            replayed_ops = await self._replay_wal_on_startup()
            logger.info("replayed_operations_from_WAL_on_first_use", replayed_ops=replayed_ops)

        try:
            # FUZZING-HARDENED INPUT VALIDATION
            key, ttl_seconds = self._validate_cache_inputs(key, value, ttl_seconds)

            # If WAL is enabled, log the operation before applying it to the cache
            if self.enable_wal and self.wal:
                wal_entry = WalEntry(
                    operation=WalOperationType.SET,
                    key=key,
                    value=value,
                    ttl=ttl_seconds,
                )
                log_success = await self.wal.log_operation(wal_entry)
                if not log_success:
                    logger.error("failed_to_log_SET_operation_to_WAL", key=key)
                    # Optionally, could raise an exception here if WAL logging is critical
                    # For now, we'll continue but log the issue

            current_time = time()
            entry = CacheEntry(data=value, timestamp=current_time, ttl=ttl_seconds)

            shard, lock = self._get_shard(key)
            async with lock:
                # Check bounds before adding - if we're already at the limit, we need to evict BEFORE adding
                # to ensure we never exceed the bounds, but avoid infinite loops

                # Add the entry first to avoid an empty cache scenario
                shard[key] = entry

                # Check bounds after adding - if we're still over the limit, start evicting
                # but limit the number of evictions to avoid infinite loops
                eviction_count = 0
                max_evictions = (
                    self.num_shards * 2
                )  # Prevent infinite loop, try at most 2 per shard

                while not self._check_cache_bounds() and eviction_count < max_evictions:
                    # Cache is too large, trigger LRU eviction
                    lru_key = self._get_lru_key(shard)
                    if lru_key and lru_key != key:  # Don't evict the entry we just added
                        # Remove LRU entry from this shard
                        del shard[lru_key]
                        runtime_metrics.cache_evictions.increment()
                        log_with_correlation(
                            logging.DEBUG,
                            f"LRU eviction removed key: {lru_key}",
                            correlation_id,
                        )
                        eviction_count += 1
                    elif lru_key == key:
                        # If the key we just added is the LRU (happens with size 1 cache),
                        # we need to remove it and raise error
                        del shard[key]
                        raise ValueError(
                            f"Cache bounds exceeded: cannot add key {repr(key)} (cache too small)"
                        )
                    else:
                        # If no LRU key found in current shard, try other shards
                        eviction_found = False
                        for i, other_shard in enumerate(self.shards):
                            if i == self.shards.index(shard):
                                continue  # Skip current shard since we just checked it

                            if not self._check_cache_bounds():
                                other_lock = self.shard_locks[i]
                                async with other_lock:
                                    lru_key = self._get_lru_key(other_shard)
                                    if lru_key:
                                        del other_shard[lru_key]
                                        runtime_metrics.cache_evictions.increment()
                                        log_with_correlation(
                                            logging.DEBUG,
                                            f"LRU eviction removed key from shard {i}: {lru_key}",
                                            correlation_id,
                                        )
                                        eviction_count += 1
                                        eviction_found = True
                                        break  # Only evict one per iteration

                        if not eviction_found and eviction_count == 0 and self.max_entries < 2:
                            # If we couldn't find anything to evict and we're in a tiny cache
                            # For tiny caches, check if we're trying to add a second item
                            if len(shard) > 1 or any(
                                len(s) > 0
                                for j, s in enumerate(self.shards)
                                if j != self.shards.index(shard)
                            ):
                                # Remove the newly added entry since it makes us exceed bounds
                                del shard[key]
                                raise ValueError(
                                    f"Cache bounds exceeded: cannot add key {repr(key)} (cache too small)"
                                )
                            break  # No more items to evict in this pass

                # Final bounds check - reject and remove entry if still over bounds
                # This is crucial to ensure we never exceed the configured limits
                if not self._check_cache_bounds() and key in shard:
                    # Remove the newly added entry if bounds still exceeded
                    del shard[key]
                    raise ValueError(
                        f"Cache bounds exceeded: cannot add key {repr(key)} (cache too large)"
                    )
                runtime_metrics.cache_sets.increment()
                runtime_metrics.cache_size.set(self.size())
                log_with_correlation(
                    logging.DEBUG, f"Cache SET for key: {repr(key)}", correlation_id
                )
        except Exception as e:
            log_with_correlation(
                logging.ERROR,
                f"Cache SET failed for key {repr(key)}: {e}",
                correlation_id,
            )
            # Re-raising the exception to make the caller aware of the failure.
            # Silent failures in cache operations are dangerous.
            raise
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _validate_cache_inputs(
        self, key: Any, value: Any, ttl_seconds: float | None
    ) -> tuple[str, float]:
        """
        Comprehensive input validation based on fuzzing failures.

        Args:
            key: Raw key input
            value: Raw value input
            ttl_seconds: Raw TTL input

        Returns:
            tuple: (validated_key, validated_ttl)

        Raises:
            ValueError: For invalid inputs
            TypeError: For incorrect types
        """
        validated_key = self._validate_cache_key(key)
        self._validate_cache_value(value)
        validated_ttl = self._validate_cache_ttl(ttl_seconds)

        return validated_key, float(validated_ttl)

    def _validate_cache_key(self, key: Any) -> str:
        """Validate the cache key."""
        # KEY VALIDATION - HARDENED AGAINST FUZZING
        if key is None:
            raise TypeError("Cache key cannot be None")

        # Convert key to string if possible, but validate it's reasonable
        if not isinstance(key, (str, int, float, bool)):
            # For complex objects, require they have a proper string representation
            try:
                str_key = str(key)
                if len(str_key) > 1000:  # Prevent extremely long keys
                    raise ValueError(f"Cache key too long: {len(str_key)} characters (max 1000)")
                if "\x00" in str_key:  # Prevent null bytes
                    raise ValueError("Cache key cannot contain null bytes")
                key = str_key
            except Exception as _e:
                raise TypeError(f"Cache key must be convertible to string: {type(key)}") from _e
        else:
            key = str(key)

        # Additional key validations
        if len(key) == 0:
            raise ValueError("Cache key cannot be empty")
        if len(key) > 1000:
            raise ValueError(f"Cache key too long: {len(key)} characters (max 1000)")
        if "\x00" in key or "\r" in key or "\n" in key:
            raise ValueError("Cache key cannot contain control characters")

        return key

    def _validate_cache_value(self, value: Any) -> None:
        """Validate the cache value."""
        # VALUE VALIDATION - DEFEND AGAINST MALICIOUS INPUTS
        if value is None:
            raise ValueError("Cache value cannot be None")

        # v5.9.7: Avoid broken "always true" isinstance guard.
        # This cache is primarily in-memory; picklability is not a reliable contract.
        # We only reject clearly dangerous/unintended values.
        import inspect

        if inspect.iscoroutine(value) or inspect.isawaitable(value):
            raise ValueError("Cache value cannot be a coroutine/awaitable")

        # Disallow callables by default (usually indicates a bug)
        if callable(value):
            raise ValueError("Cache value cannot be callable")

    def _validate_cache_ttl(self, ttl_seconds: float | None) -> float:
        """Validate the TTL value."""
        # TTL VALIDATION - PREVENT EDGE CASES
        if ttl_seconds is None:
            return self.ttl_seconds
        if not isinstance(ttl_seconds, (int, float)):
            raise TypeError(f"TTL must be numeric: {type(ttl_seconds)}")
        if ttl_seconds < 0:
            raise ValueError(f"TTL cannot be negative: {ttl_seconds}")
        if ttl_seconds > 86400 * 365:  # Max 1 year
            raise ValueError(f"TTL too large: {ttl_seconds} seconds (max 1 year)")

        return ttl_seconds

    async def delete(self, key: str) -> bool:
        """
        Asynchronously delete an item from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if item was deleted, False if not found
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "delete", "key": key}
        )

        # Perform WAL replay if needed on first use
        if hasattr(self, "_needs_wal_replay_on_first_use") and self._needs_wal_replay_on_first_use:
            # Remove the flag to prevent replay on every operation
            self._needs_wal_replay_on_first_use = False
            # Perform the WAL replay
            replayed_ops = await self._replay_wal_on_startup()
            logger.info("replayed_operations_from_WAL_on_first_use", replayed_ops=replayed_ops)

        try:
            # If WAL is enabled, log the operation before applying it to the cache
            if self.enable_wal and self.wal:
                wal_entry = WalEntry(operation=WalOperationType.DELETE, key=key)
                log_success = await self.wal.log_operation(wal_entry)
                if not log_success:
                    logger.error("failed_to_log_DELETE_operation_to_WAL", key=key)
                    # Optionally, could raise an exception here if WAL logging is critical
                    # For now, we'll continue but log the issue

            shard, lock = self._get_shard(key)
            async with lock:
                if key in shard:
                    del shard[key]
                    runtime_metrics.cache_evictions.increment()
                    runtime_metrics.cache_size.set(self.size())
                    log_with_correlation(
                        logging.DEBUG, f"Cache DELETE for key: {key}", correlation_id
                    )
                    return True
                return False
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def rollback_transaction(self, operations: list[dict[str, Any]]) -> bool:
        """
        Rollback a series of cache operations atomically with comprehensive bounds checking.

        Args:
            operations: List of operations to rollback, each with format:
                       {"operation": "set|delete", "key": str, "value": Any, "ttl": int}

        Returns:
            True if rollback successful, False otherwise
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "async_cache",
                "operation": "rollback",
                "operations_count": len(operations),
            }
        )

        try:
            # BOUNDS CHECKING - Validate operations list
            if not isinstance(operations, list):
                raise TypeError(f"Operations must be a list, got {type(operations)}")

            if len(operations) == 0:
                log_with_correlation(
                    logging.DEBUG, "Empty operations list for rollback", correlation_id
                )
                return True

            # Prevent excessive rollback operations (DoS protection)
            if len(operations) > 10000:
                raise ValueError(f"Too many operations for rollback: {len(operations)} (max 10000)")

            # Validate each operation structure
            for i, op in enumerate(operations):
                if not isinstance(op, dict):
                    raise TypeError(f"Operation at index {i} must be a dict, got {type(op)}")

                if "key" not in op:
                    raise ValueError(f"Operation at index {i} missing required 'key' field")

                if op["operation"] not in ["set", "delete"]:
                    raise ValueError(
                        f"Operation at index {i} has invalid operation: {op.get('operation')}"
                    )

                # Validate key bounds
                key = op["key"]
                if not isinstance(key, (str, int, float, bool)):
                    try:
                        key = str(key)
                        if len(key) > 1000:
                            raise ValueError(
                                f"Key too long in operation {i}: {len(key)} characters"
                            )
                    except Exception as _e:
                        raise ValueError(f"Invalid key in operation {i}: {type(key)}") from _e

            # Group operations by shard to minimize lock contention with bounds checking
            shard_operations: dict[int, list[dict[str, Any]]] = {}
            for op in operations:
                try:
                    # Use the same bounds-checked shard calculation
                    _, lock = self._get_shard(op["key"])
                    shard_idx = self.shard_locks.index(lock)  # Get shard index from lock

                    if shard_idx not in shard_operations:
                        shard_operations[shard_idx] = []
                    shard_operations[shard_idx].append(op)
                except (ValueError, IndexError) as e:
                    log_with_correlation(
                        logging.ERROR,
                        f"Failed to determine shard for key {repr(op['key'])}: {e}",
                        correlation_id,
                    )
                    return False

            # Execute rollback per shard
            for shard_idx, ops in shard_operations.items():
                shard = self.shards[shard_idx]
                lock = self.shard_locks[shard_idx]

                async with lock:
                    try:
                        # Use savepoint-like approach (in-memory rollback)
                        for op in reversed(ops):  # Reverse order for proper rollback
                            if op["operation"] == "set":
                                # Restore the previous value or delete if it didn't exist
                                if "previous_value" in op:
                                    current_time = time()
                                    entry = CacheEntry(
                                        data=op["previous_value"],
                                        timestamp=current_time,
                                        ttl=op.get("previous_ttl", self.ttl_seconds),
                                    )
                                    shard[op["key"]] = entry
                                else:
                                    shard.pop(op["key"], None)
                            elif op["operation"] == "delete" and "previous_value" in op:
                                # Restore deleted item
                                current_time = time()
                                entry = CacheEntry(
                                    data=op["previous_value"],
                                    timestamp=current_time,
                                    ttl=op.get("previous_ttl", self.ttl_seconds),
                                )
                                shard[op["key"]] = entry

                        runtime_metrics.cache_size.set(self.size())
                        log_with_correlation(
                            logging.DEBUG,
                            f"Rolled back {len(ops)} operations on shard {shard_idx}",
                            correlation_id,
                        )

                    except Exception as e:
                        logger.error("exception_caught", error=str(e), exc_info=True)
                        log_with_correlation(
                            logging.ERROR,
                            f"Failed to rollback operations on shard {shard_idx}: {e}",
                            correlation_id,
                        )
                        return False

            log_with_correlation(
                logging.INFO,
                f"Successfully rolled back {len(operations)} cache operations",
                correlation_id,
            )
            return True

        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            log_with_correlation(
                logging.ERROR,
                f"Cache rollback failed: {e}",
                correlation_id,
                exc_info=True,
            )
            return False
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def clear(self) -> None:
        """Asynchronously clear all cache entries."""
        for i in range(self.num_shards):
            shard = self.shards[i]
            lock = self.shard_locks[i]
            async with lock:
                shard.clear()
        logger.debug("Cache CLEARED")

    def size(self) -> int:
        """Get the current number of items in cache.

        Note: This method provides an approximate count as it doesn't acquire locks
        for each shard to avoid blocking. In high-concurrency scenarios, the count
        may be slightly inaccurate but sufficient for bounds checking purposes.
        """
        # Return approximate size without blocking on locks for performance
        # This is acceptable for bounds checking but may be slightly inaccurate
        # under heavy concurrent modifications
        return sum(len(shard) for shard in self.shards)

    def _check_cache_bounds(self) -> bool:
        """
        Check if cache size is within safe bounds to prevent memory exhaustion.

        Returns:
            True if within bounds, False if too large
        """
        current_size = self.size()

        # Check item count bounds
        if not self._check_item_count_bounds(current_size):
            return False

        # Check memory usage bounds
        return self._check_memory_usage_bounds(current_size)

    def _check_item_count_bounds(self, current_size: int) -> bool:
        """Check if item count is within safe bounds."""
        max_safe_size = self.max_entries  # Use configurable max entries
        # We check if current_size > max_safe_size because the check is done after adding the new entry
        if current_size > max_safe_size:
            logger.warning(
                f"Cache size {current_size} exceeds safe bounds {max_safe_size}",
                extra={
                    "correlation_id": runtime_metrics.create_correlation_id(
                        {
                            "component": "async_cache",
                            "operation": "bounds_check",
                            "current_size": current_size,
                            "max_safe_size": max_safe_size,
                        }
                    )
                },
            )
            return False
        return True

    def _check_memory_usage_bounds(self, current_size: int) -> bool:
        """Check if memory usage is within safe bounds."""
        # More accurate memory usage estimation
        try:
            import sys

            estimated_memory_mb = 0

            # Calculate more accurate memory usage by sampling some entries
            sample_size = min(100, current_size)  # Sample up to 100 entries
            sample_count = 0
            sample_memory = 0

            for shard in self.shards:
                for key, entry in shard.items():
                    if sample_count >= sample_size:
                        break
                    # Estimate memory for key and value
                    sample_memory += sys.getsizeof(key)
                    sample_memory += sys.getsizeof(entry.data)
                    sample_memory += sys.getsizeof(entry.timestamp)
                    sample_memory += sys.getsizeof(entry.ttl)
                    sample_count += 1
                if sample_count >= sample_size:
                    break

            # Extrapolate to full cache size
            if sample_count > 0:
                avg_memory_per_item = sample_memory / sample_count
                estimated_memory_mb = (avg_memory_per_item * current_size) / (1024 * 1024)
            else:
                # Fallback to rough calculation if no items sampled
                estimated_memory_mb = current_size * 0.5  # ~500KB per 1000 entries

            # Use configurable max memory limit
            max_memory_mb = self.max_memory_mb

            # Check if we're approaching the memory limit (80% threshold)
            memory_threshold = max_memory_mb * 0.8
            if estimated_memory_mb > memory_threshold:
                # Trigger graceful degradation measures when approaching the limit
                logger.warning(
                    "Cache memory usage {estimated_memory_mb:.1f}MB approaching limit of {max_memory_mb}MB",
                    extra={
                        "correlation_id": runtime_metrics.create_correlation_id(
                            {
                                "component": "async_cache",
                                "operation": "memory_bounds_approaching",
                                "estimated_mb": estimated_memory_mb,
                                "current_size": current_size,
                                "sample_count": sample_count,
                                "avg_memory_per_item": (
                                    avg_memory_per_item if sample_count > 0 else 0
                                ),
                                "max_memory_mb": max_memory_mb,
                                "threshold_reached": "80%",
                            }
                        )
                    },
                )

                # Implement auto-tuning for cache parameters to reduce memory usage
                if estimated_memory_mb > max_memory_mb:
                    logger.warning(
                        "Estimated cache memory usage {estimated_memory_mb:.1f}MB exceeds {max_memory_mb}MB limit",
                        extra={
                            "correlation_id": runtime_metrics.create_correlation_id(
                                {
                                    "component": "async_cache",
                                    "operation": "memory_bounds_exceeded",
                                    "estimated_mb": estimated_memory_mb,
                                    "current_size": current_size,
                                    "sample_count": sample_count,
                                    "avg_memory_per_item": (
                                        avg_memory_per_item if sample_count > 0 else 0
                                    ),
                                    "max_memory_mb": max_memory_mb,
                                }
                            )
                        },
                    )
                    return False
        except Exception as e:
            # If memory estimation fails, log and continue with basic size check
            logger.warning(
                f"Failed to estimate memory usage: {e}, proceeding with basic size check",
                extra={
                    "correlation_id": runtime_metrics.create_correlation_id(
                        {
                            "component": "async_cache",
                            "operation": "memory_bounds_check_error",
                            "error": str(e),
                        }
                    )
                },
            )
            # If we can't estimate memory, just check the size limit
            max_safe_size = self.max_entries  # Use configurable max entries
            if current_size > max_safe_size:
                return False

        return True

    def get_detailed_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache metrics for monitoring."""
        total_requests = runtime_metrics.cache_hits.value + runtime_metrics.cache_misses.value
        total_sets = runtime_metrics.cache_sets.value
        total_evictions = runtime_metrics.cache_evictions.value

        return {
            "size": self.size(),
            "num_shards": self.num_shards,
            "ttl_seconds": self.ttl_seconds,
            "cleanup_interval": self.cleanup_interval,
            "hits": runtime_metrics.cache_hits.value,
            "misses": runtime_metrics.cache_misses.value,
            "sets": total_sets,
            "evictions": total_evictions,
            "cleanup_cycles": runtime_metrics.cache_cleanup_cycles.value,
            "hit_rate": (
                (runtime_metrics.cache_hits.value / total_requests) if total_requests > 0 else 0
            ),
            "miss_rate": (
                (runtime_metrics.cache_misses.value / total_requests) if total_requests > 0 else 0
            ),
            "eviction_rate": (total_evictions / total_sets) if total_sets > 0 else 0,
            "shard_distribution": [len(shard) for shard in self.shards],
            "is_running": self.is_running,
            "health_status": runtime_metrics.get_health_status().get("async_cache", {}),
        }

    def create_backup_snapshot(self) -> dict[str, Any]:
        """
        Create a backup snapshot of current cache state for rollback purposes.
        WARNING: This is expensive and should be used sparingly.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "async_cache", "operation": "backup_snapshot"}
        )

        try:
            snapshot = {}
            current_time = time()

            for shard_idx, shard in enumerate(self.shards):
                shard_snapshot = {}
                for key, entry in shard.items():
                    if current_time - entry.timestamp <= entry.ttl:  # Only backup valid entries
                        shard_snapshot[key] = {
                            "data": entry.data,
                            "timestamp": entry.timestamp,
                            "ttl": entry.ttl,
                        }
                snapshot[f"shard_{shard_idx}"] = shard_snapshot

            snapshot["_metadata"] = {
                "created_at": current_time,
                "total_entries": sum(len(s) for s in snapshot.values() if isinstance(s, dict)),
                "correlation_id": correlation_id,
            }

            log_with_correlation(
                logging.INFO,
                f"Created cache backup snapshot with {snapshot['_metadata']['total_entries']} entries",
                correlation_id,
            )
            return snapshot

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def restore_from_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """
        Restore cache from a backup snapshot with comprehensive bounds checking.
        This will atomically replace all current cache contents.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "async_cache",
                "operation": "restore_snapshot",
                "snapshot_entries": snapshot.get("_metadata", {}).get("total_entries", 0),
            }
        )

        try:
            # BOUNDS CHECKING - Validate snapshot structure
            if snapshot is None:
                raise ValueError("Snapshot cannot be None")

            if not isinstance(snapshot, dict):
                raise TypeError(f"Snapshot must be a dict, got {type(snapshot)}")

            if "_metadata" not in snapshot:
                log_with_correlation(
                    logging.ERROR,
                    "Invalid snapshot format - missing metadata",
                    correlation_id,
                )
                return False

            metadata = snapshot["_metadata"]
            if not isinstance(metadata, dict):
                raise TypeError("Snapshot metadata must be a dict")

            # Validate total entries bounds (prevent memory exhaustion)
            total_entries = metadata.get("total_entries", 0)
            if not isinstance(total_entries, int) or total_entries < 0:
                raise ValueError(f"Invalid total_entries in metadata: {total_entries}")
            if total_entries > self.max_entries:  # Use configurable max entries
                raise ValueError(
                    f"Snapshot too large: {total_entries} entries (max {self.max_entries})"
                )

            # Validate snapshot age (don't restore very old snapshots)
            created_at = metadata.get("created_at")
            if not isinstance(created_at, (int, float)):
                raise ValueError(f"Invalid created_at timestamp: {created_at}")

            snapshot_age = time() - created_at
            if snapshot_age < 0:
                raise ValueError(f"Snapshot from future: age {snapshot_age}s")
            if snapshot_age > 3600:  # 1 hour max age
                log_with_correlation(
                    logging.WARNING,
                    "Snapshot too old ({snapshot_age:.0f}s) - refusing restore",
                    correlation_id,
                )
                return False

            # Clear current cache
            await self.clear()

            # Restore from snapshot
            restored_count = 0
            for shard_key, shard_data in snapshot.items():
                if shard_key.startswith("shard_") and isinstance(shard_data, dict):
                    shard_idx = int(shard_key.split("_")[1])
                    if 0 <= shard_idx < self.num_shards:
                        shard = self.shards[shard_idx]
                        lock = self.shard_locks[shard_idx]

                        async with lock:
                            for key, entry_data in shard_data.items():
                                entry = CacheEntry(
                                    data=entry_data["data"],
                                    timestamp=entry_data["timestamp"],
                                    ttl=entry_data["ttl"],
                                )
                                shard[key] = entry
                                restored_count += 1

            runtime_metrics.cache_size.set(self.size())
            log_with_correlation(
                logging.INFO,
                f"Restored {restored_count} entries from snapshot",
                correlation_id,
            )
            return True

        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            log_with_correlation(
                logging.ERROR,
                f"Failed to restore from snapshot: {e}",
                correlation_id,
                exc_info=True,
            )
            return False
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        self.is_running = False
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task
        logger.debug("AsyncTTLCache stopped")

    async def _health_check_functionality(self, correlation_id) -> dict[str, Any]:
        """
        Perform basic functionality tests for the health check.

        Args:
            correlation_id: Correlation ID for logging

        Returns:
            Dict with status or error details if functionality tests fail
        """
        test_key = f"health_check_{correlation_id.id}_{int(time())}"
        test_value = {
            "test": "data",
            "timestamp": time(),
            "correlation_id": correlation_id.id,
        }

        # Test set operation with validation
        try:
            await self.set(test_key, test_value, ttl_seconds=300)  # 5 min TTL for safety
        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            return {
                "status": "critical",
                "component": "async_cache",
                "error": f"SET operation failed: {e}",
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        # Test get operation
        try:
            retrieved = await self.get(test_key)
            if retrieved != test_value:
                return {
                    "status": "critical",
                    "component": "async_cache",
                    "error": "GET operation data mismatch",
                    "expected": test_value,
                    "received": retrieved,
                    "correlation_id": correlation_id.id,
                    "production_ready": False,
                }
        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            return {
                "status": "critical",
                "component": "async_cache",
                "error": f"GET operation failed: {e}",
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        # Test delete operation
        try:
            deleted = await self.delete(test_key)
            if not deleted:
                return {
                    "status": "warning",
                    "component": "async_cache",
                    "error": "DELETE operation returned False",
                    "correlation_id": correlation_id.id,
                    "production_ready": True,  # Not critical for production
                }
        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            return {
                "status": "error",
                "component": "async_cache",
                "error": f"DELETE operation failed: {e}",
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        return {"status": "ok"}

    async def _health_check_integrity(self, is_production: bool, correlation_id) -> dict[str, Any]:
        """
        Perform cache bounds and integrity checks.

        This method validates multiple aspects of cache integrity:
        1. Checks that cache size is valid
        2. Verifies that bounds checks pass
        3. Ensures shard sizes add up correctly
        4. Checks for shard size balance

        Args:
            is_production: Whether the environment is production
            correlation_id: Correlation ID for logging

        Returns:
            Dict with status or error details if integrity checks fail
        """
        current_size = self.size()
        if current_size < 0:
            return self._create_integrity_error_response(
                "Invalid cache size",
                {"current_size": current_size},
                correlation_id,
                False,
            )

        # Check bounds compliance
        bounds_ok = self._check_cache_bounds()
        if not bounds_ok:
            return self._create_integrity_warning_response(
                "Cache bounds exceeded",
                {"current_size": current_size},
                correlation_id,
                is_production,
            )

        # 3. SHARD INTEGRITY CHECKS
        shard_sizes = [len(shard) for shard in self.shards]
        total_from_shards = sum(shard_sizes)

        if total_from_shards != current_size:
            return self._create_integrity_error_response(
                "Shard size inconsistency",
                {
                    "total_size": current_size,
                    "shard_total": total_from_shards,
                    "shard_sizes": shard_sizes,
                },
                correlation_id,
                False,
            )

        # Check for shard size imbalances (should be roughly equal)
        imbalance_check = self._check_shard_balance(shard_sizes, current_size)
        if imbalance_check:
            return imbalance_check

        return {
            "status": "ok",
            "current_size": current_size,
            "shard_sizes": shard_sizes,
            "bounds_ok": bounds_ok,
        }

    def _create_integrity_error_response(
        self, error_msg: str, details: dict, correlation_id, production_ready: bool
    ) -> dict[str, Any]:
        """Create a standardized error response for integrity checks."""
        return {
            "status": "critical",
            "component": "async_cache",
            "error": error_msg,
            "correlation_id": correlation_id.id,
            "production_ready": production_ready,
            **details,
        }

    def _create_integrity_warning_response(
        self, error_msg: str, details: dict, correlation_id, is_production: bool
    ) -> dict[str, Any]:
        """Create a standardized warning response for integrity checks."""
        return {
            "status": "warning",
            "component": "async_cache",
            "error": error_msg,
            "correlation_id": correlation_id.id,
            "production_ready": is_production,  # Only critical in production
            **details,
        }

    def _check_shard_balance(
        self, shard_sizes: list[int], current_size: int
    ) -> dict[str, Any] | None:
        """Check if shards are balanced."""
        if self.num_shards > 1:
            avg_shard_size = current_size / self.num_shards
            max_shard_size = max(shard_sizes)
            if max_shard_size > avg_shard_size * 3:  # Allow 3x imbalance
                return {
                    "status": "warning",
                    "component": "async_cache",
                    "error": "Shard size imbalance detected",
                    "avg_shard_size": avg_shard_size,
                    "max_shard_size": max_shard_size,
                    "shard_sizes": shard_sizes,
                    "production_ready": True,  # Not critical but worth monitoring
                }
        return None

    async def _health_check_background_tasks(
        self, is_production: bool, correlation_id
    ) -> dict[str, str]:
        """
        Check the status of background tasks.

        Args:
            is_production: Whether the environment is production
            correlation_id: Correlation ID for logging

        Returns:
            Dict with status or error details if background task checks fail
        """
        cleanup_status = (
            "running" if self.cleanup_task and not self.cleanup_task.done() else "stopped"
        )
        if is_production and cleanup_status != "running":
            return {
                "status": "error",
                "component": "async_cache",
                "error": "Background cleanup task not running in production",
                "cleanup_status": cleanup_status,
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        return {"status": "ok", "cleanup_status": cleanup_status}

    async def _health_check_performance(self, correlation_id) -> dict[str, Any]:
        """
        Check performance metrics.

        Args:
            correlation_id: Correlation ID for logging

        Returns:
            Dict with status or error details if performance checks fail
        """
        metrics = self.get_detailed_metrics()
        hit_rate = metrics.get("hit_rate", 0)
        if hit_rate < 0 or hit_rate > 1:
            return {
                "status": "error",
                "component": "async_cache",
                "error": "Invalid hit rate metrics",
                "hit_rate": hit_rate,
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        return {"status": "ok", "hit_rate": hit_rate}

    async def _health_check_config(self, correlation_id) -> dict[str, Any]:
        """
        Check configuration validity.

        Args:
            correlation_id: Correlation ID for logging

        Returns:
            Dict with status or error details if config validation fails
        """
        # 6. PRODUCTION READINESS VALIDATION
        production_issues = []

        # Check configuration validity
        if self.ttl_seconds <= 0 or self.ttl_seconds > 86400:
            production_issues.append(f"Invalid TTL configuration: {self.ttl_seconds}")

        if self.num_shards <= 0 or self.num_shards > 256:
            production_issues.append(f"Invalid shard count: {self.num_shards}")

        if len(production_issues) > 0:
            return {
                "status": "error",
                "component": "async_cache",
                "error": "Production configuration issues",
                "issues": production_issues,
                "correlation_id": correlation_id.id,
                "production_ready": False,
            }

        return {"status": "ok"}

    async def health_check(self) -> dict[str, Any]:
        """
        Perform PRODUCTION-GRADE comprehensive health check of the cache with rigorous validation.
        """
        try:
            correlation_id = runtime_metrics.create_correlation_id(
                {"component": "async_cache", "operation": "health_check"}
            )

            # PRODUCTION READINESS: Check environment and security context
            from resync.core import env_detector

            is_production = env_detector.is_production()
            security_level = env_detector.get_security_level()

            # 1. BASIC FUNCTIONALITY TESTS
            functionality_result = await self._health_check_functionality(correlation_id)
            if functionality_result["status"] != "ok":
                return functionality_result

            # 2. CACHE BOUNDS AND INTEGRITY CHECKS
            integrity_result = await self._health_check_integrity(is_production, correlation_id)
            if integrity_result["status"] != "ok":
                return integrity_result
            current_size = integrity_result["current_size"]
            shard_sizes = integrity_result["shard_sizes"]
            bounds_ok = integrity_result["bounds_ok"]

            # 4. BACKGROUND TASK HEALTH
            bg_task_result = await self._health_check_background_tasks(
                is_production, correlation_id
            )
            if bg_task_result["status"] != "ok":
                return bg_task_result
            cleanup_status = bg_task_result["cleanup_status"]

            # 5. PERFORMANCE METRICS VALIDATION
            perf_result = await self._health_check_performance(correlation_id)
            if perf_result["status"] != "ok":
                return perf_result
            hit_rate = perf_result["hit_rate"]

            # 6. CONFIGURATION VALIDATION
            config_result = await self._health_check_config(correlation_id)
            if config_result["status"] != "ok":
                return config_result

            # SUCCESS - All checks passed
            result = {
                "status": "healthy",
                "component": "async_cache",
                "production_ready": True,
                "size": current_size,
                "num_shards": self.num_shards,
                "shard_sizes": shard_sizes,
                "cleanup_status": cleanup_status,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": hit_rate,
                "bounds_compliant": bounds_ok,
                "environment": env_detector._environment,
                "security_level": security_level,
                "correlation_id": correlation_id.id,
            }

            log_with_correlation(
                logging.DEBUG,
                f"Cache health check PASSED - production ready: {result['production_ready']}",
                correlation_id,
            )
            runtime_metrics.close_correlation_id(correlation_id)
            return result

        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            error_correlation = runtime_metrics.create_correlation_id(
                {
                    "component": "async_cache",
                    "operation": "health_check_critical_failure",
                }
            )
            return {
                "status": "critical",
                "component": "async_cache",
                "error": f"Health check completely failed: {e}",
                "correlation_id": str(error_correlation),
                "production_ready": False,
                "exception_type": type(e).__name__,
            }

    async def __aenter__(self) -> AsyncTTLCache:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - cleanup resources."""
        await self.stop()

    # Methods for WAL replay - these apply operations without re-logging to prevent loops
    async def apply_wal_set(self, key: str, value: Any, ttl: float | None = None):
        """
        Apply a SET operation from WAL replay without re-logging.
        This method is used during recovery to apply operations that were already logged to WAL.
        """
        # Ensure cleanup task is running
        self._start_cleanup_task()

        try:
            # Validate inputs
            validated_key, validated_ttl = self._validate_cache_inputs(key, value, ttl)

            current_time = time()
            entry = CacheEntry(data=value, timestamp=current_time, ttl=validated_ttl)

            shard, lock = self._get_shard(validated_key)
            async with lock:
                # Check bounds first - if we're already at the limit, we need to evict BEFORE adding
                # to ensure we never exceed the bounds (same logic as set method)
                while not self._check_cache_bounds():
                    lru_key = self._get_lru_key(shard)
                    if lru_key:
                        del shard[lru_key]
                        runtime_metrics.cache_evictions.increment()
                    else:
                        break

                if not self._check_cache_bounds():
                    for i, other_shard in enumerate(self.shards):
                        if i == self.shards.index(shard):
                            continue

                        if not self._check_cache_bounds():
                            other_lock = self.shard_locks[i]
                            async with other_lock:
                                lru_key = self._get_lru_key(other_shard)
                                if lru_key:
                                    del other_shard[lru_key]
                                    runtime_metrics.cache_evictions.increment()
                        else:
                            break

                # Final bounds check - reject if still over bounds
                # This is crucial to ensure we never exceed the configured limits
                if not self._check_cache_bounds():
                    logger.warning(
                        "cache_bounds_exceeded_during_WAL_replay",
                        key=repr(validated_key),
                    )
                    return

                # Only add the entry if we're within bounds
                shard[validated_key] = entry
                runtime_metrics.cache_sets.increment()
                runtime_metrics.cache_size.set(self.size())
        except Exception as e:
            logger.error("WAL_replay_SET_failed", key=repr(key), error=str(e))

    async def apply_wal_delete(self, key: str):
        """
        Apply a DELETE operation from WAL replay without re-logging.
        This method is used during recovery to apply operations that were already logged to WAL.
        """
        try:
            shard, lock = self._get_shard(key)
            async with lock:
                if key in shard:
                    del shard[key]
                    runtime_metrics.cache_evictions.increment()
                    runtime_metrics.cache_size.set(self.size())
        except Exception as e:
            logger.error("WAL_replay_DELETE_failed", key=key, error=str(e))


async def get_redis_client():
    """
    Get an async Redis client for connection validation.

    Returns:
        AsyncRedis: Connected Redis client
    """
    try:
        from redis.asyncio import Redis as AsyncRedis

        from resync.settings import settings

        # v5.9.7: Use consolidated settings attribute and honor legacy env var via Settings aliases.
        redis_url = getattr(settings, "redis_url", None) or "redis://localhost:6379/0"

        return AsyncRedis.from_url(
            redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            decode_responses=False,
        )

    except Exception as e:
        logger.error("Failed to create Redis client", error=str(e))
        raise
