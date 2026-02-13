"""
Strategy Pattern Implementations for Cache Operations

This module provides strategy pattern implementations for complex cache operations
that have been extracted from the main AsyncTTLCache class to improve modularity
and maintainability.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from time import time
from typing import Any

from resync.core.cache.memory_manager import CacheMemoryManager
from resync.core.metrics import log_with_correlation, runtime_metrics
from resync.core.shared_types import CacheEntry
from resync.core.write_ahead_log import WalEntry, WalOperationType

logger = logging.getLogger(__name__)


class CacheSetStrategy(ABC):
    """Strategy interface for cache set operations."""

    @abstractmethod
    async def execute(
        self,
        key: str,
        value: Any,
        ttl_seconds: float,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        memory_manager: CacheMemoryManager,
        enable_wal: bool = False,
        wal: Any | None = None,
    ) -> None:
        """Execute the set operation strategy."""


class StandardCacheSetStrategy(CacheSetStrategy):
    """Standard implementation of cache set strategy with bounds checking and LRU eviction."""

    async def execute(
        self,
        key: str,
        value: Any,
        ttl_seconds: float,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        memory_manager: CacheMemoryManager,
        enable_wal: bool = False,
        wal: Any | None = None,
    ) -> None:
        """
        Execute set operation with comprehensive bounds checking and LRU eviction.

        Args:
            key: Cache key to set
            value: Value to cache
            ttl_seconds: TTL for the entry
            shards: List of cache shards
            shard_locks: List of locks for each shard
            memory_manager: Memory manager for bounds checking
            enable_wal: Whether WAL is enabled
            wal: Write-ahead log instance
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "cache_set_strategy",
                "operation": "execute",
                "key": repr(key),
                "ttl_seconds": ttl_seconds,
            }
        )

        try:
            # Log to WAL if enabled
            if enable_wal and wal:
                wal_entry = WalEntry(
                    operation=WalOperationType.SET,
                    key=key,
                    value=value,
                    ttl=ttl_seconds,
                )
                log_success = await wal.log_operation(wal_entry)
                if not log_success:
                    logger.error("failed_to_log_SET_operation_to_WAL", key=key)

            current_time = time()
            entry = CacheEntry(data=value, timestamp=current_time, ttl=ttl_seconds)

            shard, lock = self._get_shard(key, shards, shard_locks)
            async with lock:
                # Add the entry first
                shard[key] = entry

                # Check bounds and evict if necessary
                await self._handle_bounds_checking(
                    shards, shard_locks, memory_manager, key, correlation_id
                )

                runtime_metrics.cache_sets.increment()
                runtime_metrics.cache_size.set(sum(len(s) for s in shards))

        except Exception as e:
            log_with_correlation(
                logging.ERROR,
                f"Cache SET failed for key {repr(key)}: {e}",
                correlation_id,
            )
            raise
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _get_shard(
        self,
        key: str,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
    ) -> tuple[dict[str, CacheEntry], asyncio.Lock]:
        """Get the shard and lock for a given key with bounds checking."""
        try:
            key_hash = hash(key)
            if key_hash == 0:
                key_hash = sum(ord(c) for c in str(key)) + len(str(key))

            shard_index = abs(key_hash) % len(shards)

            if not (0 <= shard_index < len(shards)):
                shard_index = (len(key) + (ord(key[0]) if key else 0)) % len(shards)

        except (OverflowError, ValueError):
            key_sum = sum(ord(c) for c in str(key)[:20])
            shard_index = key_sum % len(shards)
            logger.warning(
                f"Hash computation failed for key {repr(key)}, using fallback shard {shard_index}"
            )

        return shards[shard_index], shard_locks[shard_index]

    async def _handle_bounds_checking(
        self,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        memory_manager: CacheMemoryManager,
        current_key: str,
        correlation_id,
    ) -> None:
        """Handle cache bounds checking and LRU eviction."""
        eviction_count = 0
        max_evictions = len(shards) * 2

        while (
            not memory_manager.check_memory_bounds(shards, sum(len(s) for s in shards))
            and eviction_count < max_evictions
        ):
            # Find LRU key to evict
            lru_key = self._find_lru_key_for_eviction(shards, shard_locks, current_key)
            if lru_key:
                await self._evict_key(shards, shard_locks, lru_key, correlation_id)
                eviction_count += 1
            else:
                break

    def _find_lru_key_for_eviction(
        self,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        exclude_key: str,
    ) -> str | None:
        """Find the best LRU key to evict across all shards."""
        for _i, shard in enumerate(shards):
            if not shard:
                continue

            # Find LRU in current shard, excluding the key we just added
            lru_key = None
            lru_timestamp = float("inf")

            for key, entry in shard.items():
                if key == exclude_key:
                    continue
                if entry.timestamp < lru_timestamp:
                    lru_timestamp = entry.timestamp
                    lru_key = key

            if lru_key:
                return lru_key

        return None

    async def _evict_key(
        self,
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        key: str,
        correlation_id,
    ) -> None:
        """Evict a specific key from the cache."""
        for i, shard in enumerate(shards):
            if key in shard:
                lock = shard_locks[i]
                async with lock:
                    if key in shard:  # Double-check after acquiring lock
                        del shard[key]
                        runtime_metrics.cache_evictions.increment()
                        log_with_correlation(
                            logging.DEBUG,
                            f"LRU eviction removed key: {key}",
                            correlation_id,
                        )
                break


class CacheRollbackStrategy(ABC):
    """Strategy interface for cache rollback operations."""

    @abstractmethod
    async def execute(
        self,
        operations: list[dict[str, Any]],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        default_ttl: float,
    ) -> bool:
        """Execute the rollback operation strategy."""


class StandardCacheRollbackStrategy(CacheRollbackStrategy):
    """Standard implementation of cache rollback strategy."""

    async def execute(
        self,
        operations: list[dict[str, Any]],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        default_ttl: float,
    ) -> bool:
        """
        Execute rollback operation with comprehensive validation and bounds checking.

        Args:
            operations: List of operations to rollback
            shards: List of cache shards
            shard_locks: List of locks for each shard
            default_ttl: Default TTL for restored entries

        Returns:
            True if rollback successful, False otherwise
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "cache_rollback_strategy",
                "operation": "execute",
                "operations_count": len(operations),
            }
        )

        try:
            # Validate operations
            if not self._validate_operations(operations):
                return False

            # Group operations by shard
            shard_operations = self._group_operations_by_shard(operations, shards, shard_locks)

            # Execute rollback per shard
            for shard_idx, ops in shard_operations.items():
                success = await self._rollback_shard_operations(
                    shard_idx, ops, shards, shard_locks, default_ttl, correlation_id
                )
                if not success:
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

    def _validate_operations(self, operations: list[dict[str, Any]]) -> bool:
        """Validate the structure and content of rollback operations."""
        if not isinstance(operations, list):
            return False

        if len(operations) == 0:
            return True

        if len(operations) > 10000:
            return False

        for _i, op in enumerate(operations):
            if not isinstance(op, dict):
                return False

            if "key" not in op or "operation" not in op:
                return False

            if op["operation"] not in ["set", "delete"]:
                return False

            # Validate key
            key = op["key"]
            if not isinstance(key, (str, int, float, bool)):
                try:
                    str_key = str(key)
                    if len(str_key) > 1000:
                        return False
                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    return False

        return True

    def _group_operations_by_shard(
        self,
        operations: list[dict[str, Any]],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
    ) -> dict[int, list[dict[str, Any]]]:
        """Group operations by shard index for efficient processing."""
        shard_operations: dict[int, list[dict[str, Any]]] = {}

        for op in operations:
            try:
                # Calculate shard index for the key
                key_hash = hash(op["key"])
                if key_hash == 0:
                    key_hash = sum(ord(c) for c in str(op["key"])) + len(str(op["key"]))
                shard_idx = abs(key_hash) % len(shards)

                if shard_idx not in shard_operations:
                    shard_operations[shard_idx] = []
                shard_operations[shard_idx].append(op)
            except (ValueError, IndexError):
                # Skip invalid operations
                continue

        return shard_operations

    async def _rollback_shard_operations(
        self,
        shard_idx: int,
        operations: list[dict[str, Any]],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        default_ttl: float,
        correlation_id,
    ) -> bool:
        """Rollback operations for a specific shard."""
        shard = shards[shard_idx]
        lock = shard_locks[shard_idx]

        async with lock:
            try:
                # Process operations in reverse order for proper rollback
                for op in reversed(operations):
                    if op["operation"] == "set":
                        await self._rollback_set_operation(op, shard, default_ttl)
                    elif op["operation"] == "delete":
                        await self._rollback_delete_operation(op, shard, default_ttl)

                runtime_metrics.cache_size.set(sum(len(s) for s in shards))
                log_with_correlation(
                    logging.DEBUG,
                    f"Rolled back {len(operations)} operations on shard {shard_idx}",
                    correlation_id,
                )
                return True

            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                log_with_correlation(
                    logging.ERROR,
                    f"Failed to rollback operations on shard {shard_idx}: {e}",
                    correlation_id,
                )
                return False

    async def _rollback_set_operation(
        self,
        operation: dict[str, Any],
        shard: dict[str, CacheEntry],
        default_ttl: float,
    ) -> None:
        """Rollback a set operation."""
        key = operation["key"]
        if "previous_value" in operation:
            # Restore previous value
            current_time = time()
            entry = CacheEntry(
                data=operation["previous_value"],
                timestamp=current_time,
                ttl=operation.get("previous_ttl", default_ttl),
            )
            shard[key] = entry
        else:
            # Remove the key if it didn't exist before
            shard.pop(key, None)

    async def _rollback_delete_operation(
        self,
        operation: dict[str, Any],
        shard: dict[str, CacheEntry],
        default_ttl: float,
    ) -> None:
        """Rollback a delete operation."""
        key = operation["key"]
        if "previous_value" in operation:
            # Restore deleted item
            current_time = time()
            entry = CacheEntry(
                data=operation["previous_value"],
                timestamp=current_time,
                ttl=operation.get("previous_ttl", default_ttl),
            )
            shard[key] = entry


class CacheRestoreStrategy(ABC):
    """Strategy interface for cache restore operations."""

    @abstractmethod
    async def execute(
        self,
        snapshot: dict[str, Any],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        max_entries: int,
    ) -> bool:
        """Execute the restore operation strategy."""


class StandardCacheRestoreStrategy(CacheRestoreStrategy):
    """Standard implementation of cache restore strategy."""

    async def execute(
        self,
        snapshot: dict[str, Any],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        max_entries: int,
    ) -> bool:
        """
        Execute restore operation with comprehensive validation and bounds checking.

        Args:
            snapshot: Snapshot data to restore from
            shards: List of cache shards
            shard_locks: List of locks for each shard
            max_entries: Maximum allowed entries in cache

        Returns:
            True if restore successful, False otherwise
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "cache_restore_strategy",
                "operation": "execute",
                "snapshot_entries": snapshot.get("_metadata", {}).get("total_entries", 0),
            }
        )

        try:
            # Validate snapshot
            if not self._validate_snapshot(snapshot, max_entries):
                return False

            metadata = snapshot["_metadata"]
            metadata.get("total_entries", 0)

            # Clear current cache
            await self._clear_cache(shards, shard_locks)

            # Restore from snapshot
            restored_count = 0
            for shard_key, shard_data in snapshot.items():
                if shard_key.startswith("shard_") and isinstance(shard_data, dict):
                    restored_count += await self._restore_shard(
                        shard_key, shard_data, shards, shard_locks, correlation_id
                    )

            runtime_metrics.cache_size.set(sum(len(s) for s in shards))
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

    def _validate_snapshot(self, snapshot: dict[str, Any], max_entries: int) -> bool:
        """Validate snapshot structure and bounds."""
        if not isinstance(snapshot, dict) or "_metadata" not in snapshot:
            return False

        metadata = snapshot["_metadata"]
        if not isinstance(metadata, dict):
            return False

        total_entries = metadata.get("total_entries", 0)
        if not isinstance(total_entries, int) or total_entries < 0:
            return False

        if total_entries > max_entries:
            return False

        created_at = metadata.get("created_at")
        if not isinstance(created_at, (int, float)) or created_at <= 0:
            return False

        # Check snapshot age (max 1 hour)
        snapshot_age = time() - created_at
        return not (snapshot_age < 0 or snapshot_age > 3600)

    async def _clear_cache(
        self, shards: list[dict[str, CacheEntry]], shard_locks: list[asyncio.Lock]
    ) -> None:
        """Clear all cache shards."""
        tasks = []
        for i in range(len(shards)):
            shard_locks[i]

            async def clear_shard(shard_idx: int):
                async with shard_locks[shard_idx]:
                    shards[shard_idx].clear()

            tasks.append(clear_shard(i))

        await asyncio.gather(*tasks)

    async def _restore_shard(
        self,
        shard_key: str,
        shard_data: dict[str, Any],
        shards: list[dict[str, CacheEntry]],
        shard_locks: list[asyncio.Lock],
        correlation_id,
    ) -> int:
        """Restore a single shard from snapshot data."""
        try:
            shard_idx = int(shard_key.split("_")[1])
            if not (0 <= shard_idx < len(shards)):
                return 0

            shard = shards[shard_idx]
            lock = shard_locks[shard_idx]

            restored_count = 0
            async with lock:
                for key, entry_data in shard_data.items():
                    entry = CacheEntry(
                        data=entry_data["data"],
                        timestamp=entry_data["timestamp"],
                        ttl=entry_data["ttl"],
                    )
                    shard[key] = entry
                    restored_count += 1

            return restored_count

        except (ValueError, IndexError) as e:
            log_with_correlation(
                logging.WARNING,
                f"Failed to restore shard {shard_key}: {e}",
                correlation_id,
            )
            return 0
