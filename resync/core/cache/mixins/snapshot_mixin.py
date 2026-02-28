"""
Cache Snapshot Mixin.

Provides backup and restore functionality for cache implementations.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

class CacheSnapshotMixin:
    """
    Mixin providing snapshot/backup capabilities for cache.

    Requires base class to have:
    - self.shards: List of cache shards  # type: ignore[attr-defined]
    - self.shard_locks: List of shard locks  # type: ignore[attr-defined]
    """

    def create_backup_snapshot(self) -> dict[str, Any]:
        """
        Create a backup snapshot of the entire cache.

        Returns:
            Dict containing serialized cache state
        """
        snapshot = {
            "version": "1.0",
            "timestamp": time.time(),
            "shards": [],
        }

        for _i, shard in enumerate(self.shards):  # type: ignore[attr-defined]
            shard_data = {}
            for key, entry in shard.items():
                shard_data[key] = {
                    "data": entry.data,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                }
            snapshot["shards"].append(shard_data)  # type: ignore[attr-defined]

        snapshot["total_entries"] = sum(len(s) for s in self.shards)  # type: ignore[misc,attr-defined]

        logger.info("Created snapshot with %s entries", snapshot["total_entries"])

        return snapshot

    async def restore_from_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """
        Restore cache state from a snapshot.

        Args:
            snapshot: Previously created snapshot dict

        Returns:
            True if restore was successful
        """
        try:
            if "shards" not in snapshot:
                logger.error("Invalid snapshot format: missing shards")
                return False

            # Clear current cache
            await self.clear()  # type: ignore[attr-defined]

            # Restore entries
            restored_count = 0
            current_time = time.time()

            from ..async_cache import CacheEntry

            for i, shard_data in enumerate(snapshot["shards"]):
                if i >= len(self.shards):  # type: ignore[attr-defined]
                    break

                async with self.shard_locks[i]:  # type: ignore[attr-defined]
                    for key, entry_data in shard_data.items():
                        # Skip expired entries
                        if current_time > entry_data["timestamp"] + entry_data["ttl"]:
                            continue

                        self.shards[i][key] = CacheEntry(  # type: ignore[attr-defined]
                            data=entry_data["data"],
                            timestamp=entry_data["timestamp"],
                            ttl=entry_data["ttl"],
                        )
                        restored_count += 1

            logger.info("Restored %s entries from snapshot", restored_count)
            return True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Snapshot restore failed: %s", e)
            return False

    def get_snapshot_metadata(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Get metadata about a snapshot without loading all data."""
        return {
            "version": snapshot.get("version", "unknown"),
            "timestamp": snapshot.get("timestamp", 0),
            "total_entries": snapshot.get("total_entries", 0),
            "shard_count": len(snapshot.get("shards", [])),
        }
