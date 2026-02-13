"""
Persistence Management Module for Cache Operations

This module provides dedicated persistence management functionality for cache systems,
extracted from the main cache implementation to improve modularity and maintainability.
"""

import json
import logging
import os
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class CachePersistenceManager:
    """
    Manages persistence operations for cache systems.

    This class provides functionality to create backup snapshots of cache state
    and restore from those snapshots. It handles serialization, file I/O, and
    validation of cache data for reliable persistence operations.

    The persistence manager is designed to be:
    - Simple and focused on persistence operations
    - Compatible with existing cache implementations
    - Extensible for future persistence strategies
    - Production-ready with proper error handling and logging
    """

    def __init__(self, snapshot_dir: str = "./cache_snapshots"):
        """
        Initialize the persistence manager.

        Args:
            snapshot_dir: Directory path for storing snapshot files.
                         Defaults to './cache_snapshots'
        """
        self.snapshot_dir = snapshot_dir
        self._ensure_snapshot_directory()

    def _ensure_snapshot_directory(self) -> None:
        """Ensure the snapshot directory exists."""
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create snapshot directory {self.snapshot_dir}: {e}")
            raise

    def create_backup_snapshot(self, cache_data: dict[str, Any]) -> str:
        """
        Create a backup snapshot of cache state.

        This method serializes the provided cache data into a JSON format
        and saves it to a timestamped file in the snapshot directory.

        Args:
            cache_data: Dictionary containing cache entries organized by shards.
                       Expected format: {"shard_0": {...}, "shard_1": {...}, ...}

        Returns:
            str: Path to the created snapshot file

        Raises:
            ValueError: If cache_data is invalid or serialization fails
            IOError: If file writing fails
        """
        if not isinstance(cache_data, dict):
            raise ValueError(f"Cache data must be a dictionary, got {type(cache_data)}")

        # Validate cache data structure
        total_entries = 0
        for key, value in cache_data.items():
            if not key.startswith("shard_"):
                continue  # Skip metadata and other keys

            if not isinstance(value, dict):
                raise ValueError(f"Invalid shard data format for {key}")

            total_entries += len(value)

        # Create snapshot with metadata
        timestamp = int(time())
        snapshot = {
            "_metadata": {
                "created_at": timestamp,
                "total_entries": total_entries,
                "version": "1.0",
            },
            **cache_data,
        }

        # Generate filename
        filename = f"cache_snapshot_{timestamp}.json"
        filepath = os.path.join(self.snapshot_dir, filename)

        try:
            # Serialize to JSON with pretty formatting for readability
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            logger.info(f"Created cache backup snapshot: {filepath} ({total_entries} entries)")
            return filepath

        except OSError as e:
            logger.error(f"Failed to write snapshot to {filepath}: {e}")
            raise OSError(f"Failed to create snapshot: {e}") from None
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize cache data: {e}")
            raise ValueError(f"Serialization failed: {e}") from None

    def restore_from_snapshot(self, snapshot_path: str) -> dict[str, Any]:
        """
        Restore cache data from a snapshot file.

        This method reads and validates a snapshot file, then returns the
        cache data for restoration. The actual cache restoration should be
        handled by the calling cache implementation.

        Args:
            snapshot_path: Path to the snapshot file to restore from

        Returns:
            Dict containing the cache data and metadata

        Raises:
            FileNotFoundError: If snapshot file doesn't exist
            ValueError: If snapshot format is invalid
            IOError: If file reading fails
        """
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

        try:
            # Read and parse snapshot file
            with open(snapshot_path, encoding="utf-8") as f:
                snapshot = json.load(f)

        except OSError as e:
            logger.error(f"Failed to read snapshot from {snapshot_path}: {e}")
            raise OSError(f"Failed to read snapshot: {e}") from None
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse snapshot JSON from {snapshot_path}: {e}")
            raise ValueError(f"Invalid snapshot format: {e}") from None

        # Validate snapshot structure
        if not isinstance(snapshot, dict):
            raise ValueError("Snapshot must be a dictionary")

        if "_metadata" not in snapshot:
            raise ValueError("Snapshot missing required metadata")

        metadata = snapshot["_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Snapshot metadata must be a dictionary")

        # Validate metadata fields
        required_fields = ["created_at", "total_entries", "version"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Snapshot metadata missing required field: {field}")

        # Validate timestamp
        created_at = metadata["created_at"]
        if not isinstance(created_at, (int, float)) or created_at <= 0:
            raise ValueError(f"Invalid timestamp in snapshot: {created_at}")

        # Check snapshot age (max 24 hours)
        snapshot_age = time() - created_at
        if snapshot_age > 86400:  # 24 hours
            logger.warning("Snapshot is {snapshot_age:.0f}s old (max recommended: 86400s)")

        # Validate total entries
        total_entries = metadata["total_entries"]
        if not isinstance(total_entries, int) or total_entries < 0:
            raise ValueError(f"Invalid total_entries in snapshot: {total_entries}")

        # Validate cache data structure
        cache_data = {}
        for key, value in snapshot.items():
            if key == "_metadata":
                continue

            if not key.startswith("shard_"):
                logger.warning(f"Skipping unknown key in snapshot: {key}")
                continue

            if not isinstance(value, dict):
                raise ValueError(f"Invalid shard data format for {key}")

            cache_data[key] = value

        logger.info(f"Loaded snapshot from {snapshot_path}: {total_entries} entries")
        return snapshot

    def list_snapshots(self) -> list[dict[str, Any]]:
        """
        List all available snapshots in the snapshot directory.

        Returns:
            List of dictionaries containing snapshot information:
            [
                {
                    "path": "/path/to/snapshot.json",
                    "created_at": timestamp,
                    "total_entries": count,
                    "size_bytes": file_size
                },
                ...
            ]
        """
        snapshots = []

        try:
            for filename in os.listdir(self.snapshot_dir):
                if not filename.startswith("cache_snapshot_") or not filename.endswith(".json"):
                    continue

                filepath = os.path.join(self.snapshot_dir, filename)

                try:
                    # Get file stats
                    stat = os.stat(filepath)
                    size_bytes = stat.st_size

                    # Extract timestamp from filename
                    try:
                        timestamp_str = filename.replace("cache_snapshot_", "").replace(".json", "")
                        created_at = int(timestamp_str)
                    except ValueError:
                        logger.warning(f"Could not parse timestamp from filename: {filename}")
                        continue

                    # Try to read metadata for total_entries
                    try:
                        with open(filepath, encoding="utf-8") as f:
                            snapshot_data = json.load(f)
                        total_entries = snapshot_data.get("_metadata", {}).get("total_entries", 0)
                    except (json.JSONDecodeError, KeyError):
                        total_entries = 0

                    snapshots.append(
                        {
                            "path": filepath,
                            "filename": filename,
                            "created_at": created_at,
                            "total_entries": total_entries,
                            "size_bytes": size_bytes,
                        }
                    )

                except OSError as e:
                    logger.warning(f"Failed to read snapshot file {filepath}: {e}")
                    continue

        except OSError as e:
            logger.error(f"Failed to list snapshot directory {self.snapshot_dir}: {e}")

        # Sort by creation time (newest first)
        snapshots.sort(key=lambda x: x["created_at"], reverse=True)
        return snapshots

    def cleanup_old_snapshots(self, max_age_seconds: int = 86400) -> int:
        """
        Remove snapshots older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default: 24 hours)

        Returns:
            Number of snapshots removed
        """
        current_time = time()
        removed_count = 0

        try:
            for filename in os.listdir(self.snapshot_dir):
                if not filename.startswith("cache_snapshot_") or not filename.endswith(".json"):
                    continue

                filepath = os.path.join(self.snapshot_dir, filename)

                try:
                    # Extract timestamp from filename
                    timestamp_str = filename.replace("cache_snapshot_", "").replace(".json", "")
                    created_at = int(timestamp_str)

                    # Check if snapshot is too old
                    if current_time - created_at > max_age_seconds:
                        os.remove(filepath)
                        removed_count += 1
                        logger.info(f"Removed old snapshot: {filepath}")

                except (ValueError, OSError) as e:
                    logger.warning(f"Failed to process snapshot file {filepath}: {e}")
                    continue

        except OSError as e:
            logger.error(f"Failed to cleanup snapshot directory {self.snapshot_dir}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old snapshots")

        return removed_count
