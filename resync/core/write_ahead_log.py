"""
Write-Ahead Logging (WAL) system for cache persistence.

This module implements a write-ahead log that records cache operations before they're
applied to the main cache, ensuring durability and crash recovery for critical data.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Soft import for aiofiles (optional dependency)
try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None  # type: ignore

logger = logging.getLogger(__name__)


class WalOperationType(Enum):
    """Types of operations that can be logged in the WAL."""

    SET = "SET"
    DELETE = "DELETE"
    EXPIRE = "EXPIRE"


@dataclass
class WalEntry:
    """Represents a single entry in the write-ahead log."""

    operation: WalOperationType
    key: str
    value: Any | None = None
    timestamp: float = field(default_factory=time.time)
    ttl: float | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the WAL entry to a dictionary for serialization."""
        result = {
            "operation": self.operation.value,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }
        # Only include checksum if it exists
        if self.checksum is not None:
            result["checksum"] = self.checksum
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WalEntry":
        """Create a WAL entry from a dictionary."""
        operation = WalOperationType(data["operation"])
        entry = cls(
            operation=operation,
            key=data["key"],
            value=data.get("value"),
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl"),
        )
        # Add checksum if present in the data
        if "checksum" in data:
            entry.checksum = data["checksum"]
        return entry

    def calculate_checksum(self) -> str:
        """Calculate a checksum for this entry to ensure data integrity."""
        # Temporarily remove checksum from data to calculate checksum
        temp_checksum = self.checksum
        self.checksum = None
        try:
            data_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        finally:
            # Restore original checksum
            self.checksum = temp_checksum


class WriteAheadLog:
    """Write-Ahead Logging system for cache operations."""

    def __init__(
        self, log_path: str | Path, max_log_size: int = 10 * 1024 * 1024
    ):  # 10MB default
        """
        Initialize the WAL system.

        Args:
            log_path: Path to store the WAL files
            max_log_size: Maximum size of a single WAL file before rotation
        """
        self.log_path = Path(log_path)
        self.max_log_size = max_log_size
        self.log_file = None
        self.current_size = 0
        self.lock = asyncio.Lock()

        # Ensure log directory exists
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Initialize with the current log file
        self.current_log_file_path = self.log_path / f"wal_{int(time.time())}.log"
        self._file_handle = None  # Track the file handle for proper closing

    async def _ensure_log_file_open(self):
        """Ensure the log file is open for writing."""
        # Check if file handle exists and is for the correct file path
        need_new_handle = False
        if self._file_handle is None:
            need_new_handle = True
        else:
            # aiofiles doesn't expose the filename directly, so we'll assume we need to check
            # by tracking the current file path separately
            if not hasattr(self, "_current_file_path"):
                self._current_file_path = self.current_log_file_path
                need_new_handle = True
            elif self._current_file_path != self.current_log_file_path:
                # Different file path, need new handle
                need_new_handle = True
            elif self._file_handle.closed:
                # File is closed, need new handle
                need_new_handle = True

        if need_new_handle:
            # Close current file handle if it exists and is open
            if self._file_handle and not getattr(self._file_handle, "closed", False):
                try:
                    await self._file_handle.close()
                except Exception as e:
                    logger.warning("Error closing WAL file handle: %s", e)

            # Open or create the log file in append mode using aiofiles
            if aiofiles is None:
                raise RuntimeError(
                    "aiofiles is required for async WAL operations but is not installed."
                )
            self._file_handle = await aiofiles.open(
                self.current_log_file_path, mode="a", encoding="utf-8"
            )
            self._current_file_path = self.current_log_file_path
            # Get current file size
            if self.current_log_file_path.exists():
                self.current_size = self.current_log_file_path.stat().st_size
            else:
                self.current_size = 0

    async def _rotate_log_if_needed(self):
        """Rotate the log file if it exceeds the maximum size."""
        # Check if the current size (from the file itself) exceeds the max size
        if self.current_log_file_path.exists():
            actual_size = self.current_log_file_path.stat().st_size
            if actual_size >= self.max_log_size:
                # Close current file
                if self._file_handle and not self._file_handle.closed:
                    try:
                        await self._file_handle.close()
                    except Exception as e:
                        logger.warning(
                            "Error closing WAL file handle during rotation: %s", e
                        )

                # Create new log file with timestamp
                timestamp = int(time.time())
                self.current_log_file_path = self.log_path / f"wal_{timestamp}.log"
                self.current_size = 0
                # New file will be opened on next operation

    async def log_operation(self, entry: WalEntry) -> bool:
        """
        Log an operation to the write-ahead log with fsync for durability.

        Args:
            entry: The WAL entry to log

        Returns:
            True if successfully logged, False otherwise
        """
        async with self.lock:
            try:
                # Check if we need to rotate first to ensure we're writing to the right file
                await self._rotate_log_if_needed()

                # Ensure log file is open (this will open the correct file after rotation if needed)
                await self._ensure_log_file_open()

                # Calculate checksum for data integrity (before adding it to the entry)
                checksum = entry.calculate_checksum()

                # Add checksum to the entry
                entry.checksum = checksum

                # Serialize and write the entry
                serialized_entry = json.dumps(entry.to_dict()) + "\n"

                # Write to the file using aiofiles
                await self._file_handle.write(serialized_entry)
                await self._file_handle.flush()  # Ensure data is written to OS buffer

                # For aiofiles, sync the file handle itself
                # Since aiofiles doesn't have a direct fsync, we'll flush and use os.fsync on the underlying fd
                # aiofiles handles wrap a real file object in ``._file``.
                # Try to fsync the underlying file descriptor for durability.
                fd = None
                if hasattr(self._file_handle, "_file") and hasattr(
                    self._file_handle._file, "fileno"
                ):
                    fd = self._file_handle._file.fileno()
                elif hasattr(self._file_handle, "fileno"):
                    fd = self._file_handle.fileno()
                if fd is not None:
                    os.fsync(fd)

                # Update current size
                self.current_size += len(serialized_entry.encode("utf-8"))

                return True
            except Exception as e:
                logger.error("Failed to log operation to WAL: %s", e)
                return False

    async def read_log(self, log_file_path: str | Path) -> list[WalEntry]:
        """
        Read and parse all entries from a WAL file.

        Args:
            log_file_path: Path to the WAL file to read

        Returns:
            List of WAL entries from the file
        """
        entries = []

        try:
            if aiofiles is None:
                raise RuntimeError(
                    "aiofiles is required for async WAL operations but is not installed."
                )
            async with aiofiles.open(log_file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = WalEntry.from_dict(data)

                        # Verify checksum
                        expected_checksum = entry.calculate_checksum()
                        if entry.checksum != expected_checksum:
                            logger.warning(
                                f"Checksum mismatch at line {line_num} in {log_file_path}"
                            )
                            continue  # Skip corrupted entry

                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse JSON at line {line_num} in {log_file_path}: {e}"
                        )
                    except Exception as e:
                        logger.error(
                            "Error processing line %s in %s: %s",
                            line_num,
                            log_file_path,
                            e,
                        )

        except FileNotFoundError:
            logger.info("WAL file not found: %s", log_file_path)
        except Exception as e:
            logger.error("Error reading WAL file %s: %s", log_file_path, e)

        return entries

    async def replay_log(self, cache: Any) -> int:
        """
        Replay all operations in the WAL to recover cache state.

        Args:
            cache: Cache instance to replay operations on

        Returns:
            Number of operations successfully replayed
        """
        replayed_count = 0
        failed_count = 0

        # Get all WAL files, sorted by creation time
        wal_files = sorted(
            self.log_path.glob("wal_*.log"), key=lambda x: x.stat().st_mtime
        )

        for wal_file in wal_files:
            logger.info("Replaying WAL file: %s", wal_file)
            entries = await self.read_log(wal_file)

            for entry in entries:
                try:
                    # Apply the operation to the cache
                    if entry.operation == WalOperationType.SET:
                        # We need to call the cache's internal set method
                        # The cache should have a method to apply operations without logging again
                        if hasattr(cache, "apply_wal_set"):
                            await cache.apply_wal_set(entry.key, entry.value, entry.ttl)
                        else:
                            # Fallback: try to directly set with TTL if it's an AsyncTTLCache
                            await cache.set(
                                entry.key, entry.value, ttl_override=entry.ttl
                            )
                    elif entry.operation == WalOperationType.DELETE:
                        if hasattr(cache, "apply_wal_delete"):
                            await cache.apply_wal_delete(entry.key)
                        else:
                            await cache.delete(entry.key)
                    elif entry.operation == WalOperationType.EXPIRE:
                        # For expired entries, we just need to ensure they're not in the cache
                        await cache.delete(entry.key)

                    replayed_count += 1
                except Exception as e:
                    logger.error(
                        "Error replaying WAL entry for key %s: %s", entry.key, e
                    )
                    failed_count += 1

        logger.info(
            "Replayed %s operations from WAL, %s failed", replayed_count, failed_count
        )
        return replayed_count

    def cleanup_old_logs(self, retention_hours: int = 24):
        """
        Clean up old WAL files based on retention policy.

        Args:
            retention_hours: Number of hours to retain WAL files
        """
        cutoff_time = time.time() - (retention_hours * 3600)

        for wal_file in self.log_path.glob("wal_*.log"):
            if wal_file.stat().st_mtime < cutoff_time:
                try:
                    wal_file.unlink()
                    logger.info("Removed old WAL file: %s", wal_file)
                except Exception as e:
                    logger.error("Failed to remove old WAL file %s: %s", wal_file, e)

    async def close(self):
        """Close the WAL system and release resources."""
        if self._file_handle and not self._file_handle.closed:
            try:
                await self._file_handle.close()
            except Exception as e:
                logger.error("Error closing WAL file handle: %s", e)
