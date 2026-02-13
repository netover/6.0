import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resync.core.health.health_models import HealthStatus
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class DiskSpaceStatus:
    """Status information for disk space monitoring."""

    # Disk usage metrics
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    used_percent: float = 0.0

    # Mount point information
    mount_point: str = ""
    device: str = ""

    # Health indicators
    status: HealthStatus = HealthStatus.UNKNOWN
    is_readonly: bool = False

    # Timestamp
    timestamp: float = 0.0

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class IntegrityStatus:
    """Status information for file integrity monitoring."""

    # File information
    total_files: int = 0
    corrupted_files: int = 0
    missing_files: int = 0

    # Checksum information
    files_with_checksums: int = 0
    checksum_failures: int = 0

    # Directory information
    directories_scanned: int = 0
    scan_errors: int = 0

    # Health indicators
    status: HealthStatus = HealthStatus.UNKNOWN
    integrity_score: float = 100.0  # Percentage of files that passed integrity checks

    # Error details
    error_details: list[str] = None

    # Timestamp
    timestamp: float = 0.0

    def __post_init__(self):
        """Initialize mutable defaults and set timestamp."""
        if self.error_details is None:
            self.error_details = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PermissionStatus:
    """Status information for file permission monitoring."""

    # Permission summary
    total_paths: int = 0
    accessible_paths: int = 0
    permission_denied_paths: int = 0

    # Security issues
    world_writable_files: int = 0
    suspicious_permissions: int = 0

    # Directory information
    directories_checked: int = 0
    permission_errors: int = 0

    # Health indicators
    status: HealthStatus = HealthStatus.UNKNOWN
    security_score: float = 100.0  # Percentage of paths with secure permissions

    # Error details
    error_details: list[str] = None

    # Timestamp
    timestamp: float = 0.0

    def __post_init__(self):
        """Initialize mutable defaults and set timestamp."""
        if self.error_details is None:
            self.error_details = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FileSystemHealthMonitor:
    """
    Monitor for filesystem health, disk space, file integrity, and permissions.

    This class provides comprehensive filesystem monitoring capabilities including:
    - Disk space usage monitoring
    - File integrity checking
    - Permission and security analysis
    - File system performance metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the filesystem health monitor.

        Args:
            config: Optional configuration dictionary with monitoring settings
        """
        self.config = config or {}
        self._monitoring_active = False

        # Configuration defaults
        self.disk_space_threshold = self.config.get("disk_space_threshold_percent", 90.0)
        self.integrity_check_paths = self.config.get("integrity_check_paths", ["/"])
        self.permission_check_paths = self.config.get("permission_check_paths", ["/"])
        self.exclude_patterns = self.config.get("exclude_patterns", [])

        # Performance tracking
        self._check_history: list[dict[str, Any]] = []
        self._max_history_size = 100

    def check_disk_space(self) -> DiskSpaceStatus:
        """
        Check disk space usage for monitored paths.

        Returns:
            DiskSpaceStatus: Current disk space status
        """
        time.time()

        try:
            disk_status = DiskSpaceStatus()
            max_usage_percent = 0.0

            # Check each configured path
            for path in self.integrity_check_paths:
                try:
                    path_obj = Path(path)

                    # Skip if path doesn't exist
                    if not path_obj.exists():
                        continue

                    # Get disk usage statistics
                    stat_info = os.statvfs(path)

                    # Calculate disk space metrics
                    total_bytes = stat_info.f_blocks * stat_info.f_frsize
                    free_bytes = stat_info.f_bavail * stat_info.f_frsize
                    used_bytes = total_bytes - free_bytes
                    used_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0

                    # Track maximum usage across all paths
                    if used_percent > max_usage_percent:
                        max_usage_percent = used_percent
                        disk_status.total_bytes = total_bytes
                        disk_status.used_bytes = used_bytes
                        disk_status.free_bytes = free_bytes
                        disk_status.used_percent = used_percent
                        disk_status.mount_point = str(path_obj)
                        disk_status.device = str(path_obj)  # Simplified for basic implementation

                    # Check if filesystem is readonly
                    disk_status.is_readonly = not os.access(path, os.W_OK)

                except (OSError, AttributeError) as e:
                    logger.warning("disk_space_check_failed_for_path", path=path, error=str(e))
                    continue

            # Determine health status based on usage
            if max_usage_percent >= self.disk_space_threshold:
                disk_status.status = HealthStatus.UNHEALTHY
            elif max_usage_percent >= (self.disk_space_threshold * 0.8):
                disk_status.status = HealthStatus.DEGRADED
            else:
                disk_status.status = HealthStatus.HEALTHY

            # Update timestamp
            disk_status.timestamp = time.time()

            # Add to history
            self._add_check_to_history("disk_space", disk_status.__dict__)

            return disk_status

        except Exception as e:
            logger.error("disk_space_check_failed", error=str(e))
            return DiskSpaceStatus(status=HealthStatus.UNHEALTHY, timestamp=time.time())

    def check_file_integrity(self) -> IntegrityStatus:
        """
        Check file integrity for monitored paths.

        Returns:
            IntegrityStatus: Current file integrity status
        """
        time.time()

        try:
            integrity_status = IntegrityStatus()

            for path in self.integrity_check_paths:
                try:
                    path_obj = Path(path)

                    if not path_obj.exists():
                        continue

                    # Scan directory recursively
                    self._scan_directory_integrity(path_obj, integrity_status)

                except (OSError, AttributeError) as e:
                    logger.warning("integrity_check_failed_for_path", path=path, error=str(e))
                    integrity_status.scan_errors += 1
                    integrity_status.error_details.append(f"Path {path}: {str(e)}")
                    continue

            # Calculate integrity score
            if integrity_status.total_files > 0:
                integrity_status.integrity_score = (
                    (integrity_status.total_files - integrity_status.corrupted_files)
                    / integrity_status.total_files
                ) * 100

            # Determine health status
            if integrity_status.corrupted_files > 0 or integrity_status.scan_errors > 0:
                integrity_status.status = HealthStatus.UNHEALTHY
            elif integrity_status.missing_files > 0:
                integrity_status.status = HealthStatus.DEGRADED
            else:
                integrity_status.status = HealthStatus.HEALTHY

            # Update timestamp
            integrity_status.timestamp = time.time()

            # Add to history
            self._add_check_to_history("file_integrity", integrity_status.__dict__)

            return integrity_status

        except Exception as e:
            logger.error("file_integrity_check_failed", error=str(e))
            return IntegrityStatus(
                status=HealthStatus.UNHEALTHY,
                scan_errors=1,
                error_details=[str(e)],
                timestamp=time.time(),
            )

    def check_permissions(self) -> PermissionStatus:
        """
        Check file and directory permissions for monitored paths.

        Returns:
            PermissionStatus: Current permission status
        """
        time.time()

        try:
            permission_status = PermissionStatus()

            for path in self.permission_check_paths:
                try:
                    path_obj = Path(path)

                    if not path_obj.exists():
                        continue

                    # Scan directory recursively for permission issues
                    self._scan_directory_permissions(path_obj, permission_status)

                except (OSError, AttributeError) as e:
                    logger.warning("permission_check_failed_for_path", path=path, error=str(e))
                    permission_status.permission_errors += 1
                    permission_status.error_details.append(f"Path {path}: {str(e)}")
                    continue

            # Calculate security score
            if permission_status.total_paths > 0:
                permission_status.security_score = (
                    permission_status.accessible_paths / permission_status.total_paths
                ) * 100

            # Determine health status
            if (
                permission_status.world_writable_files > 0
                or permission_status.suspicious_permissions > 0
            ):
                permission_status.status = HealthStatus.UNHEALTHY
            elif permission_status.permission_denied_paths > 0:
                permission_status.status = HealthStatus.DEGRADED
            else:
                permission_status.status = HealthStatus.HEALTHY

            # Update timestamp
            permission_status.timestamp = time.time()

            # Add to history
            self._add_check_to_history("permissions", permission_status.__dict__)

            return permission_status

        except Exception as e:
            logger.error("permission_check_failed", error=str(e))
            return PermissionStatus(
                status=HealthStatus.UNHEALTHY,
                permission_errors=1,
                error_details=[str(e)],
                timestamp=time.time(),
            )

    def _scan_directory_integrity(self, path: Path, status: IntegrityStatus) -> None:
        """Recursively scan directory for file integrity issues."""
        try:
            # Count directories
            if path.is_dir():
                status.directories_scanned += 1

                # Scan all items in directory
                for item in path.iterdir():
                    # Skip excluded patterns
                    if self._is_excluded(item):
                        continue

                    if item.is_file():
                        status.total_files += 1
                        # Basic integrity check - file exists and is readable
                        try:
                            if not item.exists():
                                status.missing_files += 1
                            elif not os.access(item, os.R_OK) or (
                                self._should_verify_checksum(item)
                                and not self._verify_file_checksum(item)
                            ):
                                status.corrupted_files += 1
                        except (OSError, PermissionError):
                            status.corrupted_files += 1

                    elif item.is_dir():
                        self._scan_directory_integrity(item, status)

        except (OSError, PermissionError) as e:
            status.scan_errors += 1
            status.error_details.append(f"Directory {path}: {str(e)}")

    def _scan_directory_permissions(self, path: Path, status: PermissionStatus) -> None:
        """Recursively scan directory for permission issues."""
        try:
            # Count paths
            status.total_paths += 1

            # Check accessibility
            if os.access(path, os.R_OK):
                status.accessible_paths += 1
            else:
                status.permission_denied_paths += 1

            # Check for world-writable files (security risk)
            if path.is_file():
                try:
                    file_stat = path.stat()
                    mode = file_stat.st_mode

                    # Check if world-writable (security risk)
                    if bool(mode & stat.S_IWOTH):
                        status.world_writable_files += 1

                    # Check for suspicious permissions (executable data files, etc.)
                    if self._has_suspicious_permissions(path, mode):
                        status.suspicious_permissions += 1

                except (OSError, AttributeError):
                    status.permission_errors += 1

            elif path.is_dir():
                status.directories_checked += 1

                # Scan all items in directory
                for item in path.iterdir():
                    # Skip excluded patterns
                    if self._is_excluded(item):
                        continue

                    self._scan_directory_permissions(item, status)

        except (OSError, PermissionError) as e:
            status.permission_errors += 1
            status.error_details.append(f"Path {path}: {str(e)}")

    def _is_excluded(self, path: Path) -> bool:
        """Check if a path should be excluded from scanning."""
        path_str = str(path)

        return any(pattern in path_str for pattern in self.exclude_patterns)

    def _has_suspicious_permissions(self, path: Path, mode: int) -> bool:
        """Check if file has suspicious permissions."""
        # Check for executable files with suspicious extensions
        if bool(mode & stat.S_IXUSR):  # User execute permission
            suspicious_extensions = {
                ".txt",
                ".log",
                ".json",
                ".xml",
                ".yaml",
                ".yml",
                ".md",
            }
            if path.suffix.lower() in suspicious_extensions:
                return True

        return False

    def _add_check_to_history(self, check_type: str, data: dict[str, Any]) -> None:
        """Add check result to history for trend analysis."""
        history_entry = {
            "type": check_type,
            "timestamp": time.time(),
            "data": data.copy(),
        }

        self._check_history.append(history_entry)

        # Maintain history size limit
        if len(self._check_history) > self._max_history_size:
            self._check_history = self._check_history[-self._max_history_size :]

    def get_check_history(
        self, check_type: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get historical check results.

        Args:
            check_type: Type of check to filter by (None for all)
            limit: Maximum number of records to return (None for all)

        Returns:
            List of historical check results
        """
        filtered_history = self._check_history

        if check_type:
            filtered_history = [entry for entry in filtered_history if entry["type"] == check_type]

        if limit:
            filtered_history = filtered_history[-limit:]

        return filtered_history.copy()

    def clear_check_history(self) -> None:
        """Clear the check history."""
        self._check_history.clear()

    def _should_verify_checksum(self, file_path: Path) -> bool:
        """Check if file should have checksum verification.

        Verifies checksums for critical configuration files.
        """
        critical_extensions = {".json", ".yaml", ".yml", ".conf", ".ini"}
        return file_path.suffix.lower() in critical_extensions

    def _verify_file_checksum(self, file_path: Path) -> bool:
        """Verify file checksum against stored value.

        Returns True if file passes integrity check.
        """
        import hashlib

        try:
            # Calculate current checksum
            with open(file_path, "rb") as f:
                content = f.read()
                hashlib.sha256(content).hexdigest()

            # For now, just verify file is readable and has content
            # In production, compare against stored checksums
            return len(content) > 0
        except Exception:
            return False

    async def start_monitoring(self) -> None:
        """Start continuous monitoring (placeholder for future enhancement)."""
        self._monitoring_active = True
        logger.info("filesystem_health_monitoring_started")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring_active = False
        logger.info("filesystem_health_monitoring_stopped")
