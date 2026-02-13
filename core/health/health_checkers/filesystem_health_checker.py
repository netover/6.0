"""
File System Health Checker

This module provides health checking functionality for file system and disk space.
"""

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)

from .base_health_checker import BaseHealthChecker

logger = structlog.get_logger(__name__)


class FileSystemHealthChecker(BaseHealthChecker):
    """
    Health checker for file system health and disk space monitoring.
    """

    @property
    def component_name(self) -> str:
        return "file_system"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.FILE_SYSTEM

    async def check_health(self) -> ComponentHealth:
        """
        Check file system health and disk space monitoring.

        Returns:
            ComponentHealth: File system health status
        """
        start_time = time.time()

        try:
            # Check disk space
            import psutil

            disk_usage = psutil.disk_usage("/")
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            # Determine status
            if disk_usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critically low: {disk_usage_percent:.1f}% used"
            elif disk_usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = "Disk space getting low: {disk_usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space OK: {disk_usage_percent:.1f}% used"

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata={
                    "disk_usage_percent": disk_usage_percent,
                    "disk_free_gb": disk_usage.free / (1024**3),
                    "disk_used_gb": disk_usage.used / (1024**3),
                    "disk_total_gb": disk_usage.total / (1024**3),
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("file_system_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message=f"File system check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on filesystem exception type."""
        return ComponentType.FILE_SYSTEM

    def get_component_config(self) -> dict[str, Any]:
        """Get filesystem-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
            "disk_space_warning_percent": 85,
            "disk_space_critical_percent": 95,
        }
