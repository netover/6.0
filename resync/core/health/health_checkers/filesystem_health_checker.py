"""
File System Health Checker

This module provides health checking functionality for file system and disk space.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any

import psutil
import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)

from .base_health_checker import BaseHealthChecker
from .common import ErrorContext, ThresholdConfig, build_error_health, response_time_ms, threshold_status

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
            disk_usage = psutil.disk_usage("/" if os.name == "posix" else "C:")
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            status, message = threshold_status(
                value=disk_usage_percent,
                config=ThresholdConfig(
                    warning=85,
                    critical=95,
                    healthy_msg="Disk space OK: {value:.1f}% used",
                    degraded_msg="Disk space getting low: {value:.1f}% used",
                    critical_msg="Disk space critically low: {value:.1f}% used",
                ),
            )

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time_ms(start_time),
                last_check=datetime.now(timezone.utc),
                metadata={
                    "disk_usage_percent": disk_usage_percent,
                    "disk_free_gb": disk_usage.free / (1024**3),
                    "disk_used_gb": disk_usage.used / (1024**3),
                    "disk_total_gb": disk_usage.total / (1024**3),
                },
            )

        except Exception as e:
            return build_error_health(
                ctx=ErrorContext(
                    name=self.component_name,
                    type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="File system check failed",
                    start_time=start_time,
                    error=e,
                    log_event="file_system_health_check_failed",
                ),
                logger=logger,
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
