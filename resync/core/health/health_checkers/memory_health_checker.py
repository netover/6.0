"""
Memory Health Checker

This module provides health checking functionality for memory usage monitoring.
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


class MemoryHealthChecker(BaseHealthChecker):
    """
    Health checker for memory usage monitoring.
    """

    @property
    def component_name(self) -> str:
        return "memory"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.MEMORY

    async def check_health(self) -> ComponentHealth:
        """
        Check memory usage monitoring.

        Returns:
            ComponentHealth: Memory health status
        """
        start_time = time.time()

        try:
            import psutil

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Determine status
            if memory_usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critically high: {memory_usage_percent:.1f}%"
            elif memory_usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = "Memory usage high: {memory_usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal: {memory_usage_percent:.1f}%"

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata={
                    "memory_usage_percent": memory_usage_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("memory_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on memory exception type."""
        return ComponentType.MEMORY

    def get_component_config(self) -> dict[str, Any]:
        """Get memory-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 1,
            "warning_percent": 85,
            "critical_percent": 95,
        }
