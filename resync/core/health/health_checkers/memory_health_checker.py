# pylint: disable=all
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
from .common import (
    ErrorContext,
    ThresholdConfig,
    build_error_health,
    response_time_ms,
    threshold_status,
)

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

            status, message = threshold_status(
                value=memory_usage_percent,
                config=ThresholdConfig(
                    warning=85,
                    critical=95,
                    healthy_msg="Memory usage normal: {value:.1f}%",
                    degraded_msg="Memory usage high: {value:.1f}%",
                    critical_msg="Memory usage critically high: {value:.1f}%",
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
                    "memory_usage_percent": memory_usage_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                },
            )

        except Exception as e:
            return build_error_health(
                ctx=ErrorContext(
                    name=self.component_name,
                    type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="Memory check failed",
                    start_time=start_time,
                    error=e,
                    log_event="memory_health_check_failed",
                ),
                logger=logger,
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
