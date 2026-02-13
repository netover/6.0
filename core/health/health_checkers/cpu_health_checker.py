"""
CPU Health Checker

This module provides health checking functionality for CPU load monitoring.
"""

import asyncio
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


class CpuHealthChecker(BaseHealthChecker):
    """
    Health checker for CPU load monitoring.
    """

    @property
    def component_name(self) -> str:
        return "cpu"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.CPU

    async def check_health(self) -> ComponentHealth:
        """
        Check CPU load monitoring.

        Returns:
            ComponentHealth: CPU health status
        """
        start_time = time.time()

        try:
            import psutil

            # Multiple samples for more accurate reading
            cpu_samples = []
            cpu_samples.append(psutil.cpu_percent(interval=0))
            await asyncio.sleep(0.05)
            cpu_samples.append(psutil.cpu_percent(interval=0))
            await asyncio.sleep(0.05)
            cpu_samples.append(psutil.cpu_percent(interval=0))

            cpu_percent = sum(cpu_samples) / len(cpu_samples)

            # Determine status
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critically high: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = HealthStatus.DEGRADED
                message = "CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal: {cpu_percent:.1f}%"

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata={
                    "cpu_usage_percent": cpu_percent,
                    "cpu_samples": [round(s, 1) for s in cpu_samples],
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("cpu_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on CPU exception type."""
        return ComponentType.CPU

    def get_component_config(self) -> dict[str, Any]:
        """Get CPU-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 1,
            "warning_percent": 85,
            "critical_percent": 95,
        }
