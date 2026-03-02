# pylint
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
from .common import (
    ErrorContext,
    ThresholdConfig,
    build_error_health,
    response_time_ms,
    threshold_status,
)

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

            status, message = threshold_status(
                value=cpu_percent,
                config=ThresholdConfig(
                    warning=85,
                    critical=95,
                    healthy_msg="CPU usage normal: {value:.1f}%",
                    degraded_msg="CPU usage high: {value:.1f}%",
                    critical_msg="CPU usage critically high: {value:.1f}%",
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
                    "cpu_usage_percent": cpu_percent,
                    "cpu_samples": [round(s, 1) for s in cpu_samples],
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                },
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            return build_error_health(
                ctx=ErrorContext(
                    name=self.component_name,
                    type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="CPU check failed",
                    start_time=start_time,
                    error=e,
                    log_event="cpu_health_check_failed",
                ),
                logger=logger,
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
