# ruff: noqa: E501
# pylint: disable=all
"""
TWS Monitor Health Checker

This module provides health checking functionality for TWS monitor (external API service).
"""

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)

from .base_health_checker import BaseHealthChecker

logger = structlog.get_logger(__name__)


class TWSMonitorHealthChecker(BaseHealthChecker):
    """
    Health checker for TWS monitor health (external API service).
    """

    @property
    def component_name(self) -> str:
        return "tws_monitor"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.EXTERNAL_API

    async def check_health(self) -> ComponentHealth:
        """
        Check TWS monitor health (external API service).

        Returns:
            ComponentHealth: TWS monitor health status
        """
        start_time = time.time()

        try:
            # Check TWS configuration
            from resync.settings import settings

            tws_config = settings.get("tws_monitor", {})
            if not tws_config or not tws_config.get("enabled", False):
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="TWS monitor not configured",
                    last_check=datetime.now(timezone.utc),
                )

            # Real connectivity test
            tws_url = tws_config.get("url")
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{tws_url}/health")
                    if resp.status_code != 200:
                        raise Exception(f"HTTP {resp.status_code}")
            except Exception:
                # Re-raise to be caught by outer block
                raise

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="TWS monitor connectivity test successful",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata={
                    "tws_enabled": tws_config.get("enabled", False),
                    "tws_url": tws_url,
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("tws_monitor_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"TWS monitor connectivity failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on TWS monitor exception type."""
        return ComponentType.EXTERNAL_API

    def get_component_config(self) -> dict[str, Any]:
        """Get TWS monitor-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 1,
        }
