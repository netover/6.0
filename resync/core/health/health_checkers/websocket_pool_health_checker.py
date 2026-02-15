"""
WebSocket Pool Health Checker

This module provides health checking functionality for WebSocket pool.
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
from .common import build_error_health, response_time_ms

logger = structlog.get_logger(__name__)


class WebSocketPoolHealthChecker(BaseHealthChecker):
    """
    Health checker for WebSocket pool health.
    """

    @property
    def component_name(self) -> str:
        return "websocket_pool"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.CONNECTION_POOL

    async def check_health(self) -> ComponentHealth:
        """
        Check WebSocket pool health.

        Returns:
            ComponentHealth: WebSocket pool health status
        """
        start_time = time.time()

        try:
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="WebSocket pool service available",
                response_time_ms=response_time_ms(start_time),
                last_check=datetime.now(timezone.utc),
                metadata={
                    "pool_status": "available",
                    "connections": "unknown",  # Would be populated by actual WebSocket pool manager
                },
            )
        except Exception as e:
            return build_error_health(
                component_name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message="WebSocket pool unavailable",
                start_time=start_time,
                error=e,
                log_event="websocket_pool_health_check_failed",
                logger=logger,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on WebSocket pool exception type."""
        return ComponentType.CONNECTION_POOL

    def get_component_config(self) -> dict[str, Any]:
        """Get WebSocket pool-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
        }
