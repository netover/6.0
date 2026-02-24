# pylint: disable=all
"""
Connection Pools Health Checker

This module provides health checking functionality for connection pools.
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


class ConnectionPoolsHealthChecker(BaseHealthChecker):
    """
    Health checker for connection pools health.
    """

    @property
    def component_name(self) -> str:
        return "connection_pools"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.CONNECTION_POOL

    async def check_health(self) -> ComponentHealth:
        """
        Check connection pools health.

        Returns:
            ComponentHealth: Connection pools health status
        """
        start_time = time.time()

        try:
                        # Use pool manager from pools.pool_manager; connection_manager
            # does not define this API.
            from resync.core.pools.pool_manager import get_connection_pool_manager

            pool_manager = get_connection_pool_manager()
            if not pool_manager:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="Connection pool manager not available",
                    last_check=datetime.now(timezone.utc),
                )

            # Check pool status
            pool_stats = pool_manager.get_pool_stats()

            if not pool_stats:
                response_time = (time.time() - start_time) * 1000
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Connection pools statistics unavailable (empty/null)",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={"pool_stats": "empty or null"},
                )

            # Analyze pool health with safe defaults
            total_connections = pool_stats.get("total_connections", 0)
            active_connections = pool_stats.get("active_connections", 0)

            if total_connections == 0:
                status = HealthStatus.UNHEALTHY
                message = "No database connections available"
            else:
                # Calculate connection usage percentage
                connection_usage_percent = active_connections / total_connections * 100

                # Use database-specific threshold for database pool
                threshold_percent = self.config.database_connection_threshold_percent

                if connection_usage_percent > threshold_percent:
                    status = HealthStatus.DEGRADED
                    message = (
                        "Connection pool near capacity: "
                        f"{active_connections}/{total_connections} "
                        f"({connection_usage_percent:.1f}%, "
                        f"threshold: {threshold_percent}%)"
                    )
                else:
                    status = HealthStatus.HEALTHY
                    message = (
                        "Connection pool healthy: "
                        f"{active_connections}/{total_connections} "
                        f"({connection_usage_percent:.1f}%)"
                    )

            response_time = (time.time() - start_time) * 1000

            # Enhance metadata with calculated percentages and thresholds
            enhanced_metadata = dict(pool_stats)
            if "active_connections" in pool_stats and "total_connections" in pool_stats:
                enhanced_metadata["connection_usage_percent"] = round(
                    connection_usage_percent, 1
                )
                enhanced_metadata["threshold_percent"] = (
                    self.config.database_connection_threshold_percent
                )

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata=enhanced_metadata,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("connection_pools_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Connection pools check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on connection pool exception type."""
        return ComponentType.CONNECTION_POOL

    def get_component_config(self) -> dict[str, Any]:
        """Get connection pools-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 3,
            "connection_threshold_percent": (
                self.config.database_connection_threshold_percent
            ),
        }
