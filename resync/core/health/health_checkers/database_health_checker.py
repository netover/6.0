# pylint: skip-file
# mypy: ignore-errors
"""
Database Health Checker

This module provides health checking functionality for database connections.
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


class DatabaseHealthChecker(BaseHealthChecker):
    """
    Health checker for database connections and connection pools.
    """

    @property
    def component_name(self) -> str:
        return "database"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DATABASE

    async def check_health(self) -> ComponentHealth:
        """
        Check database health using connection pools.

        Returns:
            ComponentHealth: Database health status
        """
        start_time = time.time()

        try:
            # Use pool manager from pools.pool_manager (connection_manager does not define this)
            from resync.core.pools.pool_manager import get_connection_pool_manager

            pool_manager = get_connection_pool_manager()
            if not pool_manager:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="Database connection pool not available",
                    last_check=datetime.now(timezone.utc),
                )

            # Test database connectivity
            async with pool_manager.acquire_connection("default") as conn:
                # Simple query to test connection
                if hasattr(conn, "execute"):
                    result = await conn.execute("SELECT 1")
                    if hasattr(result, "fetchone"):
                        await result.fetchone()
                elif hasattr(conn, "cursor"):
                    # SQLite case
                    cursor = await conn.cursor()
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
                    await cursor.close()

            response_time = (time.time() - start_time) * 1000

            # Get real pool statistics from pool manager
            pool_stats = pool_manager.get_pool_stats()

            if not pool_stats:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Database pool statistics unavailable (empty/null)",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={"pool_stats": "empty or null"},
                )

            db_pool_stats = pool_stats.get("database")

            if db_pool_stats is None:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Database pool statistics missing for 'database' pool",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={"database_pool": "missing"},
                )

            # Calculate connection usage percentage
            active_connections = db_pool_stats.active_connections
            total_connections = db_pool_stats.total_connections
            connection_usage_percent = (
                (active_connections / total_connections * 100)
                if total_connections > 0
                else 0.0
            )

            # Determine status based on configurable threshold
            threshold_percent = self.config.database_connection_threshold_percent
            if connection_usage_percent > threshold_percent:
                status = HealthStatus.DEGRADED
                message = "Database connection pool near capacity: {active_connections}/{total_connections} ({connection_usage_percent:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = "Database connection pool healthy: {active_connections}/{total_connections} ({connection_usage_percent:.1f}%)"

            # Use real database pool statistics
            pool_metadata = {
                "active_connections": active_connections,
                "idle_connections": db_pool_stats.idle_connections,
                "total_connections": total_connections,
                "connection_usage_percent": round(connection_usage_percent, 1),
                "threshold_percent": threshold_percent,
                "connection_errors": db_pool_stats.connection_errors,
                "pool_hits": db_pool_stats.pool_hits,
                "pool_misses": db_pool_stats.pool_misses,
                "connection_creations": db_pool_stats.connection_creations,
                "connection_closures": db_pool_stats.connection_closures,
                "waiting_connections": db_pool_stats.waiting_connections,
                "peak_connections": db_pool_stats.peak_connections,
                "average_wait_time": round(db_pool_stats.average_wait_time, 3),
                "last_health_check": (
                    db_pool_stats.last_health_check.isoformat()
                    if db_pool_stats.last_health_check
                    else None
                ),
            }

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata=pool_metadata,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            logger.error("database_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on database exception type."""
        # For database errors, we typically want to mark as UNHEALTHY
        # since database connectivity issues are critical
        return ComponentType.DATABASE

    def get_component_config(self) -> dict[str, Any]:
        """Get database-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 3,
            "connection_threshold_percent": self.config.database_connection_threshold_percent,
        }
