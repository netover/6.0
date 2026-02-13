"""
Redis Health Checker

This module provides health checking functionality for Redis cache connections.
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


class RedisHealthChecker(BaseHealthChecker):
    """
    Health checker for Redis cache connectivity and performance.
    """

    @property
    def component_name(self) -> str:
        return "redis"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.REDIS

    async def check_health(self) -> ComponentHealth:
        """
        Check Redis health and connectivity.

        Returns:
            ComponentHealth: Redis health status
        """
        start_time = time.time()

        try:
            # Check Redis configuration
            from resync.settings import settings

            if not settings.REDIS_URL:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="Redis URL not configured",
                    last_check=datetime.now(timezone.utc),
                )

            # Test actual Redis connectivity
            from resync.core.redis_init import get_redis_client
            from redis.exceptions import RedisError
            from redis.exceptions import TimeoutError as RedisTimeoutError

            try:
                # Use shared connection pool (prevents connection churn)
                redis_client = get_redis_client()
                
                # Test connectivity with ping
                await redis_client.ping()

                # Test read/write operation
                test_key = f"health_check_{int(time.time())}"
                await redis_client.setex(test_key, 1, "test")  # Set with expiration
                value = await redis_client.get(test_key)

                if value != "test" and value != b"test":
                    raise RedisError("Redis read/write test failed")

                # Get Redis info for additional details
                redis_info = await redis_client.info()

                response_time = (time.time() - start_time) * 1000

                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Redis connectivity test successful",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={
                        "redis_version": redis_info.get("redis_version"),
                        "connected_clients": redis_info.get("connected_clients"),
                        "used_memory": redis_info.get("used_memory_human"),
                        "uptime_seconds": redis_info.get("uptime_in_seconds"),
                        "test_key_result": str(value),
                    },
                )
            except (RedisError, RedisTimeoutError) as e:
                response_time = (time.time() - start_time) * 1000

                logger.error("redis_connectivity_test_failed", error=str(e))
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Redis connectivity failed: {str(e)}",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    error_count=1,
                )
            # No finally block needed as we don't close the shared pool

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error("redis_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on Redis exception type."""
        return ComponentType.REDIS

    def get_component_config(self) -> dict[str, Any]:
        """Get Redis-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
        }
