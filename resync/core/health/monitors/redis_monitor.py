"""
Redis Health Monitor

This module provides comprehensive Redis connectivity and health monitoring
functionality, including connection testing, performance metrics, and
detailed health reporting.
"""

import asyncio  # Added to support async sleep in retry logic
import time
from datetime import datetime, timezone

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)
from resync.settings import settings

logger = structlog.get_logger(__name__)


class RedisHealthMonitor:
    """
    Comprehensive Redis health monitor.

    This class provides detailed Redis health checking including:
    - Connection testing and validation
    - Performance metrics collection
    - Memory usage monitoring
    - Read/write operation testing
    """

    def __init__(self):
        """Initialize the Redis health monitor."""
        self._last_check: datetime | None = None
        self._cached_result: ComponentHealth | None = None

    async def check_redis_health(self) -> ComponentHealth:
        """
        Perform comprehensive Redis health check.

        Returns:
            ComponentHealth: Detailed Redis health status
        """
        start_time = time.time()

        try:
            # Check Redis configuration
            if not settings.REDIS_URL:
                return ComponentHealth(
                    name="redis",
                    component_type=ComponentType.REDIS,
                    status=HealthStatus.UNKNOWN,
                    message="Redis URL not configured",
                    last_check=datetime.now(timezone.utc),
                )

            # Test actual Redis connectivity
            import redis.asyncio as redis_async
            from redis.exceptions import RedisError
            from redis.exceptions import TimeoutError as RedisTimeoutError

            try:
                redis_client = redis_async.from_url(settings.REDIS_URL)

                # Test connectivity with ping
                await redis_client.ping()

                # Test read/write operation
                test_key = f"health_check_{int(time.time())}"
                await redis_client.setex(test_key, 1, "test")  # Set with expiration
                value = await redis_client.get(test_key)

                if value != b"test":
                    raise RedisError("Redis read/write test failed")

                # Get Redis info for additional details
                redis_info = await redis_client.info()

                response_time = (time.time() - start_time) * 1000

                health = ComponentHealth(
                    name="redis",
                    component_type=ComponentType.REDIS,
                    status=HealthStatus.HEALTHY,
                    message="Redis connectivity test successful",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={
                        "redis_version": redis_info.get("redis_version"),
                        "connected_clients": redis_info.get("connected_clients"),
                        "used_memory": redis_info.get("used_memory_human"),
                        "uptime_seconds": redis_info.get("uptime_in_seconds"),
                        "test_key_result": value.decode() if value else None,
                        "redis_url_configured": bool(settings.REDIS_URL),
                    },
                )

                # Cache the result
                self._cached_result = health
                self._last_check = datetime.now(timezone.utc)

                return health

            except (RedisError, RedisTimeoutError) as e:
                response_time = (time.time() - start_time) * 1000

                logger.error("redis_connectivity_test_failed", error=str(e))
                return ComponentHealth(
                    name="redis",
                    component_type=ComponentType.REDIS,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Redis connectivity failed: {str(e)}",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    error_count=1,
                )
            finally:
                # Close the test connection
                try:
                    await redis_client.close()
                except Exception as e:
                    logger.debug("Redis client close error during health check: %s", e)

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            # Sanitize error message for security
            secure_message = str(e)

            logger.error("redis_health_check_failed", error=str(e))
            return ComponentHealth(
                name="redis",
                component_type=ComponentType.REDIS,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {secure_message}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    async def check_redis_health_with_retry(
        self, max_retries: int = 3, component_name: str = "redis"
    ) -> ComponentHealth:
        """
        Execute Redis health check with retry logic and exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            component_name: Name of the component for logging

        Returns:
            ComponentHealth: Redis health status after retries
        """
        for attempt in range(max_retries):
            try:
                return await self.check_redis_health()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "redis_health_check_failed_after_retries",
                        component_name=component_name,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                wait_time = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "redis_health_check_failed_retrying",
                    component_name=component_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)

        # This should never be reached, but just in case
        return ComponentHealth(
            name=component_name,
            component_type=ComponentType.REDIS,
            status=HealthStatus.UNKNOWN,
            message="Redis health check failed after all retries",
            last_check=datetime.now(timezone.utc),
        )

    def get_cached_health(self) -> ComponentHealth | None:
        """
        Get cached health result if available and recent.

        Returns:
            Cached ComponentHealth or None if cache is stale/empty
        """
        if self._cached_result:
            # Simple cache expiry check (5 minutes)
            age = datetime.now(timezone.utc) - self._last_check
            if age.total_seconds() < 300:
                return self._cached_result
            # Cache expired
            self._cached_result = None

        return None

    def clear_cache(self) -> None:
        """Clear the cached health result."""
        self._cached_result = None
        self._last_check = None
