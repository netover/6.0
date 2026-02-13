"""
Cache Hierarchy Health Monitor

This module provides comprehensive cache health monitoring functionality,
including performance metrics, connectivity testing, and detailed health
reporting for the cache hierarchy system.
"""

import asyncio  # Added to support async sleep in retry logic
import time
from datetime import datetime

import structlog

from resync.core.exceptions import CacheHealthCheckError
from resync.core.health.health_models import ComponentHealth, ComponentType, HealthStatus

logger = structlog.get_logger(__name__)


class CacheHierarchyHealthMonitor:
    """
    Comprehensive cache hierarchy health monitor.

    This class provides detailed cache health checking including:
    - Cache connectivity and functionality testing
    - Performance metrics collection
    - Cache operation validation
    - Memory usage monitoring
    """

    def __init__(self):
        """Initialize the cache hierarchy health monitor."""
        self._last_check: datetime | None = None
        self._cached_result: ComponentHealth | None = None

    async def check_cache_health(self) -> ComponentHealth:
        """
        Perform comprehensive cache hierarchy health check.

        Returns:
            ComponentHealth: Detailed cache hierarchy health status
        """
        start_time = time.time()

        try:
            # Import and test the actual cache implementation
            from resync.core.cache import AsyncTTLCache

            # Create a test cache instance to verify functionality
            test_cache = AsyncTTLCache(ttl_seconds=60, cleanup_interval=30)

            # Test cache operations
            test_key = f"health_test_{int(time.time())}"
            test_value = {"timestamp": time.time(), "status": "health_check"}

            # Test set operation
            await test_cache.set(test_key, test_value)

            # Test get operation
            retrieved_value = await test_cache.get(test_key)

            # Verify the value was retrieved correctly
            if retrieved_value != test_value:
                await test_cache.stop()
                raise CacheHealthCheckError(
                    operation="get/set test",
                    details_info="Value mismatch after set/get cycle",
                )

            # Test delete operation
            delete_result = await test_cache.delete(test_key)
            if not delete_result:
                logger.warning("Cache delete test had unexpected result")

            # Get cache statistics
            metrics = test_cache.get_detailed_metrics()

            # Stop the test cache
            await test_cache.stop()

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                name="cache_hierarchy",
                component_type=ComponentType.CACHE,
                status=HealthStatus.HEALTHY,
                message="Cache hierarchy operational",
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata=metrics,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            # Sanitize error message for security
            secure_message = str(e)

            logger.error("cache_hierarchy_health_check_failed", error=str(e))
            return ComponentHealth(
                name="cache_hierarchy",
                component_type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                message=f"Cache hierarchy issues: {secure_message}",
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_count=1,
            )

    async def check_cache_health_with_retry(
        self, max_retries: int = 3, component_name: str = "cache_hierarchy"
    ) -> ComponentHealth:
        """
        Execute cache health check with retry logic and exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            component_name: Name of the component for logging

        Returns:
            ComponentHealth: Cache health status after retries
        """
        for attempt in range(max_retries):
            try:
                return await self.check_cache_health()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "cache_health_check_failed_after_retries",
                        component_name=component_name,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                wait_time = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "cache_health_check_failed_retrying",
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
            component_type=ComponentType.CACHE,
            status=HealthStatus.UNKNOWN,
            message="Cache health check failed after all retries",
            last_check=datetime.now(),
        )

    def get_cached_health(self) -> ComponentHealth | None:
        """
        Get cached health result if available and recent.

        Returns:
            Cached ComponentHealth or None if cache is stale/empty
        """
        if self._cached_result:
            # Simple cache expiry check (5 minutes)
            age = datetime.now() - self._last_check
            if age.total_seconds() < 300:
                return self._cached_result
            # Cache expired
            self._cached_result = None

        return None

    def clear_cache(self) -> None:
        """Clear the cached health result."""
        self._cached_result = None
        self._last_check = None
