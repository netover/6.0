"""
Cache Health Checker

This module provides health checking functionality for cache hierarchy.
"""

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.exceptions import CacheHealthCheckError
from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)

from .base_health_checker import BaseHealthChecker

logger = structlog.get_logger(__name__)

class CacheHealthChecker(BaseHealthChecker):
    """
    Health checker for cache hierarchy functionality.
    """

    @property
    def component_name(self) -> str:
        return "cache_hierarchy"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.CACHE

    async def check_health(self) -> ComponentHealth:
        """
        Check cache hierarchy health.

        Returns:
            ComponentHealth: Cache health status
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
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="Cache hierarchy operational",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                metadata=metrics,
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            response_time = (time.time() - start_time) * 1000
            logger.error("cache_hierarchy_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.DEGRADED,
                message=f"Cache hierarchy issues: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on cache exception type."""
        return ComponentType.CACHE

    def get_component_config(self) -> dict[str, Any]:
        """Get cache-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
        }
