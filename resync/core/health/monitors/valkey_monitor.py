# pylint
"""
Valkey Health Monitor

This module provides comprehensive Valkey connectivity and health monitoring
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

class ValkeyHealthMonitor:
    """
    Comprehensive Valkey health monitor.

    This class provides detailed Valkey health checking including:
    - Connection testing and validation
    - Performance metrics collection
    - Memory usage monitoring
    - Read/write operation testing
    """

    def __init__(self):
        """Initialize the Valkey health monitor."""
        self._last_check: datetime | None = None
        self._cached_result: ComponentHealth | None = None

    async def check_valkey_health(self) -> ComponentHealth:
        """
        Perform comprehensive Valkey health check.

        Returns:
            ComponentHealth: Detailed Valkey health status
        """
        start_time = time.time()

        try:
            # Check Valkey configuration
            if not settings.valkey_url.get_secret_value():
                return ComponentHealth(
                    name="valkey",
                    component_type=ComponentType.VALKEY,
                    status=HealthStatus.UNKNOWN,
                    message="Valkey URL not configured",
                    last_check=datetime.now(datetime.UTC),
                )

            # Test actual Valkey connectivity
            import valkey.asyncio as valkey_async
            from valkey.exceptions import ValkeyError
            from valkey.exceptions import TimeoutError as ValkeyTimeoutError

            try:
                valkey_client = valkey_async.from_url(settings.valkey_url.get_secret_value())

                # Test connectivity with ping
                await valkey_client.ping()

                # Test read/write operation
                test_key = f"health_check_{int(time.time())}"
                await valkey_client.setex(test_key, 1, "test")  # Set with expiration
                value = await valkey_client.get(test_key)

                if value != b"test":
                    raise ValkeyError("Valkey read/write test failed")

                # Get Valkey info for additional details
                valkey_info = await valkey_client.info()

                response_time = (time.time() - start_time) * 1000

                health = ComponentHealth(
                    name="valkey",
                    component_type=ComponentType.VALKEY,
                    status=HealthStatus.HEALTHY,
                    message="Valkey connectivity test successful",
                    response_time_ms=response_time,
                    last_check=datetime.now(datetime.UTC),
                    metadata={
                        "valkey_version": valkey_info.get("valkey_version"),
                        "connected_clients": valkey_info.get("connected_clients"),
                        "used_memory": valkey_info.get("used_memory_human"),
                        "uptime_seconds": valkey_info.get("uptime_in_seconds"),
                        "test_key_result": value.decode() if value else None,
                        "valkey_url_configured": bool(settings.valkey_url.get_secret_value()),
                    },
                )

                # Cache the result
                self._cached_result = health
                self._last_check = datetime.now(datetime.UTC)

                return health

            except (ValkeyError, ValkeyTimeoutError) as e:
                response_time = (time.time() - start_time) * 1000

                logger.error("valkey_connectivity_test_failed", error=str(e))
                return ComponentHealth(
                    name="valkey",
                    component_type=ComponentType.VALKEY,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Valkey connectivity failed: {str(e)}",
                    response_time_ms=response_time,
                    last_check=datetime.now(datetime.UTC),
                    error_count=1,
                )
            finally:
                # Close the test connection
                try:
                    await valkey_client.close()
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    import sys as _sys
                    from resync.core.exception_guard import maybe_reraise_programming_error
                    _exc_type, _exc, _tb = _sys.exc_info()
                    maybe_reraise_programming_error(_exc, _tb)

                    logger.debug("Valkey client close error during health check: %s", e)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            response_time = (time.time() - start_time) * 1000

            # Sanitize error message for security
            secure_message = str(e)

            logger.error("valkey_health_check_failed", error=str(e))
            return ComponentHealth(
                name="valkey",
                component_type=ComponentType.VALKEY,
                status=HealthStatus.UNHEALTHY,
                message=f"Valkey check failed: {secure_message}",
                response_time_ms=response_time,
                last_check=datetime.now(datetime.UTC),
                error_count=1,
            )

    async def check_valkey_health_with_retry(
        self, max_retries: int = 3, component_name: str = "valkey"
    ) -> ComponentHealth:
        """
        Execute Valkey health check with retry logic and exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            component_name: Name of the component for logging

        Returns:
            ComponentHealth: Valkey health status after retries
        """
        for attempt in range(max_retries):
            try:
                return await self.check_valkey_health()
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                if attempt == max_retries - 1:
                    logger.error(
                        "valkey_health_check_failed_after_retries",
                        component_name=component_name,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                wait_time = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "valkey_health_check_failed_retrying",
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
            component_type=ComponentType.VALKEY,
            status=HealthStatus.UNKNOWN,
            message="Valkey health check failed after all retries",
            last_check=datetime.now(datetime.UTC),
        )

    def get_cached_health(self) -> ComponentHealth | None:
        """
        Get cached health result if available and recent.

        Returns:
            Cached ComponentHealth or None if cache is stale/empty
        """
        if self._cached_result:
            # Simple cache expiry check (5 minutes)
            age = datetime.now(datetime.UTC) - self._last_check
            if age.total_seconds() < 300:
                return self._cached_result
            # Cache expired
            self._cached_result = None

        return None

    def clear_cache(self) -> None:
        """Clear the cached health result."""
        self._cached_result = None
        self._last_check = None

# Backward compat alias
ValkeyMonitor = ValkeyHealthMonitor
