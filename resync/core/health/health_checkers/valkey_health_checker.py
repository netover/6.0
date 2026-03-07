# pylint
"""
Valkey Health Checker

This module provides health checking functionality for Valkey cache connections.
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

class ValkeyHealthChecker(BaseHealthChecker):
    """
    Health checker for Valkey cache connectivity and performance.
    """

    @property
    def component_name(self) -> str:
        return "valkey"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.VALKEY

    async def check_health(self) -> ComponentHealth:
        """
        Check Valkey health and connectivity.

        Returns:
            ComponentHealth: Valkey health status
        """
        start_time = time.time()

        try:
            # Check Valkey configuration
            from resync.settings import settings

            if not settings.valkey_url:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="Valkey URL not configured",
                    last_check=datetime.now(timezone.utc),
                )

            # Test actual Valkey connectivity
            from resync.core.valkey_init import get_valkey_client, is_valkey_available
            try:
                from valkey.exceptions import ValkeyError
                from valkey.exceptions import TimeoutError as ValkeyTimeoutError
            except ImportError:
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    message="valkey-py not installed",
                    last_check=datetime.now(timezone.utc),
                )

            try:
                if not is_valkey_available():
                    return ComponentHealth(
                        name=self.component_name,
                        component_type=self.component_type,
                        status=HealthStatus.UNKNOWN,
                        message="Valkey/Valkey library not available",
                        last_check=datetime.now(timezone.utc),
                    )
                # Use shared connection pool (prevents connection churn)
                valkey_client = get_valkey_client()

                # Test connectivity with ping
                await valkey_client.ping()

                # Test read/write operation
                test_key = f"health_check_{int(time.time())}"
                await valkey_client.setex(test_key, 1, "test")  # Set with expiration
                value = await valkey_client.get(test_key)

                if value != "test" and value != b"test":
                    raise ValkeyError("Valkey read/write test failed")

                # Get Valkey info for additional details
                valkey_info = await valkey_client.info()

                response_time = (time.time() - start_time) * 1000

                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Valkey connectivity test successful",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    metadata={
                        "valkey_version": valkey_info.get("valkey_version"),
                        "connected_clients": valkey_info.get("connected_clients"),
                        "used_memory": valkey_info.get("used_memory_human"),
                        "uptime_seconds": valkey_info.get("uptime_in_seconds"),
                        "test_key_result": str(value),
                    },
                )
            except (ValkeyError, ValkeyTimeoutError) as e:
                response_time = (time.time() - start_time) * 1000

                logger.error("valkey_connectivity_test_failed", error=str(e))
                return ComponentHealth(
                    name=self.component_name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Valkey connectivity failed: {str(e)}",
                    response_time_ms=response_time,
                    last_check=datetime.now(timezone.utc),
                    error_count=1,
                )
            # No finally block needed as we don't close the shared pool

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            response_time = (time.time() - start_time) * 1000
            logger.error("valkey_health_check_failed", error=str(e))
            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Valkey check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """Determine health status based on Valkey exception type."""
        return ComponentType.VALKEY

    def get_component_config(self) -> dict[str, Any]:
        """Get Valkey-specific configuration."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
        }
