"""
Base Health Checker

This module provides the base class for all health checker implementations.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthCheckConfig,
)

logger = structlog.get_logger(__name__)


class BaseHealthChecker(ABC):
    """
    Base class for all health checker implementations.

    Provides common functionality and defines the interface that all
    health checkers must implement.
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the health checker.

        Args:
            config: Health check configuration
        """
        self.config = config or HealthCheckConfig()
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Name of the component this checker monitors."""

    @property
    @abstractmethod
    def component_type(self) -> ComponentType:
        """Type of the component this checker monitors."""

    @abstractmethod
    async def check_health(self) -> ComponentHealth:
        """
        Perform the health check.

        Returns:
            ComponentHealth: Health status of the component
        """

    async def check_health_with_timeout(
        self, timeout_seconds: float | None = None
    ) -> ComponentHealth:
        """
        Perform health check with timeout protection.

        Args:
            timeout_seconds: Timeout in seconds (uses config default if None)

        Returns:
            ComponentHealth: Health status of the component
        """
        start_time = time.time()

        try:
            health_result = await self.check_health()
            response_time = (time.time() - start_time) * 1000

            # Update response time if not already set
            if health_result.response_time_ms is None:
                health_result.response_time_ms = response_time

            return health_result

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(
                "health_check_failed",
                component=self.component_name,
                error=str(e),
                response_time_ms=response_time,
            )

            return ComponentHealth(
                name=self.component_name,
                component_type=self.component_type,
                status=self._get_status_for_exception(e),
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_count=1,
            )

    def _get_status_for_exception(self, exception: Exception) -> ComponentType:
        """
        Determine health status based on exception type.

        Args:
            exception: The exception that occurred

        Returns:
            Appropriate health status for the exception
        """
        # Default to UNKNOWN for most exceptions
        return ComponentType.UNKNOWN

    def get_component_config(self) -> dict[str, Any]:
        """
        Get configuration specific to this component.

        Returns:
            Dictionary with component-specific configuration
        """
        # This will be implemented by the configuration manager
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "retry_attempts": 2,
        }

    def validate_config(self) -> list[str]:
        """
        Validate configuration for this component.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.config.timeout_seconds <= 0:
            errors.append(f"{self.component_name}: timeout_seconds must be positive")

        return errors
