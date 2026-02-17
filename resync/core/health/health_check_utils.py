"""
Health Check Utilities

This module provides shared utility functions and helpers for health check
operations, including retry logic, performance metrics, and common health
check patterns.
"""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

from resync.core.health.health_models import ComponentHealth, ComponentType, HealthStatus

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class HealthCheckUtils:
    """
    Utility class providing common health check functionality.

    This class contains reusable methods for:
    - Retry logic with exponential backoff
    - Performance metrics calculation
    - Common health check patterns
    - Result aggregation and analysis
    """

    @staticmethod
    async def execute_with_retry(
        func: Callable[[], Any],
        max_retries: int = 3,
        component_name: str = "unknown",
        base_delay: float = 1.0,
    ) -> Any:
        """
        Execute a function with retry logic and exponential backoff.

        Args:
            func: The async function to execute
            max_retries: Maximum number of retry attempts
            component_name: Name of the component for logging
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            The function result

        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "function_execution_failed_after_retries",
                        component_name=component_name,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                wait_time = base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    "function_execution_failed_retrying",
                    component_name=component_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)
        return None

    @staticmethod
    def calculate_overall_status(
        components: dict[str, ComponentHealth],
    ) -> HealthStatus:
        """
        Calculate overall health status from component health results.

        Args:
            components: Dictionary of component health results

        Returns:
            Overall health status (worst status wins)
        """
        # Priority order for health status (higher number = worse)
        priority = {
            HealthStatus.UNHEALTHY: 3,
            HealthStatus.DEGRADED: 2,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.HEALTHY: 0,
        }

        worst_status = HealthStatus.HEALTHY
        for component in components.values():
            if priority[component.status] > priority[worst_status]:
                worst_status = component.status

        return worst_status

    @staticmethod
    def generate_summary(components: dict[str, ComponentHealth]) -> dict[str, int]:
        """
        Generate summary statistics from component health results.

        Args:
            components: Dictionary of component health results

        Returns:
            Dictionary with health status counts
        """
        summary: dict[str, int] = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "total": len(components),
        }

        for component in components.values():
            if component.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif component.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif component.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            else:
                summary["unknown"] += 1

        return summary

    @staticmethod
    def check_alerts(
        components: dict[str, ComponentHealth],
        thresholds: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Check for alert conditions in component health results.

        Args:
            components: Dictionary of component health results
            thresholds: Optional custom thresholds for alert conditions

        Returns:
            List of alert messages
        """
        alerts: list[str] = []
        default_thresholds = {
            "database_connection_threshold_percent": 80.0,
            "memory_usage_threshold_percent": 85.0,
            "cpu_usage_threshold_percent": 85.0,
            "disk_usage_threshold_percent": 90.0,
        }
        thresholds = thresholds or default_thresholds

        for name, component in components.items():
            if component.status == HealthStatus.UNHEALTHY:
                alerts.append(f"{name} is unhealthy")
            elif component.status == HealthStatus.DEGRADED:
                # Include specific threshold breach information in alerts
                if name == "database" and "connection_usage_percent" in component.metadata:
                    threshold = thresholds.get(
                        "database_connection_threshold_percent",
                        default_thresholds["database_connection_threshold_percent"],
                    )
                    usage = component.metadata["connection_usage_percent"]
                    alerts.append(
                        f"Database connection pool usage at {usage:.1f}% (threshold: {threshold}%)"
                    )
                elif name == "memory" and "memory_usage_percent" in component.metadata:
                    threshold = thresholds.get(
                        "memory_usage_threshold_percent",
                        default_thresholds["memory_usage_threshold_percent"],
                    )
                    usage = component.metadata["memory_usage_percent"]
                    alerts.append(f"Memory usage at {usage:.1f}% (threshold: {threshold}%)")
                elif name == "cpu" and "cpu_usage_percent" in component.metadata:
                    threshold = thresholds.get(
                        "cpu_usage_threshold_percent",
                        default_thresholds["cpu_usage_threshold_percent"],
                    )
                    usage = component.metadata["cpu_usage_percent"]
                    alerts.append(f"CPU usage at {usage:.1f}% (threshold: {threshold}%)")
                else:
                    alerts.append(f"{name} is degraded")

        return alerts

    @staticmethod
    def get_component_type_mapping() -> dict[str, ComponentType]:
        """
        Get mapping of component names to component types.

        Returns:
            Dictionary mapping component names to their types
        """
        return {
            "database": ComponentType.DATABASE,
            "redis": ComponentType.REDIS,
            "cache_hierarchy": ComponentType.CACHE,
            "file_system": ComponentType.FILE_SYSTEM,
            "memory": ComponentType.MEMORY,
            "cpu": ComponentType.CPU,
            "tws_monitor": ComponentType.EXTERNAL_API,
            "connection_pools": ComponentType.CONNECTION_POOL,
            "websocket_pool": ComponentType.CONNECTION_POOL,
        }

    @staticmethod
    def get_component_type(name: str) -> ComponentType:
        """
        Get component type for a given component name.

        Args:
            name: Component name

        Returns:
            Component type
        """
        mapping = HealthCheckUtils.get_component_type_mapping()
        return mapping.get(name, ComponentType.OTHER)

    @staticmethod
    def calculate_performance_metrics(
        components: dict[str, ComponentHealth],
    ) -> dict[str, Any]:
        """
        Calculate performance metrics from component health results.

        Args:
            components: Dictionary of component health results

        Returns:
            Dictionary with performance metrics
        """
        total_response_time = 0.0
        response_times = []
        failed_checks = 0

        for component in components.values():
            if component.response_time_ms is not None:
                total_response_time += component.response_time_ms
                response_times.append(component.response_time_ms)

            if component.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
                failed_checks += 1

        component_count = len(components)
        avg_response_time = total_response_time / component_count if component_count > 0 else 0

        return {
            "total_components": component_count,
            "failed_checks": failed_checks,
            "success_rate": (
                (component_count - failed_checks) / component_count if component_count > 0 else 0
            ),
            "avg_response_time_ms": avg_response_time,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
        }

    @staticmethod
    def format_component_metadata(component: ComponentHealth) -> dict[str, Any]:
        """
        Format component metadata for consistent output.

        Args:
            component: Component health result

        Returns:
            Formatted metadata dictionary
        """
        metadata = {
            "component_type": component.component_type.value,
            "status": component.status.value,
            "response_time_ms": component.response_time_ms,
            "last_check": (component.last_check.isoformat() if component.last_check else None),
            "error_count": component.error_count,
        }

        # Add component-specific metadata
        if component.metadata:
            metadata.update(component.metadata)

        return metadata
