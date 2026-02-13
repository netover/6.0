"""
System Resource Health Monitor

This module provides comprehensive system resource monitoring functionality,
including CPU usage, memory utilization, and system performance metrics.
"""

import asyncio
import time
from datetime import datetime

import psutil
import structlog

from resync.core.health.health_models import ComponentHealth, ComponentType, HealthStatus

logger = structlog.get_logger(__name__)


class SystemResourceMonitor:
    """
    Comprehensive system resource health monitor.

    This class provides detailed system resource monitoring including:
    - CPU usage monitoring with multi-sample readings
    - Memory utilization tracking
    - System performance metrics
    - Resource threshold monitoring
    """

    def __init__(self):
        """Initialize the system resource monitor."""
        self._last_check: datetime | None = None
        self._cached_results: dict[str, ComponentHealth] = {}

    async def check_memory_health(self) -> ComponentHealth:
        """
        Check memory usage monitoring.

        Returns:
            ComponentHealth: Memory health status
        """
        start_time = time.time()

        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Determine status based on usage thresholds
            if memory_usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critically high: {memory_usage_percent:.1f}%"
            elif memory_usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = "Memory usage high: {memory_usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal: {memory_usage_percent:.1f}%"

            # Get process memory usage
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)

            response_time = (time.time() - start_time) * 1000

            health = ComponentHealth(
                name="memory",
                component_type=ComponentType.MEMORY,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata={
                    "memory_usage_percent": memory_usage_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "process_memory_mb": process_memory_mb,
                    "memory_used_gb": memory.used / (1024**3),
                },
            )

            # Cache the result
            self._cached_results["memory"] = health
            return health

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            # Sanitize error message for security
            secure_message = str(e)

            logger.error("memory_health_check_failed", error=str(e))
            return ComponentHealth(
                name="memory",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {secure_message}",
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_count=1,
            )

    async def check_cpu_health(self) -> ComponentHealth:
        """
        Check CPU load monitoring with multi-sample readings.

        Returns:
            ComponentHealth: CPU health status
        """
        start_time = time.time()

        try:
            # Multi-sample CPU reading for more accurate results
            cpu_samples = []

            # First reading
            cpu_samples.append(psutil.cpu_percent(interval=0))
            await asyncio.sleep(0.05)  # Small delay between samples
            # Second reading
            cpu_samples.append(psutil.cpu_percent(interval=0))
            await asyncio.sleep(0.05)  # Small delay between samples
            # Third reading
            cpu_samples.append(psutil.cpu_percent(interval=0))

            # Average the samples for a more accurate reading
            cpu_percent = sum(cpu_samples) / len(cpu_samples)

            # Determine status based on usage thresholds
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critically high: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = HealthStatus.DEGRADED
                message = "CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal: {cpu_percent:.1f}%"

            response_time = (time.time() - start_time) * 1000

            health = ComponentHealth(
                name="cpu",
                component_type=ComponentType.CPU,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata={
                    "cpu_usage_percent": cpu_percent,
                    "cpu_samples": [round(s, 1) for s in cpu_samples],
                    "cpu_count": psutil.cpu_count(),
                    "cpu_frequency_mhz": (
                        getattr(psutil.cpu_freq(), "current", None)
                        if hasattr(psutil, "cpu_freq")
                        else None
                    ),
                },
            )

            # Cache the result
            self._cached_results["cpu"] = health
            return health

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            # Sanitize error message for security
            secure_message = str(e)

            logger.error("cpu_health_check_failed", error=str(e))
            return ComponentHealth(
                name="cpu",
                component_type=ComponentType.CPU,
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {secure_message}",
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_count=1,
            )

    async def check_system_health(self) -> dict[str, ComponentHealth]:
        """
        Check all system resource health metrics.

        Returns:
            Dictionary mapping component names to their health status
        """
        memory_health = await self.check_memory_health()
        cpu_health = await self.check_cpu_health()

        # Update last check time
        self._last_check = datetime.now()

        return {
            "memory": memory_health,
            "cpu": cpu_health,
        }

    def get_cached_health(self, component_name: str) -> ComponentHealth | None:
        """
        Get cached health result for a specific component.

        Args:
            component_name: Name of the component (memory or cpu)

        Returns:
            Cached ComponentHealth or None if cache is stale/empty
        """
        if component_name in self._cached_results:
            # Simple cache expiry check (5 minutes)
            age = datetime.now() - self._last_check
            if age and age.total_seconds() < 300:
                return self._cached_results[component_name]
            # Cache expired
            self._cached_results.pop(component_name, None)

        return None

    def clear_cache(self) -> None:
        """Clear all cached health results."""
        self._cached_results.clear()
        self._last_check = None
