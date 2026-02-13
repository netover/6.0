"""
Health Monitoring Coordinator

This module provides continuous health monitoring coordination and lifecycle management
for the health check service, including monitoring loops and service lifecycle management.
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
import contextlib

import structlog

from resync.core.health.health_models import HealthCheckConfig

logger = structlog.get_logger(__name__)


class HealthMonitoringCoordinator:
    """
    Coordinates continuous health monitoring and service lifecycle management.

    This class manages the continuous monitoring loop, service startup/shutdown,
    and integration with the broader health monitoring ecosystem.
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the health monitoring coordinator.

        Args:
            config: Health check configuration (uses default if None)
        """
        self.config = config or HealthCheckConfig()
        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False
        self._lock = asyncio.Lock()

    async def start_monitoring(self, health_check_func) -> None:
        """
        Start continuous health monitoring.

        Args:
            health_check_func: Async function to call for health checks
        """
        async with self._lock:
            if self._is_monitoring:
                return

            self._is_monitoring = True
            self._monitoring_task = await create_tracked_task(self._monitoring_loop(health_check_func), name="monitoring_loop")
            logger.info("health_monitoring_coordinator_started")

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        async with self._lock:
            self._is_monitoring = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._monitoring_task
                self._monitoring_task = None
            logger.info("health_monitoring_coordinator_stopped")

    async def _monitoring_loop(self, health_check_func) -> None:
        """Continuous monitoring loop with error handling."""
        while self._is_monitoring:
            try:
                await health_check_func()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("error_in_health_monitoring_loop", error=str(e))
                await asyncio.sleep(10)  # Brief pause on error

    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active."""
        return self._is_monitoring
