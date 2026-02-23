"""
Unified Health Service

This module provides the consolidated health service that combines:
- Health check orchestration (from health_service_orchestrator.py)
- Enhanced health monitoring (from enhanced_health_service.py)
- Component coordination and lifecycle management

This replaces:
- health_service_orchestrator.py (867 lines)
- enhanced_health_service.py (548 lines)

Consolidated into a single, well-organized module (~600 lines).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthCheckConfig,
    HealthCheckResult,
    HealthStatus,
    HealthStatusHistory,
)
from resync.core.task_tracker import track_task

if TYPE_CHECKING:
    from .health_checkers.base_health_checker import BaseHealthChecker

logger = structlog.get_logger(__name__)


class UnifiedHealthService:
    """
    Unified health check service that consolidates orchestration and enhanced monitoring.

    This service provides:
    - Comprehensive health checking across all components
    - Modular checker integration via HealthCheckerFactory
    - Circuit breaker protection for critical components
    - Health history tracking and metrics
    - Proactive monitoring capabilities
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the unified health service.

        Args:
            config: Optional health check configuration
        """
        self.config = config or HealthCheckConfig()
        self.health_history: list[HealthStatusHistory] = []
        self.last_health_check: datetime | None = None

        # Component results cache
        self._component_results: dict[str, ComponentHealth] = {}
        self._lock = asyncio.Lock()

        # Lazy-loaded checkers (from factory)
        self._checkers: dict[str, BaseHealthChecker] | None = None

        # Circuit breakers for critical components
        self._circuit_breakers: dict[str, Any] = {}

        # Monitoring control
        self._monitoring_task: asyncio.Task | None = None
        self._is_monitoring = False

        # Metrics
        self._check_count = 0
        self._total_check_time = 0.0

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start_monitoring(self, tg: asyncio.TaskGroup | None = None, interval_seconds: int | None = None) -> None:
        """
        Start continuous health monitoring.

        Args:
            tg: Optional TaskGroup to run the background task in
            interval_seconds: Check interval (uses config default if None)
        """
        if self._is_monitoring:
            logger.warning("health_monitoring_already_running")
            return

        self._is_monitoring = True
        interval = interval_seconds or self.config.check_interval_seconds

        if tg:
            self._monitoring_task = tg.create_task(
                self._monitoring_loop(interval), name="monitoring_loop"
            )
        else:
            self._monitoring_task = track_task(
                self._monitoring_loop(interval), name="monitoring_loop"
            )

        logger.info("unified_health_monitoring_started", interval=interval, method="task_group" if tg else "track_task")

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring gracefully."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        logger.info("unified_health_monitoring_stopped")

    async def _monitoring_loop(self, interval: int) -> None:
        """
        Continuous monitoring loop.

        Args:
            interval: Seconds between health checks
        """
        while self._is_monitoring:
            try:
                await self.perform_comprehensive_health_check()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    break
                logger.error("health_monitoring_error", error=str(e))
                await asyncio.sleep(interval)

    # =========================================================================
    # Core Health Check Operations
    # =========================================================================

    async def perform_comprehensive_health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check across all components.

        Returns:
            HealthCheckResult with status of all components
        """
        start_time = time.time()
        correlation_id = f"health_{int(start_time * 1000)}"

        logger.debug(
            "starting_comprehensive_health_check", correlation_id=correlation_id
        )

        try:
            # Get all health checkers
            checkers = self._get_health_checkers()

            # Execute all checks in parallel with timeout using TaskGroup
            components: dict[str, ComponentHealth] = {}

            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    try:
                        async with asyncio.TaskGroup() as tg:
                            tasks = {
                                name: tg.create_task(
                                    self._execute_check_with_timeout(checker, name),
                                    name=f"health_check_{name}"
                                )
                                for name, checker in checkers.items()
                            }
                    except* asyncio.CancelledError:
                        raise
                    except* Exception:
                        # Results extracted below from tasks
                        pass

                    # Extração segura de resultados
                    for name, task in tasks.items():
                        if task.done() and not task.cancelled():
                            try:
                                components[name] = task.result()
                            except Exception as e:
                                if isinstance(e, asyncio.CancelledError):
                                    raise
                                components[name] = self._create_error_health(name, str(e))
                        else:
                            components[name] = self._create_error_health(name, "Check interrupted or timed out")
            except TimeoutError:
                logger.warning("comprehensive_health_check_timeout", timeout=self.config.timeout_seconds)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                logger.error("comprehensive_health_check_group_error", error=str(e))

            # Fill in missing results (failsafe)
            for name in checkers:
                if name not in components:
                    components[name] = self._create_error_health(name, "Internal health check orchestration failure")

            # Calculate overall status
            overall_status = self._calculate_overall_status(components)

            # Build result
            result = HealthCheckResult(
                overall_status=overall_status,
                components=components,
                timestamp=datetime.now(UTC),
                duration_ms=(time.time() - start_time) * 1000,
                alerts=self._generate_alerts(components),
                summary=self._generate_summary(components),
            )

            # Update tracking
            async with self._lock:
                self._component_results = components.copy()
                self.last_health_check = result.timestamp
                self._check_count += 1
                self._total_check_time += result.duration_ms

            # Update history
            await self._update_health_history(result)

            logger.info(
                "comprehensive_health_check_completed",
                status=result.status.value,
                duration_ms=result.duration_ms,
                components_checked=len(components),
            )

            return result

        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            logger.error("comprehensive_health_check_failed", error=str(e))
            return HealthCheckResult(
                overall_status=HealthStatus.UNHEALTHY,
                components={},
                timestamp=datetime.now(UTC),
                duration_ms=(time.time() - start_time) * 1000,
                alerts=[f"Health check failed: {e!s}"],
                summary={"error": 1},
            )

    async def _execute_check_with_timeout(
        self, checker: BaseHealthChecker, name: str
    ) -> ComponentHealth:
        """
        Execute a single health check with timeout protection.

        Args:
            checker: The health checker to execute
            name: Component name for logging

        Returns:
            ComponentHealth result
        """
        try:
            return await asyncio.wait_for(
                checker.check_health(),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            logger.warning("health_check_timeout", component=name)
            return self._create_error_health(name, "Check timeout")
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("health_check_error", component=name, error=str(e))
            return self._create_error_health(name, str(e))

    def _get_health_checkers(self) -> dict[str, BaseHealthChecker]:
        """
        Get all registered health checkers (lazy initialization).

        Returns:
            Dictionary of component name to checker
        """
        if self._checkers is None:
            from .health_checkers.health_checker_factory import HealthCheckerFactory

            factory = HealthCheckerFactory()
            self._checkers = factory.get_all_health_checkers()

        return self._checkers

    # =========================================================================
    # Status Calculation
    # =========================================================================

    def _calculate_overall_status(
        self, components: dict[str, ComponentHealth]
    ) -> HealthStatus:
        """
        Calculate overall health status from component results.

        Args:
            components: Dictionary of component health results

        Returns:
            Overall HealthStatus
        """
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components.values()]

        # Any critical component unhealthy = overall unhealthy
        critical_components = {"database", "redis"}
        for name, health in components.items():
            if name in critical_components and health.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY

        # Count by status
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)

        # Determine overall status
        if unhealthy_count > len(components) * 0.5:
            return HealthStatus.UNHEALTHY
        if unhealthy_count > 0 or degraded_count > len(components) * 0.3:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _generate_summary(
        self, components: dict[str, ComponentHealth]
    ) -> dict[str, int]:
        """Generate summary counts by status."""
        summary = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
        for health in components.values():
            key = health.status.value.lower()
            summary[key] = summary.get(key, 0) + 1
        return summary

    def _generate_alerts(self, components: dict[str, ComponentHealth]) -> list[str]:
        """Generate alerts for unhealthy/degraded components."""
        alerts = []
        for name, health in components.items():
            if health.status == HealthStatus.UNHEALTHY:
                alerts.append(f"CRITICAL: {name} is unhealthy - {health.message}")
            elif health.status == HealthStatus.DEGRADED:
                alerts.append(f"WARNING: {name} is degraded - {health.message}")
        return alerts

    def _create_error_health(self, name: str, error: str) -> ComponentHealth:
        """Create an error ComponentHealth result."""
        return ComponentHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=error,
            response_time_ms=0,
            component_type=ComponentType.SERVICE,
            metadata={"error": error},
        )

    # =========================================================================
    # History Management
    # =========================================================================

    async def _update_health_history(self, result: HealthCheckResult) -> None:
        """
        Update health history with new result.

        Args:
            result: Health check result to record
        """
        # Extract individual status changes
        component_changes = {
            name: health.status for name, health in result.components.items()
        }

        history_entry = HealthStatusHistory(
            timestamp=result.timestamp,
            overall_status=result.overall_status,
            component_changes=component_changes,
            component_count=len(result.components),
            duration_ms=result.duration_ms,
        )

        async with self._lock:
            self.health_history.append(history_entry)

            # Limit history size
            max_history = getattr(self.config, "max_history_entries", 1000)
            if len(self.health_history) > max_history:
                self.health_history = self.health_history[-max_history:]

    def get_health_history(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[HealthStatusHistory]:
        """
        Get health check history.

        Args:
            limit: Maximum entries to return
            since: Only return entries after this time

        Returns:
            List of health status history entries
        """
        history = self.health_history

        if since:
            history = [h for h in history if h.timestamp >= since]

        return history[-limit:]

    # =========================================================================
    # Component Access
    # =========================================================================

    async def get_component_health(self, component_name: str) -> ComponentHealth | None:
        """
        Get latest health status for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            ComponentHealth or None if not found
        """
        async with self._lock:
            return self._component_results.get(component_name)

    async def get_all_component_health(self) -> dict[str, ComponentHealth]:
        """
        Get latest health status for all components.

        Returns:
            Dictionary of component name to health
        """
        async with self._lock:
            return self._component_results.copy()

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """
        Get health service metrics.

        Returns:
            Dictionary of metrics
        """
        avg_duration = (
            self._total_check_time / self._check_count if self._check_count > 0 else 0
        )

        return {
            "check_count": self._check_count,
            "total_check_time_ms": self._total_check_time,
            "average_check_time_ms": avg_duration,
            "is_monitoring": self._is_monitoring,
            "last_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
            "history_size": len(self.health_history),
            "components_tracked": len(self._component_results),
        }


# =============================================================================
# Module-level singleton management
# =============================================================================

_unified_health_service: UnifiedHealthService | None = None
_service_lock: asyncio.Lock | None = None
_service_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_service_lock() -> asyncio.Lock:
    """Lazy-initialize service lock bound to current running loop (gunicorn --preload safe)."""
    global _service_lock, _service_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # In very early import time, no loop exists; defer creation.
        return asyncio.Lock()
    if _service_lock is None or _service_lock_loop is not loop:
        _service_lock = asyncio.Lock()
        _service_lock_loop = loop
    return _service_lock


async def get_unified_health_service() -> UnifiedHealthService:
    """
    Get or create the singleton UnifiedHealthService instance.

    Returns:
        UnifiedHealthService instance
    """
    global _unified_health_service

    if _unified_health_service is None:
        async with _get_service_lock():
            if _unified_health_service is None:
                _unified_health_service = UnifiedHealthService()

    return _unified_health_service


async def shutdown_unified_health_service() -> None:
    """Shutdown the singleton UnifiedHealthService instance."""
    global _unified_health_service

    if _unified_health_service is not None:
        await _unified_health_service.stop_monitoring()
        _unified_health_service = None


async def get_health_status() -> HealthCheckResult:
    """
    Convenience function to get a comprehensive health check result.

    Returns:
        HealthCheckResult snapshot
    """
    service = await get_unified_health_service()
    return await service.perform_comprehensive_health_check()
