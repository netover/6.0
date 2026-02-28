# ruff: noqa: E501
"""
Health Service Facade

This module provides a unified facade interface for the health service system,
coordinating between different managers, monitors, and observers to provide
a simplified and consistent API.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthStatus,
    HealthStatusHistory,
)

# Import extracted components
from .health_config_manager import HealthCheckConfigurationManager
from .health_monitoring_coordinator import HealthMonitoringCoordinator
from .health_monitoring_observer import (
    AlertingHealthObserver,
    HealthMonitoringSubject,
    HealthMonitorObserver,
    LoggingHealthObserver,
    MetricsHealthObserver,
)
from .recovery_manager import HealthRecoveryManager
from .unified_health_service import UnifiedHealthService

logger = structlog.get_logger(__name__)

class HealthServiceFacade:
    """
    Facade providing a unified interface for the health service system.

    This class coordinates between different health service components:
    - HealthCheckService: Core health checking functionality
    - HealthCheckConfigurationManager: Configuration management
    - HealthMonitoringCoordinator: Continuous monitoring coordination
    - HealthRecoveryManager: Recovery operations
    - HealthMonitoringSubject: Observer pattern coordination
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the health service facade.

        Args:
            config: Health check configuration (creates default if None)
        """
        self.config = config or HealthCheckConfig()
        self.config_manager = HealthCheckConfigurationManager(self.config)

        # Initialize core services
        self.health_service = UnifiedHealthService(self.config)
        self.recovery_manager = HealthRecoveryManager()

        # Initialize monitoring components
        self.monitoring_subject = HealthMonitoringSubject()
        self.monitoring_coordinator = HealthMonitoringCoordinator(self.config)

        # Initialize observers
        self._default_observers: list[HealthMonitorObserver] = [
            LoggingHealthObserver(),
            AlertingHealthObserver(),
            MetricsHealthObserver(),
        ]

        # State management
        self._initialized = False
        self._monitoring_active = False
        self._lock = asyncio.Lock()

        logger.info("health_service_facade_initialized")

    async def initialize(self) -> None:
        """Initialize the health service facade and all components."""
        async with self._lock:
            if self._initialized:
                return

            try:
                # Attach default observers
                for observer in self._default_observers:
                    await self.monitoring_subject.attach(observer)

                # UnifiedHealthService is ready on construction
                # No explicit initialization needed

                self._initialized = True
                logger.info("health_service_facade_components_initialized")

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error(
                    "health_service_facade_initialization_failed", error=str(e)
                )
                raise

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        async with self._lock:
            if not self._initialized:
                await self.initialize()

            if self._monitoring_active:
                return

            try:
                # Start monitoring coordinator
                await self.monitoring_coordinator.start_monitoring(
                    self._perform_comprehensive_health_check
                )

                self._monitoring_active = True
                logger.info("health_service_facade_monitoring_started")

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error(
                    "health_service_facade_monitoring_start_failed", error=str(e)
                )
                raise

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        async with self._lock:
            if not self._monitoring_active:
                return

            try:
                await self.monitoring_coordinator.stop_monitoring()
                self._monitoring_active = False
                logger.info("health_service_facade_monitoring_stopped")

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error(
                    "health_service_facade_monitoring_stop_failed", error=str(e)
                )
                raise

    async def perform_comprehensive_health_check(self) -> HealthCheckResult:
        """
        Perform a comprehensive health check using all extracted components.

        Returns:
            HealthCheckResult: Comprehensive health check result
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        correlation_id = f"facade_health_{int(start_time)}"

        logger.debug(
            "facade_comprehensive_health_check_started", correlation_id=correlation_id
        )

        try:
            # Use the unified health service directly
            result = await self.health_service.perform_comprehensive_health_check()

            # Get all components from result
            all_components = result.components
            overall_status = result.status
            summary = result.summary

            # Create comprehensive result
            result = HealthCheckResult(
                overall_status=overall_status,
                timestamp=datetime.now(timezone.utc),
                correlation_id=correlation_id,
                components=all_components,
                summary=summary,
                performance_metrics={
                    "total_check_time_ms": (time.time() - start_time) * 1000,
                    "components_checked": len(all_components),
                    "facade_orchestrated": True,
                },
            )

            # Notify observers of system summary
            await self.monitoring_subject.notify_system_summary(
                overall_status, all_components, summary
            )

            logger.info(
                "facade_comprehensive_health_check_completed",
                correlation_id=correlation_id,
                overall_status=overall_status,
                total_components=len(all_components),
                duration_ms=result.performance_metrics["total_check_time_ms"],
            )

            return result

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("facade_comprehensive_health_check_failed", error=str(e))

            # Return error result
            return HealthCheckResult(
                overall_status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(timezone.utc),
                correlation_id=correlation_id,
                components={},
                summary={"error": str(e)},
                performance_metrics={
                    "total_check_time_ms": (time.time() - start_time) * 1000,
                    "error": True,
                },
            )

    async def get_component_health(self, component_name: str) -> ComponentHealth | None:
        """
        Get health status for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            ComponentHealth or None if component not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get component health from unified service
            return await self.health_service.get_component_health(component_name)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "facade_get_component_health_failed",
                component=component_name,
                error=str(e),
            )
            return None

    async def attempt_component_recovery(self, component_name: str) -> bool:
        """
        Attempt recovery for a specific component.

        Args:
            component_name: Name of the component to recover

        Returns:
            True if recovery successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("facade_component_recovery_started", component=component_name)

            # Use recovery manager for recovery operations
            if component_name == "database":
                result = await self.recovery_manager.attempt_database_recovery()
            elif component_name == "cache_hierarchy":
                result = await self.recovery_manager.attempt_cache_recovery()
            else:
                # Generic service recovery for other components
                result = await self.recovery_manager.attempt_service_recovery()

            success = result.success

            logger.info(
                "facade_component_recovery_completed",
                component=component_name,
                success=success,
                recovery_type=result.recovery_type,
            )

            return success

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "facade_component_recovery_failed",
                component=component_name,
                error=str(e),
            )
            return False

    def get_health_history(
        self, hours: int = 24, max_entries: int | None = None
    ) -> list[HealthStatusHistory]:
        """
        Get health status history.

        Args:
            hours: Number of hours to look back
            max_entries: Maximum number of entries to return

        Returns:
            List of health status history entries
        """
        # This would integrate with a history manager if available
        # For now, return empty list as placeholder
        return []

    def get_configuration(self) -> HealthCheckConfig:
        """Get current health check configuration."""
        return self.config_manager.get_config()

    def update_configuration(self, **kwargs) -> None:
        """Update health check configuration."""
        self.config_manager.update_config(**kwargs)

    def get_service_status(self) -> dict[str, Any]:
        """Get status of all facade-managed services."""
        return {
            "initialized": self._initialized,
            "monitoring_active": self._monitoring_active,
            "observer_count": self.monitoring_subject.get_observer_count(),
            "config_valid": len(self.config_manager.validate_config()) == 0,
            "health_service_metrics": self.health_service.get_metrics(),
        }

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get metrics summary from all observers."""
        # Find the metrics observer and get its summary
        for observer in self._default_observers:
            if isinstance(observer, MetricsHealthObserver):
                return observer.get_metrics_summary()

        return {"error": "Metrics observer not found"}

    async def attach_observer(self, observer: HealthMonitorObserver) -> None:
        """Attach a custom observer to the monitoring system."""
        await self.monitoring_subject.attach(observer)

    async def detach_observer(self, observer: HealthMonitorObserver) -> None:
        """Detach a custom observer from the monitoring system."""
        await self.monitoring_subject.detach(observer)

    async def _perform_comprehensive_health_check(self) -> None:
        """Internal method for monitoring coordinator to perform health checks."""
        try:
            result = await self.perform_comprehensive_health_check()

            # Notify observers of individual component results
            for component_name, component_health in result.components.items():
                # Check for status changes (simplified - would need previous state tracking)
                await self.monitoring_subject.notify_check_completed(
                    component_name,
                    component_health,
                    (time.time() * 1000) - (result.timestamp.timestamp() * 1000),
                )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("facade_monitoring_health_check_failed", error=str(e))

    async def shutdown(self) -> None:
        """Shutdown the health service facade and all components."""
        async with self._lock:
            try:
                logger.info("health_service_facade_shutdown_started")

                # Stop monitoring
                if self._monitoring_active:
                    await self.stop_monitoring()

                # Shutdown unified health service
                await self.health_service.stop_monitoring()

                self._initialized = False

                logger.info("health_service_facade_shutdown_completed")

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("health_service_facade_shutdown_failed", error=str(e))
                raise
