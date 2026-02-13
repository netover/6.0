"""
Health Monitoring Observer Pattern Implementation

This module implements the Observer pattern for coordinating between different
health monitors and providing a unified interface for health monitoring events.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import structlog

from resync.core.health.health_models import ComponentHealth, HealthStatus

logger = structlog.get_logger(__name__)


class HealthMonitoringEvent:
    """Event data for health monitoring notifications."""

    def __init__(
        self,
        event_type: str,
        component_name: str,
        health_status: HealthStatus,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.event_type = event_type
        self.component_name = component_name
        self.health_status = health_status
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}


class HealthMonitorObserver(ABC):
    """Abstract base class for health monitor observers."""

    @abstractmethod
    async def on_health_status_changed(self, event: HealthMonitoringEvent) -> None:
        """Called when a component's health status changes."""

    @abstractmethod
    async def on_component_check_completed(self, event: HealthMonitoringEvent) -> None:
        """Called when a component health check is completed."""

    @abstractmethod
    async def on_system_health_summary(self, event: HealthMonitoringEvent) -> None:
        """Called when system health summary is generated."""


class HealthMonitoringSubject:
    """Subject class that manages health monitoring observers."""

    def __init__(self):
        """Initialize the monitoring subject."""
        self._observers: list[HealthMonitorObserver] = []
        self._lock = asyncio.Lock()

    async def attach(self, observer: HealthMonitorObserver) -> None:
        """Attach an observer to the subject."""
        async with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
                logger.debug(
                    "health_monitor_observer_attached",
                    observer_type=type(observer).__name__,
                    total_observers=len(self._observers),
                )

    async def detach(self, observer: HealthMonitorObserver) -> None:
        """Detach an observer from the subject."""
        async with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.debug(
                    "health_monitor_observer_detached",
                    observer_type=type(observer).__name__,
                    total_observers=len(self._observers),
                )

    async def notify_status_changed(
        self,
        component_name: str,
        old_status: HealthStatus,
        new_status: HealthStatus,
        component_health: ComponentHealth,
    ) -> None:
        """Notify observers of health status change."""
        if old_status == new_status:
            return  # No change

        event = HealthMonitoringEvent(
            event_type="status_changed",
            component_name=component_name,
            health_status=new_status,
            metadata={
                "old_status": old_status,
                "component_health": component_health,
                "status_changed": True,
            },
        )

        await self._notify_observers("on_health_status_changed", event, component_name)

    async def notify_check_completed(
        self,
        component_name: str,
        component_health: ComponentHealth,
        check_duration_ms: float,
    ) -> None:
        """Notify observers of completed health check."""
        event = HealthMonitoringEvent(
            event_type="check_completed",
            component_name=component_name,
            health_status=component_health.status,
            metadata={
                "component_health": component_health,
                "check_duration_ms": check_duration_ms,
                "response_time_ms": component_health.response_time_ms,
            },
        )

        await self._notify_observers("on_component_check_completed", event, component_name)

    async def notify_system_summary(
        self,
        overall_status: HealthStatus,
        components: dict[str, ComponentHealth],
        summary: dict[str, Any],
    ) -> None:
        """Notify observers of system health summary."""
        event = HealthMonitoringEvent(
            event_type="system_summary",
            component_name="system",
            health_status=overall_status,
            metadata={
                "overall_status": overall_status,
                "components": components,
                "summary": summary,
                "total_components": len(components),
            },
        )

        await self._notify_observers("on_system_health_summary", event, "system")

    async def _notify_observers(
        self,
        method_name: str,
        event: HealthMonitoringEvent,
        component_name: str,
    ) -> None:
        """Notify all observers of an event."""
        async with self._lock:
            observers_copy = self._observers.copy()

        if not observers_copy:
            return

        # Notify observers concurrently
        tasks = []
        for observer in observers_copy:
            try:
                method = getattr(observer, method_name)
                task = asyncio.create_task(method(event))
                tasks.append(task)
            except Exception as e:
                logger.error(
                    "error_getting_observer_method",
                    observer_type=type(observer).__name__,
                    method=method_name,
                    error=str(e),
                )

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(
                    "error_notifying_observers",
                    method=method_name,
                    component=component_name,
                    error=str(e),
                )

    def get_observer_count(self) -> int:
        """Get the number of attached observers."""
        return len(self._observers)


class LoggingHealthObserver(HealthMonitorObserver):
    """Observer that logs health monitoring events."""

    async def on_health_status_changed(self, event: HealthMonitoringEvent) -> None:
        """Log health status changes."""
        logger.info(
            "health_status_changed",
            component=event.component_name,
            new_status=event.health_status,
            old_status=event.metadata.get("old_status"),
            timestamp=event.timestamp.isoformat(),
        )

    async def on_component_check_completed(self, event: HealthMonitoringEvent) -> None:
        """Log component check completion."""
        logger.debug(
            "component_health_check_completed",
            component=event.component_name,
            status=event.health_status,
            duration_ms=event.metadata.get("check_duration_ms"),
            response_time_ms=event.metadata.get("response_time_ms"),
        )

    async def on_system_health_summary(self, event: HealthMonitoringEvent) -> None:
        """Log system health summary."""
        metadata = event.metadata
        logger.info(
            "system_health_summary_generated",
            overall_status=event.health_status,
            total_components=metadata.get("total_components"),
            components=metadata.get("summary"),
        )


class AlertingHealthObserver(HealthMonitorObserver):
    """Observer that handles health-based alerting."""

    def __init__(self):
        """Initialize the alerting observer."""
        self._last_alerts: dict[str, datetime] = {}
        self._alert_cooldown_minutes = 5

    async def on_health_status_changed(self, event: HealthMonitoringEvent) -> None:
        """Handle health status changes for alerting."""
        component_name = event.component_name
        new_status = event.health_status

        # Check if we should send alert (cooldown period)
        last_alert = self._last_alerts.get(component_name)
        if last_alert:
            time_since_last_alert = datetime.now() - last_alert
            if time_since_last_alert.total_seconds() < (self._alert_cooldown_minutes * 60):
                return  # Still in cooldown period

        # Send alert for unhealthy or degraded components
        if new_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            await self._send_alert(event)
            self._last_alerts[component_name] = datetime.now()

    async def on_component_check_completed(self, event: HealthMonitoringEvent) -> None:
        """Handle component check completion for alerting."""
        # Only alert on failures or slow responses
        if event.health_status == HealthStatus.UNHEALTHY:
            await self._send_alert(event)

    async def on_system_health_summary(self, event: HealthMonitoringEvent) -> None:
        """Handle system health summary for alerting."""
        # Alert if overall system health is poor
        if event.health_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
            await self._send_system_alert(event)

    async def _send_alert(self, event: HealthMonitoringEvent) -> None:
        """Send alert for component health issue."""
        logger.warning(
            "health_alert_triggered",
            component=event.component_name,
            status=event.health_status,
            event_type=event.event_type,
            metadata=event.metadata,
        )

        # In a real implementation, this would integrate with alerting systems
        # like PagerDuty, Slack, email, etc.

    async def _send_system_alert(self, event: HealthMonitoringEvent) -> None:
        """Send alert for system-wide health issues."""
        logger.error(
            "system_health_alert",
            overall_status=event.health_status,
            total_components=event.metadata.get("total_components"),
            summary=event.metadata.get("summary"),
        )


class MetricsHealthObserver(HealthMonitorObserver):
    """Observer that collects health metrics."""

    def __init__(self):
        """Initialize the metrics observer."""
        self._status_changes: list[dict[str, Any]] = []
        self._check_durations: list[dict[str, Any]] = []
        self._system_summaries: list[dict[str, Any]] = []
        self._max_metrics_history = 1000

    async def on_health_status_changed(self, event: HealthMonitoringEvent) -> None:
        """Record health status change metrics."""
        self._status_changes.append(
            {
                "timestamp": event.timestamp,
                "component": event.component_name,
                "status": event.health_status,
                "old_status": event.metadata.get("old_status"),
            }
        )

        # Maintain history size limit
        if len(self._status_changes) > self._max_metrics_history:
            self._status_changes = self._status_changes[-self._max_metrics_history :]

    async def on_component_check_completed(self, event: HealthMonitoringEvent) -> None:
        """Record component check metrics."""
        self._check_durations.append(
            {
                "timestamp": event.timestamp,
                "component": event.component_name,
                "status": event.health_status,
                "duration_ms": event.metadata.get("check_duration_ms"),
                "response_time_ms": event.metadata.get("response_time_ms"),
            }
        )

        # Maintain history size limit
        if len(self._check_durations) > self._max_metrics_history:
            self._check_durations = self._check_durations[-self._max_metrics_history :]

    async def on_system_health_summary(self, event: HealthMonitoringEvent) -> None:
        """Record system health summary metrics."""
        self._system_summaries.append(
            {
                "timestamp": event.timestamp,
                "overall_status": event.health_status,
                "total_components": event.metadata.get("total_components"),
                "summary": event.metadata.get("summary"),
            }
        )

        # Maintain history size limit
        if len(self._system_summaries) > self._max_metrics_history:
            self._system_summaries = self._system_summaries[-self._max_metrics_history :]

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            "status_changes_count": len(self._status_changes),
            "check_durations_count": len(self._check_durations),
            "system_summaries_count": len(self._system_summaries),
            "latest_status_changes": (self._status_changes[-10:] if self._status_changes else []),
            "latest_system_summary": (
                self._system_summaries[-1] if self._system_summaries else None
            ),
        }
