"""Common utilities for health checkers to reduce duplicated logic."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

from resync.core.health.health_models import ComponentHealth, ComponentType, HealthStatus


def response_time_ms(start_time: float) -> float:
    """Return elapsed time in milliseconds from a start timestamp."""
    return (time.time() - start_time) * 1000


def threshold_status(
    *,
    value: float,
    warning_threshold: float,
    critical_threshold: float,
    healthy_message: str,
    degraded_message: str,
    critical_message: str,
) -> tuple[HealthStatus, str]:
    """Resolve health status and formatted message from warning/critical thresholds."""
    if value > critical_threshold:
        return HealthStatus.UNHEALTHY, critical_message.format(value=value)
    if value > warning_threshold:
        return HealthStatus.DEGRADED, degraded_message.format(value=value)
    return HealthStatus.HEALTHY, healthy_message.format(value=value)


def build_error_health(
    *,
    component_name: str,
    component_type: ComponentType,
    status: HealthStatus,
    message: str,
    start_time: float,
    error: Exception,
    log_event: str,
    logger: structlog.stdlib.BoundLogger,
) -> ComponentHealth:
    """Build a standardized error ComponentHealth response and log failure."""
    logger.error(log_event, error_type=type(error).__name__, exc_info=True)
    return ComponentHealth(
        name=component_name,
        component_type=component_type,
        status=status,
        message=message,
        response_time_ms=response_time_ms(start_time),
        last_check=datetime.now(timezone.utc),
        error_count=1,
    )
