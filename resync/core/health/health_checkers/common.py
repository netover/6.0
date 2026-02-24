"""Common utilities for health checkers to reduce duplicated logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthStatus,
)


@dataclass(frozen=True)
class ThresholdConfig:
    """Thresholds and messages for health checks."""

    warning: float
    critical: float
    healthy_msg: str
    degraded_msg: str
    critical_msg: str


def response_time_ms(start_time: float) -> float:
    """Return elapsed time in milliseconds from a start timestamp."""
    return (time.time() - start_time) * 1000


def threshold_status(
    value: float,
    config: ThresholdConfig,
) -> tuple[HealthStatus, str]:
    """Resolve health status and formatted message from warning/critical thresholds."""
    if value > config.critical:
        return HealthStatus.UNHEALTHY, config.critical_msg.format(value=value)
    if value > config.warning:
        return HealthStatus.DEGRADED, config.degraded_msg.format(value=value)
    return HealthStatus.HEALTHY, config.healthy_msg.format(value=value)


@dataclass(frozen=True)
class ErrorContext:
    """Context for building error health results."""

    name: str
    type: ComponentType
    status: HealthStatus
    message: str
    start_time: float
    error: Exception
    log_event: str


def build_error_health(
    ctx: ErrorContext,
    logger: structlog.stdlib.BoundLogger,
) -> ComponentHealth:
    """Build a standardized error ComponentHealth response and log failure."""
    logger.error(ctx.log_event, error_type=type(ctx.error).__name__, exc_info=True)
    return ComponentHealth(
        name=ctx.name,
        component_type=ctx.type,
        status=ctx.status,
        message=ctx.message,
        response_time_ms=response_time_ms(ctx.start_time),
        last_check=datetime.now(timezone.utc),
        error_count=1,
    )
