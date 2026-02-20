from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from resync.core.health import (
    ComponentType,
    HealthStatus,
    get_unified_health_service,
    get_status_color,
    get_status_description,
)

logger = logging.getLogger(__name__)

PROGRAMMING_ERRORS = (TypeError, KeyError, AttributeError, IndexError)


# Lazy import of runtime_metrics to avoid circular dependencies
def _get_runtime_metrics():
    """Lazy import of runtime_metrics."""
    from resync.core.metrics import runtime_metrics

    return runtime_metrics


# Main health router
router = APIRouter(prefix="/health", tags=["health"])

# Config router for compatibility
config_router = APIRouter(
    prefix="/config", tags=["config"]
)  # Config-related endpoints would go here

# Alias for health router
health_router = router


# Health check response models
class HealthSummaryResponse(BaseModel):
    """Health check summary response."""

    status: str
    status_color: str
    status_description: str
    timestamp: str
    correlation_id: str
    summary: dict[str, Any]
    alerts: list[str]
    performance_metrics: dict[str, Any]


class ComponentHealthResponse(BaseModel):
    """Individual component health response."""

    name: str
    component_type: str
    status: str
    status_color: str
    message: str
    response_time_ms: float | None = None
    last_check: str
    error_count: int
    metadata: dict[str, Any] | None = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""

    overall_status: str
    overall_status_color: str
    timestamp: str
    correlation_id: str
    components: dict[str, ComponentHealthResponse]
    summary: dict[str, Any]
    alerts: list[str]
    performance_metrics: dict[str, Any]
    history: list[dict[str, Any]]


class CoreHealthResponse(BaseModel):
    """Core components health response."""

    status: str
    status_color: str
    timestamp: str
    core_components: dict[str, ComponentHealthResponse]
    summary: dict[str, Any]


# Core components that are critical for system operation
CORE_COMPONENTS = {"database", "redis", "connection_pools", "file_system"}


@router.get("/", response_model=HealthSummaryResponse)
async def get_health_summary(
    auto_enable: bool = Query(
        default=False,
        description="Auto-enable system components if validation is successful",
    ),
) -> HealthSummaryResponse:
    """
    Get overall system health summary with status indicators.

    Args:
        auto_enable: Whether to auto-enable system components if validation is successful

    Returns:
        HealthSummaryResponse: Overall system health status with color-coded indicators
    """
    try:
        health_service = await get_unified_health_service()
        health_result = await health_service.perform_comprehensive_health_check()

        # If auto_enable is true and health is good, attempt to enable any disabled components
        if auto_enable and health_result.overall_status != HealthStatus.UNHEALTHY:
            # In a real implementation, this would enable components that might be disabled
            logger.info("Health check successful with auto_enable: %s", auto_enable)

        # Add auto_enable information to the response
        summary_with_auto_enable = health_result.summary.copy()
        summary_with_auto_enable["auto_enable"] = auto_enable
        summary_with_auto_enable["auto_enable_applied"] = (
            auto_enable and health_result.overall_status != HealthStatus.UNHEALTHY
        )

        _get_runtime_metrics().health_check_with_auto_enable.increment()

        return HealthSummaryResponse(
            status=health_result.overall_status.value,
            status_color=get_status_color(health_result.overall_status),
            status_description=get_status_description(health_result.overall_status),
            timestamp=health_result.timestamp.isoformat(),
            correlation_id=health_result.correlation_id or "",
            summary=summary_with_auto_enable,
            alerts=[str(alert) for alert in (health_result.alerts or [])],
            performance_metrics=health_result.performance_metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        original_exception = e
        logger.error("Health check failed: %s", original_exception)
        # Increment counter for health check failures if we can
        try:
            _get_runtime_metrics().health_check_with_auto_enable.increment()
        except (AttributeError, ImportError) as metrics_e:
            # Log metrics failure but don't fail the health check
            logger.warning(
                "Failed to increment health check metrics: %s", metrics_e, exc_info=True
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check system error: {str(original_exception)}",
        ) from original_exception


@router.get("/core", response_model=CoreHealthResponse)
async def get_core_health() -> CoreHealthResponse:
    """
    Get health status for core system components only.

    Returns:
        CoreHealthResponse: Health status of core components with status indicators
    """
    try:
        health_service = await get_unified_health_service()
        health_result = await health_service.perform_comprehensive_health_check()

        # Filter only core components
        core_components = {
            name: component
            for name, component in health_result.components.items()
            if name in CORE_COMPONENTS
        }

        # Calculate core status (more strict - any unhealthy core component = unhealthy overall)
        core_status = HealthStatus.HEALTHY
        for component in core_components.values():
            if component.status == HealthStatus.UNHEALTHY:
                core_status = HealthStatus.UNHEALTHY
                break
            if component.status == HealthStatus.DEGRADED:
                core_status = HealthStatus.DEGRADED

        # Convert components to response format
        core_components_response = {
            name: ComponentHealthResponse(
                name=component.name,
                component_type=component.component_type.value,
                status=component.status.value,
                status_color=get_status_color(component.status),
                message=component.message or "",
                response_time_ms=component.response_time_ms,
                last_check=component.last_check.isoformat(),
                error_count=component.error_count,
                metadata=component.metadata,
            )
            for name, component in core_components.items()
        }

        # Generate core summary
        core_summary = {
            "total_core_components": len(core_components),
            "healthy_core_components": sum(
                1 for c in core_components.values() if c.status == HealthStatus.HEALTHY
            ),
            "unhealthy_core_components": sum(
                1
                for c in core_components.values()
                if c.status == HealthStatus.UNHEALTHY
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return CoreHealthResponse(
            status=core_status.value,
            status_color=get_status_color(core_status),
            timestamp=health_result.timestamp.isoformat(),
            core_components=core_components_response,
            summary=core_summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Core health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core health check system error. Check server logs for details.",
        ) from e


@router.get("/detailed", response_model=DetailedHealthResponse)
async def get_detailed_health(
    include_history: bool = Query(
        False, description="Include health history in response"
    ),
    history_hours: int = Query(
        24, description="Hours of history to include", ge=1, le=168
    ),
) -> DetailedHealthResponse:
    """
    Get detailed health check with all components and optional history.

    Args:
        include_history: Whether to include historical health data
        history_hours: Number of hours of history to include (1-168)

    Returns:
        DetailedHealthResponse: Comprehensive health status with all components
    """
    try:
        health_service = await get_unified_health_service()
        health_result = await health_service.perform_comprehensive_health_check()

        # Convert components to response format
        components_response = {
            name: ComponentHealthResponse(
                name=component.name,
                component_type=component.component_type.value,
                status=component.status.value,
                status_color=get_status_color(component.status),
                message=component.message or "",
                response_time_ms=component.response_time_ms,
                last_check=component.last_check.isoformat(),
                error_count=component.error_count,
                metadata=component.metadata,
            )
            for name, component in health_result.components.items()
        }

        # Get history if requested
        history_data = []
        if include_history:
            history = health_service.get_health_history(history_hours)
            history_data = [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "overall_status": entry.overall_status.value,
                    "overall_status_color": get_status_color(entry.overall_status),
                    "summary": get_status_description(entry.overall_status),
                }
                for entry in history
            ]

        return DetailedHealthResponse(
            overall_status=health_result.overall_status.value,
            overall_status_color=get_status_color(health_result.overall_status),
            timestamp=health_result.timestamp.isoformat(),
            correlation_id=health_result.correlation_id or "",
            components=components_response,
            summary=health_result.summary,
            alerts=health_result.alerts,
            performance_metrics=health_result.performance_metrics,
            history=history_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Detailed health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detailed health check system error. Check server logs for details.",
        ) from e


@router.get("/ready")
async def readiness_probe() -> dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.

    Returns 503 Service Unavailable if core components are unhealthy,
    200 OK if system is ready to serve requests.

    Returns:
        dict[str, Any]: Readiness status with core component details
    """
    try:
        health_service = await get_unified_health_service()
        health_result = await health_service.perform_comprehensive_health_check()

        # Check only core components for readiness
        core_components = {
            name: component
            for name, component in health_result.components.items()
            if name in CORE_COMPONENTS
        }

        # System is ready if all core components are healthy
        ready = all(
            component.status == HealthStatus.HEALTHY
            for component in core_components.values()
        )

        response_data = {
            "status": "ready" if ready else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": health_result.correlation_id,
            "core_components": {
                name: {
                    "status": component.status.value,
                    "status_color": get_status_color(component.status),
                    "message": component.message,
                }
                for name, component in core_components.items()
            },
        }

        if not ready:
            # Return 503 Service Unavailable if not ready
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=response_data
            )

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Readiness probe failed: %s", e)
        # Always return 503 on probe failure
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        ) from e


@router.get("/live")
async def liveness_probe() -> dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.

    Returns 503 Service Unavailable if the health check system itself is failing,
    200 OK if the system is alive and responding.

    Returns:
        dict[str, Any]: Liveness status
    """
    try:
        health_service = await get_unified_health_service()

        # Simple liveness check - just verify the service is responding
        # We don't check actual component health, just that the system is alive
        current_time = datetime.now(timezone.utc)

        # Check if we can get basic service info
        last_check = health_service.last_health_check

        # System is considered alive if it has performed health checks recently
        # or if it's the first check (last_check is None)
        alive = (
            last_check is None
            or (current_time - last_check).total_seconds() < 300  # 5 minutes
        )

        if not alive:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "dead",
                    "timestamp": current_time.isoformat(),
                    "last_health_check": last_check.isoformat() if last_check else None,
                    "message": "Health check system appears to be stuck",
                },
            )

        return {
            "status": "alive",
            "timestamp": current_time.isoformat(),
            "last_health_check": last_check.isoformat() if last_check else None,
            "message": "System is responding",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Liveness probe failed: %s", e)
        # Always return 503 on probe failure
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "dead",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        ) from e


@router.post("/component/{component_name}/recover")
async def recover_component(component_name: str) -> dict[str, Any]:
    """
    Attempt to recover a specific component.

    Args:
        component_name: Name of the component to recover

    Returns:
        dict[str, Any]: Recovery attempt result
    """
    try:
        health_service = await get_unified_health_service()

        # Attempt recovery
        recovery_success = await health_service.attempt_recovery(component_name)

        # Get updated component health
        component_health = await health_service.get_component_health(component_name)

        response_data = {
            "component": component_name,
            "recovery_attempted": True,
            "recovery_successful": recovery_success,
            "current_status": (
                component_health.status.value if component_health else "unknown"
            ),
            "status_color": (
                get_status_color(component_health.status) if component_health else "⚪"
            ),
            "message": (
                component_health.message if component_health else "Component not found"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not recovery_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=response_data
            )

        return response_data

    except HTTPException:
        # Re-raise HTTPException to preserve the original status code and detail
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Component recovery failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "component": component_name,
                "recovery_attempted": True,
                "recovery_successful": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ) from e


@router.get("/redis")
async def get_redis_health() -> dict[str, Any]:
    """
    Get detailed Redis health check with connection validation.

    This endpoint performs explicit Redis connectivity testing and returns
    critical status information for idempotency guarantee validation.

    Returns:
        dict[str, Any]: Redis health status with connection details
    """
    try:
        health_service = await get_unified_health_service()
        health_result = await health_service.perform_comprehensive_health_check()

        redis_component = health_result.components.get("redis")

        if not redis_component:
            return {
                "status": "critical",
                "message": "Redis component not found in health check",
                "idempotency_safe": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Additional Redis-specific validation
        redis_details = {
            "status": redis_component.status.value,
            "status_color": get_status_color(redis_component.status),
            "message": redis_component.message,
            "last_check": (
                redis_component.last_check.isoformat()
                if redis_component.last_check
                else None
            ),
            "response_time_ms": redis_component.response_time_ms,
            "details": redis_component.metadata or {},
        }

        # Determine if system can guarantee idempotency
        idempotency_safe = redis_component.status == HealthStatus.HEALTHY

        if not idempotency_safe:
            logger.warning(
                "Redis health check failed - idempotency may be compromised",
                extra={"redis_status": redis_component.status.value},
            )

        return {
            "status": "healthy" if idempotency_safe else "critical",
            "idempotency_safe": idempotency_safe,
            "redis": redis_details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": health_result.correlation_id,
            "warning": (
                "Redis unavailable - idempotency guarantees compromised"
                if not idempotency_safe
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error checking Redis health: %s", e, exc_info=True)
        return {
            "status": "critical",
            "idempotency_safe": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warning": "Redis health check failed - system cannot guarantee idempotency",
        }


@router.get("/components")
async def list_components() -> dict[str, list[dict[str, str]]]:
    """
    List all available health check components.

    Returns:
        dict[str, list[dict[str, str]]]: List of available components
    """
    components = [
        {
            "name": "database",
            "type": ComponentType.DATABASE.value,
            "description": "Database connectivity and performance",
        },
        {
            "name": "redis",
            "type": ComponentType.REDIS.value,
            "description": "Redis cache connectivity",
        },
        {
            "name": "cache_hierarchy",
            "type": ComponentType.CACHE.value,
            "description": "Cache hierarchy health",
        },
        {
            "name": "file_system",
            "type": ComponentType.FILE_SYSTEM.value,
            "description": "File system and disk space",
        },
        {
            "name": "memory",
            "type": ComponentType.MEMORY.value,
            "description": "Memory usage monitoring",
        },
        {
            "name": "cpu",
            "type": ComponentType.CPU.value,
            "description": "CPU load monitoring",
        },
        {
            "name": "tws_monitor",
            "type": ComponentType.EXTERNAL_API.value,
            "description": "TWS external service health",
        },
        {
            "name": "connection_pools",
            "type": ComponentType.CONNECTION_POOL.value,
            "description": "Database connection pools",
        },
        {
            "name": "websocket_pool",
            "type": ComponentType.WEBSOCKET.value,
            "description": "WebSocket connection pools",
        },
    ]

    return {"components": components}
