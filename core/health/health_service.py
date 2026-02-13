"""
Legacy import path for HealthCheckService.

DEPRECATED: Use `resync.core.health.unified_health_service` instead.

This module re-exports UnifiedHealthService as HealthCheckService
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.core.health.health_service import HealthCheckService
    
    # After (recommended)
    from resync.core.health.unified_health_service import UnifiedHealthService
    # or use the facade:
    from resync.core.health.health_service_facade import HealthServiceFacade
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.core.health.health_service' is deprecated. "
    "Use 'resync.core.health.unified_health_service' or "
    "'resync.core.health.health_service_facade' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export UnifiedHealthService as HealthCheckService for compatibility
from resync.core.health.unified_health_service import (  # noqa: E402
    UnifiedHealthService,
    get_unified_health_service,
    shutdown_unified_health_service,
)
from resync.core.health.health_models import (  # noqa: E402
    ComponentHealth,
    ComponentType,
    HealthCheckConfig,
    HealthCheckResult,
    HealthStatus,
)

# Alias for backward compatibility
HealthCheckService = UnifiedHealthService

__all__ = [
    "HealthCheckService",
    "UnifiedHealthService",
    "ComponentHealth",
    "ComponentType",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthStatus",
]

# ---------------------------------------------------------------------------
# Backwards-compatible module-level helpers expected by older imports.
# These wrap the unified singleton implementation.
# ---------------------------------------------------------------------------

import asyncio
from datetime import datetime


async def get_health_check_service() -> UnifiedHealthService:
    """Legacy alias for get_unified_health_service()."""
    return await get_unified_health_service()


async def shutdown_health_check_service() -> None:
    """Legacy alias for shutdown_unified_health_service()."""
    await shutdown_unified_health_service()


async def get_health_status() -> HealthCheckResult:
    """Return a snapshot health status (legacy helper).

    This keeps older call sites working by delegating to the unified service.
    """
    svc = await get_unified_health_service()

    start = asyncio.get_running_loop().time()
    components = await svc.get_all_component_health()

    # Simple aggregation rules:
    # - Any UNHEALTHY => overall UNHEALTHY
    # - Else any DEGRADED => overall DEGRADED
    # - Else => HEALTHY
    overall = HealthStatus.HEALTHY
    for ch in components.values():
        if ch.status == HealthStatus.UNHEALTHY:
            overall = HealthStatus.UNHEALTHY
            break
        if ch.status == HealthStatus.DEGRADED:
            overall = HealthStatus.DEGRADED

    duration_ms = (asyncio.get_running_loop().time() - start) * 1000.0

    return HealthCheckResult(
        overall_status=overall,
        timestamp=datetime.utcnow(),
        components=components,
        summary={
            "component_count": len(components),
        },
        duration_ms=duration_ms,
    )

