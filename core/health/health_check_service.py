"""
Legacy import path for HealthCheckService (basic version).

DEPRECATED: Use `resync.core.health.unified_health_service` instead.

This module re-exports UnifiedHealthService as HealthCheckService
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.core.health.health_check_service import HealthCheckService
    
    # After (recommended)
    from resync.core.health.unified_health_service import UnifiedHealthService
    # or use the facade:
    from resync.core.health.health_service_facade import HealthServiceFacade
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.core.health.health_check_service' is deprecated. "
    "Use 'resync.core.health.unified_health_service' or "
    "'resync.core.health.health_service_facade' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export UnifiedHealthService as HealthCheckService for compatibility
from resync.core.health.unified_health_service import (  # noqa: E402
    UnifiedHealthService,
)
from resync.core.health.health_models import (  # noqa: E402
    ComponentHealth,
    ComponentType,
    HealthCheckResult,
    HealthStatus,
    SystemHealthStatus,
)

# Alias for backward compatibility
HealthCheckService = UnifiedHealthService

__all__ = [
    "HealthCheckService",
    "UnifiedHealthService",
    "ComponentHealth",
    "ComponentType",
    "HealthCheckResult",
    "HealthStatus",
    "SystemHealthStatus",
]
