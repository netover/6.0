# __init__.py — Public API (lazy, robust, backwards-friendly)

from __future__ import annotations

import os
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Package metadata (dinâmico quando possível, com fallback estável)
# ---------------------------------------------------------------------------
try:
    # Resolve pelo nome do pacote raiz (mais resiliente em workspaces)
    __version__ = _pkg_version(__name__.split(".", 1)[0])
except PackageNotFoundError:
    # Fallback para nome comum do pacote/distribution (ajuste se necessário)
    try:
        __version__ = _pkg_version("health_service_components")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__doc__ = (
    "Health Service Components — Public API facade with lazy-loading.\n\n"
    "Design goals:\n"
    "- Lazy import (PEP 562) to minimize startup time and avoid import cycles.\n"
    "- Single source of truth for public exports (no drift with __all__).\n"
    "- Backward compatibility via per-symbol DeprecationWarning.\n"
    "- Optional fast-path: preload essentials or strict import validation.\n\n"
    "Environment flags:\n"
    "- HSC_PRELOAD_ESSENTIALS=1   -> Preloads selected hot-path symbols.\n"
    "- HSC_STRICT_IMPORTS=1       -> Imports and validates all exports at "
    "import time.\n"
)

# ---------------------------------------------------------------------------
# Public export map: Symbol -> (submodule, attribute)
# Mantém a API pública coesa e evita divergência entre imports e __all__.
# ---------------------------------------------------------------------------
_EXPORTS: dict[str, tuple[str, str]] = {
    # === MODELS (v5.4.9 - moved from core/) ===
    "ComponentHealth": ("health_models", "ComponentHealth"),
    "ComponentType": ("health_models", "ComponentType"),
    "HealthCheckConfig": ("health_models", "HealthCheckConfig"),
    "HealthCheckResult": ("health_models", "HealthCheckResult"),
    "HealthStatus": ("health_models", "HealthStatus"),
    "SystemHealthStatus": ("health_models", "SystemHealthStatus"),
    "get_status_color": ("health_models", "get_status_color"),
    "get_status_description": ("health_models", "get_status_description"),
    # === SERVICE (v5.4.9 - moved from core/) ===
    "get_health_check_service": (
        "unified_health_service",
        "get_unified_health_service",
    ),
    "get_health_status": ("unified_health_service", "get_health_status"),
    "shutdown_health_check_service": (
        "unified_health_service",
        "shutdown_unified_health_service",
    ),
    # === RECOMMENDED (v5.3.9+) ===
    # Unified service (consolidates orchestrator + enhanced)
    "UnifiedHealthService": ("unified_health_service", "UnifiedHealthService"),
    "get_unified_health_service": (
        "unified_health_service",
        "get_unified_health_service",
    ),
    "shutdown_unified_health_service": (
        "unified_health_service",
        "shutdown_unified_health_service",
    ),
    # Health check service (legacy alias → UnifiedHealthService)
    "HealthCheckService": ("unified_health_service", "UnifiedHealthService"),
    # Facade (public API)
    "HealthServiceFacade": ("health_service_facade", "HealthServiceFacade"),
    # Configuration
    "HealthCheckConfigurationManager": (
        "health_config_manager",
        "HealthCheckConfigurationManager",
    ),
    # History
    "HealthHistoryManager": ("health_history_manager", "HealthHistoryManager"),
    # === SUPPORTING COMPONENTS ===
    # Monitoring components
    "ProactiveHealthMonitor": ("proactive_monitor", "ProactiveHealthMonitor"),
    "PerformanceMetricsCollector": (
        "performance_metrics_collector",
        "PerformanceMetricsCollector",
    ),
    "HealthMonitoringCoordinator": (
        "health_monitoring_coordinator",
        "HealthMonitoringCoordinator",
    ),
    "HealthMonitoringAggregator": (
        "monitoring_aggregator",
        "HealthMonitoringAggregator",
    ),
    # Circuit breaker functionality
    "CircuitBreakerManager": (
        "circuit_breaker_manager",
        "CircuitBreakerManager",
    ),
    # Caching
    "ComponentCacheManager": (
        "component_cache_manager",
        "ComponentCacheManager",
    ),
    # Memory management
    "MemoryUsageTracker": ("memory_usage_tracker", "MemoryUsageTracker"),
    # Recovery and alerting
    "HealthRecoveryManager": ("recovery_manager", "HealthRecoveryManager"),
    "HealthAlerting": ("health_alerting", "HealthAlerting"),
    # Utilities
    "HealthCheckUtils": ("health_check_utils", "HealthCheckUtils"),
    "HealthCheckRetry": ("health_check_retry", "HealthCheckRetry"),
}

__all__ = tuple(_EXPORTS.keys())

# ---------------------------------------------------------------------------
# Deprecations — mensagens específicas por símbolo legado
# ---------------------------------------------------------------------------
_DEPRECATED: dict[str, str] = {
    "get_health_check_service": "Importing 'get_health_check_service' from 'resync.core.health' is deprecated. Use 'get_unified_health_service' instead.",
    "get_health_status": "Importing 'get_health_status' from 'resync.core.health' is deprecated. Use 'UnifiedHealthService.perform_comprehensive_health_check()' instead.",
    "shutdown_health_check_service": "Importing 'shutdown_health_check_service' from 'resync.core.health' is deprecated. Use 'shutdown_unified_health_service' instead.",
    "HealthCheckService": "Importing 'HealthCheckService' from 'resync.core.health' is deprecated. Use 'UnifiedHealthService' instead.",
}

# ---------------------------------------------------------------------------
# Helpers de ambiente (pré-carregamento opcional de essenciais)
# ---------------------------------------------------------------------------
_ESSENTIALS = (
    "get_unified_health_service",
    "shutdown_unified_health_service",
    "UnifiedHealthService",
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Lazy loader (PEP 562) com memoization e deprecação por símbolo
# ---------------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    if name in _DEPRECATED:
        warnings.warn(_DEPRECATED[name], DeprecationWarning, stacklevel=2)

    module = import_module(f".{module_name}", __name__)
    obj = getattr(module, attr_name)
    globals()[name] = obj  # memoize para chamadas futuras
    return obj


def __dir__() -> list[str]:
    # Reflete __all__ e símbolos já resolvidos (melhor DX em IDE)
    return sorted(list(set(list(globals().keys()) + list(__all__))))


# ---------------------------------------------------------------------------
# Caminhos opcionais:
# - Preload de essenciais: reduz latência no 1º acesso de hot-paths.
# - Strict imports: falha cedo se algum export não resolve (para CI).
# ---------------------------------------------------------------------------
def _preload_essentials() -> None:
    for name in _ESSENTIALS:
        # getattr dispara o lazy loader e deixa memoizado
        if name not in globals():
            _ = getattr(__import__(__name__, fromlist=[name]), name)


def _strict_validate_exports() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in __all__:
            if name not in globals():
                _ = getattr(__import__(__name__, fromlist=[name]), name)


if _env_flag("HSC_PRELOAD_ESSENTIALS"):
    _preload_essentials()

if _env_flag("HSC_STRICT_IMPORTS"):
    _strict_validate_exports()

# ---------------------------------------------------------------------------
# Suporte a type-checkers sem custo em runtime
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from .component_cache_manager import ComponentCacheManager  # noqa: F401
    from .health_alerting import HealthAlerting  # noqa: F401
    from .health_check_retry import HealthCheckRetry  # noqa: F401
    from .health_check_utils import HealthCheckUtils  # noqa: F401
    from .health_config_manager import HealthCheckConfigurationManager  # noqa: F401
    from .health_history_manager import HealthHistoryManager  # noqa: F401
    from .health_monitoring_coordinator import HealthMonitoringCoordinator  # noqa: F401
    from .health_service_facade import HealthServiceFacade  # noqa: F401
    from .memory_usage_tracker import MemoryUsageTracker  # noqa: F401
    from .monitoring_aggregator import HealthMonitoringAggregator  # noqa: F401
    from .performance_metrics_collector import PerformanceMetricsCollector  # noqa: F401
    from .proactive_monitor import ProactiveHealthMonitor  # noqa: F401
    from .recovery_manager import HealthRecoveryManager  # noqa: F401
    from .unified_health_service import (  # noqa: F401
        UnifiedHealthService,
        get_unified_health_service,
        shutdown_unified_health_service,
    )

    HealthCheckService = UnifiedHealthService  # legacy alias
