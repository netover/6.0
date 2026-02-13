"""
Observability Module - Comprehensive monitoring and tracing.

v5.6.0: Production-ready observability stack.

Components:
- OpenTelemetry distributed tracing
- Prometheus metrics
- Structured logging with correlation IDs
- Health monitoring

Usage:
    from resync.core.observability import (
        setup_telemetry,
        setup_prometheus_metrics,
        get_correlation_id,
    )
"""

from .config import (
    EvidentlyMonitor,
    ObservabilityConfig,
    get_evidently_monitor,
    get_langfuse_client,
    get_observability_config,
    get_observability_status,
    setup_observability,
)
from .telemetry import (
    add_span_attributes,
    create_span,
    get_current_span,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    inject_trace_context,
    is_telemetry_enabled,
    record_exception,
    set_span_status_error,
    setup_prometheus_metrics,
    setup_telemetry,
    shutdown_telemetry,
)

__all__ = [
    # Telemetry
    "setup_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_current_span",
    "get_current_trace_id",
    "get_current_span_id",
    "create_span",
    "add_span_attributes",
    "record_exception",
    "set_span_status_error",
    "inject_trace_context",
    "is_telemetry_enabled",
    # Metrics
    "setup_prometheus_metrics",
    # Evidently Monitor
    "EvidentlyMonitor",
    "get_evidently_monitor",
    # Langfuse
    "get_langfuse_client",
    # Config
    "ObservabilityConfig",
    "get_observability_config",
    "get_observability_status",
    "setup_observability",
]
