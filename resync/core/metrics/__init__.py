"""
Lightweight Metrics Module.

Provides SQLite-based metrics storage and collection without external dependencies.
Designed for ~13,000 jobs/day with minimal resource usage (~15MB RAM).

Components:
- LightweightMetricsStore: SQLite-based metrics storage
- ContinualLearningMetrics: Metrics collector for CL components
- Dashboard API: REST endpoints for metrics visualization

Usage:
    from resync.core.metrics import (
        get_metrics_store,
        get_cl_metrics,
        record_metric,
        increment_counter,
        record_timing,
    )

    # Record metrics
    await increment_counter("query_count")
    await record_timing("response_time_ms", 150.5)

    # Get dashboard data
    metrics = get_cl_metrics()
    data = await metrics.get_dashboard_data(hours=24)
"""

# Re-export log_with_correlation for backward compatibility
from resync.core.logger import log_with_correlation
from resync.core.metrics.continual_learning_metrics import (
    ContinualLearningMetrics,
    MetricNames,
    QueryMetrics,
    get_cl_metrics,
)
from resync.core.metrics.lightweight_store import (
    AggregatedMetric,
    AggregationPeriod,
    LightweightMetricsStore,
    MetricPoint,
    MetricType,
    get_metrics_store,
    increment_counter,
    record_metric,
    record_timing,
)
from resync.core.metrics.runtime_metrics import RuntimeMetricsCollector, runtime_metrics

__all__ = [
    # Store
    "LightweightMetricsStore",
    "MetricType",
    "MetricPoint",
    "AggregatedMetric",
    "AggregationPeriod",
    "get_metrics_store",
    # Convenience functions
    "record_metric",
    "increment_counter",
    "record_timing",
    # Runtime metrics
    "RuntimeMetricsCollector",
    "runtime_metrics",
    "get_runtime_metrics",
    # Continual Learning
    "ContinualLearningMetrics",
    "MetricNames",
    "QueryMetrics",
    "get_cl_metrics",
    # Logging
    "log_with_correlation",
]

def get_runtime_metrics() -> RuntimeMetricsCollector:
    """Return the singleton runtime metrics collector.

    Kept for backward compatibility with older imports expecting a callable.
    """
    return runtime_metrics
