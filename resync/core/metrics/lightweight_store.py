"""
Lightweight Metrics Store - PostgreSQL Implementation.

Provides metrics storage using PostgreSQL.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from resync.core.database.models import MetricDataPoint
from resync.core.database.repositories import MetricsStore
from resync.core.shared_types import MetricType

logger = logging.getLogger(__name__)

__all__ = [
    "LightweightMetricsStore",
    "MetricType",
    "MetricPoint",
    "AggregatedMetric",
    "AggregationPeriod",
    "get_metrics_store",
    "record_metric",
    "increment_counter",
    "record_timing",
]


# =============================================================================
# Data Types
# =============================================================================


class AggregationPeriod(str, Enum):
    """Time period for metric aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)
    unit: str | None = None


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""

    name: str
    period: AggregationPeriod
    count: int = 0
    sum: float = 0.0
    min: float = 0.0
    max: float = 0.0
    avg: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None


# =============================================================================
# Lightweight Metrics Store
# =============================================================================


class LightweightMetricsStore:
    """Lightweight Metrics Store - PostgreSQL Backend."""

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._store = MetricsStore()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True
        logger.info("LightweightMetricsStore initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the store."""
        self._initialized = False

    async def record(
        self,
        metric_name: str,
        value: float,
        unit: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> MetricDataPoint:
        """Record a metric data point."""
        return await self._store.record(metric_name, value, unit=unit, tags=tags)

    async def record_metric(self, metric_name: str, value: float, **kwargs) -> MetricDataPoint:
        """Alias for record."""
        return await self.record(metric_name, value, **kwargs)

    async def query(self, metric_name: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Query metric data."""
        return await self._store.query(metric_name, start, end)

    async def get_metric_values(self, metric_name: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get metric values for last N hours."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return await self.query(metric_name, start, end)

    async def get_stats(self, metric_name: str, hours: int = 24) -> dict[str, float]:
        """Get aggregate stats for a metric."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return await self._store.data_points.get_metric_stats(metric_name, start, end)

    async def cleanup(self, days: int = 30) -> int:
        """Clean up old metrics."""
        return await self._store.cleanup(days)

    # Convenience methods for common metrics
    async def record_latency(self, operation: str, latency_ms: float) -> MetricDataPoint:
        """Record latency metric."""
        return await self.record(f"latency.{operation}", latency_ms, unit="ms")

    async def record_count(self, counter_name: str, count: int = 1) -> MetricDataPoint:
        """Record count metric."""
        return await self.record(f"count.{counter_name}", float(count), unit="count")

    async def record_gauge(self, gauge_name: str, value: float) -> MetricDataPoint:
        """Record gauge metric."""
        return await self.record(f"gauge.{gauge_name}", value)


_instance: LightweightMetricsStore | None = None


def get_metrics_store() -> LightweightMetricsStore:
    """Get the singleton LightweightMetricsStore instance."""
    global _instance
    if _instance is None:
        _instance = LightweightMetricsStore()
    return _instance


# =============================================================================
# Convenience Functions
# =============================================================================


async def record_metric(name: str, value: float, **kwargs) -> MetricDataPoint:
    """Record a metric using the global store."""
    store = get_metrics_store()
    return await store.record(name, value, **kwargs)


async def increment_counter(name: str, count: int = 1) -> MetricDataPoint:
    """Increment a counter metric."""
    store = get_metrics_store()
    return await store.record_count(name, count)


async def record_timing(name: str, duration_ms: float) -> MetricDataPoint:
    """Record a timing/latency metric."""
    store = get_metrics_store()
    return await store.record_latency(name, duration_ms)
