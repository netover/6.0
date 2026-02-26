"""
Lightweight Metrics Store - PostgreSQL Implementation.

Provides metrics storage using PostgreSQL.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from threading import Lock

from sqlalchemy import func, select

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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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

    
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._state_lock = Lock()
    async def initialize(self) -> None:
        """Initialize the store.

        Kept async because multiple callers `await` this method.
        """
        self._initialized = True
        logger.info("LightweightMetricsStore initialized (PostgreSQL)")

    
    def initialize_sync(self) -> None:
        """Synchronous alias for initialization (legacy callers)."""
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

    async def record_metric(
        self, metric_name: str, value: float, **kwargs
    ) -> MetricDataPoint:
        """Alias for record."""
        return await self.record(metric_name, value, **kwargs)

    async def query(
        self, metric_name: str, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Query metric data."""
        return await self._store.query(metric_name, start, end)

    async def get_metric_values(
        self, metric_name: str, hours: int = 24
    ) -> list[dict[str, Any]]:
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
    async def record_latency(
        self, operation: str, latency_ms: float
    ) -> MetricDataPoint:
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

async def increment(self, name: str, count: int = 1) -> None:
    """Increment an in-memory counter and persist an event data point."""
    with self._state_lock:
        self._counters[name] = int(self._counters.get(name, 0)) + int(count)
        current = self._counters[name]
    # Persist current value as a datapoint (counter semantics)
    await self.record_count(name, count)

async def set_gauge(self, name: str, value: float) -> None:
    """Set a gauge value (in-memory) and persist a datapoint."""
    with self._state_lock:
        self._gauges[name] = float(value)
    await self.record_gauge(name, float(value))

def get_counter(self, name: str) -> int:
    """Get the current counter value (in-memory)."""
    with self._state_lock:
        return int(self._counters.get(name, 0))

def get_gauge(self, name: str) -> float | None:
    """Get the current gauge value (in-memory)."""
    with self._state_lock:
        return self._gauges.get(name)

async def get_metric_names(self) -> list[str]:
    """Return distinct metric names stored in the database."""
    async with self._store.data_points._get_session() as session:
        result = await session.execute(select(MetricDataPoint.metric_name).distinct())
        return sorted([row[0] for row in result.all() if row[0]])

async def get_summary(self) -> dict[str, Any]:
    """Dashboard summary used by /api/v1/metrics/data."""
    # Storage counts
    async with self._store.data_points._get_session() as session:
        result = await session.execute(select(func.count(MetricDataPoint.id)))
        raw_records = int(result.scalar() or 0)
    with self._state_lock:
        counters = dict(self._counters)
        gauges = dict(self._gauges)
    return {
        "storage": {"raw_records": raw_records, "aggregated_records": 0},
        "counters": counters,
        "gauges": gauges,
    }

@dataclass(slots=True)
class AggregatedSeriesPoint:
    """Aggregated time bucket for charts."""
    period_start: datetime
    count: int
    sum_value: float
    avg_value: float
    min_value: float
    max_value: float

async def get_aggregated(
    self,
    metric_name: str,
    period: str = "hour",
    hours: int = 24,
) -> list["LightweightMetricsStore.AggregatedSeriesPoint"]:
    """Aggregate metrics into time buckets.

    This method is built to satisfy the dashboard API contract (sum_value/avg_value).
    """
    if period not in {"minute", "hour", "day"}:
        period = "hour"
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    # Postgres date_trunc for buckets
    async with self._store.data_points._get_session() as session:
        bucket = func.date_trunc(period, MetricDataPoint.timestamp).label("bucket")
        q = (
            select(
                bucket,
                func.count(MetricDataPoint.id).label("count"),
                func.sum(MetricDataPoint.value).label("sum"),
                func.avg(MetricDataPoint.value).label("avg"),
                func.min(MetricDataPoint.value).label("min"),
                func.max(MetricDataPoint.value).label("max"),
            )
            .where(
                MetricDataPoint.metric_name == metric_name,
                MetricDataPoint.timestamp >= start,
                MetricDataPoint.timestamp <= end,
            )
            .group_by(bucket)
            .order_by(bucket.asc())
        )
        result = await session.execute(q)
        rows = result.all()

    points: list[LightweightMetricsStore.AggregatedSeriesPoint] = []
    for r in rows:
        b = r.bucket
        points.append(
            LightweightMetricsStore.AggregatedSeriesPoint(
                period_start=b if isinstance(b, datetime) else datetime.fromisoformat(str(b)),
                count=int(r.count or 0),
                sum_value=float(r.sum or 0.0),
                avg_value=float(r.avg or 0.0),
                min_value=float(r.min or 0.0),
                max_value=float(r.max or 0.0),
            )
        )
    return points
