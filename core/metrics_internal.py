"""
Internal Metrics System - Replaces Prometheus/Grafana dependencies.

Provides lightweight metrics collection without external dependencies.
Metrics can be exported to JSON for any visualization tool.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric measurement."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """A metric with its metadata and values."""

    name: str
    type: MetricType
    description: str = ""
    values: list[MetricValue] = field(default_factory=list)

    def record(self, value: float, labels: dict[str, str] | None = None):
        """Record a new value."""
        self.values.append(
            MetricValue(
                value=value,
                timestamp=time.time(),
                labels=labels or {},
            )
        )
        # Keep only last 1000 values
        if len(self.values) > 1000:
            self.values = self.values[-1000:]

    def get_current(self) -> float | None:
        """Get most recent value."""
        if self.values:
            return self.values[-1].value
        return None

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "current_value": self.get_current(),
            "values_count": len(self.values),
        }


class Counter:
    """A counter that only increases."""

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = Lock()

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None):
        """Increment the counter."""
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current value."""
        label_key = tuple(sorted((labels or {}).items()))
        return self._values.get(label_key, 0)

    def labels(self, **kwargs) -> "_LabeledCounter":
        """Return counter with specific labels (for compatibility)."""
        return _LabeledCounter(self, kwargs)


class _LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, amount: float = 1):
        self._counter.inc(amount, self._labels)


class Gauge:
    """A gauge that can go up or down."""

    def __init__(self, name: str, description: str = "", labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = Lock()

    def set(self, value: float, labels: dict[str, str] | None = None):
        """Set the gauge value."""
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None):
        """Increment the gauge."""
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1, labels: dict[str, str] = None):
        """Decrement the gauge."""
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] -= amount

    def get(self, labels: dict[str, str] = None) -> float:
        """Get current value."""
        label_key = tuple(sorted((labels or {}).items()))
        return self._values.get(label_key, 0)

    def labels(self, **kwargs) -> "Gauge":
        """Return gauge with specific labels."""
        return _LabeledGauge(self, kwargs)


class _LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float):
        self._gauge.set(value, self._labels)

    def inc(self, amount: float = 1):
        self._gauge.inc(amount, self._labels)

    def dec(self, amount: float = 1):
        self._gauge.dec(amount, self._labels)


class Histogram:
    """A histogram for measuring distributions."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] = None,
        buckets: tuple = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._observations: dict[tuple, list[float]] = defaultdict(list)
        self._lock = Lock()

    def observe(self, value: float, labels: dict[str, str] = None):
        """Record an observation."""
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._observations[label_key].append(value)
            # Keep last 10000 observations
            if len(self._observations[label_key]) > 10000:
                self._observations[label_key] = self._observations[label_key][-10000:]

    def get_percentile(self, percentile: float, labels: dict[str, str] = None) -> float | None:
        """Get a percentile value."""
        label_key = tuple(sorted((labels or {}).items()))
        observations = self._observations.get(label_key, [])
        if not observations:
            return None
        sorted_obs = sorted(observations)
        idx = int(len(sorted_obs) * percentile / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def labels(self, **kwargs) -> "Histogram":
        """Return histogram with specific labels."""
        return _LabeledHistogram(self, kwargs)

    def time(self) -> "_HistogramTimer":
        """Context manager for timing operations."""
        return _HistogramTimer(self)


class _LabeledHistogram:
    """Histogram with pre-set labels."""

    def __init__(self, histogram: Histogram, labels: dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float):
        self._histogram.observe(value, self._labels)

    def time(self) -> "_HistogramTimer":
        return _HistogramTimer(self._histogram, self._labels)


class _HistogramTimer:
    """Context manager for timing."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] = None):
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self._start
        self._histogram.observe(duration, self._labels)


class MetricsRegistry:
    """Central registry for all metrics."""

    _instance: Optional["MetricsRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics: dict[str, Any] = {}
            cls._instance._lock = Lock()
        return cls._instance

    def register(self, metric: Any) -> None:
        """Register a metric."""
        with self._lock:
            self._metrics[metric.name] = metric

    def get(self, name: str) -> Any | None:
        """Get a metric by name."""
        return self._metrics.get(name)

    def get_all(self) -> dict[str, Any]:
        """Get all metrics."""
        return dict(self._metrics)

    def export_json(self) -> dict[str, Any]:
        """Export all metrics to JSON format."""
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
        }

        for name, metric in self._metrics.items():
            if isinstance(metric, Counter):
                result["metrics"][name] = {
                    "type": "counter",
                    "description": metric.description,
                    "values": dict(metric._values),
                }
            elif isinstance(metric, Gauge):
                result["metrics"][name] = {
                    "type": "gauge",
                    "description": metric.description,
                    "values": dict(metric._values),
                }
            elif isinstance(metric, Histogram):
                result["metrics"][name] = {
                    "type": "histogram",
                    "description": metric.description,
                    "p50": metric.get_percentile(50),
                    "p95": metric.get_percentile(95),
                    "p99": metric.get_percentile(99),
                }

        return result


# Global registry
registry = MetricsRegistry()


def create_counter(name: str, description: str = "", labels: list[str] = None) -> Counter:
    """Create and register a counter."""
    counter = Counter(name, description, labels)
    registry.register(counter)
    return counter


def create_gauge(name: str, description: str = "", labels: list[str] = None) -> Gauge:
    """Create and register a gauge."""
    gauge = Gauge(name, description, labels)
    registry.register(gauge)
    return gauge


def create_histogram(
    name: str,
    description: str = "",
    labels: list[str] = None,
    buckets: tuple = None,
) -> Histogram:
    """Create and register a histogram."""
    histogram = Histogram(name, description, labels, buckets)
    registry.register(histogram)
    return histogram
