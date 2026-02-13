"""
Compatibility layer - Provides prometheus_client-like interface.

This module provides Counter and Histogram classes that match
the prometheus_client API but use our internal metrics system.
"""

from resync.core.metrics_internal import (
    Counter as InternalCounter,
)
from resync.core.metrics_internal import (
    Gauge as InternalGauge,
)
from resync.core.metrics_internal import (
    Histogram as InternalHistogram,
)


class Counter(InternalCounter):
    """prometheus_client-compatible Counter."""


class Histogram(InternalHistogram):
    """prometheus_client-compatible Histogram."""


class Gauge(InternalGauge):
    """prometheus_client-compatible Gauge."""


# Re-export for compatibility
__all__ = ["Counter", "Histogram", "Gauge"]
