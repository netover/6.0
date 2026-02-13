"""Health-related API models."""

from resync.core.health import SystemHealthStatus

from .base import BaseModelWithTime


class SystemMetric(BaseModelWithTime):
    """System metric model for API responses."""

    metric_name: str
    value: float
    status: SystemHealthStatus
