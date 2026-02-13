"""System monitoring and health check validation models for API endpoints."""

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator
from pydantic import StringConstraints as PydanticStringConstraints

from .common import BaseValidatedModel, ValidationPatterns


class MetricType(str, Enum):
    """System metric types."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    DATABASE = "database"
    CACHE = "cache"
    REQUESTS = "requests"
    RESPONSE_TIME = "response_time"
    ERRORS = "errors"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SystemMetricRequest(BaseValidatedModel):
    """System metric request validation."""

    metric_types: list[MetricType] = Field(
        default_factory=lambda: [MetricType.CPU, MetricType.MEMORY],
        description="Types of metrics to retrieve",
        min_length=1,
        max_length=10,
    )

    time_range: str = Field(
        default="1h",
        pattern=r"^(1h|6h|24h|7d|30d|90d)$",
        description="Time range for metrics",
    )

    aggregation: str = Field(
        default="avg",
        pattern=r"^(avg|min|max|sum|count|p95|p99)$",
        description="Aggregation method",
    )

    granularity: str = Field(
        default="1m", pattern=r"^(1m|5m|15m|1h|6h|24h)$", description="Data granularity"
    )

    format: str = Field(
        default="json", pattern=r"^(json|csv|prometheus)$", description="Output format"
    )

    include_alerts: bool = Field(default=True, description="Include active alerts")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("metric_types")
    @classmethod
    def validate_metric_types(cls, v):
        """Validate metric types."""
        if not v:
            raise ValueError("At least one metric type must be specified")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate metric types found")
        return v


class CustomMetricRequest(BaseValidatedModel):
    """Custom metric submission request validation."""

    metric_name: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
    ] = Field(..., description="Custom metric name")

    metric_value: float = Field(..., description="Metric value")

    metric_type: MetricType = Field(default=MetricType.CUSTOM, description="Metric type")

    timestamp: datetime | None = Field(None, description="Metric timestamp (defaults to now)")

    labels: dict[str, str] | None = Field(
        default_factory=dict, description="Metric labels/dimensions", max_length=10
    )

    unit: str | None = Field(
        None, description="Unit of measurement", pattern=r"^[a-zA-Z0-9_\-/]{1,20}$"
    )

    description: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=500, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Metric description")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v):
        """Validate metric name."""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        # Check for valid metric name format (prometheus compatible)
        if not re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$", v):
            raise ValueError("Invalid metric name format")
        # Check for reserved prefixes
        reserved_prefixes = ["__", "prometheus_", "process_", "go_"]
        for prefix in reserved_prefixes:
            if v.startswith(prefix):
                raise ValueError(f"Metric name cannot start with reserved prefix: {prefix}")
        return v

    @field_validator("metric_value")
    @classmethod
    def validate_metric_value(cls, v):
        """Validate metric value."""
        if not isinstance(v, (int, float)):
            raise ValueError("Metric value must be numeric")
        if abs(v) > 1e15:  # Reasonable limit for metric values
            raise ValueError("Metric value too large")
        if v != v:  # Check for NaN
            raise ValueError("Metric value cannot be NaN")
        if v == float("inf") or v == float("-inf"):
            raise ValueError("Metric value cannot be infinite")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v):
        """Validate metric labels."""
        if not v:
            return v
        for key, value in v.items():
            # Validate label key
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid label key: {key}")
            # Validate label value
            if not value or len(value) > 100:
                raise ValueError(f"Label value too long or empty for key '{key}'")
            if ValidationPatterns.SCRIPT_PATTERN.search(value):
                raise ValueError(f"Label value contains malicious content for key '{key}'")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Validate metric description."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Metric description contains potentially malicious content")
        return v


class AlertRequest(BaseValidatedModel):
    """Alert creation/update request validation."""

    alert_name: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
    ] = Field(..., description="Alert name")

    severity: AlertSeverity = Field(..., description="Alert severity level")

    description: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=1000, strip_whitespace=True)
    ] = Field(..., description="Detailed alert description")

    metric_name: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Related metric name")

    threshold_value: float | None = Field(None, description="Threshold value that triggered alert")

    current_value: float | None = Field(None, description="Current metric value")

    labels: dict[str, str] | None = Field(
        default_factory=dict, description="Alert labels/dimensions", max_length=10
    )

    auto_resolve: bool = Field(
        default=False, description="Whether alert auto-resolves when condition clears"
    )

    notification_channels: list[str] | None = Field(
        None, description="Notification channels to use", max_length=5
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("alert_name")
    @classmethod
    def validate_alert_name(cls, v):
        """Validate alert name."""
        if not v or not v.strip():
            raise ValueError("Alert name cannot be empty")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Alert name contains potentially malicious content")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        """Validate alert description."""
        if not v or not v.strip():
            raise ValueError("Alert description cannot be empty")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Alert description contains malicious content")
        return v

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v):
        """Validate metric name if provided."""
        if v is None:
            return v
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        if not re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$", v):
            raise ValueError("Invalid metric name format")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v):
        """Validate alert labels."""
        if not v:
            return v
        for key, value in v.items():
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid label key: {key}")
            if not value or len(value) > 100:
                raise ValueError(f"Label value too long or empty for key '{key}'")
            if ValidationPatterns.SCRIPT_PATTERN.search(value):
                raise ValueError(f"Label value contains malicious content for key '{key}'")
        return v

    @field_validator("notification_channels")
    @classmethod
    def validate_notification_channels(cls, v):
        """Validate notification channels."""
        if v is None:
            return v
        valid_channels = {"email", "slack", "webhook", "sms", "push"}
        invalid_channels = set(v) - valid_channels
        if invalid_channels:
            raise ValueError(f"Invalid notification channels: {invalid_channels}")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate notification channels found")
        return v


class AlertQueryParams(BaseValidatedModel):
    """Alert query parameters validation."""

    status: AlertStatus | None = Field(None, description="Filter by alert status")

    severity: list[AlertSeverity] | None = Field(
        None, description="Filter by severity levels", max_length=4
    )

    alert_name: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Filter by alert name (partial match)")

    metric_name: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Filter by metric name")

    time_range: str = Field(
        default="24h",
        pattern=r"^(1h|6h|24h|7d|30d|90d|all)$",
        description="Time range for alerts",
    )

    include_resolved: bool = Field(default=False, description="Include resolved alerts")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity_list(cls, v):
        """Validate severity list."""
        if v is None:
            return v
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate severity levels found")
        return v

    @field_validator("alert_name", "metric_name")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields."""
        if v is None:
            return v
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Field contains potentially malicious content")
        return v


class HealthCheckRequest(BaseValidatedModel):
    """Health check request validation."""

    component: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Specific component to check")

    depth: str = Field(
        default="basic",
        pattern=r"^(basic|detailed|deep)$",
        description="Health check depth",
    )

    timeout: int = Field(default=30, ge=1, le=300, description="Health check timeout in seconds")

    include_dependencies: bool = Field(default=True, description="Include dependency health checks")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("component")
    @classmethod
    def validate_component(cls, v):
        """Validate component name."""
        if v is None:
            return v
        if not v or not v.strip():
            raise ValueError("Component name cannot be empty")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Component name contains potentially malicious content")
        return v


class LogQueryParams(BaseValidatedModel):
    """Log query parameters validation."""

    level: list[str] | None = Field(None, description="Filter by log levels", max_length=5)

    component: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Filter by component")

    search: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=200, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Search in log messages")

    time_range: str = Field(
        default="1h", pattern=r"^(1h|6h|24h|7d|30d)$", description="Time range for logs"
    )

    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of log entries")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("level")
    @classmethod
    def validate_log_levels(cls, v):
        """Validate log levels."""
        if v is None:
            return v
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        invalid_levels = set(v) - valid_levels
        if invalid_levels:
            raise ValueError(f"Invalid log levels: {invalid_levels}")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate log levels found")
        return v

    @field_validator("component", "search")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields."""
        if v is None:
            return v
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Field contains potentially malicious content")
        return v


class PerformanceTestRequest(BaseValidatedModel):
    """Performance test request validation."""

    test_type: str = Field(
        ...,
        pattern=r"^(load|stress|spike|endurance)$",
        description="Type of performance test",
    )

    target_endpoint: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=200, strip_whitespace=True)
    ] = Field(..., description="Target endpoint to test")

    concurrent_users: int = Field(..., ge=1, le=1000, description="Number of concurrent users")

    test_duration: int = Field(..., ge=1, le=3600, description="Test duration in seconds")

    ramp_up_time: int = Field(default=0, ge=0, le=300, description="Ramp up time in seconds")

    success_criteria: dict[str, Any] | None = Field(
        default_factory=dict, description="Success criteria for the test", max_length=10
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("target_endpoint")
    @classmethod
    def validate_endpoint(cls, v):
        """Validate target endpoint."""
        if not v or not v.strip():
            raise ValueError("Target endpoint cannot be empty")
        if not v.startswith("/"):
            raise ValueError("Target endpoint must start with /")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Target endpoint contains potentially malicious content")
        return v

    @field_validator("success_criteria")
    @classmethod
    def validate_success_criteria(cls, v):
        """Validate success criteria."""
        if not v:
            return v
        valid_criteria = {
            "response_time_ms",
            "error_rate",
            "throughput_rps",
            "p95_response_time_ms",
            "p99_response_time_ms",
        }
        for key, value in v.items():
            if key not in valid_criteria:
                raise ValueError(f"Invalid success criterion: {key}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Success criterion '{key}' must be numeric")
            if value <= 0:
                raise ValueError(f"Success criterion '{key}' must be positive")
        return v


__all__ = [
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "HealthStatus",
    "SystemMetricRequest",
    "CustomMetricRequest",
    "AlertRequest",
    "AlertQueryParams",
    "HealthCheckRequest",
    "LogQueryParams",
    "PerformanceTestRequest",
]
