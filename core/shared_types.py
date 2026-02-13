"""
Shared Types Module - Canonical Definitions for Common Classes

This module provides the single source of truth for commonly used types
across the Resync application, eliminating duplicate class definitions.

All modules should import these types from here rather than defining their own.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# CACHE TYPES
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """
    Represents a single entry in the cache with timestamp and TTL.

    This is the canonical definition - all cache implementations should use this.

    Attributes:
        data: The cached value (also accessible via .value property)
        timestamp: Unix timestamp when the entry was created (also accessible via .created_at)
        ttl: Time-to-live in seconds
        access_count: Number of times this entry has been accessed (for LRU)
        last_access: Timestamp of last access (also accessible via .last_accessed)
        size_bytes: Estimated size in bytes (for memory management)
    """

    data: T
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # Default 5 minutes
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0

    # Aliases for compatibility with different cache implementations
    @property
    def value(self) -> T:
        """Alias for data field."""
        return self.data

    @property
    def created_at(self) -> float:
        """Alias for timestamp field."""
        return self.timestamp

    @property
    def last_accessed(self) -> float:
        """Alias for last_access field."""
        return self.last_access

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds."""
        if self.ttl is None:
            return float("inf")
        remaining = (self.timestamp + self.ttl) - time.time()
        return max(0.0, remaining)

    def touch(self) -> None:
        """Update last access time and increment access count."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """
    Statistics for cache performance monitoring.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        size: Current number of entries
        evictions: Number of entries evicted
        avg_ttl: Average TTL of entries
        memory_bytes: Estimated memory usage
    """

    hits: int = 0
    misses: int = 0
    size: int = 0
    evictions: int = 0
    avg_ttl: float = 0.0
    memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# CIRCUIT BREAKER TYPES
# =============================================================================


class CircuitBreakerState(str, Enum):
    """
    States for circuit breaker pattern.

    CLOSED: Normal operation, requests pass through
    OPEN: Circuit is open, requests fail fast
    HALF_OPEN: Testing if service has recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Successes needed in half-open to close
        timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Max calls allowed in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3


# =============================================================================
# ALERT TYPES
# =============================================================================


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """
    Rule for triggering alerts.

    Attributes:
        name: Unique identifier for this rule
        condition: Lambda or callable that returns True when alert should trigger
        severity: Severity level of the alert
        message_template: Template for alert message
        cooldown_seconds: Minimum time between alerts
        enabled: Whether this rule is active
    """

    name: str
    condition: Callable[..., bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = ""
    cooldown_seconds: int = 300
    enabled: bool = True
    last_triggered: float | None = None


@dataclass
class Alert:
    """
    Represents a triggered alert.

    Attributes:
        rule_name: Name of the rule that triggered this alert
        severity: Severity level
        message: Alert message
        timestamp: When the alert was triggered
        context: Additional context data
        acknowledged: Whether alert has been acknowledged
    """

    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


# =============================================================================
# METRIC TYPES
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PerformanceMetrics:
    """
    Performance metrics snapshot.

    Attributes:
        cpu_percent: CPU utilization percentage
        memory_percent: Memory utilization percentage
        disk_percent: Disk utilization percentage
        request_latency_ms: Average request latency in milliseconds
        requests_per_second: Current requests per second
        error_rate: Percentage of failed requests
        active_connections: Number of active connections
        timestamp: When metrics were collected
    """

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    request_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# REQUEST/RESPONSE TYPES
# =============================================================================


@dataclass
class LoginRequest:
    """
    Standard login request structure.

    Attributes:
        username: User's username or email
        password: User's password (should be transmitted securely)
        remember_me: Whether to create a persistent session
        mfa_code: Optional MFA code if enabled
    """

    username: str
    password: str
    remember_me: bool = False
    mfa_code: str | None = None


@dataclass
class APIKeyRequest:
    """
    Request for API key operations.

    Attributes:
        name: Descriptive name for the API key
        scopes: List of permission scopes
        expires_in_days: Number of days until expiration (None = never)
    """

    name: str
    scopes: list[str] = field(default_factory=list)
    expires_in_days: int | None = None


@dataclass
class FileUploadRequest:
    """
    Request for file upload operations.

    Attributes:
        filename: Original filename
        content_type: MIME type of the file
        size_bytes: Size of the file in bytes
        checksum: Optional checksum for verification
    """

    filename: str
    content_type: str
    size_bytes: int
    checksum: str | None = None


# =============================================================================
# HEALTH CHECK TYPES
# =============================================================================


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """
    Health status of a single component.

    Attributes:
        name: Component name
        status: Health status
        message: Optional status message
        latency_ms: Response latency in milliseconds
        last_check: Timestamp of last health check
        metadata: Additional health metadata
    """

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Cache types
    "CacheEntry",
    "CacheStats",
    # Circuit breaker types
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    # Alert types
    "AlertSeverity",
    "AlertRule",
    "Alert",
    # Metric types
    "MetricType",
    "PerformanceMetrics",
    # Request types
    "LoginRequest",
    "APIKeyRequest",
    "FileUploadRequest",
    # Health types
    "HealthStatus",
    "ComponentHealth",
]
