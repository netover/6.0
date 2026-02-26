import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

class SystemHealthStatus(str, enum.Enum):
    """Overall system health status enumeration."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"

class HealthStatus(enum.Enum):
    """Health status indicators with color coding."""

    HEALTHY = "healthy"  # Green - All components operational
    DEGRADED = "degraded"  # Yellow - Some components experiencing issues
    UNHEALTHY = "unhealthy"  # Red - Critical components failing
    UNKNOWN = "unknown"  # Gray - Component status unavailable

class ComponentType(enum.Enum):
    """Types of system components for health monitoring."""

    DATABASE = "database"
    REDIS = "redis"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    CPU = "cpu"
    WEBSOCKET = "websocket"
    CACHE = "cache"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONNECTION_POOL = "connection_pool"
    OTHER = "other"

@dataclass
class ComponentHealth:
    """Health status of an individual system component."""

    name: str
    component_type: ComponentType
    status: HealthStatus
    status_code: str | None = None
    message: str | None = None
    response_time_ms: float | None = None
    last_check: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0

@dataclass
class HealthCheckResult:
    """Result of a comprehensive health check."""

    overall_status: HealthStatus
    timestamp: datetime | None
    correlation_id: str | None = None
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    duration_ms: float | None = None  # Duration of health check in milliseconds

    @property
    def status(self) -> HealthStatus:
        """Alias for overall_status for backwards compatibility."""
        return self.overall_status

@dataclass
class HealthCheckConfig:
    """Configuration for health check monitoring."""

    # General settings
    enabled: bool = True
    check_interval_seconds: int = 60
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # Component-specific thresholds
    database_timeout_seconds: int = 10
    database_connection_threshold_percent: float = (
        90.0  # Active connections warning threshold (percentage of total)
    )
    redis_timeout_seconds: int = 5
    external_api_timeout_seconds: int = 15
    file_system_threshold_percent: float = 90.0  # Disk usage warning threshold
    memory_threshold_percent: float = 85.0  # Memory usage warning threshold
    cpu_threshold_percent: float = 80.0  # CPU usage warning threshold

    # Alert settings
    alert_enabled: bool = True
    alert_threshold_degraded: int = 1  # Number of degraded components to trigger alert
    alert_threshold_unhealthy: int = (
        1  # Number of unhealthy components to trigger alert
    )
    alert_cooldown_seconds: int = 300  # Minimum time between alerts

    # Performance monitoring
    track_response_times: bool = True
    track_error_rates: bool = True
    history_retention_hours: int = 24
    error_threshold_count: int = 10  # Number of errors to trigger alert
    response_time_threshold_ms: int = 1000  # Response time threshold in milliseconds

    # Memory bounds configuration
    max_history_entries: int = 1000  # Maximum number of health history entries
    history_cleanup_threshold: float = 0.8  # Cleanup when 80% of max entries reached
    history_cleanup_batch_size: int = 100  # Number of entries to remove per cleanup
    enable_memory_monitoring: bool = True  # Enable memory usage monitoring
    memory_usage_threshold_mb: int = 50  # Alert when memory usage exceeds this (MB)
    history_compression_enabled: bool = False  # Enable compression for old entries
    history_retention_days: int = 7  # Maximum days to retain history

@dataclass
class HealthStatusHistory:
    """Historical record of health status changes."""

    timestamp: datetime | None
    overall_status: HealthStatus
    component_changes: dict[str, HealthStatus] = field(default_factory=dict)
    alerts_triggered: list[str] = field(default_factory=list)
    component_count: int | None = None
    duration_ms: float | None = None

    @property
    def status(self) -> HealthStatus:
        """Alias for overall_status for backwards compatibility."""
        return self.overall_status

class HealthCheckError(Exception):
    """Exception raised when health check fails."""

    def __init__(self, component: str, message: str, status_code: str | None = None):
        self.component = component
        self.message = message
        self.status_code = status_code
        super().__init__(f"Health check failed for {component}: {message}")

def get_status_color(status: HealthStatus) -> str:
    """Get the color associated with a health status."""
    color_map = {
        HealthStatus.HEALTHY: "🟢",
        HealthStatus.DEGRADED: "🟡",
        HealthStatus.UNHEALTHY: "🔴",
        HealthStatus.UNKNOWN: "⚪",
    }
    return color_map.get(status, "⚪")

def get_status_description(status: HealthStatus) -> str:
    """Get human-readable description of health status."""
    description_map = {
        HealthStatus.HEALTHY: "All systems operational",
        HealthStatus.DEGRADED: "Some components experiencing issues",
        HealthStatus.UNHEALTHY: "Critical components failing",
        HealthStatus.UNKNOWN: "Component status unavailable",
    }
    return description_map.get(status, "Unknown status")

@dataclass
class RecoveryResult:
    """Result of a recovery attempt operation."""

    success: bool
    component_name: str
    recovery_type: str
    timestamp: datetime | None
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error_details: str | None = None
    recovery_time_ms: float | None = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
