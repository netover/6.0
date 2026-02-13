"""
TWS Instance Model.

Represents a single TWS/HWA server connection with its configuration.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TWSInstanceStatus(str, Enum):
    """Status of a TWS instance."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TWSEnvironment(str, Enum):
    """TWS environment type."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    DR = "disaster_recovery"


@dataclass
class TWSInstanceConfig:
    """
    Configuration for a TWS instance.

    Each instance has its own connection settings, credentials,
    and behavioral configuration.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # e.g., "SAZ", "NAZ", "MAZ"
    display_name: str = ""  # e.g., "São Paulo - SAZ"
    description: str = ""

    # Connection
    host: str = "localhost"
    port: int = 31116
    ssl_enabled: bool = True
    ssl_verify: bool = True
    ssl_cert_path: str | None = None

    # Authentication
    username: str = ""
    password: str = ""
    api_key: str | None = None

    # Timeouts
    connect_timeout: int = 30
    read_timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 5

    # Environment
    environment: TWSEnvironment = TWSEnvironment.PRODUCTION
    datacenter: str = ""  # e.g., "SAZ", "NAZ"
    region: str = ""  # e.g., "South America", "North America"

    # Features
    enabled: bool = True
    read_only: bool = False
    allow_job_execution: bool = True
    allow_schedule_changes: bool = False

    # Learning
    learning_enabled: bool = True
    learning_retention_days: int = 90

    # UI
    color: str = "#3B82F6"  # Blue default
    icon: str = "server"
    sort_order: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "host": self.host,
            "port": self.port,
            "ssl_enabled": self.ssl_enabled,
            "environment": self.environment.value,
            "datacenter": self.datacenter,
            "region": self.region,
            "enabled": self.enabled,
            "read_only": self.read_only,
            "allow_job_execution": self.allow_job_execution,
            "learning_enabled": self.learning_enabled,
            "color": self.color,
            "icon": self.icon,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TWSInstanceConfig":
        """Create from dictionary."""
        # Handle enum conversion
        if "environment" in data and isinstance(data["environment"], str):
            data["environment"] = TWSEnvironment(data["environment"])

        # Handle datetime conversion
        for dt_field in ["created_at", "updated_at"]:
            if dt_field in data and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class TWSInstance:
    """
    A TWS instance with its configuration and runtime state.
    """

    config: TWSInstanceConfig
    status: TWSInstanceStatus = TWSInstanceStatus.DISCONNECTED

    # Runtime state
    last_connected: datetime | None = None
    last_error: str | None = None
    error_count: int = 0

    # Metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0

    # Active sessions
    active_sessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with runtime state."""
        result = self.config.to_dict()
        result.update(
            {
                "status": self.status.value,
                "last_connected": self.last_connected.isoformat() if self.last_connected else None,
                "last_error": self.last_error,
                "error_count": self.error_count,
                "metrics": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "success_rate": (self.successful_requests / self.total_requests * 100)
                    if self.total_requests > 0
                    else 0,
                    "avg_response_time_ms": self.avg_response_time_ms,
                },
                "active_sessions": self.active_sessions,
            }
        )
        return result

    @property
    def connection_url(self) -> str:
        """Get connection URL."""
        protocol = "https" if self.config.ssl_enabled else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return self.status == TWSInstanceStatus.CONNECTED and self.error_count < 3
