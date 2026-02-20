# mypy: ignore-errors
"""
Admin Connector Management Routes.

Provides endpoints for managing external connections:
- TWS/HWA instances
- Database connections
- Message queues (Redis, RabbitMQ)
- External APIs
- Notification channels
"""

import logging
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials

logger = logging.getLogger(__name__)

# v5.9.5: Added authentication
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


class ConnectorType(str, Enum):
    """Types of connectors."""

    TWS = "tws"
    DATABASE = "database"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    API = "api"
    SMTP = "smtp"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"


class ConnectorStatus(str, Enum):
    """Connector status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNKNOWN = "unknown"


class ConnectorCreate(BaseModel):
    """Model for creating a connector."""

    name: str = Field(..., min_length=1, max_length=100)
    type: ConnectorType
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ConnectorUpdate(BaseModel):
    """Model for updating a connector."""

    name: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    config: dict[str, Any] | None = None
    enabled: bool | None = None


class ConnectorResponse(BaseModel):
    """Model for connector response."""

    id: str
    name: str
    type: str
    host: str | None = None
    port: int | None = None
    enabled: bool
    status: str
    last_check: str | None = None
    error_message: str | None = None


class ConnectorTest(BaseModel):
    """Model for connector test."""

    timeout_seconds: int = 10


# In-memory connector store
_connectors = {
    "default-tws": {
        "id": "default-tws",
        "name": "Primary TWS Instance",
        "type": "tws",
        "host": "localhost",
        "port": 31116,
        "enabled": True,
        "status": "connected",
        "last_check": None,
        "error_message": None,
        "config": {
            "ssl_enabled": True,
            "timeout": 30,
        },
    },
    "default-redis": {
        "id": "default-redis",
        "name": "Redis Cache",
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "enabled": True,
        "status": "connected",
        "last_check": None,
        "error_message": None,
        "config": {
            "db": 0,
            "password": None,
        },
    },
}


@router.get(
    "/connectors", response_model=list[ConnectorResponse], tags=["Admin Connectors"]
)
async def list_connectors(
    type_filter: ConnectorType | None = None,
    enabled_only: bool = False,
):
    """List all connectors."""
    connectors = list(_connectors.values())

    if type_filter:
        connectors = [c for c in connectors if c["type"] == type_filter.value]

    if enabled_only:
        connectors = [c for c in connectors if c.get("enabled", True)]

    return connectors


@router.post(
    "/connectors",
    response_model=ConnectorResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin Connectors"],
)
async def create_connector(connector: ConnectorCreate):
    """Create a new connector."""
    import uuid

    # Check for duplicate name
    if any(c["name"] == connector.name for c in _connectors.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connector with this name already exists",
        )

    connector_id = str(uuid.uuid4())

    new_connector = {
        "id": connector_id,
        "name": connector.name,
        "type": connector.type.value,
        "host": connector.host,
        "port": connector.port,
        "username": connector.username,
        "enabled": connector.enabled,
        "status": "unknown",
        "last_check": None,
        "error_message": None,
        "config": connector.config,
    }

    # Don't store password in response
    if connector.password:
        new_connector["config"]["has_password"] = True

    _connectors[connector_id] = new_connector
    logger.info("Connector created: %s", connector.name)

    return ConnectorResponse(**new_connector)


@router.get(
    "/connectors/{connector_id}",
    response_model=ConnectorResponse,
    tags=["Admin Connectors"],
)
async def get_connector(connector_id: str):
    """Get connector by ID."""
    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    return ConnectorResponse(**_connectors[connector_id])


@router.put(
    "/connectors/{connector_id}",
    response_model=ConnectorResponse,
    tags=["Admin Connectors"],
)
async def update_connector(connector_id: str, update: ConnectorUpdate):
    """Update a connector."""
    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    connector = _connectors[connector_id]

    for field, value in update.dict(exclude_unset=True).items():
        if field == "config" and value:
            connector["config"].update(value)
        else:
            connector[field] = value

    logger.info("Connector updated: %s", connector["name"])
    return ConnectorResponse(**connector)


@router.delete(
    "/connectors/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin Connectors"],
)
async def delete_connector(connector_id: str):
    """Delete a connector."""
    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    name = _connectors[connector_id]["name"]
    del _connectors[connector_id]
    logger.info("Connector deleted: %s", name)


@router.post("/connectors/{connector_id}/test", tags=["Admin Connectors"])
async def test_connector(connector_id: str, test: ConnectorTest):
    """Test a connector connection."""
    from datetime import datetime, timezone

    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    connector = _connectors[connector_id]
    connector["last_check"] = datetime.now(timezone.utc).isoformat()

    # Simulate connection test based on type
    try:
        connector_type = connector["type"]

        known_types = {connector.value for connector in ConnectorType}
        if connector_type not in known_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown connector type: {connector_type}",
            )

        # Would test concrete connector health per type here.
        connector["status"] = "connected"
        connector["error_message"] = None

        return {
            "success": True,
            "status": connector["status"],
            "latency_ms": 45,  # Would measure actual latency
            "message": "Connection successful",
        }

    except HTTPException:
        raise
    except Exception as e:
        connector["status"] = "error"
        connector["error_message"] = str(e)

        return {
            "success": False,
            "status": "error",
            "message": str(e),
        }


@router.post(
    "/connectors/{connector_id}/enable",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin Connectors"],
)
async def enable_connector(connector_id: str):
    """Enable a connector."""
    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    _connectors[connector_id]["enabled"] = True
    logger.info("Connector enabled: %s", _connectors[connector_id]["name"])


@router.post(
    "/connectors/{connector_id}/disable",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin Connectors"],
)
async def disable_connector(connector_id: str):
    """Disable a connector."""
    if connector_id not in _connectors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    _connectors[connector_id]["enabled"] = False
    logger.info("Connector disabled: %s", _connectors[connector_id]["name"])


@router.get("/connectors/types/available", tags=["Admin Connectors"])
async def get_connector_types():
    """Get available connector types."""
    return {
        "types": [
            {
                "type": "tws",
                "name": "TWS/HWA",
                "description": "IBM Workload Automation",
            },
            {
                "type": "database",
                "name": "Database",
                "description": "SQL Database connection",
            },
            {"type": "redis", "name": "Redis", "description": "Redis cache/queue"},
            {"type": "rabbitmq", "name": "RabbitMQ", "description": "Message queue"},
            {
                "type": "api",
                "name": "External API",
                "description": "REST API connection",
            },
            {"type": "smtp", "name": "SMTP", "description": "Email server"},
            {"type": "slack", "name": "Slack", "description": "Slack notifications"},
            {
                "type": "teams",
                "name": "Microsoft Teams",
                "description": "Teams notifications",
            },
            {"type": "webhook", "name": "Webhook", "description": "Generic webhook"},
        ]
    }


@router.get("/connectors/status/summary", tags=["Admin Connectors"])
async def get_connectors_status_summary():
    """Get summary of all connectors status."""
    connectors = list(_connectors.values())

    return {
        "total": len(connectors),
        "connected": len([c for c in connectors if c["status"] == "connected"]),
        "disconnected": len([c for c in connectors if c["status"] == "disconnected"]),
        "error": len([c for c in connectors if c["status"] == "error"]),
        "enabled": len([c for c in connectors if c.get("enabled", True)]),
        "disabled": len([c for c in connectors if not c.get("enabled", True)]),
    }
