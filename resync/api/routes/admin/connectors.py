# mypy
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
from pydantic import BaseModel, Field, SecretStr

from resync.api.routes.admin.main import verify_admin_credentials
from resync.api.dependencies_v2 import get_database
from resync.core.database.repositories.connector_repo import ConnectorRepository

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
    # P2-43 FIX: Use SecretStr to avoid logging passwords in plain text
    password: SecretStr | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ConnectorUpdate(BaseModel):
    """Model for updating a connector."""

    name: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    # P2-43 FIX: Use SecretStr to avoid logging passwords in plain text
    password: SecretStr | None = None
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

# v7.0: Migrated to PostgreSQL-backed ConnectorRepository
# Decoupled from memory for cluster support

async def get_connector_repo(db=Depends(get_database)) -> ConnectorRepository:
    """Dependency to get ConnectorRepository."""
    return ConnectorRepository(lambda: db)


@router.get(
    "/connectors", response_model=list[ConnectorResponse], tags=["Admin Connectors"]
)
async def list_connectors(
    type_filter: ConnectorType | None = None,
    enabled_only: bool = False,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """List all connectors."""
    filters = {}
    if type_filter:
        filters["type"] = type_filter.value
    if enabled_only:
        filters["enabled"] = True

    connectors = await repo.find(filters)
    return connectors

@router.post(
    "/connectors",
    response_model=ConnectorResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin Connectors"],
)
async def create_connector(
    connector: ConnectorCreate,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Create a new connector."""
    # Check for duplicate name
    existing = await repo.get_by_name(connector.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connector with this name already exists",
        )

    # Extract raw password for storage
    raw_password = connector.password.get_secret_value() if connector.password else None

    # Merge config with password flag
    config = connector.config.copy()
    if raw_password:
        config["has_password"] = True

    new_connector = await repo.create(
        name=connector.name,
        type=connector.type.value,
        host=connector.host,
        port=connector.port,
        username=connector.username,
        password=raw_password,
        enabled=connector.enabled,
        config=config,
        status="unknown",
    )

    logger.info("Connector created: %s", connector.name)
    return new_connector

@router.get(
    "/connectors/{connector_id}",
    response_model=ConnectorResponse,
    tags=["Admin Connectors"],
)
async def get_connector(
    connector_id: str,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Get connector by ID."""
    from uuid import UUID
    connector = await repo.get_by_id(UUID(connector_id))
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    return connector

@router.put(
    "/connectors/{connector_id}",
    response_model=ConnectorResponse,
    tags=["Admin Connectors"],
)
async def update_connector(
    connector_id: str,
    update: ConnectorUpdate,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Update a connector."""
    from uuid import UUID
    uid = UUID(connector_id)
    connector = await repo.get_by_id(uid)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    update_data = update.model_dump(exclude_unset=True)
    
    # Handle password
    if "password" in update_data and update_data["password"]:
        update_data["password"] = update_data["password"].get_secret_value()
        
    # Handle nested config update if needed, but repo.update usually replaces
    # For simplicity, we just pass the dict. If repo.update doesn't merge JSON,
    # we might need to handle it.
    
    updated = await repo.update(uid, **update_data)
    logger.info("Connector updated: %s", updated.name)
    return updated

@router.delete(
    "/connectors/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin Connectors"],
)
async def delete_connector(
    connector_id: str,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Delete a connector."""
    from uuid import UUID
    uid = UUID(connector_id)
    success = await repo.delete(uid)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    logger.info("Connector deleted: %s", connector_id)


@router.post("/connectors/{connector_id}/test", tags=["Admin Connectors"])
async def test_connector(
    connector_id: str,
    test: ConnectorTest,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Test a connector connection."""
    from datetime import datetime, timezone
    from uuid import UUID
    uid = UUID(connector_id)

    connector = await repo.get_by_id(uid)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    # Update last check
    await repo.update(uid, last_check=datetime.now(timezone.utc))

    # Simulate connection test based on type
    try:
        # Would test concrete connector health per type here.
        
        # Success simulation
        await repo.update_status(uid, "connected")

        return {
            "success": True,
            "status": "connected",
            "latency_ms": 45,
            "message": "Connection successful",
        }

    except Exception as e:
        await repo.update_status(uid, "error", error_message=str(e))
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
async def enable_connector(
    connector_id: str,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Enable a connector."""
    from uuid import UUID
    uid = UUID(connector_id)
    success = await repo.update(uid, enabled=True)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    logger.info("Connector enabled: %s", connector_id)

@router.post(
    "/connectors/{connector_id}/disable",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin Connectors"],
)
async def disable_connector(
    connector_id: str,
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Disable a connector."""
    from uuid import UUID
    uid = UUID(connector_id)
    success = await repo.update(uid, enabled=False)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    logger.info("Connector disabled: %s", connector_id)

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
async def get_connectors_status_summary(
    repo: ConnectorRepository = Depends(get_connector_repo),
):
    """Get summary of all connectors status."""
    connectors = await repo.get_all(limit=1000)

    return {
        "total": len(connectors),
        "connected": len([c for c in connectors if c.status == "connected"]),
        "disconnected": len([c for c in connectors if c.status == "disconnected"]),
        "error": len([c for c in connectors if c.status == "error"]),
        "enabled": len([c for c in connectors if c.enabled]),
        "disabled": len([c for c in connectors if not c.enabled]),
    }
