"""
Admin TWS Multi-Instance Management Routes.

Provides endpoints for managing multiple TWS/HWA server connections:
- Instance CRUD
- Connection management
- Learning data per instance
- Session management (tabs)
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials

logger = logging.getLogger(__name__)

# v5.9.5: Added authentication
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


# Pydantic Models


class TWSInstanceCreate(BaseModel):
    """Model for creating a TWS instance."""

    name: str = Field(
        ..., min_length=1, max_length=50, description="Instance name (e.g., SAZ, NAZ)"
    )
    display_name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    host: str = Field(..., min_length=1)
    port: int = Field(default=31116, ge=1, le=65535)
    username: str = ""
    password: str = ""
    ssl_enabled: bool = True
    environment: str = "production"
    datacenter: str = ""
    region: str = ""
    color: str = "#3B82F6"
    read_only: bool = False
    learning_enabled: bool = True


class TWSInstanceUpdate(BaseModel):
    """Model for updating a TWS instance."""

    display_name: str | None = None
    description: str | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    ssl_enabled: bool | None = None
    enabled: bool | None = None
    color: str | None = None
    sort_order: int | None = None
    read_only: bool | None = None
    learning_enabled: bool | None = None


class SessionCreate(BaseModel):
    """Model for creating a session."""

    instance_id: str


# Lazy import to avoid circular dependencies
def _get_manager():
    from resync.core.tws_multi import get_tws_manager

    return get_tws_manager()


def _get_config_class():
    from resync.core.tws_multi.instance import TWSEnvironment, TWSInstanceConfig

    return TWSInstanceConfig, TWSEnvironment


# Instance Endpoints


@router.get("/tws-instances", tags=["TWS Instances"])
async def list_tws_instances():
    """
    List all TWS instances.

    Returns all configured TWS/HWA servers with their status.
    """
    manager = _get_manager()
    instances = manager.get_all_instances()

    return {
        "instances": [inst.to_dict() for inst in instances],
        "total": len(instances),
    }


@router.get("/tws-instances/summary", tags=["TWS Instances"])
async def get_instances_summary():
    """Get summary of all TWS instances."""
    manager = _get_manager()
    return manager.get_summary()


@router.post(
    "/tws-instances", status_code=status.HTTP_201_CREATED, tags=["TWS Instances"]
)
async def create_tws_instance(instance: TWSInstanceCreate):
    """
    Create a new TWS instance.

    Example:
    - SAZ → tws.saz.com.br:31116
    - NAZ → tws.naz.com:31116
    """
    manager = _get_manager()
    tws_instance_config_cls, tws_environment_cls = _get_config_class()

    try:
        config = tws_instance_config_cls(
            name=instance.name.upper(),
            display_name=instance.display_name,
            description=instance.description,
            host=instance.host,
            port=instance.port,
            username=instance.username,
            password=instance.password,
            ssl_enabled=instance.ssl_enabled,
            environment=tws_environment_cls(instance.environment),
            datacenter=instance.datacenter or instance.name.upper(),
            region=instance.region,
            color=instance.color,
            read_only=instance.read_only,
            learning_enabled=instance.learning_enabled,
        )

        new_instance = manager.add_instance(config)

        return {
            "success": True,
            "instance": new_instance.to_dict(),
            "message": f"TWS instance '{instance.name}' created successfully",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Internal server error. Check server logs for details.",
        ) from e


@router.get("/tws-instances/{instance_id}", tags=["TWS Instances"])
async def get_tws_instance(instance_id: str):
    """Get a specific TWS instance."""
    manager = _get_manager()
    instance = manager.get_instance(instance_id)

    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    # Include learning summary
    learning_store = manager.get_learning_store(instance_id)
    learning_summary = learning_store.get_learning_summary() if learning_store else None

    result = instance.to_dict()
    result["learning"] = learning_summary

    return result


@router.put("/tws-instances/{instance_id}", tags=["TWS Instances"])
async def update_tws_instance(instance_id: str, update: TWSInstanceUpdate):
    """Update a TWS instance."""
    manager = _get_manager()

    updates = update.dict(exclude_unset=True)
    instance = manager.update_instance(instance_id, updates)

    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    return {
        "success": True,
        "instance": instance.to_dict(),
        "message": f"Instance '{instance.config.name}' updated successfully",
    }


@router.delete("/tws-instances/{instance_id}", tags=["TWS Instances"])
async def delete_tws_instance(instance_id: str):
    """Delete a TWS instance."""
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    name = instance.config.name
    manager.delete_instance(instance_id)

    return {
        "success": True,
        "message": f"Instance '{name}' deleted successfully",
    }


# Connection Endpoints


@router.post("/tws-instances/{instance_id}/connect", tags=["TWS Instances"])
async def connect_tws_instance(instance_id: str):
    """Connect to a TWS instance."""
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    success = await manager.connect_instance(instance_id)

    return {
        "success": success,
        "status": instance.status.value,
        "message": "Connected successfully"
        if success
        else f"Connection failed: {instance.last_error}",
    }


@router.post("/tws-instances/{instance_id}/disconnect", tags=["TWS Instances"])
async def disconnect_tws_instance(instance_id: str):
    """Disconnect from a TWS instance."""
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    await manager.disconnect_instance(instance_id)

    return {
        "success": True,
        "status": "disconnected",
        "message": f"Disconnected from '{instance.config.name}'",
    }


@router.post("/tws-instances/{instance_id}/test", tags=["TWS Instances"])
async def test_tws_connection(instance_id: str):
    """Test connection to a TWS instance."""
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    return await manager.test_connection(instance_id)


# Session Endpoints (for tabs)


@router.post("/tws-instances/{instance_id}/sessions", tags=["TWS Sessions"])
async def create_session(
    instance_id: str,
    user_id: str = "default",
    username: str = "operator",
):
    """
    Create a new session (tab) for a TWS instance.

    Each session is isolated and allows the operator to work
    independently on that specific TWS server.
    """
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    session = manager.create_session(
        instance_id=instance_id,
        user_id=user_id,
        username=username,
    )

    return {
        "success": True,
        "session": session.to_dict(),
        "message": f"Session created for '{instance.config.name}'",
    }


@router.get("/tws-sessions", tags=["TWS Sessions"])
async def list_user_sessions(user_id: str = "default"):
    """List all sessions for a user."""
    manager = _get_manager()
    sessions = manager.get_user_sessions(user_id)

    return {
        "sessions": [s.to_dict() for s in sessions],
        "total": len(sessions),
    }


@router.delete("/tws-sessions/{session_id}", tags=["TWS Sessions"])
async def close_session(session_id: str):
    """Close a session (tab)."""
    manager = _get_manager()
    manager.close_session(session_id)

    return {
        "success": True,
        "message": "Session closed",
    }


# Learning Endpoints


@router.get("/tws-instances/{instance_id}/learning", tags=["TWS Learning"])
async def get_instance_learning(instance_id: str):
    """Get learning data for a specific instance."""
    manager = _get_manager()

    instance = manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    learning_store = manager.get_learning_store(instance_id)
    if not learning_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning store not found",
        )

    return {
        "instance_id": instance_id,
        "instance_name": instance.config.name,
        "summary": learning_store.get_learning_summary(),
        "job_patterns_count": len(learning_store.job_patterns),
        "known_error_codes": len(learning_store.failure_resolutions),
    }


@router.get("/tws-instances/{instance_id}/learning/patterns", tags=["TWS Learning"])
async def get_job_patterns(instance_id: str, limit: int = 100):
    """Get learned job patterns for an instance."""
    manager = _get_manager()
    learning_store = manager.get_learning_store(instance_id)

    if not learning_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    patterns = learning_store.get_all_patterns()[:limit]

    return {
        "instance_id": instance_id,
        "patterns": [p.to_dict() for p in patterns],
        "total": len(patterns),
    }


@router.post("/tws-instances/{instance_id}/learning/export", tags=["TWS Learning"])
async def export_learning_data(instance_id: str):
    """Export learning data for an instance."""
    manager = _get_manager()
    learning_store = manager.get_learning_store(instance_id)

    if not learning_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    return learning_store.export_learning_data()


@router.post("/tws-instances/{instance_id}/learning/import", tags=["TWS Learning"])
async def import_learning_data(instance_id: str, data: dict[str, Any]):
    """Import learning data to an instance."""
    manager = _get_manager()
    learning_store = manager.get_learning_store(instance_id)

    if not learning_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    learning_store.import_learning_data(data)

    return {
        "success": True,
        "message": "Learning data imported successfully",
    }


@router.delete("/tws-instances/{instance_id}/learning", tags=["TWS Learning"])
async def clear_learning_data(instance_id: str, confirm: bool = False):
    """Clear all learning data for an instance."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Set confirm=true to confirm deletion",
        )

    manager = _get_manager()
    learning_store = manager.get_learning_store(instance_id)

    if not learning_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found",
        )

    learning_store.clear_learning_data()

    return {
        "success": True,
        "message": "Learning data cleared",
    }


# Bulk Operations


@router.post("/tws-instances/bulk/connect", tags=["TWS Instances"])
async def connect_all_instances():
    """Connect to all enabled instances."""
    manager = _get_manager()
    results = {}

    for instance in manager.get_all_instances():
        if instance.config.enabled:
            success = await manager.connect_instance(instance.config.id)
            results[instance.config.name] = {
                "success": success,
                "status": instance.status.value,
            }

    return {
        "results": results,
        "connected": len([r for r in results.values() if r["success"]]),
        "failed": len([r for r in results.values() if not r["success"]]),
    }


@router.post("/tws-instances/bulk/disconnect", tags=["TWS Instances"])
async def disconnect_all_instances():
    """Disconnect from all instances."""
    manager = _get_manager()

    for instance in manager.get_all_instances():
        await manager.disconnect_instance(instance.config.id)

    return {
        "success": True,
        "message": "All instances disconnected",
    }
