"""
Admin User Management Routes.

Replaces the old in-memory `_users = {}` store (P0-05 in audit report) with the
database-backed AuthService/UserRepository implementation.

Python: 3.14
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from resync.api.auth.models import User, UserCreate, UserRole, UserUpdate
from resync.api.auth.service import get_auth_service
from resync.api.routes.admin.main import verify_admin_credentials

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


class AdminUserCreate(BaseModel):
    """Admin request model for creating a user."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str | None = None
    role: UserRole = UserRole.USER
    is_active: bool = True


class AdminUserResponse(BaseModel):
    """Admin response model (safe for clients)."""

    id: str
    username: str
    email: str
    full_name: str | None = None
    role: str
    is_active: bool
    created_at: str
    permissions: list[str] = []


class BulkUserAction(BaseModel):
    """Model for bulk actions."""

    user_ids: list[str] = Field(..., min_length=1)
    action: str = Field(..., pattern=r"^(activate|deactivate|delete)$")


def _to_admin_response(u: User) -> AdminUserResponse:
    return AdminUserResponse(
        id=u.id,
        username=u.username,
        email=u.email,
        full_name=u.full_name,
        role=u.role.value if hasattr(u.role, "value") else str(u.role),
        is_active=u.is_active,
        created_at=u.created_at.isoformat(),
        permissions=u.permissions,
    )


@router.get("/users", response_model=list[AdminUserResponse], tags=["Admin Users"])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> list[AdminUserResponse]:
    """List users with pagination."""
    service = get_auth_service()
    users = await service.list_users(limit=limit, offset=skip)
    return [_to_admin_response(u) for u in users]


@router.post(
    "/users",
    response_model=AdminUserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin Users"],
)
async def create_user(user: AdminUserCreate) -> AdminUserResponse:
    """Create a new user."""
    service = get_auth_service()
    try:
        created = await service.create_user(
            UserCreate(
                username=user.username,
                email=user.email,
                password=user.password,
                full_name=user.full_name,
                role=user.role,
                is_active=user.is_active,
            )
        )
        return _to_admin_response(created)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/users/{user_id}", response_model=AdminUserResponse, tags=["Admin Users"])
async def get_user(user_id: str) -> AdminUserResponse:
    """Get a user by ID."""
    service = get_auth_service()
    u = await service.get_user(user_id)
    if u is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_admin_response(u)


@router.put("/users/{user_id}", response_model=AdminUserResponse, tags=["Admin Users"])
async def update_user(user_id: str, updates: UserUpdate) -> AdminUserResponse:
    """Update a user."""
    service = get_auth_service()
    updated = await service.update_user(user_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_admin_response(updated)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin Users"])
async def delete_user(user_id: str) -> None:
    """Delete a user."""
    service = get_auth_service()
    ok = await service.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return None


@router.post("/users/bulk", tags=["Admin Users"])
async def bulk_user_action(action: BulkUserAction) -> dict[str, int]:
    """Perform bulk user actions."""
    service = get_auth_service()
    processed = 0
    for uid in action.user_ids:
        if action.action == "delete":
            if await service.delete_user(uid):
                processed += 1
        elif action.action == "activate":
            if await service.update_user(uid, UserUpdate(is_active=True)):
                processed += 1
        elif action.action == "deactivate":
            if await service.update_user(uid, UserUpdate(is_active=False)):
                processed += 1
    return {"processed": processed}
