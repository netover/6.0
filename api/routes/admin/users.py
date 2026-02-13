"""
Admin User Management Routes.

Provides endpoints for:
- User CRUD operations
- Password management
- Role/permission management
- Account status management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from resync.api.routes.admin.main import verify_admin_credentials

logger = logging.getLogger(__name__)

# v5.9.5: Added authentication
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


# Pydantic models for API
class UserCreate(BaseModel):
    """Model for creating a user."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str | None = None
    role: str = "user"


class UserUpdate(BaseModel):
    """Model for updating a user."""

    email: EmailStr | None = None
    full_name: str | None = None
    role: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    """Model for user response."""

    id: str
    username: str
    email: str
    full_name: str | None
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: str | None


class PasswordChange(BaseModel):
    """Model for password change."""

    current_password: str
    new_password: str = Field(..., min_length=8)


class BulkUserAction(BaseModel):
    """Model for bulk user actions."""

    user_ids: list[str]
    action: str  # activate, deactivate, delete


# In-memory user store (replace with database in production)
_users = {}


@router.get("/users", response_model=list[UserResponse], tags=["Admin Users"])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = False,
):
    """List all users with pagination."""
    users = list(_users.values())

    if active_only:
        users = [u for u in users if u.get("is_active", True)]

    return users[skip : skip + limit]


@router.post(
    "/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Admin Users"]
)
async def create_user(user: UserCreate):
    """Create a new user."""
    import uuid
    from datetime import datetime, timezone

    # Check if username exists
    if any(u["username"] == user.username for u in _users.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Check if email exists
    if any(u["email"] == user.email for u in _users.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    user_id = str(uuid.uuid4())

    # Hash password (in production, use proper hashing)
    from resync.api.core.security import get_password_hash

    hashed_password = get_password_hash(user.password)

    new_user = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_login": None,
    }

    _users[user_id] = new_user
    logger.info("User created: %s", user.username)

    return UserResponse(**new_user)


@router.get("/users/{user_id}", response_model=UserResponse, tags=["Admin Users"])
async def get_user(user_id: str):
    """Get user by ID."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(**_users[user_id])


@router.put("/users/{user_id}", response_model=UserResponse, tags=["Admin Users"])
async def update_user(user_id: str, user_update: UserUpdate):
    """Update user details."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user = _users[user_id]

    for field, value in user_update.dict(exclude_unset=True).items():
        user[field] = value

    logger.info("User updated: %s", user['username'])
    return UserResponse(**user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin Users"])
async def delete_user(user_id: str):
    """Delete a user."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    username = _users[user_id]["username"]
    del _users[user_id]
    logger.info("User deleted: %s", username)


@router.post(
    "/users/{user_id}/change-password", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin Users"]
)
async def change_password(user_id: str, password_change: PasswordChange):
    """Change user password."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user = _users[user_id]

    # Verify current password
    from resync.api.core.security import get_password_hash, verify_password

    if not verify_password(password_change.current_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user["hashed_password"] = get_password_hash(password_change.new_password)
    logger.info("Password changed for user: %s", user['username'])


@router.post(
    "/users/{user_id}/activate", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin Users"]
)
async def activate_user(user_id: str):
    """Activate a user account."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    _users[user_id]["is_active"] = True
    logger.info("User activated: %s", _users[user_id]['username'])


@router.post(
    "/users/{user_id}/deactivate", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin Users"]
)
async def deactivate_user(user_id: str):
    """Deactivate a user account."""
    if user_id not in _users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    _users[user_id]["is_active"] = False
    logger.info("User deactivated: %s", _users[user_id]['username'])


@router.post("/users/bulk-action", tags=["Admin Users"])
async def bulk_user_action(action: BulkUserAction):
    """Perform bulk action on multiple users."""
    results = {"success": [], "failed": []}

    for user_id in action.user_ids:
        if user_id not in _users:
            results["failed"].append({"id": user_id, "reason": "Not found"})
            continue

        try:
            if action.action == "activate":
                _users[user_id]["is_active"] = True
            elif action.action == "deactivate":
                _users[user_id]["is_active"] = False
            elif action.action == "delete":
                del _users[user_id]
            else:
                results["failed"].append({"id": user_id, "reason": "Unknown action"})
                continue

            results["success"].append(user_id)
        except Exception as e:
            results["failed"].append({"id": user_id, "reason": str(e)})

    return results
