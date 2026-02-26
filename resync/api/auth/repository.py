"""
User Repository - PostgreSQL Implementation (Async).

This repository is the DB-backed implementation used by AuthService.
It provides both the legacy method names (create_user/get_user/...) and the
newer interface expected by AuthService (get_by_username/create/get_by_id/...).

Python: 3.14
Stack: SQLAlchemy 2.0 async (AsyncSession via BaseRepository)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from resync.core.database.models.auth import User as DBUser
from resync.core.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

__all__ = ["UserRepository", "get_user_repository"]


def _user_to_dict(user: DBUser) -> dict[str, Any]:
    """Serialize DB user to a stable dict used across service layers."""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "is_verified": getattr(user, "is_verified", False),
        "hashed_password": user.hashed_password,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "permissions": user.permissions,
    }


class UserRepository:
    """
    User Repository - PostgreSQL Backend.

    Uses BaseRepository[DBUser] for async CRUD.
    """

    def __init__(self) -> None:
        self._repo: BaseRepository[DBUser] = BaseRepository(DBUser)
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("UserRepository initialized (PostgreSQL)")

    async def close(self) -> None:
        self._initialized = False

    # ---------------------------------------------------------------------
    # Interface expected by AuthService (P0-01 in audit report)
    # ---------------------------------------------------------------------

    async def get_by_username(self, username: str) -> dict[str, Any] | None:
        user = await self._repo.find_one({"username": username})
        return _user_to_dict(user) if user else None

    async def create(self, user_dict: dict[str, Any]) -> dict[str, Any]:
        # DBUser.id is not auto-generated; ensure we have an ID.
        if not user_dict.get("id"):
            user_dict["id"] = str(uuid4())
        user = await self._repo.create(**user_dict)
        return _user_to_dict(user)

    async def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        user = await self._repo.find_one({"id": user_id})
        return _user_to_dict(user) if user else None

    async def update(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        user = await self._repo.find_one({"id": user_id})
        if not user:
            return None
        # BaseRepository.update expects DB primary key id (same as user.id here)
        updated = await self._repo.update(user.id, **updates)
        return _user_to_dict(updated) if updated else None

    async def delete(self, user_id: str) -> bool:
        user = await self._repo.find_one({"id": user_id})
        if not user:
            return False
        return await self._repo.delete(user.id)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        users = await self._repo.get_all(limit=limit, offset=offset, order_by="username", desc=False)
        return [_user_to_dict(u) for u in users]

    # ---------------------------------------------------------------------
    # Legacy interface (kept for backward compatibility)
    # ---------------------------------------------------------------------

    async def create_user(self, user_id: str, preferences: dict | None = None) -> Any:
        # Historically this operated on UserProfile. Keep minimal behavior:
        # Create an auth user with user_id as username if called.
        return await self.create(
            {
                "id": str(uuid4()),
                "username": user_id,
                "email": f"{user_id}@local.invalid",
                "hashed_password": "!" ,  # unusable placeholder (legacy path)
                "role": "user",
                "is_active": True,
                "full_name": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    async def get_user(self, user_id: str) -> Any | None:
        return await self.get_by_id(user_id)

    async def get_user_by_id(self, user_id: str) -> Any | None:
        return await self.get_by_id(user_id)

    async def update_user(self, user_id: str, **kwargs) -> Any | None:
        return await self.update(user_id, kwargs)

    async def delete_user(self, user_id: str) -> bool:
        return await self.delete(user_id)

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[Any]:
        return await self.list_all(limit=limit, offset=offset)

    async def user_exists(self, user_id: str) -> bool:
        return (await self.get_by_id(user_id)) is not None

    async def get_or_create_user(self, user_id: str) -> Any:
        user = await self.get_by_id(user_id)
        if not user:
            user = await self.create_user(user_id)
        return user

    async def update_last_login(self, user_id: str) -> Any | None:
        return await self.update(user_id, {"last_login": datetime.now(timezone.utc)})


_instance: UserRepository | None = None


def get_user_repository() -> UserRepository:
    """Get singleton repository instance."""
    global _instance
    if _instance is None:
        _instance = UserRepository()
    return _instance
