"""
User Repository - PostgreSQL Implementation.

Provides user authentication storage using PostgreSQL.
"""

import logging
from datetime import datetime, timezone

from resync.core.database.models import UserProfile
from resync.core.database.repositories import BaseRepository

logger = logging.getLogger(__name__)

__all__ = ["UserRepository", "get_user_repository"]


class UserRepository:
    """
    User Repository - PostgreSQL Backend.

    Provides user authentication and profile management.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize. db_path is ignored - uses PostgreSQL."""
        if db_path:
            logger.debug("db_path ignored, using PostgreSQL: %s", db_path)
        self._repo = BaseRepository(UserProfile)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the repository."""
        self._initialized = True
        logger.info("UserRepository initialized (PostgreSQL)")

    async def close(self) -> None:
        """Close the repository."""
        self._initialized = False

    async def create_user(self, user_id: str, preferences: dict | None = None) -> UserProfile:
        """Create a new user."""
        return await self._repo.create(user_id=user_id, preferences=preferences or {})

    async def get_user(self, user_id: str) -> UserProfile | None:
        """Get user by ID."""
        return await self._repo.find_one({"user_id": user_id})

    async def get_user_by_id(self, user_id: str) -> UserProfile | None:
        """Alias for get_user."""
        return await self.get_user(user_id)

    async def update_user(self, user_id: str, **kwargs) -> UserProfile | None:
        """Update user data."""
        user = await self.get_user(user_id)
        if user:
            return await self._repo.update(user.id, **kwargs)
        return None

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = await self.get_user(user_id)
        if user:
            return await self._repo.delete(user.id)
        return False

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[UserProfile]:
        """List all users."""
        return await self._repo.get_all(limit=limit, offset=offset)

    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists."""
        return await self._repo.exists({"user_id": user_id})

    async def get_or_create_user(self, user_id: str) -> UserProfile:
        """Get existing user or create new one."""
        user = await self.get_user(user_id)
        if not user:
            user = await self.create_user(user_id)
        return user

    async def update_last_login(self, user_id: str) -> UserProfile | None:
        """Update user's last login timestamp."""
        return await self.update_user(user_id, last_active=datetime.now(timezone.utc))

    # Sync methods for backward compatibility
    def create_user_sync(self, user_id: str, **kwargs) -> None:
        """Sync version - deprecated."""
        logger.warning("create_user_sync is deprecated, use async create_user")

    def get_user_sync(self, user_id: str) -> dict | None:
        """Sync version - deprecated."""
        logger.warning("get_user_sync is deprecated, use async get_user")
        return None


_instance: UserRepository | None = None


def get_user_repository() -> UserRepository:
    """Get the singleton UserRepository instance."""
    global _instance
    if _instance is None:
        _instance = UserRepository()
    return _instance
