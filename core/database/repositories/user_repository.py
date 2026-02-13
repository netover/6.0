"""
User Repository - Database operations for user management.

Migrated from fastapi_app/db/user_service.py for consolidation.
Uses the Repository pattern consistent with other repositories.

Part of Resync v5.4.7 - Database Consolidation
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.models.auth import User, UserRole


class UserRepository:
    """
    Repository for user database operations.

    Provides CRUD operations and authentication support for User model.
    Replaces the legacy UserService from fastapi_app/db.
    """

    # Account lockout settings
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=15)

    def __init__(self, session: AsyncSession):
        """
        Initialize the repository.

        Args:
            session: Async SQLAlchemy session
        """
        self.session = session

    async def create(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: str | None = None,
        role: UserRole = UserRole.USER,
    ) -> User:
        """
        Create a new user.

        Args:
            username: Unique username
            email: Unique email address
            hashed_password: Pre-hashed password
            full_name: Optional full name
            role: User role (default: USER)

        Returns:
            Created User instance
        """
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role.value if isinstance(role, UserRole) else role,
        )

        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def authenticate(
        self,
        username: str,
        password: str,
        verify_password_func: Callable[[str, str], bool],
    ) -> User | None:
        """
        Authenticate user with username and password.

        Returns user if authentication successful, None otherwise.
        Handles account lockout for too many failed attempts.

        Args:
            username: Username to authenticate
            password: Plain text password
            verify_password_func: Function to verify password against hash

        Returns:
            User if authentication successful, None otherwise
        """
        user = await self.get_by_username(username)

        if not user:
            return None

        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return None

        # Check if user is active
        if not user.is_active:
            return None

        # Verify password
        if not verify_password_func(password, user.hashed_password):
            await self._handle_failed_login(user)
            return None

        # Successful login - reset failed attempts and update last login
        await self._handle_successful_login(user)

        return user

    async def _handle_failed_login(self, user: User) -> None:
        """Handle failed login attempt."""
        user.failed_login_attempts += 1

        # Lock account if too many failed attempts
        if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.now(timezone.utc) + self.LOCKOUT_DURATION

        await self.session.commit()

    async def _handle_successful_login(self, user: User) -> None:
        """Handle successful login."""
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)

        await self.session.commit()

    async def update(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> User | None:
        """
        Update user attributes.

        Args:
            user_id: User ID to update
            **kwargs: Attributes to update

        Returns:
            Updated User or None if not found
        """
        user = await self.get_by_id(user_id)

        if not user:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def deactivate(self, user_id: str) -> bool:
        """Deactivate user account."""
        user = await self.get_by_id(user_id)

        if not user:
            return False

        user.is_active = False
        await self.session.commit()

        return True

    async def verify(self, user_id: str) -> bool:
        """Mark user as verified."""
        user = await self.get_by_id(user_id)

        if not user:
            return False

        user.is_verified = True
        await self.session.commit()

        return True

    async def change_password(
        self,
        user_id: str,
        new_hashed_password: str,
    ) -> bool:
        """Change user password."""
        user = await self.get_by_id(user_id)

        if not user:
            return False

        user.hashed_password = new_hashed_password
        user.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        return True

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[User]:
        """List users with pagination."""
        query = select(User)

        if active_only:
            query = query.where(User.is_active)

        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def unlock(self, user_id: str) -> bool:
        """Manually unlock a user account."""
        user = await self.get_by_id(user_id)

        if not user:
            return False

        user.locked_until = None
        user.failed_login_attempts = 0
        await self.session.commit()

        return True

    async def delete(self, user_id: str) -> bool:
        """
        Delete a user (hard delete).

        Consider using deactivate() for soft delete instead.
        """
        user = await self.get_by_id(user_id)

        if not user:
            return False

        await self.session.delete(user)
        await self.session.commit()

        return True


# Backward compatibility alias
UserService = UserRepository


__all__ = ["UserRepository", "UserService"]
