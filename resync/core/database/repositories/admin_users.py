"""
Admin User Repository for PostgreSQL.

Provides async CRUD operations for admin users with:
- Password hashing (using hashlib for now, recommend bcrypt in production)
- Role-based queries
- Account locking after failed attempts
- Login tracking
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from resync.api.models import AdminUser

from .base import BaseRepository

logger = logging.getLogger(__name__)

# Security constants
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30
PASSWORD_SALT_LENGTH = 32


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with salt.

    NOTE: For production, use bcrypt or argon2 instead of SHA256.
    This is a basic implementation for demonstration.

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(PASSWORD_SALT_LENGTH)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}", salt


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against stored hash.

    Args:
        password: Plain text password to verify
        stored_hash: Stored hash in format "salt:hash"

    Returns:
        True if password matches
    """
    try:
        salt, _ = stored_hash.split(":", 1)
        new_hash, _ = hash_password(password, salt)
        return secrets.compare_digest(new_hash, stored_hash)
    except (ValueError, AttributeError):
        return False


class AdminUserRepository(BaseRepository[AdminUser]):
    """Repository for admin user operations."""

    def __init__(self, session_factory=None):
        """Initialize repository."""
        super().__init__(AdminUser, session_factory)

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str | None = None,
        role: str = "user",
    ) -> AdminUser:
        """Create a new admin user.

        Args:
            username: Unique username
            email: Unique email address
            password: Plain text password (will be hashed)
            full_name: Optional full name
            role: User role (default: "user")

        Returns:
            Created AdminUser instance
        """
        password_hash, _ = hash_password(password)
        return await self.create(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            role=role,
            is_active=True,
            is_verified=False,
        )

    async def get_by_username(self, username: str) -> AdminUser | None:
        """Get user by username."""
        async with self._get_session() as session:
            result = await session.execute(select(AdminUser).where(AdminUser.username == username))
            return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> AdminUser | None:
        """Get user by email."""
        async with self._get_session() as session:
            result = await session.execute(select(AdminUser).where(AdminUser.email == email))
            return result.scalar_one_or_none()

    async def authenticate(self, username: str, password: str) -> AdminUser | None:
        """Authenticate a user.

        Handles:
        - Password verification
        - Account locking after failed attempts
        - Login tracking

        Args:
            username: Username to authenticate
            password: Plain text password

        Returns:
            AdminUser if authentication successful, None otherwise
        """
        user = await self.get_by_username(username)
        if not user:
            return None

        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            logger.warning("Account locked: %s", username)
            return None

        # Verify password
        if not verify_password(password, user.password_hash):
            await self._handle_failed_login(user)
            return None

        # Check if active
        if not user.is_active:
            logger.warning("Inactive account login attempt: %s", username)
            return None

        # Successful login
        await self._handle_successful_login(user)
        return user

    async def _handle_failed_login(self, user: AdminUser) -> None:
        """Handle failed login attempt."""
        async with self._get_session() as session:
            new_attempts = user.failed_login_attempts + 1
            updates = {"failed_login_attempts": new_attempts}

            # Lock account if max attempts exceeded
            if new_attempts >= MAX_FAILED_ATTEMPTS:
                updates["locked_until"] = datetime.now(timezone.utc) + timedelta(
                    minutes=LOCKOUT_DURATION_MINUTES
                )
                logger.warning("Account locked due to failed attempts: %s", user.username)

            await session.execute(
                update(AdminUser).where(AdminUser.id == user.id).values(**updates)
            )
            await session.commit()

    async def _handle_successful_login(self, user: AdminUser) -> None:
        """Handle successful login."""
        async with self._get_session() as session:
            await session.execute(
                update(AdminUser)
                .where(AdminUser.id == user.id)
                .values(
                    failed_login_attempts=0,
                    locked_until=None,
                    last_login=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    async def change_password(self, user_id: int, new_password: str) -> bool:
        """Change user password.

        Args:
            user_id: User ID
            new_password: New plain text password

        Returns:
            True if successful
        """
        password_hash, _ = hash_password(new_password)
        async with self._get_session() as session:
            result = await session.execute(
                update(AdminUser).where(AdminUser.id == user_id).values(password_hash=password_hash)
            )
            await session.commit()
            return result.rowcount > 0

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = False,
        role: str | None = None,
    ) -> list[AdminUser]:
        """List users with pagination and filters.

        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            active_only: Only return active users
            role: Filter by role

        Returns:
            List of AdminUser instances
        """
        async with self._get_session() as session:
            query = select(AdminUser)

            if active_only:
                query = query.where(AdminUser.is_active.is_(True))

            if role:
                query = query.where(AdminUser.role == role)

            query = query.offset(skip).limit(limit).order_by(AdminUser.created_at.desc())

            result = await session.execute(query)
            return list(result.scalars().all())

    async def activate_user(self, user_id: int) -> bool:
        """Activate a user account."""
        async with self._get_session() as session:
            result = await session.execute(
                update(AdminUser).where(AdminUser.id == user_id).values(is_active=True)
            )
            await session.commit()
            return result.rowcount > 0

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account."""
        async with self._get_session() as session:
            result = await session.execute(
                update(AdminUser).where(AdminUser.id == user_id).values(is_active=False)
            )
            await session.commit()
            return result.rowcount > 0

    async def verify_user(self, user_id: int) -> bool:
        """Mark user as verified."""
        async with self._get_session() as session:
            result = await session.execute(
                update(AdminUser).where(AdminUser.id == user_id).values(is_verified=True)
            )
            await session.commit()
            return result.rowcount > 0

    async def update_user(
        self,
        user_id: int,
        email: str | None = None,
        full_name: str | None = None,
        role: str | None = None,
    ) -> AdminUser | None:
        """Update user details.

        Args:
            user_id: User ID
            email: New email (optional)
            full_name: New full name (optional)
            role: New role (optional)

        Returns:
            Updated AdminUser or None if not found
        """
        updates = {}
        if email is not None:
            updates["email"] = email
        if full_name is not None:
            updates["full_name"] = full_name
        if role is not None:
            updates["role"] = role

        if not updates:
            return await self.get_by_id(user_id)

        async with self._get_session() as session:
            await session.execute(
                update(AdminUser).where(AdminUser.id == user_id).values(**updates)
            )
            await session.commit()

        return await self.get_by_id(user_id)

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user (soft delete by deactivating).

        For actual deletion, use the base class delete method.
        """
        return await self.deactivate_user(user_id)

    async def unlock_user(self, user_id: int) -> bool:
        """Unlock a locked user account."""
        async with self._get_session() as session:
            result = await session.execute(
                update(AdminUser)
                .where(AdminUser.id == user_id)
                .values(locked_until=None, failed_login_attempts=0)
            )
            await session.commit()
            return result.rowcount > 0


# Singleton instance
_repository: AdminUserRepository | None = None


def get_admin_user_repository() -> AdminUserRepository:
    """Get singleton AdminUserRepository instance."""
    global _repository
    if _repository is None:
        _repository = AdminUserRepository()
    return _repository
