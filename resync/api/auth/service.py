# pylint: disable=all
# mypy: no-rerun
"""
Authentication service with database support.
"""

import logging
import os
from datetime import datetime, timezone, timedelta

from .models import (
    Token,
    TokenPayload,
    User,
    UserCreate,
    UserInDB,
    UserRole,
    UserUpdate,
)
from .repository import UserRepository

logger = logging.getLogger(__name__)

# Get secret key from environment with secure default handling
_DEFAULT_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")
if not _DEFAULT_SECRET_KEY:
    _env = os.getenv("ENVIRONMENT", "development").lower()
    if _env in ("production", "prod", "staging"):
        raise RuntimeError(
            "AUTH_SECRET_KEY must be set in production/staging environments. "
            "Set the AUTH_SECRET_KEY environment variable."
        )
    logger.warning(
        "AUTH_SECRET_KEY not set — using insecure default. "
        "This is only acceptable in development."
    )
    _DEFAULT_SECRET_KEY = "insecure-dev-key-do-not-use-in-production"


class AuthService:
    """
    Authentication service providing user management and authentication.
    """

    def __init__(
        self,
        repository: UserRepository | None = None,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
    ):
        """
        Initialize authentication service.

        Args:
            repository: User repository (created if not provided)
            secret_key: JWT signing key
            algorithm: JWT algorithm
            access_token_expire_minutes: Token expiration time
        """
        self.repository = repository or UserRepository()
        self.secret_key = secret_key or _DEFAULT_SECRET_KEY
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

        # Password hashing
        try:
            from passlib.context import CryptContext

            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        except ImportError:
            self.pwd_context = None
            logger.warning("passlib not available, using fallback hashing")

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        raise RuntimeError(
            "passlib[bcrypt] is not installed. Cannot hash passwords safely. "
            "Install with: pip install passlib[bcrypt]"
        )

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if self.pwd_context:
            return self.pwd_context.verify(plain_password, hashed_password)
        raise RuntimeError(
            "passlib[bcrypt] is not installed. Cannot verify passwords safely. "
            "Install with: pip install passlib[bcrypt]"
        )

    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user

        Raises:
            ValueError: If username already exists
        """
        # Check if username exists
        existing = await self.repository.get_by_username(user_data.username)
        if existing:
            raise ValueError(f"Username '{user_data.username}' already exists")

        # Hash password and create user
        user_dict = user_data.model_dump()
        user_dict["hashed_password"] = self._hash_password(user_dict.pop("password"))

        created = await self.repository.create(user_dict)

        return User(
            id=created["id"],
            username=created["username"],
            email=created["email"],
            full_name=created["full_name"],
            role=UserRole(created["role"]),
            is_active=created["is_active"],
            created_at=datetime.fromisoformat(created["created_at"]),
            permissions=created["permissions"],
        )

    async def authenticate(self, username: str, password: str) -> UserInDB | None:
        """
        Authenticate a user.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User if authentication successful, None otherwise
        """
        user_data = await self.repository.get_by_username(username)
        if not user_data:
            logger.warning(
                "Authentication failed: user not found",
                extra={"username_prefix": username[:3] + "***"},
            )
            return None

        if not self._verify_password(password, user_data["hashed_password"]):
            logger.warning(
                "Authentication failed: invalid password",
                extra={"username_prefix": username[:3] + "***"},
            )
            return None

        if not user_data["is_active"]:
            logger.warning(
                "Authentication failed: inactive user",
                extra={"username_prefix": username[:3] + "***"},
            )
            return None

        # Update last login
        await self.repository.update(
            user_data["id"], {"last_login": datetime.now(timezone.utc).isoformat()}
        )

        logger.info("User %s authenticated successfully", username)

        return UserInDB(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            hashed_password=user_data["hashed_password"],
            role=UserRole(user_data["role"]),
            is_active=user_data["is_active"],
            created_at=datetime.fromisoformat(user_data["created_at"]),
            updated_at=datetime.fromisoformat(user_data["updated_at"]),
            last_login=datetime.fromisoformat(user_data["last_login"])
            if user_data.get("last_login")
            else None,
            permissions=user_data["permissions"],
        )

    def create_access_token(self, user: UserInDB) -> Token:
        """
        Create JWT access token for user.

        Args:
            user: Authenticated user

        Returns:
            Token response
        """
        from resync.core.jwt_utils import jwt

        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.access_token_expire_minutes
        )

        payload = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
            "exp": expire,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return Token(
            access_token=token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
        )

    def verify_token(self, token: str) -> TokenPayload | None:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            from resync.core.jwt_utils import JWTError, jwt

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            return TokenPayload(
                sub=payload["sub"],
                username=payload["username"],
                role=payload["role"],
                permissions=payload.get("permissions", []),
                exp=datetime.fromtimestamp(payload["exp"]),
            )

        except JWTError:
            logger.warning("Token verification failed")
            return None

    async def get_user(self, user_id: str) -> User | None:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User if found, None otherwise
        """
        user_data = await self.repository.get_by_id(user_id)
        if not user_data:
            return None

        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            role=UserRole(user_data["role"]),
            is_active=user_data["is_active"],
            created_at=datetime.fromisoformat(user_data["created_at"]),
            permissions=user_data["permissions"],
        )

    async def update_user(self, user_id: str, updates: UserUpdate) -> User | None:
        """
        Update user data.

        Args:
            user_id: User ID to update
            updates: Fields to update

        Returns:
            Updated user if found, None otherwise
        """
        update_dict = updates.model_dump(exclude_unset=True)

        # Hash password if provided
        if "password" in update_dict:
            update_dict["hashed_password"] = self._hash_password(
                update_dict.pop("password")
            )

        updated = await self.repository.update(user_id, update_dict)
        if not updated:
            return None

        return await self.get_user(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: User ID to delete

        Returns:
            True if deleted, False if not found
        """
        return await self.repository.delete(user_id)

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """
        List all users.

        Args:
            limit: Maximum users to return
            offset: Number to skip

        Returns:
            List of users
        """
        users_data = await self.repository.list_all(limit, offset)

        return [
            User(
                id=u["id"],
                username=u["username"],
                email=u["email"],
                full_name=u["full_name"],
                role=UserRole(u["role"]),
                is_active=u["is_active"],
                created_at=datetime.fromisoformat(u["created_at"]),
                permissions=u["permissions"],
            )
            for u in users_data
        ]

    async def grant_permission(self, user_id: str, permission: str) -> bool:
        """
        Grant permission to user.

        Args:
            user_id: User ID
            permission: Permission to grant

        Returns:
            True if granted
        """
        user_data = await self.repository.get_by_id(user_id)
        if not user_data:
            return False

        permissions = user_data.get("permissions", [])
        if permission not in permissions:
            permissions.append(permission)
            await self.repository.update(user_id, {"permissions": permissions})

        return True

    async def revoke_permission(self, user_id: str, permission: str) -> bool:
        """
        Revoke permission from user.

        Args:
            user_id: User ID
            permission: Permission to revoke

        Returns:
            True if revoked
        """
        user_data = await self.repository.get_by_id(user_id)
        if not user_data:
            return False

        permissions = user_data.get("permissions", [])
        if permission in permissions:
            permissions.remove(permission)
            await self.repository.update(user_id, {"permissions": permissions})

        return True


# Global service instance and lock
_auth_service: AuthService | None = None
_auth_lock = __import__("threading").Lock()


def get_auth_service() -> AuthService:
    """Get or create auth service instance (thread-safe)."""
    global _auth_service
    if _auth_service is None:
        with _auth_lock:
            if _auth_service is None:
                _auth_service = AuthService()
    return _auth_service


def reset_auth_service():
    """Reset auth service (for testing)."""
    global _auth_service
    _auth_service = None
