"""
Security utilities for FastAPI application.

Provides:
- Password hashing with bcrypt
- JWT token creation and verification
- Database-backed user authentication
- Permission checking

v5.4.7: Consolidated to use PyJWT and core/database
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

# Use unified JWT module (PyJWT)
from resync.core.jwt_utils import JWTError, jwt

# v5.3.20: Import from config.py which re-exports from resync.settings
from .config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    # v5.3.20: Use jwt_algorithm from consolidated settings
    return jwt.encode(
        to_encode,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def verify_token(token: str) -> dict | None:
    """Verify JWT token and return payload."""
    try:
        # v5.3.20: Use jwt_algorithm from consolidated settings
        return jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
):
    """Get current authenticated user from JWT token."""
    token = credentials.credentials

    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user info from token
    username = payload.get("username", user_id)
    role = payload.get("role", "user")

    # Basic validation: user_id must be non-empty string
    if not isinstance(user_id, str) or len(user_id) < 1:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return validated user info from token
    return {
        "user_id": user_id,
        "username": username,
        "role": role,
        "permissions": payload.get("permissions", []),
    }


async def get_current_user_from_db(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: AsyncSession = None,
):
    """
    Get current authenticated user with database validation.

    Use this dependency when you need to verify the user still exists
    and is active in the database.
    """
    token = credentials.credentials
    payload = verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If database session provided, validate user exists
    if db:
        from resync.core.database.repositories import UserRepository

        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user.to_dict()

    # Fallback to token-based user info
    return {
        "user_id": user_id,
        "username": payload.get("username", user_id),
        "role": payload.get("role", "user"),
        "permissions": payload.get("permissions", []),
    }


def check_permissions(required_permissions: list, user_permissions: list) -> bool:
    """Check if user has required permissions."""
    return all(perm in user_permissions for perm in required_permissions)


def require_permissions(required_permissions: list):
    """Dependency to check user permissions."""

    def permission_checker(current_user: dict = Depends(get_current_user)):
        if not check_permissions(required_permissions, current_user.get("permissions", [])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return permission_checker


def require_role(required_roles: list):
    """Dependency to check user role."""

    def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role", "")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role}' not authorized",
            )
        return current_user

    return role_checker
