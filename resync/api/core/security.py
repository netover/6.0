# pylint
# mypy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union
import bcrypt
import logging
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from resync.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
ALGORITHM = settings.jwt_algorithm


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if the provided plain text password matches the cryptographic hash."""
    try:
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode("utf-8")
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password)
    except Exception:
        logger.exception("password_verification_failed")
        return False


def get_password_hash(password: str) -> str:
    """Generate a secure cryptographic hash of the password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Generate a new JWT access token signed with the application's secret key.

    Args:
        subject: The unique identifier for the token (usually username or user_id)
        expires_delta: Optional custom expiration time window

    Returns:
        A signed JWT string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    # Convert expire to Unix timestamp for JWT standard compliance
    to_encode = {"exp": int(expire.timestamp()), "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key.get_secret_value(), algorithm=ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Validate and decode a JWT access token.

    Returns:
        The decoded payload dictionary if valid, otherwise None.
    """
    try:
        payload = jwt.decode(
            token, settings.secret_key.get_secret_value(), algorithms=[ALGORITHM]
        )
        return payload
    except (JWTError, ValidationError):
        return None


verify_token = decode_access_token


def check_permissions(required_permissions: list, user_permissions: list) -> bool:
    """Check if user has required permissions."""
    return all((perm in user_permissions for perm in required_permissions))


def require_permissions(required_permissions: list):
    """FastAPI dependency to check user permissions."""
    from fastapi import Depends, HTTPException, status

    def permission_checker(current_user: dict = Depends(get_current_user)):
        if not check_permissions(
            required_permissions, current_user.get("permissions", [])
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return permission_checker


def require_role(required_roles: list):
    """FastAPI dependency to check user role."""
    from fastapi import Depends, HTTPException, status

    def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role", "")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role}' not authorized",
            )
        return current_user

    return role_checker


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Backend dependency to extract user from token without DB hit."""
    from fastapi import HTTPException, status

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username", payload.get("sub")),
        "role": payload.get("role", "user"),
        "permissions": payload.get("permissions", []),
    }
