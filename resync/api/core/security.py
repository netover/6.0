import bcrypt
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from resync.core.exception_guard import maybe_reraise_programming_error
from resync.core.jwt_utils import JWTError, create_token, decode_token, unwrap_secret
from resync.settings import get_settings

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _get_algorithm() -> str:
    """Lazy accessor — avoids module-level settings cache."""
    return get_settings().jwt_algorithm


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if the provided plain text password matches the cryptographic hash."""
    try:
        hashed_password_bytes = (
            hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password
        )
        return cast(
            bool,
            bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password_bytes),
        )
    except (ValueError, TypeError) as exc:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error

        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("password_verification_failed", exc_info=exc)
        return False


async def verify_password_async(plain_password: str, hashed_password: str) -> bool:
    """Async wrapper for verify_password() to avoid blocking the event loop."""
    return await asyncio.to_thread(verify_password, plain_password, hashed_password)


async def get_password_hash_async(password: str) -> str:
    """Async wrapper for get_password_hash() to avoid blocking the event loop."""
    return await asyncio.to_thread(get_password_hash, password)


def get_password_hash(password: str) -> str:
    """Generate a secure cryptographic hash of the password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return cast(str, hashed.decode("utf-8"))


def create_access_token(subject: Any, expires_delta: timedelta | None = None) -> str:
    """Generate a JWT access token with iss/aud claims."""
    settings = get_settings()
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.project_name,
        "aud": settings.environment.value if hasattr(settings.environment, "value") else str(settings.environment),
    }

    return create_token(
        payload,
        secret_key=unwrap_secret(settings.secret_key),
        algorithm=_get_algorithm(),
        expires_in=max(1, int(expire.timestamp() - now.timestamp())),
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Validate and decode a JWT access token.

    Returns the decoded payload dict, or None if the token is invalid / expired.
    Handles SecretStr unwrapping and PyJWT audience validation.
    """
    settings = get_settings()
    try:
        # Resolve the expected audience
        audience = settings.environment.value if hasattr(settings.environment, "value") else str(settings.environment)

        payload = decode_token(
            token,
            secret_key=unwrap_secret(settings.secret_key),
            algorithms=[_get_algorithm()],
            audience=audience,
            options={
                "leeway": int(getattr(settings, "jwt_leeway_seconds", 0)),
            },
        )
        return payload
    except JWTError:
        return None
    except (OSError, ValueError, TypeError, RuntimeError, ConnectionError) as exc:
        maybe_reraise_programming_error(exc, exc.__traceback__)
        logger.error("decode_access_token_failed", exc_info=exc)
        raise


verify_token = decode_access_token


def check_permissions(required_permissions: list[str], user_permissions: list[str]) -> bool:
    """Check if user has required permissions."""
    return all((perm in user_permissions for perm in required_permissions))


def require_permissions(
    required_permissions: list[str],
) -> Any:
    """FastAPI dependency to check user permissions."""
    from fastapi import Depends, HTTPException, status

    def permission_checker(
        current_user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
    ) -> dict[str, Any]:
        if not check_permissions(required_permissions, current_user.get("permissions", [])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return permission_checker


def require_role(required_roles: list[str]) -> Any:
    """FastAPI dependency to check user role."""
    from fastapi import Depends, HTTPException, status

    def role_checker(
        current_user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
    ) -> dict[str, Any]:
        user_role = current_user.get("role", "")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges",
            )
        return current_user

    return role_checker


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
) -> dict[str, Any]:
    """Backend dependency to extract user from token without DB hit."""
    from fastapi import HTTPException, status

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

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


async def verify_token_async(token: str) -> dict[str, Any] | None:
    """Async token verification with optional revocation (enterprise-grade).

    - Decodes JWT
    - Checks expiration
    - Checks Valkey-backed JTI revocation list (if enabled)

    Prefer this from FastAPI dependencies.
    """
    payload = decode_access_token(token)
    if not payload:
        return None
    jti = payload.get("jti")
    if jti:
        from resync.core.token_revocation import is_jti_revoked

        if await is_jti_revoked(str(jti)):
            return None
    return payload
