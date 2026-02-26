# pylint
"""
Security module for JWT authentication.

SECURITY NOTES (v5.4.1):
- PyJWT is a REQUIRED dependency - system fails closed if unavailable
- Uses main settings system with production validators
- No fail-open paths - all auth failures result in 401/403

This module provides:
- JWT token decoding and validation
- FastAPI dependencies for authentication
- Role-based access control
"""

import logging
from typing import Any

from pydantic import SecretStr


def _unwrap_secret(val: SecretStr | str | None) -> str | None:
    """Return the raw secret string from Pydantic SecretStr or a plain string."""
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    return val

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from resync.api.security.models import (
    LoginRequest,
    OAuthToken,
    TokenRequest,
    UserCreate,
    UserResponse,
    password_hasher,
)
from resync.api.security.validations import (
    EnhancedLoginRequest,
    SensitiveFieldValidator,
    TokenRequestWithValidation,
    UserCreateWithValidation,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CRITICAL: PyJWT is REQUIRED - fail closed if not available
# =============================================================================
try:
    import jwt  # pyjwt

    JWT_AVAILABLE = True
    logger.info("pyjwt_loaded version=%s", getattr(jwt, "__version__", "unknown"))
except ImportError as e:
    JWT_AVAILABLE = False
    logger.critical(
        "pyjwt_import_failed error=%s action=%s",
        str(e),
        "Authentication will be DISABLED - install pyjwt",
    )
    jwt = None  # type: ignore

# =============================================================================
# Settings Integration - Use main validated settings, not AppSettings
# =============================================================================
def get_settings():
    """
    Get application settings with production validators.

    Uses the main settings system which includes:
    - Secret key validation (not default in production)
    - Password strength requirements
    - Environment-specific rules
    """
    from resync.settings import get_settings as _get_settings

    return _get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# =============================================================================
# Startup Validation
# =============================================================================
def validate_auth_requirements() -> None:
    """
    Validate that all authentication requirements are met.

    Called at startup to ensure fail-closed behavior.
    Raises RuntimeError if requirements not met.
    """
    errors = []

    # Check PyJWT availability
    if not JWT_AVAILABLE:
        errors.append("PyJWT library is not installed. Run: pip install pyjwt")

    # Check settings
    try:
        settings = get_settings()

        # Validate secret key is not default
        secret_key = getattr(settings, "secret_key", None) or getattr(
            settings, "jwt_secret_key", None
        )
        if secret_key in (None, "", "change-me", "change-me-in-production"):
            errors.append(
                "JWT secret key is not configured. "
                "Set SECRET_KEY or JWT_SECRET_KEY environment variable."
            )
        elif len(str(secret_key)) < 32:
            errors.append(
                f"JWT secret key is too short ({len(str(secret_key))} chars). "
                "Use at least 32 characters for security."
            )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        errors.append(f"Failed to load settings: {e}")

    if errors:
        error_msg = "Authentication requirements not met:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        logger.critical("auth_validation_failed errors=%s", errors)
        raise RuntimeError(error_msg)

    logger.info("auth_validation_passed")

# =============================================================================
# Token Decoding - No Fail-Open Paths
# =============================================================================
def decode_token(token: str, settings: Any = None) -> dict[str, Any]:
    """
    Decode and verify a JWT token.

    SECURITY: This function has NO fail-open paths.
    - Missing token → 401
    - Invalid token → 401
    - PyJWT unavailable → 503
    - Invalid secret → 401

    Args:
        token: The JWT token string
        settings: Application settings (auto-resolved if None)

    Returns:
        Decoded token payload

    Raises:
        HTTPException: 401 for auth failures, 503 for system errors
    """
    # Validate token presence
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # CRITICAL: Fail closed if PyJWT not available
    if not JWT_AVAILABLE:
        logger.error("auth_unavailable reason=pyjwt_not_installed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable. Contact administrator.",
        )

    # Get settings if not provided
    if settings is None:
        settings = get_settings()

    # Get secret key from settings (support both naming conventions)
    raw_key = getattr(settings, "secret_key", None) or getattr(
        settings, "jwt_secret_key", None
    )
    secret_key = _unwrap_secret(raw_key)
    algorithm = getattr(settings, "jwt_algorithm", "HS256")

    # Validate secret key
    if not secret_key or secret_key in ("change-me", "change-me-in-production"):
        logger.error("auth_misconfigured reason=invalid_secret_key")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service misconfigured. Contact administrator.",
        )

    # Decode and verify token
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
        )

        # Validate required claims
        if "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject claim",
                headers={"WWW-Authenticate": "Bearer"},
            )

        logger.debug(
            "token_decoded sub=%s role=%s", payload.get("sub"), payload.get("role")
        )
        return payload

    except jwt.ExpiredSignatureError as e:
        logger.warning("token_expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except jwt.InvalidTokenError as e:
        logger.warning("token_invalid error=%s", str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("token_decode_error error=%s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

# =============================================================================
# FastAPI Dependencies
# =============================================================================
async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> dict[str, Any]:
    """
    FastAPI dependency that returns the decoded JWT payload for the current request.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user": user["sub"]}
    """
    settings = get_settings()
    return decode_token(token, settings)

async def get_current_user_optional(
    token: str | None = Depends(
        OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
    ),
) -> dict[str, Any] | None:
    """
    FastAPI dependency that optionally returns user if token present.

    Returns None if no token provided (for mixed auth/public endpoints).
    Still validates token if present.
    """
    if not token:
        return None

    settings = get_settings()
    return decode_token(token, settings)

def require_role(required_role: str):
    """
    Dependency factory that enforces a specific role in the JWT payload.

    Usage:
        @app.get("/admin-only")
        async def admin_route(user: dict = Depends(require_role("admin"))):
            return {"admin": user["sub"]}

    Args:
        required_role: Role name that must be present in token

    Returns:
        FastAPI dependency function
    """

    async def role_dependency(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        # Get role(s) from token - support both single role and role list
        user_role = user.get("role")
        user_roles = user.get("roles", [])

        # Normalize to list
        all_roles = []
        if user_role:
            all_roles.append(user_role)
        if isinstance(user_roles, list):
            all_roles.extend(user_roles)
        elif user_roles:
            all_roles.append(user_roles)

        # Admin role bypasses all checks
        if "admin" in all_roles:
            return user

        # Check for required role
        if required_role not in all_roles:
            logger.warning(
                "insufficient_permissions user=%s required=%s actual=%s",
                user.get("sub"),
                required_role,
                all_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {required_role} role required",
            )

        return user

    return role_dependency

def require_any_role(*roles: str):
    """
    Dependency factory that enforces any of the specified roles.

    Usage:
        @app.get("/staff-only")
        async def staff_route(
            user: dict = Depends(require_any_role("admin", "operator"))
        ):
            return {"staff": user["sub"]}
    """

    async def role_dependency(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        user_role = user.get("role")
        user_roles = user.get("roles", [])

        all_roles = set()
        if user_role:
            all_roles.add(user_role)
        if isinstance(user_roles, list):
            all_roles.update(user_roles)

        # Admin bypasses
        if "admin" in all_roles:
            return user

        # Check for any required role
        if not all_roles.intersection(set(roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: one of {roles} required",
            )

        return user

    return role_dependency

# =============================================================================
# Utility Functions
# =============================================================================
def create_access_token(
    data: dict[str, Any],
    expires_delta: int | None = None,
) -> str:
    """
    Create a new JWT access token.

    Args:
        data: Payload data (must include 'sub')
        expires_delta: Token lifetime in seconds (default: 3600)

    Returns:
        Encoded JWT string
    """
    import time

    if not JWT_AVAILABLE:
        raise RuntimeError("PyJWT is required to create tokens")

    settings = get_settings()
    raw_key = getattr(settings, "secret_key", None) or getattr(
        settings, "jwt_secret_key", None
    )
    secret_key = _unwrap_secret(raw_key)
    algorithm = getattr(settings, "jwt_algorithm", "HS256")

    if not secret_key or secret_key in ("change-me", "change-me-in-production"):
        raise RuntimeError("JWT secret key not configured")

    # Unwrap Pydantic SecretStr (or SecretStr-like) keys for PyJWT.
    if not isinstance(secret_key, str):
        get_secret_value = getattr(secret_key, "get_secret_value", None)
        if callable(get_secret_value):
            secret_key = get_secret_value()
        if not isinstance(secret_key, str):
            raise TypeError("JWT secret key must be a string")

    to_encode = data.copy()

    # Add expiration
    expire = time.time() + (expires_delta or 3600)
    to_encode.update({"exp": expire, "iat": time.time()})

    return jwt.encode(to_encode, secret_key, algorithm=algorithm)

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    # Auth functions
    "JWT_AVAILABLE",
    "validate_auth_requirements",
    "decode_token",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "require_any_role",
    "create_access_token",
    "oauth2_scheme",
    # Models
    "LoginRequest",
    "UserCreate",
    "UserResponse",
    "TokenRequest",
    "OAuthToken",
    "password_hasher",
    # Validations
    "EnhancedLoginRequest",
    "SensitiveFieldValidator",
    "TokenRequestWithValidation",
    "UserCreateWithValidation",
]
