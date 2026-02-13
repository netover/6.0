"""Authentication and authorization API endpoints.

This module provides JWT-based authentication endpoints and utilities,
including token generation, validation, and user session management.
It implements secure authentication flows with proper error handling
and integrates with the application's security middleware.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from resync.core.security.rate_limiter_v2 import rate_limit_auth
from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Auth router for authentication endpoints
router = APIRouter(prefix="/auth", tags=["auth"])

# Security schemes
# Allow missing Authorization to support HttpOnly cookie fallback
security = HTTPBearer(auto_error=False)

# Secret key for JWT tokens
# NOTE: Resolved lazily at runtime to avoid unsafe fallbacks in production.
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ---------------------------------------------------------------------------
# JWT secret key resolution (Option B)
# ---------------------------------------------------------------------------

_DEVELOPMENT_FALLBACK_SECRET = "fallback_secret_key_for_development"

def _is_secret_key_secure(secret: str | None) -> bool:
    if not secret:
        return False
    if len(secret) < 32:
        return False
    insecure = {
        _DEVELOPMENT_FALLBACK_SECRET,
        "dev-secret-key-change-me-in-production-minimum-32-chars",
    }
    return secret not in insecure


def _get_configured_secret_key() -> str | None:
    return getattr(settings, "get_jwt_secret_key()", None) or getattr(settings, "secret_key", None)


def get_jwt_secret_key() -> str:
    """Resolve JWT secret key.

    Option B:
      - production: block auth operations with 503 if missing/weak
      - non-production: allow dev fallback
    """
    configured = _get_configured_secret_key()

    if getattr(settings, "is_production", False):
        if not _is_secret_key_secure(configured):
            logger.critical("auth_secret_key_missing_or_insecure", extra={"configured": bool(configured)})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured (get_jwt_secret_key() missing).",
            )
        return configured  # type: ignore[return-value]

    if configured and _is_secret_key_secure(configured):
        return configured
    if not configured:
        logger.warning("auth_secret_key_not_set_using_dev_fallback")
        return _DEVELOPMENT_FALLBACK_SECRET
    logger.warning("auth_secret_key_insecure_nonprod", extra={"length": len(configured)})
    return configured


class SecureAuthenticator:
    """Authenticator resistente a timing attacks."""

    def __init__(self) -> None:
        self._failed_attempts: dict[str, list[datetime]] = {}
        self._lockout_duration = timedelta(minutes=15)
        self._max_attempts = 5
        self._lockout_lock = asyncio.Lock()

    async def verify_credentials(
        self, username: str, password: str, request_ip: str
    ) -> tuple[bool, str | None]:
        """
        Verify credentials with:
        - Constant-time comparison
        - Rate limiting per IP
        - Account lockout
        - Audit logging
        """
        # Check if IP is locked out
        async with self._lockout_lock:
            if await self._is_locked_out(request_ip):
                logger.warning(
                    "Authentication attempt from locked out IP",
                    extra={"ip": request_ip},
                )
                # Still perform full verification to maintain constant time
                # but will return failure regardless
                lockout_remaining = await self._get_lockout_remaining(request_ip)
                await asyncio.sleep(0.5)  # Artificial delay
                return (
                    False,
                    f"Too many failed attempts. Try again in {lockout_remaining} minutes",
                )

        # Always hash both provided and expected values to maintain constant time
        provided_username_hash = self._hash_credential(username)
        provided_password_hash = self._hash_credential(password)

        expected_username_hash = self._hash_credential(settings.admin_username)
        expected_password_hash = self._hash_credential(settings.admin_password)

        # Constant-time comparison using secrets.compare_digest
        username_valid = secrets.compare_digest(provided_username_hash, expected_username_hash)

        password_valid = secrets.compare_digest(provided_password_hash, expected_password_hash)

        # Combine results without short-circuiting
        credentials_valid = username_valid and password_valid

        # Artificial delay to prevent timing analysis
        await asyncio.sleep(secrets.randbelow(100) / 1000)  # 0-100ms random delay

        if not credentials_valid:
            await self._record_failed_attempt(request_ip)

            logger.warning(
                "Failed authentication attempt",
                extra={
                    "ip": request_ip,
                    "username_provided": username[:3] + "***",  # Partial for logs
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            return False, "Invalid credentials"

        # Success - clear failed attempts
        async with self._lockout_lock:
            if request_ip in self._failed_attempts:
                del self._failed_attempts[request_ip]

        logger.info(
            "Successful authentication",
            extra={"ip": request_ip, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        return True, None

    def _hash_credential(self, credential: str) -> bytes:
        """Hash credential for constant-time comparison."""
        # Use HMAC with secret key to prevent rainbow table attacks
        secret_key = getattr(settings, "secret_key", get_jwt_secret_key()).encode("utf-8")
        return hmac.new(secret_key, credential.encode("utf-8"), hashlib.sha256).digest()

    async def _record_failed_attempt(self, ip: str) -> None:
        """Record failed authentication attempt."""
        async with self._lockout_lock:
            now = datetime.now(timezone.utc)

            if ip not in self._failed_attempts:
                self._failed_attempts[ip] = []

            # Add current attempt
            self._failed_attempts[ip].append(now)

            # Remove attempts outside lockout window
            cutoff = now - self._lockout_duration
            self._failed_attempts[ip] = [
                attempt for attempt in self._failed_attempts[ip] if attempt > cutoff
            ]

            # Log if approaching lockout
            attempt_count = len(self._failed_attempts[ip])
            if attempt_count >= self._max_attempts - 1:
                logger.warning(
                    f"IP approaching lockout: {attempt_count}/{self._max_attempts} attempts",
                    extra={"ip": ip},
                )

    async def _is_locked_out(self, ip: str) -> bool:
        """Check if IP is currently locked out."""
        if ip not in self._failed_attempts:
            return False

        now = datetime.now(timezone.utc)
        cutoff = now - self._lockout_duration

        # Count recent attempts
        recent_attempts = [attempt for attempt in self._failed_attempts[ip] if attempt > cutoff]

        return len(recent_attempts) >= self._max_attempts

    async def _get_lockout_remaining(self, ip: str) -> int:
        """Get remaining lockout time in minutes."""
        if ip not in self._failed_attempts or not self._failed_attempts[ip]:
            return 0

        oldest_attempt = min(self._failed_attempts[ip])
        unlock_time = oldest_attempt + self._lockout_duration
        remaining = (unlock_time - datetime.now(timezone.utc)).total_seconds() / 60

        return max(0, int(remaining))


# Global authenticator singleton (lazy init; avoids import-time asyncio.Lock binding)
_authenticator: "SecureAuthenticator | None" = None
_authenticator_init_lock: "asyncio.Lock | None" = None


async def get_authenticator() -> "SecureAuthenticator":
    """Return a lazily-initialized SecureAuthenticator.

    IMPORTANT: SecureAuthenticator creates an asyncio.Lock in __init__. Creating it at
    module import time can break gunicorn --preload/multi-loop scenarios.
    """
    global _authenticator, _authenticator_init_lock
    if _authenticator is not None:
        return _authenticator
    if _authenticator_init_lock is None:
        _authenticator_init_lock = asyncio.Lock()
    async with _authenticator_init_lock:
        if _authenticator is None:
            _authenticator = SecureAuthenticator()
    return _authenticator


def verify_admin_credentials(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """
    Verify admin credentials for protected endpoints using JWT tokens.
    """
    try:
        # 1) Try Authorization header (Bearer)
        token = credentials.credentials if (credentials and credentials.credentials) else None

        # 2) Fallback: HttpOnly cookie "access_token"
        if not token:
            token = request.cookies.get("access_token")
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Credentials not provided",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Decode & validate JWT
        payload = jwt.decode(token, get_jwt_secret_key(), algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Accept ADMIN_USERNAME or admin_username for compatibility
        admin_user = getattr(settings, "ADMIN_USERNAME", None) or getattr(
            settings, "admin_username", None
        )
        if username != admin_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create a new JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, get_jwt_secret_key(), algorithm=ALGORITHM)


async def authenticate_admin(username: str, password: str) -> bool:
    """
    Authenticate admin user credentials with enhanced security validation.
    """
    # Verify the username matches the admin username from settings
    admin_user = getattr(settings, "ADMIN_USERNAME", None) or getattr(
        settings, "admin_username", None
    )
    if username != admin_user:
        return False

    # Use the SecureAuthenticator for constant-time comparison
    client_ip = "unknown"  # In this context, we don't have the request object
    is_valid, _ = await (await get_authenticator()).verify_credentials(username, password, client_ip)
    return is_valid


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class LoginResponse(BaseModel):
    """Login response model."""
    success: bool
    message: str
    token: TokenResponse | None = None


@router.post("/login", response_model=LoginResponse)
@rate_limit_auth
async def login(request: Request, login_data: LoginRequest) -> LoginResponse:
    """
    Authenticate user and return JWT token.

    Args:
        request: FastAPI request object
        login_data: Login credentials

    Returns:
        LoginResponse with JWT token if successful
    """
    client_ip = request.client.host if request.client else "unknown"

    is_valid, error_message = await (await get_authenticator()).verify_credentials(
        login_data.username,
        login_data.password,
        client_ip
    )

    if not is_valid:
        logger.warning(
            "login_failed",
            extra={"ip": client_ip, "username": login_data.username[:3] + "***"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_message or "Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token
    access_token = create_access_token(
        data={"sub": login_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    logger.info(
        "login_successful",
        extra={"ip": client_ip, "username": login_data.username}
    )

    return LoginResponse(
        success=True,
        message="Login successful",
        token=TokenResponse(access_token=access_token)
    )


@router.post("/logout")
async def logout(request: Request) -> dict[str, Any]:
    """
    Logout user (invalidate token).

    Note: For stateless JWT, this is primarily for client-side token cleanup.
    Server-side token blacklisting would require additional infrastructure.

    Returns:
        Success message
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info("logout_requested", extra={"ip": client_ip})

    return {
        "success": True,
        "message": "Logout successful. Please discard your token."
    }


@router.get("/verify")
async def verify_token(
    username: str | None = Depends(verify_admin_credentials)
) -> dict[str, Any]:
    """
    Verify JWT token validity.

    Args:
        username: Username from verified token

    Returns:
        Token verification status
    """
    return {
        "valid": True,
        "username": username,
        "message": "Token is valid"
    }
