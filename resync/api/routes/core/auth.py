"""Authentication and authorization API endpoints.

This module provides JWT-based authentication endpoints and utilities,
including token generation, validation, and user session management.
It implements secure authentication flows with proper error handling
and integrates with the application's security middleware.

Security Features:
- IP spoofing protection
- Redis Lua injection prevention
- CSRF protection via SameSite cookies
- JWT token leakage prevention
- Constant-time authentication
- Account enumeration prevention
- HKDF key derivation for password hashing
- Thread-safe singleton pattern
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from resync.api.core import security as jwt_security
from resync.api.routes.core.ip_utils import (
    get_trusted_client_ip,
    sanitize_ip_for_redis_key,
)
from resync.core.security.rate_limiter_v2 import rate_limit_auth
from resync.core.redis_init import get_redis_client
from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Auth router for authentication endpoints
router = APIRouter(prefix="/auth", tags=["auth"])

# Security schemes
# Allow missing Authorization to support HttpOnly cookie fallback
bearer_scheme = HTTPBearer(auto_error=False)

# Secret key for JWT tokens
# NOTE: Resolved lazily at runtime to avoid unsafe fallbacks in production.
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Generic error message to prevent account enumeration (CWE-204)
GENERIC_AUTH_ERROR = "Invalid credentials"

# HKDF for key derivation (CWE-916)
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    HKDF_AVAILABLE = True
except ImportError:
    HKDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# JWT secret key resolution
# ---------------------------------------------------------------------------

_DEVELOPMENT_FALLBACK_SECRET: str | None = None

def _get_dev_fallback_secret() -> str:
    """Generate a secure random fallback secret for development only."""
    global _DEVELOPMENT_FALLBACK_SECRET
    if _DEVELOPMENT_FALLBACK_SECRET is None:
        import os

        _DEVELOPMENT_FALLBACK_SECRET = os.urandom(32).hex()
    return _DEVELOPMENT_FALLBACK_SECRET

def _is_secret_key_secure(secret: str | None) -> bool:
    if not secret:
        return False
    if len(secret) < 32:
        return False
    known_insecure = {
        "fallback_secret_key_for_development",
        "dev-secret-key-change-me-in-production-minimum-32-chars",
    }
    return secret not in known_insecure and secret != _get_dev_fallback_secret()

def _get_configured_secret_key() -> str | None:
    """Fetch the configured JWT secret key as a raw string (never masked).

    This project uses Pydantic SecretStr for secrets. Converting it to `str()`
    yields a masked value ("**********"), which would silently break auth.

    Returns:
        The secret value as a string, or None if not configured.
    """
    key = getattr(settings, "secret_key", None)
    if key is None:
        return None
    if hasattr(key, "get_secret_value"):
        return key.get_secret_value()
    return str(key)

def get_jwt_secret_key() -> str:
    """Resolve JWT secret key.

    Option B:
      - production: block auth operations with 503 if missing/weak
      - non-production: allow dev fallback
    """
    configured = _get_configured_secret_key()

    if getattr(settings, "is_production", False):
        if not _is_secret_key_secure(configured):
            logger.critical(
                "auth_secret_key_missing_or_insecure",
                extra={"configured": bool(configured)},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured (get_jwt_secret_key() missing).",
            )
        return configured  # type: ignore[return-value]

    if configured and _is_secret_key_secure(configured):
        return configured
    if not configured:
        logger.warning("auth_secret_key_not_set_using_dev_fallback")
        return _get_dev_fallback_secret()
    logger.warning(
        "auth_secret_key_insecure_nonprod", extra={"length": len(configured)}
    )
    return configured

class SecureAuthenticator:
    """Authenticator resistant to timing attacks with secure key derivation."""

    def __init__(self) -> None:
        self._lockout_duration_seconds = 15 * 60  # 15 minutes
        self._max_attempts = 5
        self._redis_prefix = "resync:auth:lockout"

        # HKDF-derived auth key (CWE-916)
        self._auth_key: bytes | None = None
        self._auth_key_lock = threading.Lock()

    def _get_auth_key(self) -> bytes:
        """Get derived authentication key (cached).

        Security:
            - Derived from JWT secret using HKDF
            - Separate from JWT signing key
            - 256-bit entropy
        """
        if self._auth_key is not None:
            return self._auth_key

        with self._auth_key_lock:
            if self._auth_key is not None:
                return self._auth_key

            jwt_secret = _get_configured_secret_key() or get_jwt_secret_key()

            if HKDF_AVAILABLE:
                # HKDF key derivation (CWE-916)
                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,  # 256 bits
                    salt=b"resync-auth-credential-hashing-v1",
                    info=b"credential-verification",
                )
                self._auth_key = hkdf.derive(jwt_secret.encode("utf-8"))
            else:
                # Fallback: use JWT secret directly (less secure)
                self._auth_key = jwt_secret.encode("utf-8")

            return self._auth_key

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
        # Fail-closed: check configuration first (TASK-012)
        if not settings.admin_username or not settings.admin_password:
            logger.critical("admin_credentials_not_configured")
            if getattr(settings, "is_production", False):
                return False, "Authentication unavailable"
            return False, GENERIC_AUTH_ERROR

        # Atomically check lockout and prepare to record attempt (TASK-010)
        # Sanitize IP to prevent Lua injection (TASK-004)
        sanitized_ip = sanitize_ip_for_redis_key(request_ip)
        is_locked, remaining, _ = await self._check_and_record_attempt(
            sanitized_ip, success=True
        )

        if is_locked:
            # Use constant delay (TASK-007)
            await asyncio.sleep(0.050)

            # Use generic error message to prevent account enumeration (TASK-008)
            logger.warning(
                "auth_locked",
                extra={"ip": request_ip, "lockout_remaining_minutes": remaining},
            )
            return False, GENERIC_AUTH_ERROR

        # Always hash both provided and expected values to maintain constant time
        provided_username_hash = self._hash_credential(username)
        provided_password_hash = self._hash_credential(password)

        expected_username_hash = self._hash_credential(settings.admin_username)
        expected_password_hash = self._hash_credential(
            settings.admin_password.get_secret_value()
        )

        # Constant-time comparison using secrets.compare_digest
        username_valid = secrets.compare_digest(
            provided_username_hash, expected_username_hash
        )

        password_valid = secrets.compare_digest(
            provided_password_hash, expected_password_hash
        )

        # Combine results without short-circuiting
        credentials_valid = username_valid and password_valid

        # Record failed attempt atomically with sanitized IP
        await self._check_and_record_attempt(sanitized_ip, success=credentials_valid)

        # Constant-time delay (TASK-007)
        await asyncio.sleep(0.050)

        if not credentials_valid:
            logger.warning(
                "auth_failed",
                extra={
                    "ip": request_ip,
                    "username_prefix": username[:3] + "***",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Use generic error message (TASK-008)
            return False, GENERIC_AUTH_ERROR

        # Success - clear failed attempts (distributed) with sanitized IP
        try:
            redis = get_redis_client()
            await redis.delete(f"{self._redis_prefix}:{sanitized_ip}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
            # Non-fatal: if Redis is down, lockout TTL will expire naturally
            logger.debug("redis_lockout_clear_failed", error=str(exc))

        logger.info(
            "auth_success",
            extra={
                "ip": request_ip,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return True, None

    def _hash_credential(self, credential: str) -> bytes:
        """Hash credential with derived key for constant-time comparison.

        Security (CWE-916):
            - Uses HKDF-derived key instead of JWT secret directly
            - HMAC-SHA256 for rainbow table resistance
        """
        auth_key = self._get_auth_key()
        return hmac.digest(auth_key, credential.encode("utf-8"), hashlib.sha256)

    async def _check_lockout_state(self, ip: str) -> tuple[bool, int]:
        """Check if IP is locked out and return remaining minutes.

        Uses the same atomic Lua path as `_check_and_record_attempt` (with success=True),
        avoiding TOCTOU races between Redis calls.
        """
        try:
            is_locked, remaining_minutes, _ = await self._check_and_record_attempt(
                ip, success=True
            )
            return is_locked, remaining_minutes
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
            logger.error("redis_lockout_check_failed", error=str(exc))
            return False, 0
    # Removed: _record_failed_attempt is now part of _check_and_record_attempt
    # to prevent TOCTOU race conditions

    async def _check_and_record_attempt(
        self, ip: str, success: bool
    ) -> tuple[bool, int, int]:
        """Atomically check lockout state and record attempt.

        This prevents TOCTOU race conditions by combining check + record in a single
        Redis Lua script that executes atomically.

        Args:
            ip: Client IP address (should be sanitized via sanitize_ip_for_redis_key)
            success: Whether the authentication succeeded

        Returns:
            (is_locked, remaining_minutes, attempt_count)
        """
        try:
            redis = get_redis_client()
            # Use sanitized IP for Redis key (TASK-004)
            key = f"{self._redis_prefix}:{ip}"
            now = time.time()
            cutoff = now - self._lockout_duration_seconds

            # Lua script for atomic check + record
            # Returns: [is_locked, remaining_seconds, attempt_count]
            lua_script = """
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local cutoff = tonumber(ARGV[2])
            local max_attempts = tonumber(ARGV[3])
            local lockout_duration = tonumber(ARGV[4])
            local success = ARGV[5] == 'true'

            -- Remove old attempts
            redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

            -- Count recent attempts
            local count = redis.call('ZCARD', key)

            -- Check if locked
            local is_locked = 0
            local remaining = 0
            if count >= max_attempts then
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                if #oldest > 0 then
                    local oldest_time = tonumber(oldest[2])
                    local unlock_time = oldest_time + lockout_duration
                    remaining = math.max(0, unlock_time - now)
                    is_locked = 1
                end
            end

            -- Record failed attempt (only if not success)
            if not success and is_locked == 0 then
                redis.call('ZADD', key, now, tostring(now))
                redis.call('EXPIRE', key, lockout_duration * 2)
                count = count + 1
            end

            return {is_locked, remaining, count}
            """

            result = await redis.eval(
                lua_script,
                1,  # number of keys
                key,
                str(now),
                str(cutoff),
                str(self._max_attempts),
                str(self._lockout_duration_seconds),
                str(success),
            )

            is_locked = bool(result[0])
            remaining_minutes = int(result[1] / 60) + 1 if result[1] > 0 else 0
            attempt_count = int(result[2])

            # Log warning if approaching lockout
            if not success and attempt_count >= self._max_attempts - 1:
                logger.warning(
                    "ip_approaching_lockout",
                    extra={"ip": ip, "count": attempt_count, "max": self._max_attempts},
                )

            return is_locked, remaining_minutes, attempt_count

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("redis_lockout_check_failed", error=str(e))
            return False, 0, 0

# Global authenticator singleton (lazy init; thread-safe)
# Using threading.Lock for thread-safety across event loops (TASK-013)
_authenticator: SecureAuthenticator | None = None
_authenticator_init_lock: threading.Lock | None = None

def _get_authenticator_lock() -> threading.Lock:
    """Get or create the authenticator initialization lock."""
    global _authenticator_init_lock
    if _authenticator_init_lock is None:
        _authenticator_init_lock = threading.Lock()
    return _authenticator_init_lock

async def get_authenticator() -> SecureAuthenticator:
    """Return a lazily-initialized SecureAuthenticator (thread-safe).

    Uses threading.Lock instead of asyncio.Lock for thread-safety
    across multiple event loops (gunicorn --preload scenario).
    """
    global _authenticator

    # Fast path: already initialized
    if _authenticator is not None:
        return _authenticator

    # Slow path: thread-safe initialization
    lock = _get_authenticator_lock()
    with lock:
        # Double-check (another thread may have initialized)
        if _authenticator is None:
            _authenticator = SecureAuthenticator()

    return _authenticator

async def verify_admin_credentials(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str | None:
    """Verify admin credentials for protected endpoints using JWT tokens.

    Security (TASK-006):
        - Specific JWT exception handling (no token in traceback)
        - Constant-time delay to prevent timing attacks
        - Generic error messages to prevent enumeration
    """
    # Track if we have a token for logging
    has_token = False

    try:
        # 1) Try Authorization header (Bearer)
        token: str | None = None
        if credentials and credentials.credentials:
            token = credentials.credentials
            has_token = True

        # 2) Fallback: HttpOnly cookie "access_token"
        if not token:
            token = request.cookies.get("access_token")
            if token:
                has_token = True

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credentials not provided",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Decode & validate JWT - specific exception handling (TASK-006)
        try:
            # Import JWT exceptions for specific handling
            from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

            payload = jwt_security.decode_access_token(token)

        except ExpiredSignatureError:
            # Log without token
            logger.info(
                "token_expired",
                extra={
                    "token_prefix": token[:8] + "..." if token else "None",
                    "ip": request.client.host if request.client else "unknown",
                },
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        except InvalidTokenError as e:
            # Log error type only (no token in message)
            logger.warning(
                "token_invalid",
                extra={
                    "error_type": type(e).__name__,
                    "has_token": has_token,
                    "ip": request.client.host if request.client else "unknown",
                },
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        username: str | None = payload.get("sub")
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
                detail="Could not validate credentials",  # Generic (TASK-008)
                headers={"WWW-Authenticate": "Bearer"},
            )

        return username

    except HTTPException:
        raise

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Generic handler - no sensitive data in error
        logger.error(
            "auth_unexpected_error",
            extra={
                "error_type": type(e).__name__,
                "has_token": has_token,
                "ip": request.client.host if request.client else "unknown",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    finally:
        # P0-01 fix: Use asyncio.sleep() instead of time.sleep() to avoid
        # blocking the event loop. Maintains constant-time defense (TASK-007).
        await asyncio.sleep(0.050)

def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """
    Create a new JWT access token using centralized security utility.
    """
    return jwt_security.create_access_token(
        subject=data.get("sub", ""), expires_delta=expires_delta
    )

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
    is_valid, _ = await (await get_authenticator()).verify_credentials(
        username, password, client_ip
    )
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

@router.post("/login")
@rate_limit_auth
async def login(request: Request, login_data: LoginRequest) -> Response:
    """
    Authenticate user and return JWT token with secure cookie (CSRF protected).

    Args:
        request: FastAPI request object
        login_data: Login credentials

    Returns:
        Response with JWT token in HttpOnly cookie (TASK-005)
    """
    # Use trusted client IP (TASK-003)
    client_ip = get_trusted_client_ip(request)

    is_valid, error_message = await (await get_authenticator()).verify_credentials(
        login_data.username, login_data.password, client_ip
    )

    if not is_valid:
        logger.warning(
            "login_failed",
            extra={"ip": client_ip, "username": login_data.username[:3] + "***"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_message or GENERIC_AUTH_ERROR,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token
    access_token = create_access_token(
        data={"sub": login_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info(
        "login_successful", extra={"ip": client_ip, "username": login_data.username}
    )

    # Return JSON response with SameSite cookie (CSRF protection - TASK-005)
    response_data = LoginResponse(
        success=True,
        message="Login successful",
        token=TokenResponse(access_token=access_token),
    )

    response = JSONResponse(content=response_data.model_dump())

    # Set secure cookie with CSRF protection
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # Prevents XSS
        secure=getattr(settings, "is_production", False),  # HTTPS only in production
        samesite="strict",  # CSRF protection
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )

    return response

@router.post("/logout")
async def logout(request: Request) -> Response:
    """
    Logout user (invalidate token) and clear secure cookie.

    Note: For stateless JWT, this is primarily for client-side token cleanup.
    Server-side token blacklisting would require additional infrastructure.

    Returns:
        Success message with cookie cleared
    """
    # Use trusted client IP (TASK-003)
    client_ip = get_trusted_client_ip(request)
    logger.info("logout_requested", extra={"ip": client_ip})

    response = JSONResponse(
        content={
            "success": True,
            "message": "Logout successful. Please discard your token.",
        }
    )

    # Clear cookie with SameSite (must match set_cookie)
    response.delete_cookie(
        key="access_token",
        path="/",
        samesite="strict",
    )

    return response

@router.get("/verify")
async def verify_token(
    username: str | None = Depends(verify_admin_credentials),
) -> dict[str, Any]:
    """
    Verify JWT token validity.

    Args:
        username: Username from verified token

    Returns:
        Token verification status
    """
    return {"valid": True, "username": username, "message": "Token is valid"}
