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
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from resync.api.core import security as jwt_security
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
        return _get_dev_fallback_secret()
    logger.warning("auth_secret_key_insecure_nonprod", extra={"length": len(configured)})
    return configured


class SecureAuthenticator:
    """Authenticator resistant to timing attacks."""

    def __init__(self) -> None:
        self._lockout_duration_seconds = 15 * 60  # 15 minutes
        self._max_attempts = 5
        self._redis_prefix = "resync:auth:lockout"

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
        # Atomically check lockout and prepare to record attempt
        # Note: We check first with success=True to see if locked, then record actual result
        is_locked, remaining, _ = await self._check_and_record_attempt(request_ip, success=True)
        if is_locked:
            logger.warning(
                "Authentication attempt from locked out IP",
                extra={"ip": request_ip, "lockout_remaining_minutes": remaining},
            )
            # Artificial delay to maintain constant-time-ish appearance
            await asyncio.sleep(0.5)
            return (
                False,
                f"Too many failed attempts. Try again in {remaining} minutes",
            )

        # Always hash both provided and expected values to maintain constant time
        provided_username_hash = self._hash_credential(username)
        provided_password_hash = self._hash_credential(password)

        if not settings.admin_username or not settings.admin_password:
            logger.error("admin_credentials_not_configured")
            return False

        expected_username_hash = self._hash_credential(settings.admin_username)
        expected_password_hash = self._hash_credential(settings.admin_password.get_secret_value())

        # Constant-time comparison using secrets.compare_digest
        username_valid = secrets.compare_digest(provided_username_hash, expected_username_hash)

        password_valid = secrets.compare_digest(provided_password_hash, expected_password_hash)

        # Combine results without short-circuiting
        credentials_valid = username_valid and password_valid

        # Artificial delay to prevent timing analysis
        await asyncio.sleep(secrets.randbelow(100) / 1000)  # 0-100ms random delay

        if not credentials_valid:
            # Record failed attempt atomically (already checked lockout above)
            await self._check_and_record_attempt(request_ip, success=False)

            logger.warning(
                "Failed authentication attempt",
                extra={
                    "ip": request_ip,
                    "username_provided": username[:3] + "***",  # Partial for logs
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            return False, "Invalid credentials"

        # Success - clear failed attempts (distributed)
        try:
            redis = get_redis_client()
            await redis.delete(f"{self._redis_prefix}:{request_ip}")
        except Exception:
            # Non-fatal: if Redis is down, we just can't clear the lockout
            pass

        logger.info(
            "Successful authentication",
            extra={"ip": request_ip, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        return True, None

    def _hash_credential(self, credential: str) -> bytes:
        """Hash credential for constant-time comparison."""
        # Use HMAC with secret key to prevent rainbow table attacks
        secret = _get_configured_secret_key() or get_jwt_secret_key()
        secret_key = secret.encode("utf-8")
        return hmac.new(secret_key, credential.encode("utf-8"), hashlib.sha256).digest()

    async def _check_lockout_state(self, ip: str) -> tuple[bool, int]:
        """Check if IP is locked out and return remaining time."""
        try:
            redis = get_redis_client()
            key = f"{self._redis_prefix}:{ip}"
            
            now = time.time()
            cutoff = now - self._lockout_duration_seconds
            
            # Remove old attempts
            await redis.zremrangebyscore(key, "-inf", cutoff)
            
            # Get current attempt count
            attempt_count = await redis.zcard(key)
            
            if attempt_count >= self._max_attempts:
                # Calculate remaining time from the oldest attempt still in window
                oldest = await redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    _, ts = oldest[0]
                    remaining = int((ts + self._lockout_duration_seconds - now) / 60)
                    return True, max(1, remaining)
                return True, 1
            
            return False, 0
        except Exception as e:
            logger.error("redis_lockout_check_failed", error=str(e))
            return False, 0


    # Removed: _record_failed_attempt is now part of _check_and_record_attempt
    # to prevent TOCTOU race conditions

    async def _check_and_record_attempt(self, ip: str, success: bool) -> tuple[bool, int, int]:
        """Atomically check lockout state and record attempt.
        
        This prevents TOCTOU race conditions by combining check + record in a single
        Redis Lua script that executes atomically.
        
        Args:
            ip: Client IP address
            success: Whether the authentication succeeded
            
        Returns:
            (is_locked, remaining_minutes, attempt_count)
        """
        try:
            redis = get_redis_client()
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
            redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
            
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
                str(success)
            )
            
            is_locked = bool(result[0])
            remaining_minutes = int(result[1] / 60) + 1 if result[1] > 0 else 0
            attempt_count = int(result[2])
            
            # Log warning if approaching lockout
            if not success and attempt_count >= self._max_attempts - 1:
                logger.warning(
                    "ip_approaching_lockout",
                    extra={"ip": ip, "count": attempt_count, "max": self._max_attempts}
                )
            
            return is_locked, remaining_minutes, attempt_count
            
        except Exception as e:
            logger.error("redis_lockout_check_failed", error=str(e))
            return False, 0, 0


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
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
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

        # Decode & validate JWT using centralized security utility
        payload = jwt_security.decode_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

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
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create a new JWT access token using centralized security utility.
    """
    return jwt_security.create_access_token(subject=data.get("sub", ""), expires_delta=expires_delta)


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
