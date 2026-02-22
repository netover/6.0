# pylint: skip-file
# mypy: ignore-errors
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

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from resync.core.jwt_utils import JWTError, decode_token, create_token
from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Security schemes
# Allow missing Authorization to support HttpOnly cookie fallback
security = HTTPBearer(auto_error=False)


# Secret key for JWT tokens
# v5.9.5: Fixed SECRET_KEY case mismatch - settings uses lowercase
# SECURITY: No fallback in production - must fail fast if not configured
def _get_secret_key() -> str:
    """Get secret key from settings, fail fast if not configured in production."""
    import os

    from resync.settings import settings

    # Try to get secret_key (lowercase - correct attribute name)
    secret = getattr(settings, "secret_key", None)

    if secret is not None:
        # Handle SecretStr from Pydantic
        if hasattr(secret, "get_secret_value"):
            return secret.get_secret_value()
        return str(secret)

    # In production, MUST have secret_key configured
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        raise RuntimeError(
            "CRITICAL: SECRET_KEY/secret_key must be configured in production. "
            "Application cannot start with default/fallback secret."
        )

    # Development only - log warning
    import logging

    logging.getLogger(__name__).warning(
        "Using fallback SECRET_KEY - this is INSECURE and only for development"
    )
    return "dev_fallback_secret_key_NOT_FOR_PRODUCTION"


SECRET_KEY = _get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class SecureAuthenticator:
    """Authenticator resistente a timing attacks."""

    def __init__(self) -> None:
        self._failed_attempts: dict[str, list[datetime]] = {}
        self._lockout_duration = timedelta(minutes=15)
        self._max_attempts = 5
        # Lazy-init to avoid event-loop binding during module import (gunicorn --preload).
        self._lockout_lock: asyncio.Lock | None = None

    def _get_lockout_lock(self) -> asyncio.Lock:
        """Return the lock used to protect lockout state.

        The lock is created lazily to avoid binding it to an event loop at
        module import time (which can break under gunicorn --preload).
        """
        if self._lockout_lock is None:
            # Ensure we're inside an event loop.
            asyncio.get_running_loop()
            self._lockout_lock = asyncio.Lock()
        return self._lockout_lock

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
        # Check if IP is locked out (outside lock to avoid holding lock during I/O)
        is_locked = await self._is_locked_out(request_ip)
        if is_locked:
            lockout_remaining = await self._get_lockout_remaining(request_ip)
            logger.warning(
                "Authentication attempt from locked out IP",
                extra={"ip": request_ip},
            )
            await asyncio.sleep(0.5)
            return (
                False,
                f"Too many failed attempts. Try again in {lockout_remaining} minutes",
            )

        # Always hash both provided and expected values to maintain constant time
        provided_username_hash = self._hash_credential(username)
        provided_password_hash = self._hash_credential(password)

        expected_username_hash = self._hash_credential(settings.admin_username)

        # v5.9.7: settings.admin_password is SecretStr|None; avoid TypeError
        expected_password_hash = self._hash_credential(settings.admin_password)

        # Constant-time comparison using secrets.compare_digest
        username_valid = secrets.compare_digest(
            provided_username_hash, expected_username_hash
        )

        password_valid = secrets.compare_digest(
            provided_password_hash, expected_password_hash
        )

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
        async with self._get_lockout_lock():
            if request_ip in self._failed_attempts:
                del self._failed_attempts[request_ip]

        logger.info(
            "Successful authentication",
            extra={
                "ip": request_ip,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return True, None

    def _hash_credential(self, credential: Any) -> bytes:
        """Hash credential for constant-time comparison.

        Accepts ``str``, ``SecretStr`` or ``None``. ``None`` is treated as empty string
        to avoid boot-time crashes when admin password is not configured.
        """
        # Use HMAC with secret key to prevent rainbow table attacks
        raw_secret = getattr(settings, "secret_key", SECRET_KEY)
        if hasattr(raw_secret, "get_secret_value"):
            secret_key = raw_secret.get_secret_value().encode("utf-8")
        else:
            secret_key = str(raw_secret).encode("utf-8")

        if credential is None:
            c = ""
        elif hasattr(credential, "get_secret_value"):
            c = credential.get_secret_value()
        else:
            c = str(credential)

        return hmac.new(secret_key, c.encode("utf-8"), hashlib.sha256).digest()

    async def _record_failed_attempt(self, ip: str) -> None:
        """Record failed authentication attempt."""
        async with self._get_lockout_lock():
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
        recent_attempts = [
            attempt for attempt in self._failed_attempts[ip] if attempt > cutoff
        ]

        return len(recent_attempts) >= self._max_attempts

    async def _get_lockout_remaining(self, ip: str) -> int:
        """Get remaining lockout time in minutes."""
        if ip not in self._failed_attempts or not self._failed_attempts[ip]:
            return 0

        oldest_attempt = min(self._failed_attempts[ip])
        unlock_time = oldest_attempt + self._lockout_duration
        remaining = (unlock_time - datetime.now(timezone.utc)).total_seconds() / 60

        return max(0, int(remaining))


# Global authenticator instance
authenticator = SecureAuthenticator()


def verify_admin_credentials(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """
    Verify admin credentials for protected endpoints using JWT tokens.
    """
    try:
        # 1) Try Authorization header (Bearer)
        token = (
            credentials.credentials
            if (credentials and credentials.credentials)
            else None
        )

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
        payload = decode_token(token, SECRET_KEY, algorithms=[ALGORITHM])
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
    except (JWTError, RuntimeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """
    Create a new JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    return create_token(to_encode, SECRET_KEY, algorithm=ALGORITHM, expires_in=None)


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
    is_valid, _ = await authenticator.verify_credentials(username, password, client_ip)
    return is_valid
