"""
Security Module.

Provides input validation, sanitization, and security utilities.
"""

import hashlib
import hmac
import secrets
from typing import Any

from resync.core.security_main import (
    # Patterns
    DANGEROUS_CHARS_PATTERN,
    EMAIL_PATTERN,
    SAFE_CHARS_ONLY,
    SAFE_STRING_PATTERN,
    STRICT_ALPHANUMERIC_PATTERN,
    STRICT_CHARS_ONLY,
    TWS_JOB_PATTERN,
    TWS_WORKSTATION_PATTERN,
    # Classes
    InputSanitizer,
    # Type aliases
    SafeAgentID,
    SafeEmail,
    SafeTWSJobName,
    SafeTWSWorkstation,
    ValidationResult,
    # Functions
    sanitize_input,
    sanitize_input_strict,
    sanitize_tws_job_name,
    sanitize_tws_workstation,
    validate_email,
    validate_input,
)

# API Key security constants
_API_KEY_SECRET = secrets.token_bytes(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash using constant-time comparison."""
    computed_hash = hash_api_key(api_key)
    return hmac.compare_digest(computed_hash, hashed_key)


async def verify_metrics_api_key(api_key: str) -> bool:
    """
    Verify the metrics API key against the configured hash.
    
    This is used by the workstation metrics endpoint.
    """
    from resync.settings import get_settings
    settings = get_settings()
    
    if not settings.metrics_api_key_hash:
        # If no hash is configured, reject all (default secure)
        # In development, use a dummy hash if needed
        return False
        
    return verify_api_key(api_key, settings.metrics_api_key_hash)


def verify_admin_token(token: str) -> dict[str, Any] | None:
    """Verify an admin token and return payload if valid."""
    try:
        from resync.core.jwt_utils import decode_token
        valid, payload = decode_token(token)
        if valid and isinstance(payload, dict):
            if payload.get("role") == "admin":
                return payload
        return None
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return None


def generate_api_key(prefix: str = "rsk") -> tuple[str, str]:
    """Generate a new API key and its hash.

    Returns:
        Tuple of (api_key, hashed_key)
    """
    key = f"{prefix}_{secrets.token_urlsafe(32)}"
    return key, hash_api_key(key)

__all__ = [
    # Classes
    "InputSanitizer",
    "ValidationResult",
    # Functions
    "sanitize_input",
    "sanitize_input_strict",
    "sanitize_tws_job_name",
    "sanitize_tws_workstation",
    "validate_email",
    "validate_input",
    # API Key functions
    "hash_api_key",
    "verify_api_key",
    "verify_metrics_api_key",
    "verify_admin_token",
    "generate_api_key",
    # Type aliases
    "SafeAgentID",
    "SafeEmail",
    "SafeTWSJobName",
    "SafeTWSWorkstation",
]
