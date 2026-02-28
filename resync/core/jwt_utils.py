"""
Unified JWT Module

Consolidates JWT handling to use PyJWT as primary library.
Provides fallback to python-jose for backward compatibility.

Usage:
    from resync.core.jwt_utils import jwt, JWTError, decode_token, create_token

Part of Resync v5.4.5 - JWT Consolidation
"""

from __future__ import annotations

import logging
import time
import types
from typing import Any

logger = logging.getLogger(__name__)


def _unwrap_secret(value: object) -> str:
    """Return a real secret string for libraries that require ``str`` keys.

    Supports Pydantic v2 ``SecretStr``-like objects (anything with a callable
    ``get_secret_value``). Fails closed on unknown types to avoid accidental
    secret exposure via ``str(value)``.
    """
    if isinstance(value, str):
        secret = value.strip()
        if not secret:
            raise ValueError("JWT secret key must be non-empty")
        return secret

    get_secret_value = getattr(value, "get_secret_value", None)
    if callable(get_secret_value):
        secret = get_secret_value()
        if not isinstance(secret, str):
            raise TypeError("Secret value must be a str")
        secret = secret.strip()
        if not secret:
            raise ValueError("JWT secret key must be non-empty")
        return secret

    raise TypeError("JWT secret key must be str or SecretStr-like")


# =============================================================================
# JWT LIBRARY SELECTION
# Priority: PyJWT > python-jose
# =============================================================================

JWT_LIBRARY: str = "none"
JWTError: type[Exception]
jwt: types.ModuleType | None = None

try:
    # Prefer PyJWT (more actively maintained, simpler API)
    import jwt as _pyjwt  # type: ignore[import-not-found]

    jwt = _pyjwt
    JWTError = _pyjwt.PyJWTError
    JWT_LIBRARY = "pyjwt"
    logger.info(
        "jwt_library_loaded library=pyjwt version=%s",
        getattr(_pyjwt, "__version__", "unknown"),
    )

except ImportError:
    try:
        # Fallback to python-jose
        from jose import JWTError as _JoseJWTError  # type: ignore[import-untyped]
        from jose import jwt as _jose_jwt  # type: ignore[import-untyped]

        jwt = _jose_jwt
        JWTError = _JoseJWTError
        JWT_LIBRARY = "python-jose"
        logger.warning(
            "jwt_library_fallback library=python-jose "
            "recommendation=Install PyJWT for better performance"
        )

    except ImportError:
        # No JWT library available
        jwt = None  # type: ignore
        JWTError = Exception
        JWT_LIBRARY = "none"
        logger.critical(
            "jwt_library_missing action=Authentication will fail. "
            "Install with: pip install PyJWT>=2.10.1"
        )

def is_jwt_available() -> bool:
    """Check if any JWT library is available."""
    return JWT_LIBRARY != "none"

def get_jwt_library() -> str:
    """Get the name of the JWT library in use."""
    return JWT_LIBRARY

# =============================================================================
# UNIFIED JWT FUNCTIONS
# =============================================================================

def decode_token(
    token: str,
    secret_key: str | object,
    algorithms: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Decode and verify a JWT token.

    Works with both PyJWT and python-jose.

    Args:
        token: The JWT token string
        secret_key: Secret key for verification
        algorithms: Allowed algorithms (default: ["HS256"])
        **kwargs: Additional options passed to the underlying library

    Returns:
        Decoded token payload

    Raises:
        JWTError: If token is invalid or expired
        RuntimeError: If no JWT library is available
    """
    if not is_jwt_available():
        raise RuntimeError(
            "No JWT library available. Install PyJWT: pip install PyJWT>=2.10.1"
        )

    secret_key = _unwrap_secret(secret_key)

    if algorithms is None:
        algorithms = ["HS256"]

    jwt_module = jwt
    if jwt_module is None:
        raise RuntimeError(
            "No JWT library available. Install PyJWT: pip install PyJWT>=2.10.1"
        )
    return jwt_module.decode(token, secret_key, algorithms=algorithms, **kwargs)

def create_token(
    payload: dict[str, Any],
    secret_key: str | object,
    algorithm: str = "HS256",
    expires_in: int | None = 3600,
) -> str:
    """
    Create a JWT token.

    Works with both PyJWT and python-jose.

    Args:
        payload: Token payload data
        secret_key: Secret key for signing
        algorithm: Signing algorithm (default: HS256)
        expires_in: Expiration time in seconds (default: 3600)

    Returns:
        Encoded JWT string

    Raises:
        RuntimeError: If no JWT library is available
    """
    if not is_jwt_available():
        raise RuntimeError(
            "No JWT library available. Install PyJWT: pip install PyJWT>=2.10.1"
        )

    to_encode = payload.copy()

    secret_key = _unwrap_secret(secret_key)

    # Add timing claims (only if not already set by caller)
    now = time.time()
    to_encode.setdefault("iat", int(now))

    if expires_in is not None:
        to_encode.setdefault("exp", int(now + expires_in))

    jwt_module = jwt
    if jwt_module is None:
        raise RuntimeError(
            "No JWT library available. Install PyJWT: pip install PyJWT>=2.10.1"
        )
    return jwt_module.encode(to_encode, secret_key, algorithm=algorithm)

def verify_token(
    token: str,
    secret_key: str | object,
    algorithms: list[str] | None = None,
    verify_exp: bool = True,
) -> tuple[bool, dict[str, Any] | str]:
    """
    Verify a JWT token and return result.

    Non-raising version of decode_token for cases where you want
    to handle errors differently.

    Args:
        token: The JWT token string
        secret_key: Secret key for verification
        algorithms: Allowed algorithms (default: ["HS256"])
        verify_exp: Whether to verify expiration (default: True)

    Returns:
        Tuple of (is_valid, payload_or_error)
        - (True, payload_dict) if valid
        - (False, error_message) if invalid
    """
    if not is_jwt_available():
        return False, "No JWT library available"

    try:
        secret_key = _unwrap_secret(secret_key)
        if algorithms is None:
            algorithms = ["HS256"]

        options = {}
        if not verify_exp:
            options["verify_exp"] = False

        jwt_module = jwt
        if jwt_module is None:
            return False, "No JWT library available"

        payload = jwt_module.decode(
            token,
            secret_key,
            algorithms=algorithms,
            options=options if options else None,
        )
        return True, payload

    except JWTError as e:
        error_msg = str(e).lower()
        if "expired" in error_msg or "expir" in error_msg:
            return False, "Token has expired"
        return False, f"Invalid token: {str(e)}"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        return False, f"Token verification failed: {str(e)}"

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "jwt",
    "JWTError",
    "JWT_LIBRARY",
    "is_jwt_available",
    "get_jwt_library",
    "decode_token",
    "create_token",
    "verify_token",
]

# Alias for backward compat
def decode_access_token(token: str, secret_key: str = "", algorithm: str = "HS256"):
    """Alias for decode_token — backward compatibility."""
    return decode_token(token, secret_key=secret_key, algorithm=algorithm)
