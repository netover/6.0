"""
Rate Limiting Middleware - Enhanced rate limiting with slowapi.

v5.6.0: Production-ready rate limiting.

Features:
- Configurable rate limits per endpoint
- Redis-backed storage for distributed deployments
- Different limits for auth vs regular endpoints
- Custom key functions (IP, user, API key)
- Bypass for internal/health endpoints

Usage:
    from resync.core.security.rate_limiter_v2 import (
        limiter,
        rate_limit,
        rate_limit_auth,
        setup_rate_limiting,
    )

    @router.post("/login")
    @rate_limit_auth  # 5/minute for auth endpoints
    async def login(request: Request):
        ...

    @router.get("/data")
    @rate_limit("100/minute")  # Custom limit
    async def get_data(request: Request):
        ...
"""

from __future__ import annotations

import os
import ipaddress
import warnings as _warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = structlog.get_logger(__name__)


def _parse_trusted_proxies(raw: str | None) -> tuple[bool, list[str]]:
    """Parse a trusted proxy allow-list.

    The list accepts:
      - `*` to trust all proxies (NOT recommended)
      - individual IPs (e.g. `10.0.0.10`)
      - CIDRs (e.g. `10.0.0.0/8`)

    This mirrors the idea behind uvicorn's `--forwarded-allow-ips`.
    """
    if not raw:
        return False, []
    items = [p.strip() for p in raw.split(",") if p.strip()]
    if not items:
        return False, []
    if any(p == "*" for p in items):
        return True, []
    return False, items


_TRUST_ALL_PROXIES, _TRUSTED_PROXIES = _parse_trusted_proxies(
    os.getenv("RATE_LIMIT_TRUSTED_PROXIES") or os.getenv("FORWARDED_ALLOW_IPS")
)


def _is_trusted_proxy(client_ip: str) -> bool:
    """Return True if the immediate peer is a trusted proxy."""
    if _TRUST_ALL_PROXIES:
        return True
    if not _TRUSTED_PROXIES:
        return False
    try:
        ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for entry in _TRUSTED_PROXIES:
        try:
            net = ipaddress.ip_network(entry, strict=False)
            if ip in net:
                return True
        except ValueError:
            if entry == client_ip:
                return True
    return False


# =============================================================================
# Configuration
# =============================================================================


def get_rate_limit_enabled() -> bool:
    """Check if rate limiting is enabled."""
    # Defensive parsing: env vars sometimes include whitespace/newlines.
    return os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in ("true", "1", "yes")


def get_redis_url() -> str | None:
    """Get Redis URL for distributed rate limiting."""
    return os.getenv("REDIS_URL", os.getenv("RATE_LIMIT_REDIS_URL"))


def get_default_limit() -> str:
    """Get default rate limit."""
    return os.getenv("RATE_LIMIT_DEFAULT", "100/minute").strip()


def get_auth_limit() -> str:
    """Get rate limit for authentication endpoints."""
    return os.getenv("RATE_LIMIT_AUTH", "5/minute").strip()


def get_strict_limit() -> str:
    """Get strict rate limit for sensitive endpoints."""
    return os.getenv("RATE_LIMIT_STRICT", "3/minute").strip()


def get_bypass_paths() -> list[str]:
    """Get list of paths that bypass rate limiting."""
    default_paths = ["/health", "/health/", "/metrics", "/docs", "/redoc", "/openapi.json"]
    raw = os.getenv("RATE_LIMIT_BYPASS_PATHS", ",".join(default_paths))
    return [p.strip() for p in raw.split(",") if p.strip()]


def _is_bypass_path(request: Request) -> bool:
    """Check if request path should bypass rate limiting."""
    path = request.url.path
    bypass_paths = get_bypass_paths()
    return any(path.startswith(p) for p in bypass_paths)


def _is_cors_preflight(request: Request) -> bool:
    """Check if request is a CORS preflight (OPTIONS)."""
    return request.method == "OPTIONS"


# =============================================================================
# Key Functions
# =============================================================================


def get_remote_address(request: Request) -> str:
    """
    Get client IP address.

    Handles X-Forwarded-For header for proxied requests.
    """
    # IMPORTANT: Forwarded headers are **spoofable** unless you only trust them
    # when the immediate peer is a known proxy.
    peer_ip = request.client.host if request.client else None

    if peer_ip and _is_trusted_proxy(peer_ip):
        # Check for forwarded header (common in reverse proxy setups)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

    # Default/fallback to direct peer address
    if peer_ip:
        return peer_ip

    return "unknown"


def get_user_identifier(request: Request) -> str:
    """
    Get user identifier for rate limiting.

    Uses authenticated user ID if available, otherwise IP address.
    """
    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"

    # Try to get API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api:{api_key[:16]}"  # Use prefix only

    # Fallback to IP
    return f"ip:{get_remote_address(request)}"


def get_api_key(request: Request) -> str:
    """Get API key for rate limiting."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"api:{api_key[:16]}"
    return get_remote_address(request)


# =============================================================================
# Limiter Setup
# =============================================================================

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address as slowapi_get_remote_address  # noqa: F401

    # Create limiter instance
    # Create limiter instance using SafeMemoryStorage to prevent import-time threads
    try:
        from limits.storage import registry
        from .storage import SafeMemoryStorage
        if SafeMemoryStorage:
             # Manually register scheme if register_scheme helper is available
             if hasattr(registry, "register_scheme"):
                 registry.register_scheme("safememory", SafeMemoryStorage)
             elif hasattr(registry, "SCHEMES") and isinstance(registry.SCHEMES, dict):
                 registry.SCHEMES["safememory"] = SafeMemoryStorage
             else:
                 # Fallback: cannot register safe storage, use memory but log warning
                 logger.warning("could_not_register_safememory", reason="incompatible_limits_library")
                 SafeMemoryStorage = None
                 
             if SafeMemoryStorage:
                 _storage_uri = "safememory://"
             else:
                 _storage_uri = "memory://"
        else:
             _storage_uri = "memory://"
    except ImportError:
        _storage_uri = "memory://"

    RATE_LIMIT_STRATEGY = os.getenv("RATE_LIMIT_STRATEGY", "sliding-window-counter")

    try:
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[get_default_limit()],
            # Use our safe storage implementation to avoid import-time threads
            storage_uri=_storage_uri,
            strategy=RATE_LIMIT_STRATEGY,
            headers_enabled=True,
        )
    except Exception as exc:
        # Strategy strings vary across limits/slowapi versions; fall back to fixed-window
        # rather than crashing the application in dev/test.
        logger.warning(
            "rate_limit_strategy_fallback",
            requested=RATE_LIMIT_STRATEGY,
            fallback="fixed-window",
            error=type(exc).__name__,
            detail=str(exc),
        )
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[get_default_limit()],
            storage_uri=_storage_uri,
            strategy="fixed-window",
            headers_enabled=True,
        )

    SLOWAPI_AVAILABLE = True


except ImportError:
    logger.warning(
        "slowapi not installed. Rate limiting disabled. Install with: pip install slowapi"
    )
    limiter = None
    SLOWAPI_AVAILABLE = False


# =============================================================================
# Rate Limit Decorators
# =============================================================================


def rate_limit(limit: str | None = None, key_func: Callable | None = None):
    """
    Rate limit decorator with custom limit.

    Args:
        limit: Rate limit string (e.g., "100/minute", "10/second")
        key_func: Custom key function for rate limit identification

    Usage:
        @router.get("/data")
        @rate_limit("50/minute")
        async def get_data(request: Request):
            ...
    """

    def decorator(func):
        if not SLOWAPI_AVAILABLE or not get_rate_limit_enabled():
            return func

        # Apply slowapi limiter with bypass for health/docs paths and CORS preflight
        limit_str = limit or get_default_limit()
        key = key_func or get_remote_address

        def exempt_condition(request: Request) -> bool:
            """Bypass rate limit for health endpoints, docs, and CORS preflight."""
            return _is_bypass_path(request) or _is_cors_preflight(request)

        return limiter.limit(limit_str, key_func=key, exempt_when=exempt_condition)(func)

    return decorator


def rate_limit_auth(func):
    """
    Rate limit decorator for authentication endpoints.

    Applies stricter limits (default: 5/minute) to prevent brute force.

    Usage:
        @router.post("/login")
        @rate_limit_auth
        async def login(request: Request):
            ...
    """
    if not SLOWAPI_AVAILABLE or not get_rate_limit_enabled():
        return func

    def exempt_condition(request: Request) -> bool:
        """Bypass rate limit for health endpoints, docs, and CORS preflight."""
        return _is_bypass_path(request) or _is_cors_preflight(request)

    return limiter.limit(get_auth_limit(), key_func=get_remote_address, exempt_when=exempt_condition)(func)


def rate_limit_strict(func):
    """
    Strict rate limit decorator for sensitive operations.

    Applies very strict limits (default: 3/minute).

    Usage:
        @router.post("/password-reset")
        @rate_limit_strict
        async def password_reset(request: Request):
            ...
    """
    if not SLOWAPI_AVAILABLE or not get_rate_limit_enabled():
        return func

    def exempt_condition(request: Request) -> bool:
        """Bypass rate limit for health endpoints, docs, and CORS preflight."""
        return _is_bypass_path(request) or _is_cors_preflight(request)

    return limiter.limit(get_strict_limit(), key_func=get_remote_address, exempt_when=exempt_condition)(func)


def rate_limit_by_user(limit: str | None = None):
    """
    Rate limit by authenticated user.

    Args:
        limit: Rate limit string

    Usage:
        @router.post("/expensive-operation")
        @rate_limit_by_user("10/hour")
        async def expensive_op(request: Request):
            ...
    """

    def decorator(func):
        if not SLOWAPI_AVAILABLE or not get_rate_limit_enabled():
            return func

        def exempt_condition(request: Request) -> bool:
            """Bypass rate limit for health endpoints, docs, and CORS preflight."""
            return _is_bypass_path(request) or _is_cors_preflight(request)

        limit_str = limit or get_default_limit()
        return limiter.limit(limit_str, key_func=get_user_identifier, exempt_when=exempt_condition)(func)

    return decorator


# =============================================================================
# FastAPI Integration
# =============================================================================


def setup_rate_limiting(app: FastAPI) -> None:
    from resync.settings import get_settings

    get_settings().is_production
    """
    Setup rate limiting for FastAPI application.

    Call this in your application startup:
        from resync.core.security.rate_limiter_v2 import setup_rate_limiting
        setup_rate_limiting(app)
    """
    if not SLOWAPI_AVAILABLE:
        logger.warning("Rate limiting not available - slowapi not installed")
        return

    if not get_rate_limit_enabled():
        logger.info("Rate limiting disabled via RATE_LIMIT_ENABLED=false")
        return

    # Upgrade storage to Redis if configured (post-import)
    redis_url = get_redis_url()
    if redis_url and limiter:
        try:
            from limits.storage import storage_from_string
            limiter._storage = storage_from_string(redis_url)
            logger.info("Rate limiter storage upgraded to Redis")
        except Exception as e:
            logger.error("Failed to upgrade rate limiter storage to Redis", error=str(e))

    # Add limiter to app state
    app.state.limiter = limiter

    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info(
        "Rate limiting enabled",
        default_limit=get_default_limit(),
        auth_limit=get_auth_limit(),
        storage="redis" if get_redis_url() else "memory",
    )


# =============================================================================
# Middleware (DEPRECATED — does not perform actual rate limiting)
# =============================================================================
class RateLimitMiddleware:
    """Rate limiting middleware for global limits.

    .. deprecated::
        This middleware is a no-op — it never rejects requests.
        Use ``setup_rate_limiting()`` for real rate limiting via slowapi.
    """

    def __init__(
        self,
        app,
        default_limit: str | None = None,
        exclude_paths: list[str] | None = None,
    ):
        _warnings.warn(
            "RateLimitMiddleware is a no-op and will be removed in v7.0. "
            "Use setup_rate_limiting() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.app = app
        self.default_limit = default_limit or get_default_limit()
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def __call__(self, scope, receive, send):
        # Pass-through — no actual rate limiting is performed.
        await self.app(scope, receive, send)



# =============================================================================
# Export
# =============================================================================

__all__ = [
    "limiter",
    "rate_limit",
    "rate_limit_auth",
    "rate_limit_strict",
    "rate_limit_by_user",
    "setup_rate_limiting",
    "get_remote_address",
    "get_user_identifier",
    "get_bypass_paths",
    "SLOWAPI_AVAILABLE",
]
