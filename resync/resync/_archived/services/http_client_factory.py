"""HTTP client factory.

Centralizes creation of httpx clients with consistent timeouts/limits.

Production notes
- Do not rely on non-existent legacy imports (e.g. config.base).
- Keep import-time side effects minimal; settings are imported lazily.

Lifecycle management
- Callers are responsible for closing returned clients via ``await client.aclose()``.
- Use ``async with create_http_client(...) as client:`` pattern when possible,
  or ensure cleanup in application shutdown handlers.
"""

from __future__ import annotations

import logging

import httpx
from httpx import AsyncClient

from resync.core.constants import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _get_settings():
    """Lazily import and return the Resync settings proxy.

    Returns:
        settings proxy instance or None if it cannot be imported.

    Rationale:
        Some tooling / isolated test contexts may import this module without the
        full application settings being available.
    """
    try:
        from resync.settings import settings  # type: ignore

        return settings
    except Exception as e:
        # Avoid raising at import-time. Call sites that require settings should
        # provide explicit parameters (e.g. base_url).
        logger.debug("settings_import_failed", exc_info=e)
        return None


def create_async_http_client(
    base_url: str,
    auth: httpx.BasicAuth | None = None,
    verify: bool = True,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
    pool_timeout: float | None = None,
    max_connections: int | None = None,
    max_keepalive: int | None = None,
) -> httpx.AsyncClient:
    """Create a configured httpx.AsyncClient.

    Args:
        base_url: Base URL for the client.
        auth: Optional authentication.
        verify: Whether to verify TLS.
        connect_timeout/read_timeout/write_timeout/pool_timeout: Optional overrides.
        max_connections/max_keepalive: Optional overrides.

    Returns:
        Configured httpx.AsyncClient.
    """

    s = _get_settings()

    timeout = httpx.Timeout(
        connect=connect_timeout
        or (getattr(s, "TWS_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT) if s else DEFAULT_CONNECT_TIMEOUT),
        read=read_timeout
        or (getattr(s, "TWS_READ_TIMEOUT", DEFAULT_READ_TIMEOUT) if s else DEFAULT_READ_TIMEOUT),
        write=write_timeout
        or (getattr(s, "TWS_WRITE_TIMEOUT", DEFAULT_WRITE_TIMEOUT) if s else DEFAULT_WRITE_TIMEOUT),
        pool=pool_timeout
        or (getattr(s, "TWS_POOL_TIMEOUT", DEFAULT_POOL_TIMEOUT) if s else DEFAULT_POOL_TIMEOUT),
    )

    limits = httpx.Limits(
        max_connections=max_connections
        or (getattr(s, "TWS_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS) if s else DEFAULT_MAX_CONNECTIONS),
        max_keepalive_connections=max_keepalive
        or (
            getattr(s, "TWS_MAX_KEEPALIVE", DEFAULT_MAX_KEEPALIVE_CONNECTIONS)
            if s
            else DEFAULT_MAX_KEEPALIVE_CONNECTIONS
        ),
    )

    return httpx.AsyncClient(
        base_url=base_url,
        auth=auth,
        verify=verify,
        timeout=timeout,
        limits=limits,
    )


def create_tws_http_client(
    *,
    base_url: str | None = None,
    auth=None,
    verify: bool | str | None = None,
    **kwargs,
) -> AsyncClient:
    """Create an httpx.AsyncClient configured for the TWS API.

    If base_url is not provided, resync.settings must be importable to build one.
    """

    s = _get_settings()

    if base_url is None:
        if s is None:
            raise RuntimeError(
                "create_tws_http_client requires base_url when resync.settings cannot be imported"
            )
        # Use http by default for TWS
        host = getattr(s, "TWS_HOST", None)
        port = getattr(s, "TWS_PORT", None)
        if not host or not port:
            raise RuntimeError(
                "TWS_HOST/TWS_PORT are not configured; provide base_url explicitly or configure settings"
            )
        base_url = f"http://{host}:{port}"

    # Verification and timeouts
    verify_param = (
        getattr(s, "TWS_VERIFY", True) if (verify is None and s is not None) else (verify if verify is not None else True)
    )

    timeout = httpx.Timeout(
        connect=getattr(s, "TWS_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT) if s else DEFAULT_CONNECT_TIMEOUT,
        read=getattr(s, "TWS_READ_TIMEOUT", DEFAULT_READ_TIMEOUT) if s else DEFAULT_READ_TIMEOUT,
        write=getattr(s, "TWS_WRITE_TIMEOUT", DEFAULT_WRITE_TIMEOUT) if s else DEFAULT_WRITE_TIMEOUT,
        pool=getattr(s, "TWS_POOL_TIMEOUT", DEFAULT_POOL_TIMEOUT) if s else DEFAULT_POOL_TIMEOUT,
    )

    limits = httpx.Limits(
        max_connections=getattr(s, "TWS_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS) if s else DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections=(
            getattr(s, "TWS_MAX_KEEPALIVE", DEFAULT_MAX_KEEPALIVE_CONNECTIONS)
            if s
            else DEFAULT_MAX_KEEPALIVE_CONNECTIONS
        ),
    )

    return AsyncClient(
        base_url=base_url,
        auth=auth,
        timeout=timeout,
        limits=limits,
        verify=verify_param,
        **kwargs,
    )
