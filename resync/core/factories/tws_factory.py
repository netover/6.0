"""
Centralized TWS Client Factory.

This module provides a single source of truth for TWS client creation
and dependency injection throughout the Resync application.

All other modules should import from this factory instead of creating
their own TWS client instances.

Usage:
    # For FastAPI dependency injection
    from resync.core.factories.tws_factory import get_tws_client

    @router.get("/endpoint")
    async def endpoint(tws_client = Depends(get_tws_client)):
        ...

    # For direct singleton access
    from resync.core.factories.tws_factory import get_tws_client_singleton

    client = get_tws_client_singleton()
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from fastapi import Depends

if TYPE_CHECKING:
    from resync.settings import Settings as AppSettings
    from resync.services.tws_service import OptimizedTWSClient

# Thread-safe singleton instance
_tws_client_instance: Optional["OptimizedTWSClient"] = None
_tws_client_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_settings() -> "AppSettings":
    """
    Get cached AppSettings instance.

    Returns:
        AppSettings: Application settings singleton
    """
    from resync.settings import settings

    return settings  # type: ignore[return-value]


def _create_tws_client(settings: "AppSettings") -> Any:
    """
    Create a new resilient TWS client instance.

    Args:
        settings: Application settings

    Returns:
        UnifiedTWSClient: Resilient TWS client
    """
    from resync.services.tws_unified import UnifiedTWSClient

    return UnifiedTWSClient()


def get_tws_client_singleton(
    settings: Optional["AppSettings"] = None,
) -> "OptimizedTWSClient":
    """
    Get or create the TWS client singleton.

    Thread-safe singleton access to the TWS client.

    Args:
        settings: Optional settings override (uses default if not provided)

    Returns:
        OptimizedTWSClient: The singleton TWS client instance
    """
    global _tws_client_instance

    if _tws_client_instance is None:
        with _tws_client_lock:
            # Double-check locking pattern
            if _tws_client_instance is None:
                effective_settings = settings or _get_settings()
                _tws_client_instance = _create_tws_client(effective_settings)

    return _tws_client_instance


def get_tws_client(
    settings: "AppSettings" = Depends(_get_settings),
) -> "OptimizedTWSClient":
    """
    FastAPI dependency that returns the TWS client singleton.

    This is the primary way to access the TWS client in FastAPI routes.

    Args:
        settings: Injected settings (provided by FastAPI DI)

    Returns:
        OptimizedTWSClient: The singleton TWS client instance

    Example:
        @router.get("/jobs")
        async def list_jobs(tws_client = Depends(get_tws_client)):
            return await tws_client.list_jobs()
    """
    return get_tws_client_singleton(settings)


def get_tws_client_factory():
    """
    Get the TWS client factory function.

    Returns a callable that creates TWS clients - useful for
    lazy initialization or testing.

    Returns:
        Callable that returns OptimizedTWSClient

    Example:
        factory = get_tws_client_factory()
        client = factory()
    """
    return get_tws_client_singleton


def reset_tws_client() -> None:
    """
    Reset the TWS client singleton.

    Useful for testing or when settings change.
    Should be called with caution in production.
    """
    global _tws_client_instance

    with _tws_client_lock:
        _tws_client_instance = None
        _get_settings.cache_clear()


# Aliases for backward compatibility
tws_client_factory = get_tws_client_factory
get_client = get_tws_client_singleton


__all__ = [
    "get_tws_client",
    "get_tws_client_singleton",
    "get_tws_client_factory",
    "reset_tws_client",
    "tws_client_factory",
    "get_client",
]
