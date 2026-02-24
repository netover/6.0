# pylint: disable=all
# mypy: no-rerun
"""
Centralized Redis Client Factory.

This module provides a single source of truth for Redis client creation
and dependency injection throughout the Resync application.

All other modules should import from this factory instead of creating
their own Redis client instances.

Supported Redis client types:
- Async Redis client (aioredis/redis.asyncio)
- Sync Redis client (redis)
- Connection pool access
- Sentinel support (optional)

Usage:
    # For FastAPI dependency injection
    from resync.core.factories.redis_factory import get_redis_client

    @router.get("/endpoint")
    async def endpoint(redis = Depends(get_redis_client)):
        await redis.get("key")

    # For direct singleton access
    from resync.core.factories.redis_factory import get_redis_client_singleton

    redis = await get_redis_client_singleton()
"""

from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from fastapi import Depends
from redis import Redis as SyncRedis

from resync.settings import Settings as AppSettings

if TYPE_CHECKING:
    from redis import Redis as SyncRedis
    from redis.asyncio import Redis as AsyncRedis

    from resync.settings import Settings as AppSettings

# Thread-safe singleton instances
_async_redis_instance: Optional["AsyncRedis"] = None
_sync_redis_instance: Optional["SyncRedis"] = None
_redis_lock = threading.Lock()
_async_lock = None  # lazy-initialized asyncio.Lock (gunicorn --preload safe)
_async_lock_loop = None


def _get_async_lock() -> asyncio.Lock:
    """Return a process-global asyncio.Lock bound to the current event loop.

    Avoids creating asyncio primitives at import time (which can bind to the
    wrong loop under gunicorn --preload).
    """
    global _async_lock, _async_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet; create a lock anyway. It will be recreated when
        # accessed from a running loop.
        _async_lock = asyncio.Lock()
        _async_lock_loop = None
        return _async_lock

    if _async_lock is None or _async_lock_loop is not loop:
        _async_lock = asyncio.Lock()
        _async_lock_loop = loop
    return _async_lock


@lru_cache(maxsize=1)
def _get_settings() -> "AppSettings":
    """
    Get cached AppSettings instance.

    Returns:
        AppSettings: Application settings singleton
    """
    from resync.settings import settings

    return settings


def _get_redis_url(settings: "AppSettings") -> str:
    """
    Build Redis URL from settings.

    Args:
        settings: Application settings

    Returns:
        str: Redis connection URL
    """
    host = getattr(settings, "redis_host", "localhost")
    port = getattr(settings, "redis_port", 6379)
    password = getattr(settings, "redis_password", None)
    db = getattr(settings, "redis_db", 0)

    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def _create_async_redis(settings: "AppSettings") -> "AsyncRedis":
    """
    Create a new async Redis client instance.

    Args:
        settings: Application settings

    Returns:
        AsyncRedis: Configured async Redis client
    """
    from redis.asyncio import ConnectionPool, Redis

    redis_url = _get_redis_url(settings)
    max_connections = getattr(settings, "redis_max_connections", 50)
    socket_timeout = getattr(settings, "redis_socket_timeout", 5.0)
    socket_connect_timeout = getattr(settings, "redis_connect_timeout", 5.0)

    pool = ConnectionPool.from_url(
        redis_url,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
        decode_responses=True,
    )

    return Redis(connection_pool=pool)


def _create_sync_redis(settings: "AppSettings") -> "SyncRedis":
    """
    Create a new sync Redis client instance.

    Args:
        settings: Application settings

    Returns:
        SyncRedis: Configured sync Redis client
    """
    from redis import ConnectionPool, Redis

    redis_url = _get_redis_url(settings)
    max_connections = getattr(settings, "redis_max_connections", 50)
    socket_timeout = getattr(settings, "redis_socket_timeout", 5.0)
    socket_connect_timeout = getattr(settings, "redis_connect_timeout", 5.0)

    pool = ConnectionPool.from_url(
        redis_url,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
        decode_responses=True,
    )

    return Redis(connection_pool=pool)


async def get_async_redis_singleton(
    settings: Optional["AppSettings"] = None,
) -> "AsyncRedis":
    """
    Get or create the async Redis client singleton.

    Thread-safe and async-safe singleton access to the Redis client.

    Args:
        settings: Optional settings override (uses default if not provided)

    Returns:
        AsyncRedis: The singleton async Redis client instance
    """
    global _async_redis_instance

    if _async_redis_instance is None:
        async with _get_async_lock():
            # Double-check locking pattern
            if _async_redis_instance is None:
                effective_settings = settings or _get_settings()
                _async_redis_instance = _create_async_redis(effective_settings)

    return _async_redis_instance


def get_sync_redis_singleton(
    settings: Optional["AppSettings"] = None,
) -> "SyncRedis":
    """
    Get or create the sync Redis client singleton.

    Thread-safe singleton access to the sync Redis client.

    Args:
        settings: Optional settings override (uses default if not provided)

    Returns:
        SyncRedis: The singleton sync Redis client instance
    """
    global _sync_redis_instance

    if _sync_redis_instance is None:
        with _redis_lock:
            # Double-check locking pattern
            if _sync_redis_instance is None:
                effective_settings = settings or _get_settings()
                _sync_redis_instance = _create_sync_redis(effective_settings)

    return _sync_redis_instance


async def get_redis_client(
    settings: "AppSettings" = Depends(_get_settings),
) -> "AsyncRedis":
    """
    FastAPI dependency that returns the async Redis client singleton.

    Delegates to the canonical client from ``resync.core.redis_init``,
    which is initialized during application startup (lifespan).

    Args:
        settings: Injected settings (unused — kept for API compatibility)

    Returns:
        AsyncRedis: The singleton async Redis client instance

    Example:
        @router.get("/cache/{key}")
        async def get_cache(key: str, redis = Depends(get_redis_client)):
            return await redis.get(key)
    """
    from resync.core.redis_init import get_redis_client as _canonical

    return _canonical()


def get_redis_client_sync(
    settings: "AppSettings" = Depends(_get_settings),
) -> "SyncRedis":
    """
    FastAPI dependency that returns the sync Redis client singleton.

    Use this for synchronous contexts or background tasks.

    Args:
        settings: Injected settings (provided by FastAPI DI)

    Returns:
        SyncRedis: The singleton sync Redis client instance
    """
    return get_sync_redis_singleton(settings)


async def reset_redis_clients() -> None:
    """
    Reset all Redis client singletons.

    Closes existing connections and clears instances.
    Useful for testing or reconnection after errors.
    """
    global _async_redis_instance, _sync_redis_instance

    async with _get_async_lock():
        if _async_redis_instance is not None:
            await _async_redis_instance.close()
            _async_redis_instance = None

    with _redis_lock:
        if _sync_redis_instance is not None:
            _sync_redis_instance.close()
            _sync_redis_instance = None

        _get_settings.cache_clear()


async def check_redis_health() -> dict:
    """
    Check Redis connection health.

    Returns:
        dict: Health status with ping result and latency
    """
    import time

    try:
        redis = await get_async_redis_singleton()
        start = time.perf_counter()
        await redis.ping()
        latency = (time.perf_counter() - start) * 1000

        return {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "connected": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False,
        }


# Aliases for backward compatibility
get_redis = get_redis_client
get_redis_pool = get_async_redis_singleton
get_redis_client_singleton = get_async_redis_singleton


__all__ = [
    "get_redis_client",
    "get_redis_client_sync",
    "get_async_redis_singleton",
    "get_sync_redis_singleton",
    "get_redis_client_singleton",
    "reset_redis_clients",
    "check_redis_health",
    "get_redis",
    "get_redis_pool",
]
