"""
Centralized Factory Module for Resync.

This package provides centralized factory functions for creating
and managing singleton instances of various clients and services.

All other modules should import from these factories instead of
creating their own instances.

Usage:
    # TWS Client
    from resync.core.factories import get_tws_client, get_tws_client_singleton

    # Redis Client
    from resync.core.factories import get_redis_client, get_async_redis_singleton

Available Factories:
    - TWS Factory: TWS client creation and dependency injection
    - Redis Factory: Redis client creation and connection pooling
"""

from resync.core.factories.tws_factory import (
    get_tws_client,
    get_tws_client_singleton,
    get_tws_client_factory,
    reset_tws_client,
)

from resync.core.factories.redis_factory import (
    get_redis_client,
    get_redis_client_sync,
    get_async_redis_singleton,
    get_sync_redis_singleton,
    reset_redis_clients,
    check_redis_health,
)


__all__ = [
    # TWS Factory
    "get_tws_client",
    "get_tws_client_singleton",
    "get_tws_client_factory",
    "reset_tws_client",
    # Redis Factory
    "get_redis_client",
    "get_redis_client_sync",
    "get_async_redis_singleton",
    "get_sync_redis_singleton",
    "reset_redis_clients",
    "check_redis_health",
]
