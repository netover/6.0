"""
Pool Manager - PostgreSQL Implementation.

Manages various connection pools using PostgreSQL.
"""

import logging
from typing import Any

from .db_pool import DatabasePool, get_db_pool

logger = logging.getLogger(__name__)

__all__ = [
    "PoolManager",
    "ConnectionPoolManager",
    "get_pool_manager",
    "get_connection_pool_manager",
]


class PoolManager:
    """
    Pool Manager - PostgreSQL Backend.

    Manages database and other connection pools.
    """

    def __init__(self):
        """Initialize the pool manager."""
        self._db_pool: DatabasePool | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all pools."""
        self._db_pool = get_db_pool()
        await self._db_pool.initialize()
        self._initialized = True
        logger.info("PoolManager initialized (PostgreSQL)")

    async def close(self) -> None:
        """Close all pools."""
        if self._db_pool:
            await self._db_pool.close()
        self._initialized = False

    @property
    def db_pool(self) -> DatabasePool:
        """Get the database pool."""
        if not self._db_pool:
            self._db_pool = get_db_pool()
        return self._db_pool

    def get_status(self) -> dict[str, Any]:
        """Get status of all pools."""
        return {
            "database": self._db_pool.get_pool_status()
            if self._db_pool
            else {"status": "not_initialized"},
            "initialized": self._initialized,
        }


_instance: PoolManager | None = None


def get_pool_manager() -> PoolManager:
    """Get the singleton PoolManager instance."""
    global _instance
    if _instance is None:
        _instance = PoolManager()
    return _instance


async def initialize_pool_manager() -> PoolManager:
    """Initialize and return the PoolManager."""
    manager = get_pool_manager()
    await manager.initialize()
    return manager


# Aliases for backward compatibility
ConnectionPoolManager = PoolManager
get_connection_pool_manager = get_pool_manager
