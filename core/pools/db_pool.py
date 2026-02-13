"""
Database Pool - PostgreSQL Implementation.

Provides database connection pooling using PostgreSQL/SQLAlchemy.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from resync.core.database.engine import get_engine, get_session, get_session_factory

logger = logging.getLogger(__name__)

__all__ = ["DatabasePool", "DatabaseConnectionPool", "get_db_pool"]


class DatabasePool:
    """
    Database Pool - PostgreSQL Backend.

    Uses SQLAlchemy's built-in connection pooling.
    """

    def __init__(self, db_path: str | None = None, pool_size: int = 10):
        """
        Initialize. db_path is ignored - uses PostgreSQL.
        Pool size is configured via SQLAlchemy engine settings.
        """
        if db_path:
            logger.debug("db_path ignored, using PostgreSQL: %s", db_path)
        self._engine = None
        self._session_factory = None
        self._initialized = False
        self._pool_size = pool_size

    def initialize(self) -> None:
        """Initialize the pool."""
        self._engine = get_engine()
        self._session_factory = get_session_factory()
        self._initialized = True
        logger.info("DatabasePool initialized (PostgreSQL + SQLAlchemy pooling)")

    async def close(self) -> None:
        """Close the pool."""
        if self._engine:
            await self._engine.dispose()
        self._initialized = False

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        async with get_session() as session:
            yield session

    @asynccontextmanager
    async def acquire(self):
        """Alias for get_connection."""
        async with self.get_connection() as conn:
            yield conn

    def get_pool_status(self) -> dict[str, Any]:
        """Get pool status information."""
        if self._engine:
            pool = self._engine.pool
            return {
                "size": pool.size() if hasattr(pool, "size") else self._pool_size,
                "checked_in": pool.checkedin() if hasattr(pool, "checkedin") else 0,
                "checked_out": pool.checkedout() if hasattr(pool, "checkedout") else 0,
                "overflow": pool.overflow() if hasattr(pool, "overflow") else 0,
                "invalid": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
            }
        return {"status": "not_initialized"}


_instance: DatabasePool | None = None


def get_db_pool() -> DatabasePool:
    """Get the singleton DatabasePool instance."""
    global _instance
    if _instance is None:
        _instance = DatabasePool()
    return _instance


async def initialize_db_pool() -> DatabasePool:
    """Initialize and return the DatabasePool."""
    pool = get_db_pool()
    pool.initialize()
    return pool


# Alias for backward compatibility
DatabaseConnectionPool = DatabasePool
