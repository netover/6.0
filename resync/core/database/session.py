"""
Database Session Module.

Provides database session management utilities.
Re-exports from main database module for backwards compatibility.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession


# P2-42 FIX: Don't import _session_factory at module level (early binding)
# Always use lazy import inside the proxy
class _SessionLocalProxy:
    """Lazy proxy for the async session factory (replaces legacy SessionLocal)."""

    def __call__(self, *a: object, **kw: object) -> AsyncSession:
        # Always get the factory fresh to avoid early binding issues
        from resync.core.database.engine import get_session_factory

        factory = get_session_factory()
        return factory()


SessionLocal = _SessionLocalProxy()


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Backward-compatible dependency provider for async DB sessions."""
    from resync.core.database.engine import get_db_session as _engine_get_db_session

    async for session in _engine_get_db_session():
        yield session


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Legacy alias retained for imports expecting `get_db`."""
    async for session in get_db_session():
        yield session


__all__ = ["get_db", "get_db_session", "SessionLocal"]
