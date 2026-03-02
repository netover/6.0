"""
Database Session Module.

Provides database session management utilities.
Re-exports from main database module for backwards compatibility.
"""

# P2-42 FIX: Don't import _session_factory at module level (early binding)
# Always use lazy import inside the proxy

class _SessionLocalProxy:
    """Lazy proxy for the async session factory (replaces legacy SessionLocal)."""
    def __call__(self, *a, **kw):
        # Always get the factory fresh to avoid early binding issues
        from resync.core.database.engine import get_session_factory
        factory = get_session_factory()
        return factory()


SessionLocal = _SessionLocalProxy()

__all__ = ["get_db", "get_db_session", "SessionLocal"]
