"""
Database Session Module.

Provides database session management utilities.
Re-exports from main database module for backwards compatibility.
"""

from resync.core.database import get_db, get_db_session

__all__ = ["get_db", "get_db_session"]

# Backward compat: SessionLocal is the legacy SQLAlchemy session factory name
from resync.core.database.engine import _session_factory as _sf

class _SessionLocalProxy:
    """Lazy proxy for the async session factory (replaces legacy SessionLocal)."""
    def __call__(self, *a, **kw):
        if _sf is None:
            from resync.core.database.engine import get_session_factory
            return get_session_factory()()
        return _sf()
SessionLocal = _SessionLocalProxy()
