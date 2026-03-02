"""
resync.models.base
==================
Re-exports the canonical SQLAlchemy ``Base`` from the database engine module.

This shim exists to satisfy legacy imports like::

    from resync.models.base import Base

The real ``Base`` lives in :mod:`resync.core.database.engine`.
"""
from __future__ import annotations

from resync.core.database.engine import Base

__all__ = ["Base"]
