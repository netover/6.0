"""
Audit Log - PostgreSQL Implementation.

Provides audit logging functionality using PostgreSQL.
This is a thin wrapper around AuditDB for backward compatibility.
"""

import logging

from resync.core.audit_db import AuditDB, log_audit_action
from resync.core.database.models import AuditEntry

logger = logging.getLogger(__name__)

__all__ = [
    "AuditLog",
    "get_audit_log",
    "get_audit_log_manager",
    "log_audit_action",
    "AuditEntry",
]


class AuditLog:
    """
    Audit Log - PostgreSQL Backend.

    This is an alias for AuditDB to maintain backward compatibility.
    """

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._db = AuditDB()

    async def initialize(self) -> None:
        """Initialize the audit log."""
        # AuditDB.initialize() is synchronous.
        # Keep this method async for backward compatibility, but do not
        # await a non-coroutine.
        self._db.initialize()

    async def close(self) -> None:
        """Close the audit log."""
        # AuditDB.close() is synchronous.
        self._db.close()

    async def log(self, action: str, **kwargs) -> AuditEntry:
        """Log an audit action."""
        return await self._db.log_action(action, **kwargs)

    async def log_action(self, action: str, **kwargs) -> AuditEntry:
        """Log an audit action (alias)."""
        return await self.log(action, **kwargs)

    async def get_recent(self, limit: int = 100) -> list[AuditEntry]:
        """Get recent audit entries."""
        return await self._db.get_recent_actions(limit)

    async def search(self, **kwargs) -> list[AuditEntry]:
        """Search audit entries."""
        return await self._db.search_actions(**kwargs)


_instance: AuditLog | None = None


def get_audit_log() -> AuditLog:
    """Get the singleton AuditLog instance."""
    global _instance
    if _instance is None:
        _instance = AuditLog()
    return _instance


# Alias for backward compatibility
get_audit_log_manager = get_audit_log
