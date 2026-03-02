"""Audit Log - PostgreSQL Implementation.

Provides audit logging functionality using PostgreSQL.
This is a thin wrapper around AuditDB for backward compatibility.
"""

import asyncio
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
    """Audit Log - PostgreSQL Backend.

    This is an alias for AuditDB to maintain backward compatibility.
    """

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._db = AuditDB()

    async def initialize(self) -> None:
        """Initialize the audit log."""
        await asyncio.to_thread(self._db.initialize)

    async def close(self) -> None:
        """Close the audit log."""
        await asyncio.to_thread(self._db.close)

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

    async def log_audit_event(
        self,
        action: str,
        user_id: str = "system",
        details: dict | None = None,
        correlation_id: str | None = None,
        source_component: str = "main",
        severity: str = "INFO",
    ) -> AuditEntry:
        """Persist an audit event to the database.

        This is the async entry-point used by log_audit_event_async() in logger.py
        and by any caller that has already awaited get_audit_log_manager().
        """
        return await self._db.log_action(
            action,
            user_id=user_id,
            details=details or {},
            correlation_id=correlation_id,
            source_component=source_component,
            severity=severity,
        )

    async def get_audit_metrics(self) -> dict:
        """Return basic audit metrics (total records, pending, etc.).

        Delegates to AuditDB if the method exists there; otherwise builds
        a lightweight summary so callers in chaos_engineering.py can rely on
        the ``total_records`` key always being present.
        """
        try:
            if hasattr(self._db, "get_audit_metrics"):
                return await asyncio.to_thread(self._db.get_audit_metrics)  # type: ignore[attr-defined]
        except Exception as exc:
            # Never swallow cancellations: it breaks graceful shutdown/timeouts.
            if isinstance(exc, asyncio.CancelledError):
                raise
            pass

        # Fallback: derive metrics from the recent-entries count.
        try:
            recent = await self._db.get_recent_actions(limit=1000)
            return {
                "total_records": len(recent),
                "source": "fallback",
            }
        except Exception as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            return {
                "total_records": 0,
                "error": str(exc),
                "source": "fallback",
            }


_instance: AuditLog | None = None
_instance_lock = asyncio.Lock()


async def get_audit_log() -> AuditLog:
    """Get the singleton AuditLog instance (thread-safe)."""
    global _instance
    
    if _instance is None:
        async with _instance_lock:
            if _instance is None:
                _instance = AuditLog()
                await _instance.initialize()
    
    return _instance


# Alias for backward compatibility
get_audit_log_manager = get_audit_log
