"""
Proactive Monitoring Initialization - PostgreSQL Implementation.

Initializes proactive monitoring components using PostgreSQL.
"""

import logging
from typing import Any

from resync.core.database.repositories import TWSStore

logger = logging.getLogger(__name__)

__all__ = ["ProactiveMonitor", "initialize_proactive_monitoring"]


class ProactiveMonitor:
    """Proactive Monitor using PostgreSQL backend."""

    def __init__(self, db_path: str | None = None, **kwargs):
        """Initialize. db_path is ignored - uses PostgreSQL."""
        if db_path:
            logger.debug(f"db_path ignored, using PostgreSQL: {db_path}")
        self._store = TWSStore()
        self._initialized = False
        self._config = kwargs

    async def initialize(self) -> None:
        """Initialize the monitor."""
        await self._store.initialize()
        self._initialized = True
        logger.info("ProactiveMonitor initialized (PostgreSQL)")

    async def close(self) -> None:
        """Close the monitor."""
        await self._store.close()
        self._initialized = False

    async def check_status(self) -> dict[str, Any]:
        """Check TWS status."""
        summary = await self._store.get_status_summary()
        return {"status": "ok", "summary": summary}

    async def get_alerts(self, limit: int = 100) -> list:
        """Get active alerts."""
        events = await self._store.events.get_unacknowledged(limit)
        return [
            {
                "id": e.id,
                "type": e.event_type,
                "severity": e.severity,
                "message": e.message,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]


_instance: ProactiveMonitor | None = None


async def initialize_proactive_monitoring(db_path: str | None = None, **kwargs) -> ProactiveMonitor:
    """Initialize and return the ProactiveMonitor."""
    global _instance
    if _instance is None:
        _instance = ProactiveMonitor(db_path=db_path, **kwargs)
        await _instance.initialize()
    return _instance


def get_proactive_monitor() -> ProactiveMonitor | None:
    """Get the ProactiveMonitor instance."""
    return _instance
