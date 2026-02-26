"""Audit Queue - PostgreSQL Implementation.

Provides audit queue functionality using PostgreSQL.
Replaces the original SQLite implementation.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from resync.core.database.models import AuditQueueItem
from resync.core.database.repositories import AuditQueueRepository

logger = logging.getLogger(__name__)

__all__ = ["AuditQueue", "AsyncAuditQueue", "IAuditQueue", "get_audit_queue"]


class AuditQueue:
    """Audit Queue - PostgreSQL Backend."""

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._repo = AuditQueueRepository()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the queue."""
        self._initialized = True
        logger.info("AuditQueue initialized (PostgreSQL)")

    def close(self) -> None:
        """Close the queue."""
        self._initialized = False

    async def enqueue(
        self, action: str, payload: dict[str, Any], priority: int = 0
    ) -> AuditQueueItem:
        """Add item to queue."""
        return await self._repo.enqueue(action, payload, priority)

    async def get_pending(self, limit: int = 10) -> list[AuditQueueItem]:
        """Get pending items ordered by priority."""
        return await self._repo.get_pending(limit)

    async def get_pending_audits(self, limit: int = 10) -> list[AuditQueueItem]:
        """Alias for get_pending."""
        return await self.get_pending(limit)

    async def mark_processing(self, item_id: int) -> AuditQueueItem | None:
        """Mark item as processing."""
        return await self._repo.mark_processing(item_id)

    async def mark_completed(self, item_id: int) -> AuditQueueItem | None:
        """Mark item as completed."""
        return await self._repo.mark_completed(item_id)

    async def mark_failed(
        self, item_id: int, error_message: str
    ) -> AuditQueueItem | None:
        """Mark item as failed."""
        return await self._repo.mark_failed(item_id, error_message)

    async def get_queue_length(self) -> int:
        """Get number of pending items."""
        return await self._repo.count({"status": "pending"})

    async def cleanup_processed_audits(self, days: int = 7) -> int:
        """Clean up old processed items."""
        if days < 1 or days > 365:
            raise ValueError("days must be between 1 and 365")
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return await self._repo.delete_many(
            {"status": "completed", "completed_at": {"$lt": cutoff}}
        )

    async def process_next(self, processor_fn: Callable) -> bool:
        """Process next item in queue."""
        items = await self.get_pending(limit=1)
        if not items:
            return False

        item = items[0]
        await self.mark_processing(item.id)

        try:
            await processor_fn(item.action, item.payload)
            await self.mark_completed(item.id)
            return True
        except (ValueError, KeyError, AttributeError, TimeoutError, ConnectionError) as e:
            await self.mark_failed(item.id, str(e))
            return False


_instance: AuditQueue | None = None
_instance_lock = asyncio.Lock()


async def get_audit_queue() -> AuditQueue:
    """Get the singleton AuditQueue instance (thread-safe)."""
    global _instance
    
    if _instance is None:
        async with _instance_lock:
            if _instance is None:
                _instance = AuditQueue()
                await asyncio.to_thread(_instance.initialize)
    
    return _instance


async def initialize_audit_queue() -> AuditQueue:
    """Initialize and return the AuditQueue."""
    queue = await get_audit_queue()
    return queue


# Aliases for backward compatibility
AsyncAuditQueue = AuditQueue
IAuditQueue = AuditQueue  # Interface alias
