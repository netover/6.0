"""
Background Task Utilities

Provides helpers for background tasks that need database access.
Ensures proper session handling after request context closes.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_db_session

logger = logging.getLogger(__name__)

# =============================================================================
# Background DB Session Helper
# =============================================================================


@asynccontextmanager
async def get_background_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a database session specifically for background tasks.
    
    This session is independent of the request lifecycle and can be safely
    used in BackgroundTasks that execute after the request context closes.
    
    Usage:
        async def background_task(user_id: int, data: dict):
            async with get_background_db_session() as session:
                await session.execute(insert(Table).values(...))
                await session.commit()
    """
    async with get_db_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# Conversation Memory Singleton with Thread Safety
# =============================================================================


_conversation_memory_instance: "ConversationMemory | None" = None
_conversation_memory_lock: asyncio.Lock | None = None
_conversation_memory_thread_lock: threading.Lock | None = None


def _get_memory_lock() -> asyncio.Lock:
    """Get or create the async lock for memory singleton."""
    global _conversation_memory_lock
    if _conversation_memory_lock is None:
        _conversation_memory_lock = asyncio.Lock()
    return _conversation_memory_lock


def _get_thread_lock() -> threading.Lock:
    """Get or create the thread lock for synchronous memory access."""
    global _conversation_memory_thread_lock
    if _conversation_memory_thread_lock is None:
        import threading
        _conversation_memory_thread_lock = threading.Lock()
    return _conversation_memory_thread_lock


async def get_conversation_memory_safe() -> "ConversationMemory":
    """
    Thread-safe singleton getter for ConversationMemory.
    
    Uses double-checked locking pattern with asyncio.Lock to prevent
    race conditions during concurrent initialization.
    
    Returns:
        The singleton ConversationMemory instance.
    """
    global _conversation_memory_instance
    
    # First check (without lock)
    if _conversation_memory_instance is not None:
        return _conversation_memory_instance
    
    # Acquire lock and check again
    lock = _get_memory_lock()
    async with lock:
        # Double-check after acquiring lock
        if _conversation_memory_instance is None:
            from resync.core.memory.conversation_memory import ConversationMemory
            
            _conversation_memory_instance = ConversationMemory()
            logger.debug("Created new ConversationMemory singleton instance")
    
    return _conversation_memory_instance


def get_conversation_memory() -> "ConversationMemory":
    """
    Synchronous singleton getter for ConversationMemory.
    
    Note: For async contexts, prefer get_conversation_memory_safe().
    This function is kept for backward compatibility with synchronous code.
    
    Returns:
        The singleton ConversationMemory instance.
    """
    global _conversation_memory_instance
    
    if _conversation_memory_instance is None:
        from resync.core.memory.conversation_memory import ConversationMemory
        
        # Use threading lock for sync access (asyncio.Lock is not sync-safe)
        lock = _get_thread_lock()
        with lock:
            if _conversation_memory_instance is None:
                _conversation_memory_instance = ConversationMemory()
    
    return _conversation_memory_instance


__all__ = [
    "get_background_db_session",
    "get_conversation_memory_safe",
    "get_conversation_memory",
]
