# ruff: noqa: E501
# pylint: disable=all
"""
PostgreSQL Checkpointer for LangGraph v6.0.0.

This module provides a simplified wrapper around LangGraph's native
AsyncPostgresSaver for conversation state persistence.

Changes from v5.x:
- Uses native langgraph-checkpoint-postgres instead of custom implementation
- Removed 500+ lines of custom code
- Full compatibility with LangGraph Studio
- Automatic schema management

Usage:
    checkpointer = await get_checkpointer()
    graph = await create_tws_agent_graph(checkpointer=checkpointer)

    # State is automatically saved after each step
    result = await graph.invoke(
        {"message": "..."},
        config={"configurable": {"thread_id": "user-123"}}
    )
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Try to import native LangGraph checkpointer
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    NATIVE_CHECKPOINTER_AVAILABLE = True
except ImportError:
    NATIVE_CHECKPOINTER_AVAILABLE = False
    AsyncPostgresSaver = None

# Fallback: try older import path
if not NATIVE_CHECKPOINTER_AVAILABLE:
    try:
        from langgraph_checkpoint_postgres import AsyncPostgresSaver

        NATIVE_CHECKPOINTER_AVAILABLE = True
    except ImportError:
        pass


# =============================================================================
# CONFIGURATION
# =============================================================================


def get_database_url() -> str:
    """Get PostgreSQL connection URL for checkpointer."""
    # Try settings first
    if hasattr(settings, "database_url") and settings.database_url:
        return settings.database_url

    # Try environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Build from components
    host = getattr(settings, "db_host", None) or os.getenv("DB_HOST", "localhost")
    port = getattr(settings, "db_port", None) or os.getenv("DB_PORT", "5432")
    user = getattr(settings, "db_user", None) or os.getenv("DB_USER", "postgres")
    password = getattr(settings, "db_password", None) or os.getenv("DB_PASSWORD", "")
    database = getattr(settings, "db_name", None) or os.getenv("DB_NAME", "resync")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


# =============================================================================
# CHECKPOINTER FACTORY
# =============================================================================


_checkpointer_instance: AsyncPostgresSaver | None = None
_checkpointer_lock: asyncio.Lock | None = None


def _get_checkpointer_lock() -> asyncio.Lock:
    """Lazily create the asyncio lock (must be inside a running event loop)."""
    global _checkpointer_lock
    if _checkpointer_lock is None:
        _checkpointer_lock = asyncio.Lock()
    return _checkpointer_lock


async def get_checkpointer() -> AsyncPostgresSaver | None:
    """
    Get or create the PostgreSQL checkpointer instance.

    Thread-safe: uses asyncio.Lock to prevent duplicate initialization
    when multiple coroutines call this concurrently during startup.

    Returns:
        AsyncPostgresSaver instance or None if unavailable

    Example:
        checkpointer = await get_checkpointer()
        if checkpointer:
            graph = create_graph(checkpointer=checkpointer)
    """
    global _checkpointer_instance

    if not NATIVE_CHECKPOINTER_AVAILABLE:
        logger.warning(
            "native_checkpointer_unavailable",
            message="Install langgraph-checkpoint-postgres for persistence",
        )
        return None

    if _checkpointer_instance is not None:
        return _checkpointer_instance

    async with _get_checkpointer_lock():
        # Double-check after acquiring lock
        if _checkpointer_instance is not None:
            return _checkpointer_instance

        try:
            db_url = get_database_url()

            # Create native checkpointer
            _checkpointer_instance = AsyncPostgresSaver.from_conn_string(db_url)

            # Setup schema (creates tables if needed)
            await _checkpointer_instance.setup()

            logger.info(
                "checkpointer_initialized",
                type="AsyncPostgresSaver",
                database=db_url.split("@")[-1] if "@" in db_url else "localhost",
            )

            return _checkpointer_instance

        except Exception as e:
            logger.error("checkpointer_init_failed", error=str(e))
            return None


def close_checkpointer() -> None:
    """Close the checkpointer connection."""
    global _checkpointer_instance

    if _checkpointer_instance is not None:
        try:
            # Native checkpointer handles cleanup
            _checkpointer_instance = None
            logger.info("checkpointer_closed")
        except Exception as e:
            logger.warning("checkpointer_close_error", error=str(e))


@asynccontextmanager
async def checkpointer_context():
    """
    Context manager for checkpointer lifecycle.

    Usage:
        async with checkpointer_context() as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            result = await graph.invoke(...)
    """
    checkpointer = await get_checkpointer()
    try:
        yield checkpointer
    finally:
        close_checkpointer()


# =============================================================================
# MEMORY STORE (NEW IN LANGGRAPH 0.3)
# =============================================================================


try:
    from langgraph.store.memory import InMemoryStore

    MEMORY_STORE_AVAILABLE = True
except ImportError:
    MEMORY_STORE_AVAILABLE = False
    InMemoryStore = None


_memory_store_instance = None
_memory_store_lock = threading.Lock()


def get_memory_store():
    """
    Get in-memory store for cross-thread memory.

    Thread-safe: uses double-checked locking to prevent duplicate instances.
    LangGraph 0.3 feature: Allows sharing state across threads.

    Returns:
        InMemoryStore instance or None if unavailable
    """
    global _memory_store_instance

    if not MEMORY_STORE_AVAILABLE:
        return None

    if _memory_store_instance is None:
        with _memory_store_lock:
            if _memory_store_instance is None:
                _memory_store_instance = InMemoryStore()
                logger.info("memory_store_initialized")

    return _memory_store_instance


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================


class PostgresCheckpointer:
    """
    Legacy compatibility wrapper.

    Deprecated: Use get_checkpointer() directly instead.
    This class is kept for backward compatibility with existing code.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ttl_hours: int = 24, compress_threshold: int = 10000):
        """Initialize wrapper (parameters ignored - using native defaults)."""
        self._native_checkpointer = None
        logger.warning(
            "legacy_checkpointer_used",
            message="PostgresCheckpointer is deprecated. Use get_checkpointer() instead.",
        )

    async def get_native(self) -> AsyncPostgresSaver | None:
        """Get the underlying native checkpointer."""
        if self._native_checkpointer is None:
            self._native_checkpointer = await get_checkpointer()
        return self._native_checkpointer

    # Legacy method signatures - delegate to native
    async def aget(self, config: dict[str, Any]) -> dict | None:
        """Legacy get method."""
        native = await self.get_native()
        if native:
            return await native.aget(config)
        return None

    async def aput(
        self, config: dict[str, Any], checkpoint: dict, metadata: dict
    ) -> dict:
        """Legacy put method."""
        native = await self.get_native()
        if native:
            return await native.aput(config, checkpoint, metadata)
        return config


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "get_checkpointer",
    "close_checkpointer",
    "checkpointer_context",
    "get_memory_store",
    "PostgresCheckpointer",  # Legacy
    "NATIVE_CHECKPOINTER_AVAILABLE",
]
