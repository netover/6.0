"""
FastAPI Dependencies with Concurrency Protection.

Provides dependency injection for FastAPI routes using PostgreSQL.
v5.3.20: Added asyncio.Lock() to prevent race conditions during lazy initialization.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_db_session
from resync.core.database.repositories import (
    ContextStore,
    FeedbackStore,
    MetricsStore,
    TWSStore,
)
from resync.core.structured_logger import get_logger as _get_logger

logger = logging.getLogger(__name__)


# ============================================================================
# Re-export get_logger for backward compatibility
# ============================================================================
def get_logger(name: str = __name__) -> logging.Logger:
    """Get a structured logger instance."""
    return _get_logger(name)


# ============================================================================
# User Authentication Dependencies
# ============================================================================
async def get_current_user(request: Request) -> dict[str, Any] | None:
    """
    Get current authenticated user from request.

    Returns None if no user is authenticated.
    For routes that require authentication, use verify_admin_credentials instead.
    """
    try:
        from resync.api.routes.core.auth import verify_admin_credentials

        username = verify_admin_credentials(request)
        if username:
            return {"username": username, "authenticated": True}
    except HTTPException:
        pass
    return None


# ============================================================================
# Concurrency Locks for Thread-Safe Initialization
# ============================================================================
# These locks prevent race conditions when multiple requests arrive
# simultaneously during application startup (common in production restarts)
_tws_store_lock: asyncio.Lock | None = None
_context_store_lock: asyncio.Lock | None = None
_metrics_store_lock: asyncio.Lock | None = None
_feedback_store_lock: asyncio.Lock | None = None


def _get_lock(name: str) -> asyncio.Lock:
    """Lazy-create asyncio locks to avoid import-time loop binding."""
    global \
        _tws_store_lock, \
        _context_store_lock, \
        _metrics_store_lock, \
        _feedback_store_lock
    if name == "tws":
        if _tws_store_lock is None:
            _tws_store_lock = asyncio.Lock()
        return _tws_store_lock
    if name == "context":
        if _context_store_lock is None:
            _context_store_lock = asyncio.Lock()
        return _context_store_lock
    if name == "metrics":
        if _metrics_store_lock is None:
            _metrics_store_lock = asyncio.Lock()
        return _metrics_store_lock
    if name == "feedback":
        if _feedback_store_lock is None:
            _feedback_store_lock = asyncio.Lock()
        return _feedback_store_lock
    raise ValueError(f"unknown lock name: {name}")


# ============================================================================
# Store Singletons
# ============================================================================
_tws_store: TWSStore | None = None
_context_store: ContextStore | None = None
_metrics_store: MetricsStore | None = None
_feedback_store: FeedbackStore | None = None


# ============================================================================
# Database Session Dependencies
# ============================================================================
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with get_db_session() as session:
        yield session


# ============================================================================
# Store Dependencies (Thread-Safe with Double-Check Pattern)
# ============================================================================
async def get_tws_store() -> TWSStore:
    """
    Get TWS store dependency (Thread-Safe).

    Uses double-check locking pattern to ensure only one instance
    is created even under concurrent access.
    """
    global _tws_store
    if _tws_store is None:
        async with _get_lock("tws"):
            if _tws_store is None:
                logger.info("Initializing TWSStore singleton...")
                store = TWSStore()
                await store.initialize()
                _tws_store = store
                logger.info("TWSStore singleton initialized successfully")
    return _tws_store


async def get_context_store() -> ContextStore:
    """
    Get context store dependency (Thread-Safe).

    Uses double-check locking pattern.
    """
    global _context_store
    if _context_store is None:
        async with _get_lock("context"):
            if _context_store is None:
                logger.debug("Initializing ContextStore singleton...")
                _context_store = ContextStore()
    return _context_store


async def get_metrics_store() -> MetricsStore:
    """
    Get metrics store dependency (Thread-Safe).

    Uses double-check locking pattern.
    """
    global _metrics_store
    if _metrics_store is None:
        async with _get_lock("metrics"):
            if _metrics_store is None:
                logger.debug("Initializing MetricsStore singleton...")
                _metrics_store = MetricsStore()
    return _metrics_store


async def get_feedback_store() -> FeedbackStore:
    """
    Get feedback store dependency (Thread-Safe).

    Uses double-check locking pattern.
    """
    global _feedback_store
    if _feedback_store is None:
        async with _get_lock("feedback"):
            if _feedback_store is None:
                logger.debug("Initializing FeedbackStore singleton...")
                _feedback_store = FeedbackStore()
    return _feedback_store


# ============================================================================
# Cleanup (Shutdown)
# ============================================================================
async def cleanup_dependencies():
    """
    Cleanup all store dependencies.

    Called during application shutdown. No locking needed as this
    runs in a single-threaded shutdown context.
    """
    global _tws_store, _context_store, _metrics_store, _feedback_store

    if _tws_store:
        try:
            await _tws_store.close()
            logger.info("TWSStore closed successfully")
        except Exception as e:
            logger.warning("Error closing TWSStore: %s", e)
        _tws_store = None

    _context_store = None
    _metrics_store = None
    _feedback_store = None

    logger.info("All dependencies cleaned up")
