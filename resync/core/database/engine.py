"""
Database Engine and Session Management.

Provides async database connections with connection pooling for PostgreSQL.

v5.9.4: Implementação de graceful shutdown com draining de requisições.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import AsyncAdaptedQueuePool

from .config import get_database_config

logger = logging.getLogger(__name__)

# Base class for all SQLAlchemy models
class Base(DeclarativeBase):
    """Base class for all database models."""

# Module-level engine singleton — intentional for connection pool reuse.
# _engine_lock guards the lazy initialization only (runs once at startup).
# All async I/O uses the async session factory after initialization.
_engine = None
_session_factory = None
_engine_lock = threading.Lock()  # Sync lock guards one-time lazy init only

# v5.9.4: Contador de sessões ativas para graceful shutdown
_active_sessions = 0
_active_sessions_lock: asyncio.Lock | None = None
_active_sessions_lock_loop: asyncio.AbstractEventLoop | None = None

def _get_active_sessions_lock() -> asyncio.Lock:
    """Return a lock bound to the current running event loop.

    This avoids creating asyncio primitives at import time, which can break
    under pre-fork servers (e.g., gunicorn --preload).
    """
    global _active_sessions_lock, _active_sessions_lock_loop  # pylint
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet; create a standalone lock.
        if _active_sessions_lock is None:
            _active_sessions_lock = asyncio.Lock()
        return _active_sessions_lock

    if _active_sessions_lock is None or _active_sessions_lock_loop is not loop:
        _active_sessions_lock = asyncio.Lock()
        _active_sessions_lock_loop = loop
    return _active_sessions_lock

_shutdown_event: asyncio.Event | None = None

async def _increment_sessions():
    """Incrementa contador de sessões ativas."""
    global _active_sessions
    async with _get_active_sessions_lock():
        _active_sessions += 1

async def _decrement_sessions():
    """Decrementa contador de sessões ativas."""
    global _active_sessions
    async with _get_active_sessions_lock():
        _active_sessions -= 1

def get_engine():
    """
    Get or create the database engine.

    Returns an async engine configured for PostgreSQL with connection pooling.
    """
    global _engine

    # HARDENING [P2]: Proteção Thread-safe contra Double-Checked Locking para inicialização do Engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                config = get_database_config()

                # Engine options with connection pooling
                options = {
                    "poolclass": AsyncAdaptedQueuePool,
                    "pool_size": config.pool_size,
                    "max_overflow": config.max_overflow,
                    "pool_timeout": config.pool_timeout,
                    "pool_recycle": config.pool_recycle,
                    "pool_pre_ping": config.pool_pre_ping,
                    # Timeouts críticos para produção
                    "connect_args": {
                        "timeout": 5,  # Connection timeout (segundos)
                        "command_timeout": 60,  # Query timeout (segundos)
                        "server_settings": {
                            "jit": "off",  # Desabilita JIT para queries simples (menor overhead)
                        },
                    },
                }

                logger.info(
                    "Creating PostgreSQL database engine",
                    extra={
                        "host": config.host,
                        "database": config.name,
                        "pool_size": config.pool_size,
                    },
                )

                _engine = create_async_engine(config.url, **options)

    return _engine

def get_session_factory():
    """Get or create session factory."""
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _session_factory

def is_engine_available() -> bool:
    """Check if engine is available (not shut down)."""
    return _engine is not None and _shutdown_event is None

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.

    v5.9.4: Protegido contra uso durante shutdown.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)

    Raises:
        RuntimeError: Se engine está em processo de shutdown.
    """
    if _shutdown_event is not None:
        raise RuntimeError(
            "Database engine is shutting down. Cannot create new sessions."
        )

    if _engine is None:
        # Lazy initialization
        get_engine()

    factory = get_session_factory()
    await _increment_sessions()

    try:
        async with factory() as session:
            try:
                yield session
            except asyncio.CancelledError:
                # Do not swallow cancellations; let the server shut down promptly.
                raise
            except Exception:
                # Roll back so the connection is clean for pool reuse.
                await session.rollback()
                raise
    finally:
        await _decrement_sessions()

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Alias for get_session (FastAPI dependency compatible).
    """
    async with get_session() as session:
        yield session

async def close_engine(timeout: float = 30.0):
    """
    Close the database engine and all connections with graceful draining.

    v5.9.4: Implementa graceful shutdown que espera requisições em andamento.

    Args:
        timeout: Tempo máximo (segundos) para aguardar sessões ativas.
    """
    global _engine, _session_factory, _shutdown_event, _active_sessions

    if _engine is None:
        return

    # Sinalizar que estamos em shutdown (novas sessões serão rejeitadas)
    _shutdown_event = asyncio.Event()
    logger.info("Database shutdown initiated, waiting for active sessions...")

    # Aguardar sessões ativas com timeout
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    while True:
        async with _get_active_sessions_lock():
            active = _active_sessions

        if active == 0:
            logger.info("All database sessions completed")
            break

        elapsed = loop.time() - start_time
        if elapsed >= timeout:
            logger.warning(
                f"Shutdown timeout reached with {active} active sessions. Forcing close."
            )
            break

        logger.debug("Waiting for %s active sessions to complete...", active)
        await asyncio.sleep(0.5)

    # Fechar engine
    try:
        await _engine.dispose()
        logger.info("Database engine closed successfully")
    except SQLAlchemyError as e:
        logger.error("Error closing database engine: %s", e)
    finally:
        _engine = None
        _session_factory = None
        _shutdown_event = None
        _active_sessions = 0

def get_engine_status() -> dict:
    """Get engine status information."""
    if _engine is None:
        return {"status": "not_initialized"}

    status = "shutting_down" if _shutdown_event is not None else "running"

    pool = _engine.pool
    return {
        "status": status,
        "active_sessions": _active_sessions,
        "pool_size": pool.size() if hasattr(pool, "size") else 0,
        "checked_in": pool.checkedin() if hasattr(pool, "checkedin") else 0,
        "checked_out": pool.checkedout() if hasattr(pool, "checkedout") else 0,
        "overflow": pool.overflow() if hasattr(pool, "overflow") else 0,
    }
