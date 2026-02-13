"""
Resource management utilities with context managers for clean resource handling.

This module provides context managers for managing resources like database connections,
LLM clients, file handles, and other resources that need proper cleanup.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceManager(ABC, Generic[T]):
    """Abstract base class for resource managers."""

    @abstractmethod
    async def acquire(self) -> T:
        """Acquire a resource."""

    @abstractmethod
    async def release(self, resource: T) -> None:
        """Release a resource."""

    @abstractmethod
    async def health_check(self, resource: T) -> bool:
        """Check if resource is healthy."""


class LLMResourceManager(ResourceManager):
    """Resource manager for LLM clients."""

    def __init__(self, client_factory: Callable[[], Any]):
        self.client_factory = client_factory

    async def acquire(self) -> Any:
        """Create and return an LLM client."""
        try:
            client = self.client_factory()
            logger.debug("LLM client acquired")
            return client
        except Exception as e:
            logger.error(f"Failed to acquire LLM client: {e}")
            raise

    async def release(self, client: Any) -> None:
        """Close LLM client."""
        try:
            if hasattr(client, "close"):
                await client.close()
            elif hasattr(client, "aclose"):
                await client.aclose()
            logger.debug("LLM client released")
        except Exception as e:
            logger.warning(f"Error releasing LLM client: {e}")

    async def health_check(self, client: Any) -> bool:
        """Check if LLM client is healthy."""
        try:
            # Basic health check - client should exist and not be closed
            return client is not None
        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            return False


@asynccontextmanager
async def managed_llm_call(
    client_factory: Callable[[], Any] | None = None,
) -> AsyncIterator[Any]:
    """
    Context manager for LLM client lifecycle.

    Usage:
        async with managed_llm_call(create_client) as client:
            result = await client.generate(prompt)
    """
    if client_factory is None:
        # Default factory - can be customized
        from resync.core.utils.llm import create_llm_client

        client_factory = create_llm_client

    manager = LLMResourceManager(client_factory)
    client = None

    try:
        client = await manager.acquire()
        yield client
    except Exception as e:
        logger.error(f"Error in managed LLM call: {e}")
        raise
    finally:
        if client:
            await manager.release(client)


@asynccontextmanager
async def managed_database_connection(
    connection_factory: Callable[[], Any],
) -> AsyncIterator[Any]:
    """
    Context manager for database connection lifecycle.

    Usage:
        async with managed_database_connection(create_connection) as conn:
            result = await conn.execute(query)
    """
    conn = None
    try:
        conn = await connection_factory()
        yield conn
    except Exception as e:
        logger.error(f"Database operation error: {e}")
        raise
    finally:
        if conn:
            try:
                await conn.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")


@contextmanager
def managed_file_operation(file_path: str, mode: str = "r", **kwargs) -> Iterator[Any]:
    """
    Context manager for file operations.

    Usage:
        with managed_file_operation('file.txt', 'r') as f:
            content = f.read()
    """
    file_obj = None
    try:
        file_obj = open(file_path, mode, **kwargs)
        yield file_obj
    except Exception as e:
        logger.error(f"File operation error for {file_path}: {e}")
        raise
    finally:
        if file_obj:
            try:
                file_obj.close()
                logger.debug(f"File {file_path} closed")
            except Exception as e:
                logger.warning(f"Error closing file {file_path}: {e}")


@asynccontextmanager
async def managed_http_session(
    session_factory: Callable[[], Any] | None = None,
) -> AsyncIterator[Any]:
    """
    Context manager for HTTP session lifecycle.

    Usage:
        async with managed_http_session() as session:
            async with session.get(url) as response:
                data = await response.json()
    """
    if session_factory is None:
        # Default to aiohttp if available
        try:
            import aiohttp

            def session_factory():
                return aiohttp.ClientSession()
        except ImportError:
            try:
                import httpx

                def session_factory():
                    return httpx.AsyncClient()
            except ImportError:
                raise ImportError(
                    "No HTTP client library available. Install aiohttp or httpx."
                ) from None

    session = None
    try:
        session = await session_factory()
        yield session
    except Exception as e:
        logger.error(f"HTTP session error: {e}")
        raise
    finally:
        if session:
            try:
                await session.close()
                logger.debug("HTTP session closed")
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {e}")


class ResourcePoolManager:
    """
    Generic resource pool manager with context manager support.

    Provides connection pooling for any resource type.
    """

    def __init__(self, resource_manager: ResourceManager[T], max_size: int = 10):
        self.resource_manager = resource_manager
        self.max_size = max_size
        self.pool: asyncio.Queue[T | None] = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the resource pool."""
        # Pre-populate pool if needed

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        """Acquire a resource from the pool."""
        resource = await self._get_resource()
        try:
            yield resource
        finally:
            await self._return_resource(resource)

    async def _get_resource(self) -> T:
        """Get a resource from pool or create new one."""
        try:
            # Try to get from pool first
            resource = self.pool.get_nowait()
            if resource and await self.resource_manager.health_check(resource):
                return resource
            # Resource unhealthy, create new one
        except asyncio.QueueEmpty as exc:
            logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        # Create new resource
        async with self._lock:
            if self.size < self.max_size:
                resource = await self.resource_manager.acquire()
                self.size += 1
                return resource

        # Pool full, wait for available resource
        resource = await self.pool.get()
        if resource and await self.resource_manager.health_check(resource):
            return resource

        # Create new resource even if pool is "full" but resource was unhealthy
        return await self.resource_manager.acquire()

    async def _return_resource(self, resource: T) -> None:
        """Return resource to pool."""
        try:
            if await self.resource_manager.health_check(resource):
                await self.pool.put(resource)
            else:
                # Resource unhealthy, don't return to pool
                await self.resource_manager.release(resource)
                async with self._lock:
                    self.size -= 1
        except Exception as e:
            logger.warning(f"Error returning resource to pool: {e}")
            async with self._lock:
                self.size -= 1

    async def close(self) -> None:
        """Close all resources in pool."""
        resources = []
        try:
            while not self.pool.empty():
                resource = self.pool.get_nowait()
                if resource:
                    resources.append(resource)
        except asyncio.QueueEmpty as exc:
            logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        for resource in resources:
            try:
                await self.resource_manager.release(resource)
            except Exception as e:
                logger.warning(f"Error closing pooled resource: {e}")


# Convenience functions for common use cases


async def create_llm_resource_pool(max_size: int = 5) -> ResourcePoolManager:
    """Create a resource pool for LLM clients."""
    from resync.core.utils.llm import create_llm_client

    manager = LLMResourceManager(create_llm_client)
    pool = ResourcePoolManager(manager, max_size)
    await pool.initialize()
    return pool


def create_database_pool_manager(
    connection_factory: Callable[[], Any], max_size: int = 10
) -> ResourcePoolManager:
    """Create a resource pool for database connections."""

    class DatabaseResourceManager(ResourceManager):
        """Manager class for handling database resource operations."""

        def __init__(self, factory):
            self.factory = factory

        async def acquire(self):
            return await self.factory()

        async def release(self, conn):
            await conn.close()

        async def health_check(self, conn):
            try:
                # Basic health check - try a simple query
                if hasattr(conn, "execute"):
                    await conn.execute("SELECT 1")
                return True
            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                return False

    manager = DatabaseResourceManager(connection_factory)
    return ResourcePoolManager(manager, max_size)


@asynccontextmanager
async def managed_transaction(connection) -> AsyncIterator[Any]:
    """
    Context manager for database transactions.

    Usage:
        async with managed_transaction(conn) as tx:
            await tx.execute("INSERT INTO...")
            await tx.commit()
    """
    try:
        yield connection
        await connection.commit()
    except Exception as _e:
        await connection.rollback()
        raise
