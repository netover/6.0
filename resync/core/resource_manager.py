# pylint
"""
Resource Management Utilities for Phase 2 Performance Optimizations.

This module provides utilities for deterministic resource cleanup,
resource tracking, and leak detection following best practices from
Java's try-with-resources and Python's context managers.
"""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class ResourceInfo:
    """Information about a managed resource."""

    resource_id: str
    resource_type: str
    resource_instance: Any = field(default=None, repr=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_lifetime_seconds(self) -> float:
        """Get the lifetime of the resource in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

class ManagedResource[T]:
    """
    Base class for managed resources with automatic cleanup.

    Provides deterministic cleanup similar to Java's try-with-resources.
    """

    def __init__(self, resource_id: str, resource_type: str):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.created_at = datetime.now(timezone.utc)
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        logger.debug(
            "Acquiring resource: %s (%s)", self.resource_id, self.resource_type
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with automatic cleanup."""
        await self.close()
        return False

    def __enter__(self):
        """Sync context manager entry."""
        logger.debug(
            "Acquiring resource: %s (%s)", self.resource_id, self.resource_type
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit with automatic cleanup."""
        self.close_sync()
        return False

    async def close(self) -> None:
        """Close the resource asynchronously."""
        if not self._closed:
            try:
                await self._cleanup()
                self._closed = True
                lifetime = (
                    datetime.now(timezone.utc) - self.created_at
                ).total_seconds()
                logger.debug(
                    f"Closed resource: {self.resource_id} ({self.resource_type}), "
                    f"lifetime: {lifetime:.2f}s"
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Error closing resource %s: %s", self.resource_id, e)
                raise

    def close_sync(self) -> None:
        """Close the resource synchronously."""
        if not self._closed:
            try:
                self._cleanup_sync()
                self._closed = True
                lifetime = (
                    datetime.now(timezone.utc) - self.created_at
                ).total_seconds()
                logger.debug(
                    f"Closed resource: {self.resource_id} ({self.resource_type}), "
                    f"lifetime: {lifetime:.2f}s"
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Error closing resource %s: %s", self.resource_id, e)
                raise

    async def _cleanup(self) -> None:
        """Override this method to implement async cleanup logic."""
        pass

    def _cleanup_sync(self) -> None:
        """Override this method to implement sync cleanup logic."""
        pass

class DatabaseConnectionResource(ManagedResource[Any]):
    """Managed database connection resource."""

    def __init__(self, connection: Any, resource_id: str = "db_connection"):
        super().__init__(resource_id, "database_connection")
        self.connection = connection

    async def _cleanup(self) -> None:
        """Close the database connection."""
        try:
            if hasattr(self.connection, "close"):
                if inspect.iscoroutinefunction(self.connection.close):
                    await self.connection.close()
                else:
                    close_result = await asyncio.to_thread(self.connection.close)
                    if inspect.isawaitable(close_result):
                        await close_result
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Error closing database connection: %s", e)
            raise

class FileResource(ManagedResource[Any]):
    """Managed file resource."""

    def __init__(self, file_handle: Any, resource_id: str = "file"):
        super().__init__(resource_id, "file")
        self.file_handle = file_handle

    async def _cleanup(self) -> None:
        """Close the file handle."""
        try:
            if hasattr(self.file_handle, "close"):
                if inspect.iscoroutinefunction(self.file_handle.close):
                    await self.file_handle.close()
                else:
                    close_result = await asyncio.to_thread(self.file_handle.close)
                    if inspect.isawaitable(close_result):
                        await close_result
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Error closing file handle: %s", e)
            raise

    def _cleanup_sync(self) -> None:
        """Close the file handle synchronously."""
        try:
            if hasattr(self.file_handle, "close"):
                self.file_handle.close()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Error closing file handle synchronously: %s", e)
            raise

class NetworkSocketResource(ManagedResource[Any]):
    """Managed network socket resource."""

    def __init__(self, socket: Any, resource_id: str = "socket"):
        super().__init__(resource_id, "network_socket")
        self.socket = socket

    async def _cleanup(self) -> None:
        """Close the network socket."""
        try:
            if hasattr(self.socket, "close"):
                if inspect.iscoroutinefunction(self.socket.close):
                    await self.socket.close()
                else:
                    close_result = await asyncio.to_thread(self.socket.close)
                    if inspect.isawaitable(close_result):
                        await close_result
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Error closing network socket: %s", e)
            raise

@asynccontextmanager
async def managed_database_connection(pool: Any) -> AsyncIterator[Any]:
    """
    Context manager for database connections with automatic cleanup.
    """
    try:
        async with pool.get_connection() as conn:
            yield conn
    finally:
        pass

@asynccontextmanager
async def managed_file(file_path: str, mode: str = "r") -> AsyncIterator[Any]:
    """
    Context manager for file operations with automatic cleanup.
    """
    try:
        import aiofiles  # type: ignore
    except ImportError:
        aiofiles = None  # type: ignore

    if aiofiles is None:
        raise RuntimeError(
            "aiofiles is required for async file operations but is not installed."
        )

    file_handle = None
    try:
        file_handle = await aiofiles.open(file_path, mode)
        yield file_handle
    finally:
        if file_handle:
            await file_handle.close()

@contextmanager
def managed_file_sync(file_path: str, mode: str = "r") -> Iterator[Any]:
    """
    Synchronous context manager for file operations with automatic cleanup.
    """
    file_handle = None
    try:
        file_handle = open(file_path, mode)
        yield file_handle
    finally:
        if file_handle:
            file_handle.close()

class ResourcePool[T]:
    """
    Generic resource pool with automatic cleanup and leak detection.
    """

    def __init__(self, max_resources: int = 100):
        self.max_resources = max_resources
        self.active_resources: dict[str, ResourceInfo] = {}
        self._lock = asyncio.Lock()
        self._resource_counter = 0

    async def acquire(
        self,
        resource_type: str,
        factory: Callable[[], T],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, T]:
        """
        Acquire a resource from the pool.
        """
        async with self._lock:
            if len(self.active_resources) >= self.max_resources:
                raise RuntimeError(
                    "Resource pool exhausted: "
                    f"{len(self.active_resources)}/{self.max_resources}"
                )

            self._resource_counter += 1
            resource_id = f"{resource_type}_{self._resource_counter}"

            if inspect.iscoroutinefunction(factory):
                resource = await factory()
            else:
                resource = await asyncio.to_thread(factory)
                if inspect.isawaitable(resource):
                    resource = await resource

            # Track the resource
            info = ResourceInfo(
                resource_id=resource_id,
                resource_type=resource_type,
                resource_instance=resource,
                metadata=metadata or {},
            )
            self.active_resources[resource_id] = info

            logger.debug("Acquired resource: %s (%s)", resource_id, resource_type)
            return resource_id, resource

    async def release(self, resource_id: str, resource: Any) -> None:
        """
        Release a resource back to the pool.
        """
        async with self._lock:
            await self._release_unsafe(resource_id, resource)

    async def _release_unsafe(self, resource_id: str, resource: Any) -> None:
        """Helper for unsafe release to avoid deadlocks."""
        if resource_id in self.active_resources:
            info = self.active_resources[resource_id]
            lifetime = info.get_lifetime_seconds()

            # Cleanup the resource
            try:
                if hasattr(resource, "close"):
                    if inspect.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        close_result = await asyncio.to_thread(resource.close)
                        if inspect.isawaitable(close_result):
                            await close_result
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Error closing resource %s: %s", resource_id, e)

            del self.active_resources[resource_id]
            logger.debug(
                f"Released resource: {resource_id} ({info.resource_type}), "
                f"lifetime: {lifetime:.2f}s"
            )

    async def cleanup_all(self) -> None:
        """Cleanup all active resources."""
        async with self._lock:
            resource_ids = list(self.active_resources.keys())
            for res_id in resource_ids:
                info = self.active_resources[res_id]
                logger.warning(
                    "Force cleaning up resource: %s (%s)",
                    res_id,
                    info.resource_type,
                )
                await self._release_unsafe(res_id, info.resource_instance)

    async def detect_leaks(
        self, max_lifetime_seconds: int = 3600
    ) -> list[ResourceInfo]:
        """
        Detect potential resource leaks.
        """
        async with self._lock:
            leaks = []
            for info in self.active_resources.values():
                lifetime = info.get_lifetime_seconds()
                if lifetime > max_lifetime_seconds:
                    leaks.append(info)
                    logger.warning(
                        "Potential resource leak: "
                        f"{info.resource_id} ({info.resource_type}), "
                        f"lifetime: {lifetime:.2f}s"
                    )
            return leaks

    def get_stats(self) -> dict[str, Any]:
        """Get resource pool statistics."""
        return {
            "active_resources": len(self.active_resources),
            "max_resources": self.max_resources,
            "utilization": len(self.active_resources) / self.max_resources * 100
            if self.max_resources > 0
            else 0,
            "resource_types": {
                res_type: sum(
                    1
                    for i in self.active_resources.values()
                    if i.resource_type == res_type
                )
                for res_type in set(
                    i.resource_type for i in self.active_resources.values()
                )
            },
        }

async def resource_scope[T](
    pool: ResourcePool,
    resource_type: str,
    factory: Callable[[], T],
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[T]:
    """
    Context manager for scoped resource management.
    """
    resource_id, resource = await pool.acquire(resource_type, factory, metadata)
    try:
        yield resource
    finally:
        await pool.release(resource_id, resource)

class BatchResourceManager:
    """
    Manager for batch resource operations with automatic cleanup.
    """

    def __init__(self):
        self.resources: list[tuple[str, Any, Callable | None]] = []

    def add_resource(
        self, resource_id: str, resource: Any, cleanup_fn: Callable | None = None
    ) -> None:
        """Add a resource to the batch."""
        self.resources.append((resource_id, resource, cleanup_fn))
        logger.debug("Added resource to batch: %s", resource_id)

    async def cleanup_all(self) -> None:
        """Cleanup all resources in the batch."""
        for resource_id, resource, cleanup_fn in reversed(self.resources):
            try:
                if cleanup_fn:
                    if inspect.iscoroutinefunction(cleanup_fn):
                        await cleanup_fn(resource)
                    else:
                        cleanup_result = await asyncio.to_thread(cleanup_fn, resource)
                        if inspect.isawaitable(cleanup_result):
                            await cleanup_result
                elif hasattr(resource, "close"):
                    if inspect.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        close_result = await asyncio.to_thread(resource.close)
                        if inspect.isawaitable(close_result):
                            await close_result

                logger.debug("Cleaned up batch resource: %s", resource_id)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Error cleaning up batch resource %s: %s", resource_id, e)

        self.resources.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with automatic cleanup."""
        await self.cleanup_all()
        return False

# Global resource pool instance
_global_resource_pool: ResourcePool | None = None

def get_global_resource_pool() -> ResourcePool:
    """Get the global resource pool instance."""
    global _global_resource_pool
    if _global_resource_pool is None:
        _global_resource_pool = ResourcePool(max_resources=1000)
    return _global_resource_pool
