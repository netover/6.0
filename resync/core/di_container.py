"""
Dependency Injection Container with Context-Safe Scoping.

This module provides a thread-safe and async-safe DI container using
contextvars to properly isolate scoped services per request/task.

v5.3.2 - Fixed ServiceScope leak between concurrent requests
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import Any, TypeVar

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# --- Type Variables ---
T = TypeVar("T")
TInterface = TypeVar("TInterface")
TImplementation = TypeVar("TImplementation")

# --- Context Variable for Request-Scoped Services ---
# This ensures each async task/request has its own isolated scope
_current_scope: ContextVar[ServiceScope | None] = ContextVar(
    "di_current_scope", default=None
)


class ServiceLifetime(Enum):
    """Defines the lifecycle scope of a registered service."""

    SINGLETON = "singleton"  # Single instance for entire app lifetime
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # One instance per request/scope


class ServiceScope:
    """
    Context manager for scoped service lifetimes.

    Each request gets its own ServiceScope via contextvars,
    ensuring complete isolation between concurrent requests.
    """

    def __init__(self):
        self._services: dict[type, Any] = {}
        self._token = None

    def __enter__(self) -> ServiceScope:
        """Enter scope context and set as current."""
        self._token = _current_scope.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit scope context and cleanup."""
        # Clear all scoped services
        self._services.clear()
        # Reset context var to previous value
        if self._token is not None:
            _current_scope.reset(self._token)
            self._token = None

    async def __aenter__(self) -> ServiceScope:
        """Async enter for use with 'async with'."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit for use with 'async with'."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def get_service(self, interface: type[T]) -> T | None:
        """Get service from current scope."""
        return self._services.get(interface)

    def set_service(self, interface: type[T], instance: T) -> None:
        """Set service in current scope."""
        self._services[interface] = instance


class DIContainer:
    """
    Thread-safe and async-safe DI container with proper lifecycle management.

    Features:
    - Singleton: One instance for entire application
    - Transient: New instance on every request
    - Scoped: One instance per request, isolated via contextvars

    Example:
        container = DIContainer()
        container.register(IUserService, UserService, ServiceLifetime.SCOPED)

        async with container.create_scope():
            user_svc = await container.get(IUserService)  # Same instance in this scope
    """

    def __init__(self):
        self._factories: dict[type, tuple[Callable, ServiceLifetime]] = {}
        self._singletons: dict[type, Any] = {}
        self._locks: dict[type, asyncio.Lock] = {}
        self._global_lock: asyncio.Lock | None = None

    @property
    def global_lock(self) -> asyncio.Lock:
        """Lazy initialization of global lock."""
        if self._global_lock is None:
            self._global_lock = asyncio.Lock()
        return self._global_lock

    def register(
        self,
        interface: type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> None:
        """
        Register a service factory with specified lifetime.

        Args:
            interface: The interface/base class type
            factory: Callable that creates the service instance
            lifetime: How long the instance should live
        """
        self._factories[interface] = (factory, lifetime)
        if lifetime == ServiceLifetime.SINGLETON:
            self._locks[interface] = asyncio.Lock()
        logger.debug(
            "service_registered", interface=interface.__name__, lifetime=lifetime.value
        )

    def register_instance(self, interface: type[T], instance: T) -> None:
        """
        Register a pre-created instance as singleton.

        Args:
            interface: The interface/base class type
            instance: The pre-created instance
        """
        self._singletons[interface] = instance
        self._factories[interface] = (lambda: instance, ServiceLifetime.SINGLETON)
        logger.debug("instance_registered", interface=interface.__name__)

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> None:
        """
        Register a factory function to create service instances.

        Alias for register() for compatibility.
        """
        self.register(interface, factory, lifetime)

    async def get(self, interface: type[T]) -> T:
        """
        Resolve a service by interface with proper lifecycle management.

        Args:
            interface: The interface/base class type to resolve

        Returns:
            The service instance

        Raises:
            ValueError: If service is not registered
            RuntimeError: If scoped service requested outside of scope
        """
        if interface not in self._factories:
            raise ValueError(f"Service {interface.__name__} not registered")

        factory, lifetime = self._factories[interface]

        if lifetime == ServiceLifetime.SINGLETON:
            return await self._get_singleton(interface, factory)
        if lifetime == ServiceLifetime.TRANSIENT:
            return await self._create_instance(factory)
        # SCOPED
        return await self._get_scoped(interface, factory)

    async def _get_singleton(self, interface: type[T], factory: Callable) -> T:
        """Get or create singleton instance with double-checked locking."""
        # Fast path: already exists
        if interface in self._singletons:
            return self._singletons[interface]

        # Slow path: acquire lock and create
        async with self._locks[interface]:
            # Double-check after acquiring lock
            if interface in self._singletons:
                return self._singletons[interface]

            instance = await self._create_instance(factory)
            self._singletons[interface] = instance
            logger.debug("singleton_created", interface=interface.__name__)
            return instance

    async def _get_scoped(self, interface: type[T], factory: Callable) -> T:
        """Get or create scoped instance from current context."""
        scope = _current_scope.get()

        if scope is None:
            # No scope active - this is a programming error.
            # We fail-fast to avoid non-deterministic behavior.
            raise RuntimeError(
                f"Scoped service {interface.__name__} requested outside an active scope. "
                "Use 'async with container.create_scope()' in non-HTTP execution contexts."
            )

        # Check if already in scope
        service = scope.get_service(interface)
        if service is not None:
            return service

        # Create and store in scope
        service = await self._create_instance(factory)
        scope.set_service(interface, service)
        logger.debug("scoped_created", interface=interface.__name__)
        return service

    async def _create_instance(self, factory: Callable) -> Any:
        """Create instance, handling both sync and async factories."""
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        return factory()

    def create_scope(self) -> ServiceScope:
        """
        Create a new scope for request-level service isolation.

        Usage:
            async with container.create_scope():
                # All scoped services resolved here share the same scope
                svc = await container.get(IScopedService)

        Returns:
            ServiceScope context manager
        """
        return ServiceScope()

    def get_current_scope(self) -> ServiceScope | None:
        """
        Get the current active scope, if any.

        Returns:
            Current ServiceScope or None if not in a scope
        """
        return _current_scope.get()

    def is_registered(self, interface: type) -> bool:
        """Check if a service is registered."""
        return interface in self._factories

    def clear(self) -> None:
        """Clear all registrations. Use with caution."""
        self._factories.clear()
        self._singletons.clear()
        self._locks.clear()
        logger.info("container_cleared")


# --- Global Container Instance ---

_container_singleton: "DIContainer | None" = None


class _LazyContainerProxy:
    """Proxy that lazily creates the DIContainer on first use.

    This avoids import-time asyncio primitives being bound to a non-worker
    event loop when running under gunicorn --preload.
    """

    def _get(self) -> "DIContainer":
        global _container_singleton  # pylint: disable=global-statement
        if _container_singleton is None:
            _container_singleton = DIContainer()
        return _container_singleton

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


# Backward-compatible global name used across the codebase.
container = _LazyContainerProxy()


def get_container() -> "DIContainer":
    """Return the concrete DIContainer instance."""
    return container._get()  # type: ignore[attr-defined]


def register_default_services() -> None:
    """Register default services with the container."""
    from resync.core.audit_queue import AsyncAuditQueue, IAuditQueue

    container.register(IAuditQueue, AsyncAuditQueue, lifetime=ServiceLifetime.SINGLETON)
