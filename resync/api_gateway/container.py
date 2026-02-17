"""
Dependency injection container for Resync.

This module provides a dependency injection container to manage the lifecycle
and dependencies of services, reducing tight coupling between modules.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Protocol

from resync.api_gateway.services import (
    IAgentService,
    IKnowledgeService,
    ITWSService,
    ServiceFactory,
)
from resync.core.interfaces import (
    IAgentManager,
    IKnowledgeGraph,
    ITWSClient,
)


class IContainer(Protocol):
    """Protocol for the dependency injection container."""

    @abstractmethod
    def register(self, interface: type, implementation: Any) -> None:
        """Register an implementation for an interface."""

    @abstractmethod
    def resolve(self, interface: type) -> Any:
        """Resolve an implementation for an interface."""

    @abstractmethod
    def dispose(self) -> None:
        """Dispose of all managed resources."""


class Container:
    """Implementation of the dependency injection container."""

    def __init__(self) -> None:
        self._registrations: dict[type, Any] = {}
        self._instances: dict[type, Any] = {}
        self._logger = logging.getLogger(__name__)

    def register(self, interface: type, implementation: Any) -> None:
        """Register an implementation for an interface."""
        self._registrations[interface] = implementation
        # If implementation is a singleton, create it immediately
        if hasattr(implementation, "_singleton") and implementation._singleton:
            self._instances[interface] = implementation

    def register_singleton(self, interface: type, implementation: Any) -> None:
        """Register a singleton implementation for an interface."""
        implementation._singleton = True
        self.register(interface, implementation)

    def register_factory(self, interface: type, factory: Any) -> None:
        """Register a factory function for an interface."""
        self._registrations[interface] = factory

    def resolve(self, interface: type) -> Any:
        """Resolve an implementation for an interface."""
        if interface in self._instances:
            return self._instances[interface]

        if interface not in self._registrations:
            raise ValueError(f"No registration found for interface: {interface}")

        implementation = self._registrations[interface]

        # If it's a factory, call it to create the instance
        if callable(implementation) and not isinstance(implementation, type):
            instance = implementation(self)
            self._instances[interface] = instance
            return instance

        # If it's a class, instantiate it
        if isinstance(implementation, type):
            # Check if dependencies need to be injected
            instance = self._instantiate_with_dependencies(implementation)
            self._instances[interface] = instance
            return instance

        # Otherwise, return as is
        self._instances[interface] = implementation
        return implementation

    def _instantiate_with_dependencies(self, cls: type) -> Any:
        """Instantiate a class with its dependencies."""
        # Get the constructor signature
        import inspect

        sig = inspect.signature(cls.__init__)

        # Prepare dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Check if the parameter type is registered in our container
            if param.annotation != inspect.Parameter.empty:
                try:
                    dependency = self.resolve(param.annotation)
                    kwargs[param_name] = dependency
                except ValueError:
                    # If no registration found, use the default value if available
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise ValueError(
                            f"No registration found for dependency: {param.annotation}"
                        ) from None

        return cls(**kwargs)

    async def dispose(self) -> None:
        """Dispose of all managed resources."""
        for instance in self._instances.values():
            if hasattr(instance, "close") and callable(instance.close):
                try:
                    await instance.close()
                except Exception as e:
                    self._logger.error("Error disposing instance: %s", e)

        self._instances.clear()


# Global container instance
container = Container()


def setup_dependencies(
    tws_client: ITWSClient, agent_manager: IAgentManager, knowledge_graph: IKnowledgeGraph
) -> None:
    """Setup the dependency injection container with all necessary services."""
    # Register core components
    container.register_singleton(ITWSClient, tws_client)
    container.register_singleton(IAgentManager, agent_manager)
    container.register_singleton(IKnowledgeGraph, knowledge_graph)

    # Register services using the factory
    container.register_singleton(ITWSService, ServiceFactory.create_tws_service(tws_client))
    container.register_singleton(
        IAgentService,
        ServiceFactory.create_agent_service(agent_manager),
    )
    container.register_singleton(
        IKnowledgeService,
        ServiceFactory.create_knowledge_service(knowledge_graph),
    )


# Context manager for container lifecycle
@asynccontextmanager
async def container_lifespan() -> Any:
    """Context manager to handle container lifecycle."""
    try:
        yield container
    finally:
        await container.dispose()
