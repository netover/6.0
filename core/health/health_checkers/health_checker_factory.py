"""
Health Checker Factory

This module provides a factory for creating and managing health checker instances.
"""

from resync.core.health.health_models import ComponentType, HealthCheckConfig

from .base_health_checker import BaseHealthChecker
from .cache_health_checker import CacheHealthChecker
from .connection_pools_health_checker import ConnectionPoolsHealthChecker
from .cpu_health_checker import CpuHealthChecker
from .database_health_checker import DatabaseHealthChecker
from .filesystem_health_checker import FileSystemHealthChecker
from .memory_health_checker import MemoryHealthChecker
from .redis_health_checker import RedisHealthChecker
from .tws_monitor_health_checker import TWSMonitorHealthChecker
from .websocket_pool_health_checker import WebSocketPoolHealthChecker


class HealthCheckerFactory:
    """
    Factory for creating and managing health checker instances.

    Provides dependency injection and centralized management of all health checkers.
    """

    def __init__(self, config: HealthCheckConfig | None = None):
        """
        Initialize the health checker factory.

        Args:
            config: Health check configuration
        """
        self.config = config
        self._checkers: dict[str, BaseHealthChecker] = {}
        self._checker_classes: dict[str, type[BaseHealthChecker]] = {
            "database": DatabaseHealthChecker,
            "redis": RedisHealthChecker,
            "cache_hierarchy": CacheHealthChecker,
            "file_system": FileSystemHealthChecker,
            "memory": MemoryHealthChecker,
            "cpu": CpuHealthChecker,
            "tws_monitor": TWSMonitorHealthChecker,
            "connection_pools": ConnectionPoolsHealthChecker,
            "websocket_pool": WebSocketPoolHealthChecker,
        }
        self._initialize_checkers()

    def _initialize_checkers(self) -> None:
        """Initialize all health checker instances."""
        for name, checker_class in self._checker_classes.items():
            self._checkers[name] = checker_class(self.config)

    def get_health_checker(self, component_name: str) -> BaseHealthChecker | None:
        """
        Get a health checker instance by component name.

        Args:
            component_name: Name of the component

        Returns:
            Health checker instance or None if not found
        """
        return self._checkers.get(component_name)

    def get_all_health_checkers(self) -> dict[str, BaseHealthChecker]:
        """
        Get all health checker instances.

        Returns:
            Dictionary mapping component names to health checker instances
        """
        return self._checkers.copy()

    def get_enabled_health_checkers(self) -> dict[str, BaseHealthChecker]:
        """
        Get all enabled health checker instances.

        Returns:
            Dictionary mapping component names to enabled health checker instances
        """
        enabled = {}
        for name, checker in self._checkers.items():
            if self._is_component_enabled(name):
                enabled[name] = checker
        return enabled

    def _is_component_enabled(self, component_name: str) -> bool:
        """
        Check if a component is enabled for health checking.

        Args:
            component_name: Name of the component

        Returns:
            True if component is enabled
        """
        # All core components are enabled by default
        enabled_components = {
            "database",
            "redis",
            "cache_hierarchy",
            "file_system",
            "memory",
            "cpu",
            "tws_monitor",
            "connection_pools",
            "websocket_pool",
        }
        return component_name in enabled_components

    def get_health_checker_names(self) -> list[str]:
        """
        Get list of all available health checker names.

        Returns:
            List of component names
        """
        return list(self._checkers.keys())

    def get_enabled_health_checker_names(self) -> list[str]:
        """
        Get list of enabled health checker names.

        Returns:
            List of enabled component names
        """
        return [name for name in self._checkers if self._is_component_enabled(name)]

    def register_health_checker(
        self, component_name: str, checker_class: type[BaseHealthChecker]
    ) -> None:
        """
        Register a new health checker class.

        Args:
            component_name: Name of the component
            checker_class: Health checker class
        """
        self._checker_classes[component_name] = checker_class
        self._checkers[component_name] = checker_class(self.config)

    def unregister_health_checker(self, component_name: str) -> None:
        """
        Unregister a health checker.

        Args:
            component_name: Name of the component to unregister
        """
        self._checkers.pop(component_name, None)
        self._checker_classes.pop(component_name, None)

    def validate_all_checkers(self) -> dict[str, list[str]]:
        """
        Validate configuration for all health checkers.

        Returns:
            Dictionary mapping component names to validation error lists
        """
        validation_results = {}
        for name, checker in self._checkers.items():
            validation_results[name] = checker.validate_config()
        return validation_results

    def get_component_type_mapping(self) -> dict[str, ComponentType]:
        """
        Get mapping of component names to component types.

        Returns:
            Dictionary mapping component names to their types
        """
        mapping = {}
        for name, checker in self._checkers.items():
            mapping[name] = checker.component_type
        return mapping
