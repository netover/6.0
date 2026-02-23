# pylint: disable=all
# mypy: no-rerun
"""
Component Cache Manager

This module provides comprehensive caching functionality for health check
components, including cache management, expiry handling, and performance
optimization for health monitoring systems.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from resync.core.health.health_models import ComponentHealth

logger = structlog.get_logger(__name__)


class ComponentCacheManager:
    """
    Manages caching of component health results.

    This class provides functionality for:
    - Caching component health results with expiry
    - Thread-safe cache operations
    - Cache performance tracking (hits/misses)
    - Automatic cache cleanup and maintenance
    """

    def __init__(self, default_cache_expiry_seconds: int = 300):
        """
        Initialize the component cache manager.

        Args:
            default_cache_expiry_seconds: Default cache expiry time in seconds
        """
        self.default_cache_expiry = timedelta(seconds=default_cache_expiry_seconds)
        self.component_cache: dict[str, ComponentHealth] = {}
        self._cache_lock = asyncio.Lock()

        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

        # Cache maintenance
        self._last_cleanup: datetime | None = None
        self.cleanup_interval = timedelta(minutes=5)  # Cleanup every 5 minutes

    async def get_component(self, component_name: str) -> ComponentHealth | None:
        """
        Get a component from cache with expiry validation.

        Args:
            component_name: Name of the component to retrieve

        Returns:
            Component health if found and not expired, None otherwise
        """
        async with self._cache_lock:
            health = self.component_cache.get(component_name)
            if health:
                age = datetime.now(timezone.utc) - health.last_check
                if age < self.default_cache_expiry:
                    self._cache_hits += 1
                    logger.debug(
                        "cache_hit",
                        component=component_name,
                        age_seconds=age.total_seconds(),
                    )
                    return health
                # Cache expired, remove from cache
                self.component_cache.pop(component_name, None)
                self._cache_evictions += 1
                logger.debug(
                    "cache_expired",
                    component=component_name,
                    age_seconds=age.total_seconds(),
                )

            self._cache_misses += 1
            logger.debug("cache_miss", component=component_name)
            return None

    async def set_component(self, component_name: str, health: ComponentHealth) -> None:
        """
        Set a component in the cache.

        Args:
            component_name: Name of the component
            health: Component health result to cache
        """
        async with self._cache_lock:
            self.component_cache[component_name] = health
            logger.debug(
                "component_cached", component=component_name, status=health.status.value
            )

    async def update_component(
        self, component_name: str, health: ComponentHealth
    ) -> None:
        """
        Update a component in the cache if it exists.

        Args:
            component_name: Name of the component
            health: Updated component health result
        """
        async with self._cache_lock:
            if component_name in self.component_cache:
                self.component_cache[component_name] = health
                logger.debug(
                    "component_updated_in_cache",
                    component=component_name,
                    status=health.status.value,
                )

    async def remove_component(self, component_name: str) -> bool:
        """
        Remove a component from the cache.

        Args:
            component_name: Name of the component to remove

        Returns:
            True if component was removed, False if not found
        """
        async with self._cache_lock:
            if component_name in self.component_cache:
                self.component_cache.pop(component_name)
                self._cache_evictions += 1
                logger.debug("component_removed_from_cache", component=component_name)
                return True
            return False

    async def get_all_components(self) -> dict[str, ComponentHealth]:
        """
        Get all cached components with expiry validation.

        Returns:
            Dictionary of all valid cached components
        """
        async with self._cache_lock:
            expired_components = []
            valid_components = {}

            for name, health in self.component_cache.items():
                age = datetime.now(timezone.utc) - health.last_check
                if age < self.default_cache_expiry:
                    valid_components[name] = health
                else:
                    expired_components.append(name)
                    self._cache_evictions += 1

            # Remove expired components
            for name in expired_components:
                self.component_cache.pop(name, None)

            if expired_components:
                logger.debug(
                    "expired_components_cleaned", expired_count=len(expired_components)
                )

            return valid_components.copy()

    async def clear_cache(self) -> int:
        """
        Clear all components from the cache.

        Returns:
            Number of components that were cleared
        """
        async with self._cache_lock:
            cleared_count = len(self.component_cache)
            self.component_cache.clear()
            self._cache_evictions += cleared_count

            logger.info("cache_cleared", cleared_components=cleared_count)
            return cleared_count

    async def cleanup_expired(self) -> int:
        """
        Manually trigger cleanup of expired cache entries.

        Returns:
            Number of expired entries removed
        """
        async with self._cache_lock:
            expired_components = []
            current_time = datetime.now(timezone.utc)

            for name, health in self.component_cache.items():
                age = current_time - health.last_check
                if age >= self.default_cache_expiry:
                    expired_components.append(name)

            # Remove expired components
            for name in expired_components:
                self.component_cache.pop(name, None)

            self._cache_evictions += len(expired_components)

            if expired_components:
                logger.debug(
                    "manual_cache_cleanup", expired_count=len(expired_components)
                )

            return len(expired_components)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_operations = self._cache_hits + self._cache_misses

        return {
            "cached_components": len(self.component_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_evictions": self._cache_evictions,
            "total_operations": total_operations,
            "hit_rate": (
                (self._cache_hits / total_operations * 100)
                if total_operations > 0
                else 0.0
            ),
            "miss_rate": (
                (self._cache_misses / total_operations * 100)
                if total_operations > 0
                else 0.0
            ),
            "expiry_seconds": self.default_cache_expiry.total_seconds(),
            "last_cleanup": (
                self._last_cleanup.isoformat() if self._last_cleanup else None
            ),
        }

    async def get_components_by_status(self, status: str) -> dict[str, ComponentHealth]:
        """
        Get all cached components with a specific status.

        Args:
            status: Health status to filter by

        Returns:
            Dictionary of components with the specified status
        """
        all_components = await self.get_all_components()
        return {
            name: health
            for name, health in all_components.items()
            if health.status.value.lower() == status.lower()
        }

    async def get_stale_components(
        self, max_age_seconds: int | None = None
    ) -> dict[str, ComponentHealth]:
        """
        Get components that are stale based on age.

        Args:
            max_age_seconds: Maximum age in seconds (uses default expiry if None)

        Returns:
            Dictionary of stale components
        """
        if max_age_seconds is None:
            max_age = self.default_cache_expiry
        else:
            max_age = timedelta(seconds=max_age_seconds)

        async with self._cache_lock:
            stale_components = {}
            current_time = datetime.now(timezone.utc)

            for name, health in self.component_cache.items():
                age = current_time - health.last_check
                if age >= max_age:
                    stale_components[name] = health

            return stale_components.copy()

    def reset_stats(self) -> None:
        """Reset cache performance statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        logger.debug("cache_stats_reset")


class ComponentCacheConfig:
    """
    Configuration for component cache behavior.

    This class provides configuration options for:
    - Cache expiry times for different component types
    - Cache size limits and cleanup settings
    - Performance tuning parameters
    """

    def __init__(self):
        """Initialize cache configuration with default values."""
        # Default expiry times for different component types
        self.default_expiry_seconds = 300  # 5 minutes
        self.database_expiry_seconds = 60  # 1 minute (databases change frequently)
        self.redis_expiry_seconds = 120  # 2 minutes
        self.cache_expiry_seconds = 180  # 3 minutes
        self.filesystem_expiry_seconds = 600  # 10 minutes (less dynamic)
        self.memory_expiry_seconds = 30  # 30 seconds (very dynamic)
        self.cpu_expiry_seconds = 15  # 15 seconds (very dynamic)

        # Cache maintenance settings
        self.max_cache_size = 1000
        self.cleanup_interval_minutes = 5
        self.enable_performance_tracking = True

        # Advanced settings
        self.enable_cache_warnings = True
        self.cache_warning_threshold_percent = 80  # Warn when cache is 80% full

    def get_expiry_for_component(self, component_name: str) -> int:
        """
        Get expiry time for a specific component type.

        Args:
            component_name: Name of the component

        Returns:
            Expiry time in seconds
        """
        expiry_map = {
            "database": self.database_expiry_seconds,
            "redis": self.redis_expiry_seconds,
            "cache_hierarchy": self.cache_expiry_seconds,
            "file_system": self.filesystem_expiry_seconds,
            "memory": self.memory_expiry_seconds,
            "cpu": self.cpu_expiry_seconds,
        }

        return expiry_map.get(component_name, self.default_expiry_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "default_expiry_seconds": self.default_expiry_seconds,
            "database_expiry_seconds": self.database_expiry_seconds,
            "redis_expiry_seconds": self.redis_expiry_seconds,
            "cache_expiry_seconds": self.cache_expiry_seconds,
            "filesystem_expiry_seconds": self.filesystem_expiry_seconds,
            "memory_expiry_seconds": self.memory_expiry_seconds,
            "cpu_expiry_seconds": self.cpu_expiry_seconds,
            "max_cache_size": self.max_cache_size,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "enable_performance_tracking": self.enable_performance_tracking,
            "enable_cache_warnings": self.enable_cache_warnings,
            "cache_warning_threshold_percent": self.cache_warning_threshold_percent,
        }
