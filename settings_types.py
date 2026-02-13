"""Shared types for settings module.

This module contains shared types used by multiple settings-related modules
to avoid circular imports.
"""

from enum import Enum


class Environment(str, Enum):
    """Ambientes suportados."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class CacheHierarchyConfig:
    """Configuration object for cache hierarchy settings (snake_case internally).

    Backward-compatible UPPERCASE aliases are exposed as read-only properties.
    """

    def __init__(
        self,
        l1_max_size: int,
        l2_ttl_seconds: int,
        l2_cleanup_interval: int,
        num_shards: int = 8,
        max_workers: int = 4,
        key_prefix: str = "cache:",
    ) -> None:
        # canonical snake_case
        self.l1_max_size = l1_max_size
        self.l2_ttl_seconds = l2_ttl_seconds
        self.l2_cleanup_interval = l2_cleanup_interval
        self.num_shards = num_shards
        self.max_workers = max_workers
        self.cache_key_prefix = key_prefix

    # --- Legacy UPPERCASE aliases (read-only) ---
    # pylint: disable=invalid-name
    @property
    def L1_MAX_SIZE(self) -> int:
        """Legacy alias for l1_max_size."""
        return self.l1_max_size

    @property
    def L2_TTL_SECONDS(self) -> int:
        """Legacy alias for l2_ttl_seconds."""
        return self.l2_ttl_seconds

    @property
    def L2_CLEANUP_INTERVAL(self) -> int:
        """Legacy alias for l2_cleanup_interval."""
        return self.l2_cleanup_interval

    @property
    def NUM_SHARDS(self) -> int:
        """Legacy alias for num_shards."""
        return self.num_shards

    @property
    def MAX_WORKERS(self) -> int:
        """Legacy alias for max_workers."""
        return self.max_workers

    @property
    def CACHE_KEY_PREFIX(self) -> str:
        """Legacy alias for cache_key_prefix."""
        return self.cache_key_prefix

    # pylint: enable=invalid-name
