"""Shared types for settings module.

This module contains shared types used by multiple settings-related modules
to avoid circular imports.
"""

from enum import Enum
from pydantic import BaseModel, Field

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
    # pylint
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

    # pylint


class CacheConfig(BaseModel):
    """Unified cache configuration with TTL differentiation.

    v6.3.0: Consolidates cache settings into a single Pydantic model
    for better validation and documentation.
    """

    # TTL Configuration (Near Real-Time strategy)
    ttl_job_status: int = Field(
        default=10,
        ge=5,
        le=60,
        description="TTL in seconds for job status cache (near real-time)",
    )
    ttl_job_logs: int = Field(
        default=30,
        ge=10,
        le=120,
        description="TTL in seconds for job logs/stdlist cache",
    )
    ttl_static_structure: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="TTL in seconds for static structure (dependencies, definitions)",
    )
    ttl_graph: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="TTL in seconds for dependency graph cache",
    )

    # Cache Hierarchy (L1 + L2)
    hierarchy_l1_max_size: int = Field(
        default=5000,
        ge=1,
        description="Maximum number of entries in L1 cache (memory)",
    )
    hierarchy_l2_ttl: int = Field(
        default=600,
        gt=0,
        description="Time-To-Live for L2 cache entries in seconds (Valkey)",
    )
    hierarchy_l2_cleanup_interval: int = Field(
        default=60,
        gt=0,
        description="Cleanup interval for L2 cache in seconds",
    )
    hierarchy_num_shards: int = Field(
        default=8,
        description="Number of shards for cache (parallelism)",
    )
    hierarchy_max_workers: int = Field(
        default=4,
        ge=1,
        description="Max workers for cache operations",
    )

    # Stampede Protection & Mutex
    enable_swr: bool = Field(
        default=True,
        description="Enable Stale-While-Revalidate to prevent thundering herd",
    )
    ttl_jitter_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of TTL to use as jitter to prevent thundering herd",
    )
    enable_mutex: bool = Field(
        default=True,
        description="Enable cache mutex to prevent duplicate computations",
    )

    class Config:
        """Pydantic config."""
        frozen = True  # Immutable after construction
