"""
Cache Module for Resync.

This package provides a hierarchical caching system with multiple layers:

Cache Hierarchy:
    BaseCache (ABC)
    ├── AsyncTTLCache (L1 - in-memory with TTL and sharding)
    ├── SemanticCache (L2 - embedding-based similarity)
    └── QueryCacheManager (L3 - query result caching)

    CacheHierarchy - Orchestrates L1 → L2 → L3
    CacheFactory - Creates configured instances

Primary Classes:
    - AsyncTTLCache: High-performance async cache with sharding
    - SemanticCache: Semantic similarity-based caching
    - CacheHierarchy: Multi-level cache orchestration
    - CacheFactory: Factory for creating cache instances

Usage:
    # Simple in-memory cache
    from resync.core.cache import AsyncTTLCache
    
    cache = AsyncTTLCache(ttl_seconds=300)
    await cache.start()
    await cache.set("key", "value")
    value = await cache.get("key")
    await cache.stop()
    
    # Or use factory
    from resync.core.cache import CacheFactory
    
    cache = CacheFactory.create_memory_cache()

Deprecated Aliases:
    - ImprovedAsyncCache → AsyncTTLCache
"""

from __future__ import annotations

# Core cache implementations
from resync.core.cache.async_cache import (
    AsyncTTLCache,
    CacheEntry,
    CacheStats,
    create_cache,
    ImprovedAsyncCache,  # Backward compatibility
)

from resync.core.cache.base_cache import BaseCache

# Hierarchical caching
from resync.core.cache.cache_hierarchy import CacheHierarchy, get_cache_hierarchy

# Factory
from resync.core.cache.cache_factory import CacheFactory

# Specialized caches
from resync.core.cache.semantic_cache import SemanticCache
from resync.core.cache.query_cache import QueryCacheManager

# Advanced features
from resync.core.cache.advanced_cache import (
    AdvancedApplicationCache,
    CacheInvalidator,
)

from resync.core.cache.cache_warmer import CacheWarmer

# LLM caching
from resync.core.cache.llm_cache_wrapper import LLMCacheWrapper

# Cache with protection
from resync.core.cache.cache_with_stampede_protection import (
    CacheWithStampedeProtection,
)

# Re-export for backward compatibility
from resync.core.cache.improved_cache import ImprovedAsyncCache


__all__ = [
    # Core
    "AsyncTTLCache",
    "BaseCache",
    "CacheEntry",
    "CacheStats",
    "create_cache",
    # Hierarchy
    "CacheHierarchy",
    "get_cache_hierarchy",
    "CacheFactory",
    # Specialized
    "SemanticCache",
    "QueryCacheManager",
    # Advanced
    "AdvancedApplicationCache",
    "CacheInvalidator",
    "CacheWarmer",
    "LLMCacheWrapper",
    "CacheWithStampedeProtection",
    # Backward compatibility
    "ImprovedAsyncCache",
]
