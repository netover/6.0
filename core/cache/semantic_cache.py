"""Compatibility shim for legacy core.cache.semantic_cache imports."""

from resync.core.cache.semantic_cache import (
    CacheEntry,
    CacheResult,
    SemanticCache,
    get_semantic_cache,
)

__all__ = [
    "CacheEntry",
    "CacheResult",
    "SemanticCache",
    "get_semantic_cache",
]
