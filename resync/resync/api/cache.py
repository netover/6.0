"""
Legacy import path for cache routes.

DEPRECATED: Use `resync.api.routes.cache` instead.

This module re-exports all symbols from the canonical location
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.api.cache import router, cache_router
    
    # After (recommended)
    from resync.api.routes.cache import router, cache_router
"""
from __future__ import annotations

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.api.cache' is deprecated. "
    "Use 'resync.api.routes.cache' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from resync.api.routes.cache import *  # noqa: F401, F403, E402
from resync.api.routes.cache import router  # noqa: E402

# Explicit exports for IDE support
cache_router = router

__all__ = [
    "router",
    "cache_router",
]
