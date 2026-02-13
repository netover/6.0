"""
Legacy import path for endpoints routes.

DEPRECATED: Use `resync.api.routes.endpoints` instead.

This module re-exports all symbols from the canonical location
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.api.endpoints import router
    
    # After (recommended)
    from resync.api.routes.endpoints import router
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.api.endpoints' is deprecated. "
    "Use 'resync.api.routes.endpoints' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from resync.api.routes.endpoints import *  # noqa: F401, F403, E402
from resync.api.routes.endpoints import router  # noqa: E402

# Explicit exports for IDE support
endpoints_router = router

__all__ = [
    "router",
    "endpoints_router",
]
