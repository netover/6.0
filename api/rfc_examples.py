"""
Legacy import path for RFC examples routes.

DEPRECATED: Use `resync.api.routes.rfc_examples` instead.

This module re-exports all symbols from the canonical location
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.api.rfc_examples import router
    
    # After (recommended)
    from resync.api.routes.rfc_examples import router
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.api.rfc_examples' is deprecated. "
    "Use 'resync.api.routes.rfc_examples' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from resync.api.routes.rfc_examples import *  # noqa: F401, F403, E402
from resync.api.routes.rfc_examples import router  # noqa: E402

# Explicit exports for IDE support
rfc_examples_router = router

__all__ = [
    "router",
    "rfc_examples_router",
]
