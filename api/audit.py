"""
Legacy import path for audit routes.

DEPRECATED: Use `resync.api.routes.audit` instead.

This module re-exports all symbols from the canonical location
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.api.audit import router, audit_router
    
    # After (recommended)
    from resync.api.routes.audit import router, audit_router
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.api.audit' is deprecated. "
    "Use 'resync.api.routes.audit' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from resync.api.routes.audit import *  # noqa: F401, F403, E402
from resync.api.routes.audit import router  # noqa: E402

# Explicit exports for IDE support
audit_router = router

__all__ = [
    "router",
    "audit_router",
]
