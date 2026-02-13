"""
Legacy import path for TWS client factory.

DEPRECATED: Use `resync.core.factories.tws_factory` instead.

This module re-exports all symbols from the canonical location
for backward compatibility.

Migration:
    # Before (deprecated)
    from resync.services.tws_client_factory import get_tws_client
    
    # After (recommended)
    from resync.core.factories import get_tws_client
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'resync.services.tws_client_factory' is deprecated. "
    "Use 'resync.core.factories' instead. "
    "This module will be removed in v7.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location
from resync.core.factories.tws_factory import (  # noqa: E402
    get_tws_client,
    get_tws_client_singleton,
    get_tws_client_factory,
    reset_tws_client,
)

# Legacy alias
_build_client = get_tws_client_singleton

__all__ = [
    "get_tws_client",
    "get_tws_client_singleton",
    "get_tws_client_factory",
    "reset_tws_client",
    "_build_client",
]
