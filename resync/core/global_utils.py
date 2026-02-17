"""Utility functions for global context and environment tags.

This module re-exports selected helpers from the :mod:`resync.core` package to
avoid import cycles. The functions provided here are commonly used by other
modules (such as :mod:`resync.core.agent_manager`) to access the global
correlation identifier and environment tagging without pulling in the entire
``resync.core`` package at import time. Keeping these helpers in a separate
module makes the dependency graph explicit and helps prevent circular import
issues.

The functions are simple proxies to the corresponding implementations in
``resync.core.__init__``. Refer to that module for implementation details.
"""

from __future__ import annotations

from resync.core import get_environment_tags, get_global_correlation_id

__all__ = [
    "get_environment_tags",
    "get_global_correlation_id",
]
