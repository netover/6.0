"""Improved async cache implementation.

This project historically referenced ``ImprovedAsyncCache`` from
``resync.core.cache.improved_cache``. In v5.9.x, the primary cache
implementation is ``AsyncTTLCache`` (``resync.core.cache.async_cache``).

To keep backward compatibility while preventing import errors, this module
re-exports ``AsyncTTLCache`` under the legacy name.
"""

from __future__ import annotations

from resync.core.cache.async_cache import AsyncTTLCache


class ImprovedAsyncCache(AsyncTTLCache):
    """Backward compatible alias for :class:`~resync.core.cache.async_cache.AsyncTTLCache`."""

