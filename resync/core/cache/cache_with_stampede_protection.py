import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheEntry:
    """Represents a cache entry with metadata."""

    def __init__(self, value: Any, expiry: float, is_loading: bool = False):
        self.value = value
        self.expiry = expiry
        self.is_loading = is_loading
        self.created_at = time.time()


class StampedeProtectionLevel(Enum):
    """Levels of stampede protection."""

    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"


@dataclass
class CacheConfig:
    """Configuration for cache with stampede protection."""

    default_ttl: int = 300  # 5 minutes
    stampede_protection_level: StampedeProtectionLevel = StampedeProtectionLevel.BASIC
    max_concurrent_loads: int = 3
    load_timeout: int = 30  # seconds


class CacheWithStampedeProtection(Generic[T]):
    """Cache implementation with stampede protection."""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._loading: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str, loader: Callable[[], T], ttl: int | None = None) -> T:
        """Get value from cache or load it with stampede protection."""

        current_time = time.time()
        ttl = ttl or self.config.default_ttl
        expiry = current_time + ttl

        # Check if we have a valid cached value
        if key in self._cache:
            entry = self._cache[key]
            if current_time < entry.expiry and not entry.is_loading:
                return entry.value

        # Implement stampede protection
        if self.config.stampede_protection_level != StampedeProtectionLevel.NONE:
            return await self._get_with_stampede_protection(key, loader, expiry)
        return await self._get_without_protection(key, loader, expiry)

    async def _get_with_stampede_protection(
        self, key: str, loader: Callable[[], T], expiry: float
    ) -> T:
        """Get with stampede protection."""

        async with self._lock:
            current_time = time.time()

            # Check again after acquiring lock
            if key in self._cache:
                entry = self._cache[key]
                if current_time < entry.expiry and not entry.is_loading:
                    return entry.value

            # Check if already loading
            if key in self._loading:
                # Wait for the loading to complete
                event = self._loading[key]
                await event.wait()

                # Return the cached value
                if key in self._cache:
                    entry = self._cache[key]
                    if current_time < entry.expiry:
                        return entry.value
                # If loading failed, try again
                return await self._load_value(key, loader, expiry)

            # Start loading
            event = asyncio.Event()
            self._loading[key] = event

        try:
            return await self._load_value(key, loader, expiry)
        finally:
            # Signal that loading is complete
            event.set()
            if key in self._loading:
                del self._loading[key]

    async def _get_without_protection(
        self, key: str, loader: Callable[[], T], expiry: float
    ) -> T:
        """Get without stampede protection."""

        return await self._load_value(key, loader, expiry)

    async def _load_value(self, key: str, loader: Callable[[], T], expiry: float) -> T:
        """Load value and cache it."""

        try:
            if inspect.iscoroutinefunction(loader):
                value = await loader()
            else:
                # Run sync function in thread pool
                loop = asyncio.get_running_loop()
                value = await loop.run_in_executor(None, loader)

            # Cache the value
            entry = CacheEntry(value, expiry, is_loading=False)
            self._cache[key] = entry

            return value

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to load cache key %s: %s", key, e)
            # Cache the exception for a short time to prevent repeated failures
            error_entry = CacheEntry(e, time.time() + 30, is_loading=False)
            self._cache[key] = error_entry
            raise

    def invalidate(self, key: str):
        """Invalidate a cache entry."""

        with self._lock:  # type: ignore[attr-defined]
            self._cache.pop(key, None)
            self._loading.pop(key, None)

    def clear(self):
        """Clear all cache entries."""

        with self._lock:
            self._cache.clear()
            self._loading.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""

        current_time = time.time()
        valid_entries = sum(
            1 for entry in self._cache.values() if current_time < entry.expiry
        )

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "loading_operations": len(self._loading),
            "stampede_protection_level": self.config.stampede_protection_level.value,
        }
