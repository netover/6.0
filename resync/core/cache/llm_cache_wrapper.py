"""
Cached LLM Wrapper - Integration layer for semantic caching.

v5.3.16 - LLM caching integration with:
- Transparent caching of LLM responses
- Configurable cache behavior per endpoint
- Metrics and logging
- Graceful degradation when cache fails

Design philosophy (30 years taught me):
1. Cache is an optimization, not a dependency
2. If cache fails, system must still work
3. Make it easy to enable/disable per endpoint
4. Never break existing functionality

Usage:
    # Instead of calling LLM directly:
    response = await llm_service.generate(prompt)

    # Use the cached wrapper:
    response = await cached_llm_call(prompt, llm_service.generate)
"""

from resync.core.task_tracker import create_tracked_task
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from .semantic_cache import CacheResult, SemanticCache, get_semantic_cache

logger = logging.getLogger(__name__)

T = TypeVar("T")

@dataclass
class CachedResponse:
    """
    Response from cached LLM call.

    Attributes:
        content: The actual response text
        cached: Whether this came from cache
        cache_distance: Semantic distance (0 = exact match, only if cached)
        latency_ms: Total time to get response
        cache_lookup_ms: Time spent on cache lookup
        llm_call_ms: Time spent on LLM call (0 if cached)
        metadata: Additional info
    """

    content: str
    cached: bool = False
    cache_distance: float = 1.0
    latency_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    llm_call_ms: float = 0.0
    metadata: dict[str, Any] | None = None

# TTL classification patterns
_TTL_PATTERNS = {
    # Queries about current state - cache briefly
    "short": [
        "hoje",
        "agora",
        "atual",
        "último",
        "última",
        "recente",
        "today",
        "now",
        "current",
        "last",
        "recent",
        "running",
    ],
    # FAQ / documentation queries - cache longer
    "long": [
        "o que é",
        "como fazer",
        "como funciona",
        "explicar",
        "definição",
        "what is",
        "how to",
        "how do",
        "explain",
        "definition",
        "difference between",
    ],
    # Action commands - don't cache
    "no_cache": [
        "executar",
        "rodar",
        "parar",
        "cancelar",
        "restart",
        "rerun",
        "run",
        "stop",
        "cancel",
        "execute",
        "kill",
        "abort",
    ],
}

def classify_ttl(query: str) -> int | None:
    """
    Classify query to determine appropriate TTL.

    Returns:
        TTL in seconds, or None for no-cache queries
    """
    query_lower = query.lower()

    # Check no-cache patterns first
    for pattern in _TTL_PATTERNS["no_cache"]:
        if pattern in query_lower:
            return None  # Don't cache action commands

    # Check short-TTL patterns
    for pattern in _TTL_PATTERNS["short"]:
        if pattern in query_lower:
            return 3600  # 1 hour

    # Check long-TTL patterns
    for pattern in _TTL_PATTERNS["long"]:
        if pattern in query_lower:
            return 604800  # 7 days

    # Default: 24 hours
    return 86400

async def cached_llm_call(
    query: str,
    llm_func: Callable[..., Awaitable[str]],
    *args: Any,
    cache_enabled: bool = True,
    cache: SemanticCache | None = None,
    ttl: int | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CachedResponse:
    """
    Execute LLM call with semantic caching.

    Args:
        query: User's query (used as cache key)
        llm_func: Async function that calls the LLM
        *args: Positional arguments for llm_func
        cache_enabled: Whether to use cache
        cache: SemanticCache instance (uses singleton if None)
        ttl: TTL override (None = auto-classify)
        metadata: Additional metadata to store with cache entry
        **kwargs: Keyword arguments for llm_func

    Returns:
        CachedResponse with content and cache metadata
    """
    start_time = time.perf_counter()
    cache_lookup_ms = 0.0
    llm_call_ms = 0.0

    # Classify TTL if not provided
    effective_ttl = ttl
    if effective_ttl is None:
        effective_ttl = classify_ttl(query)

    # If TTL is None (no-cache pattern), skip cache
    if effective_ttl is None:
        cache_enabled = False
        logger.debug("Cache disabled for action query: '%s...'", query[:50])

    # Try cache first if enabled
    if cache_enabled:
        try:
            if cache is None:
                cache = await get_semantic_cache()

            cache_start = time.perf_counter()
            result: CacheResult = await cache.get(query)
            cache_lookup_ms = (time.perf_counter() - cache_start) * 1000

            if result.hit and result.response:
                total_ms = (time.perf_counter() - start_time) * 1000

                logger.info(
                    "Cache HIT: distance={result.distance:.4f}, "
                    "latency={total_ms:.1f}ms, query='{query[:50]}...'"
                )

                return CachedResponse(
                    content=result.response,
                    cached=True,
                    cache_distance=result.distance,
                    latency_ms=total_ms,
                    cache_lookup_ms=cache_lookup_ms,
                    llm_call_ms=0.0,
                    metadata={
                        "cache_hit": True,
                        "distance": result.distance,
                        "original_query": result.entry.query if result.entry else None,
                        "hit_count": result.entry.hit_count if result.entry else 0,
                    },
                )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.warning("Cache lookup failed, proceeding to LLM: %s", e)

    # Cache miss or cache disabled - call LLM
    try:
        llm_start = time.perf_counter()
        response = await llm_func(*args, **kwargs)
        llm_call_ms = (time.perf_counter() - llm_start) * 1000

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("LLM call failed: %s", e)
        raise

    # Store in cache for future (async, don't wait)
    if cache_enabled and effective_ttl:
        create_tracked_task(
            _store_in_cache(
                query=query,
                response=response,
                cache=cache,
                ttl=effective_ttl,
                metadata={
                    **(metadata or {}),
                    "llm_latency_ms": llm_call_ms,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    total_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Cache MISS: llm_latency={llm_call_ms:.1f}ms, "
        "total={total_ms:.1f}ms, query='{query[:50]}...'"
    )

    return CachedResponse(
        content=response,
        cached=False,
        cache_distance=1.0,
        latency_ms=total_ms,
        cache_lookup_ms=cache_lookup_ms,
        llm_call_ms=llm_call_ms,
        metadata={
            "cache_hit": False,
            "llm_latency_ms": llm_call_ms,
            "cache_enabled": cache_enabled,
        },
    )

async def _store_in_cache(
    query: str,
    response: str,
    cache: SemanticCache | None,
    ttl: int,
    metadata: dict[str, Any] | None,
) -> None:
    """Background task to store response in cache."""
    try:
        if cache is None:
            cache = await get_semantic_cache()
        await cache.set(query, response, ttl=ttl, metadata=metadata)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("Failed to store response in cache: %s", e)

def with_semantic_cache(
    query_param: str = "query",
    cache_enabled: bool = True,
    ttl: int | None = None,
):
    """
    Decorator to add semantic caching to an async function.

    The function must accept a query parameter (configurable name)
    and return a string response.

    Args:
        query_param: Name of the parameter containing the query text
        cache_enabled: Whether caching is enabled
        ttl: TTL override (None = auto-classify)

    Usage:
        @with_semantic_cache(query_param="message")
        async def chat_endpoint(message: str, context: dict) -> str:
            return await llm.generate(message, context)
    """

    def decorator(
        func: Callable[..., Awaitable[str]],
    ) -> Callable[..., Awaitable[CachedResponse]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> CachedResponse:
            # Extract query from kwargs or args
            query = kwargs.get(query_param)
            if query is None and args:
                # Try to find by position using function signature
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if query_param in params:
                    idx = params.index(query_param)
                    if idx < len(args):
                        query = args[idx]

            if query is None:
                raise ValueError(f"Could not find query parameter '{query_param}'")

            # Create a wrapper function that calls the original
            async def call_original() -> str:
                return await func(*args, **kwargs)

            return await cached_llm_call(
                query=query,
                llm_func=call_original,
                cache_enabled=cache_enabled,
                ttl=ttl,
            )

        return wrapper

    return decorator

class CachedLLMService:
    """
    Wrapper around existing LLM service to add semantic caching.

    Use this to wrap your existing LLM service without modifying it.

    Example:
        original_service = LLMService()
        cached_service = CachedLLMService(original_service)

        # Use cached_service instead of original_service
        response = await cached_service.generate(query)
    """

    def __init__(
        self,
        llm_service: Any,
        generate_method: str = "generate",
        query_extractor: Callable[[tuple, dict], str] | None = None,
        cache_enabled: bool = True,
        default_ttl: int | None = None,
    ):
        """
        Initialize cached LLM service wrapper.

        Args:
            llm_service: The original LLM service instance
            generate_method: Name of the method that generates responses
            query_extractor: Function to extract query from args/kwargs
            cache_enabled: Whether caching is enabled by default
            default_ttl: Default TTL (None = auto-classify)
        """
        self._service = llm_service
        self._generate_method = generate_method
        self._query_extractor = query_extractor or self._default_query_extractor
        self._cache_enabled = cache_enabled
        self._default_ttl = default_ttl
        self._cache: SemanticCache | None = None

    @staticmethod
    def _default_query_extractor(args: tuple, kwargs: dict) -> str:
        """Default: first positional arg or 'query'/'prompt'/'message' kwarg."""
        if args:
            return str(args[0])
        for key in ["query", "prompt", "message", "text", "input"]:
            if key in kwargs:
                return str(kwargs[key])
        raise ValueError("Could not extract query from arguments")

    async def generate(
        self,
        *args: Any,
        cache_enabled: bool | None = None,
        ttl: int | None = None,
        **kwargs: Any,
    ) -> CachedResponse:
        """
        Generate response with caching.

        Accepts same arguments as the wrapped service's generate method.
        """
        # Extract query
        query = self._query_extractor(args, kwargs)

        # Get original method
        original_method = getattr(self._service, self._generate_method)

        # Use provided or default cache settings
        effective_cache_enabled = (
            cache_enabled if cache_enabled is not None else self._cache_enabled
        )
        effective_ttl = ttl if ttl is not None else self._default_ttl

        return await cached_llm_call(
            query,
            original_method,
            *args,
            cache_enabled=effective_cache_enabled,
            cache=self._cache,
            ttl=effective_ttl,
            **kwargs,
        )

    def disable_cache(self) -> None:
        """Disable caching."""
        self._cache_enabled = False
        logger.info("LLM cache disabled")

    def enable_cache(self) -> None:
        """Enable caching."""
        self._cache_enabled = True
        logger.info("LLM cache enabled")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache is None:
            self._cache = await get_semantic_cache()
        return await self._cache.get_stats()

    async def clear_cache(self) -> bool:
        """Clear all cached responses."""
        if self._cache is None:
            self._cache = await get_semantic_cache()
        return await self._cache.clear()

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to wrapped service."""
        return getattr(self._service, name)

# Convenience function for one-off cached calls
async def query_with_cache(
    query: str,
    llm_func: Callable[[str], Awaitable[str]],
    cache_enabled: bool = True,
    ttl: int | None = None,
) -> str:
    """
    Simple wrapper for making a cached LLM query.

    This is the simplest way to add caching to an existing LLM call.

    Args:
        query: The user's query
        llm_func: Async function that takes query and returns response
        cache_enabled: Whether to use cache
        ttl: TTL in seconds

    Returns:
        The response string (from cache or LLM)

    Example:
        response = await query_with_cache(
            "How do I restart a TWS job?",
            lambda q: llm_service.generate(q),
        )
    """
    result = await cached_llm_call(
        query=query,
        llm_func=lambda: llm_func(query),
        cache_enabled=cache_enabled,
        ttl=ttl,
    )
    return result.content

# Backward compatibility alias
LLMCacheWrapper = CachedLLMService

__all__ = [
    "CachedResponse",
    "cached_llm_call",
    "with_semantic_cache",
    "CachedLLMService",
    "LLMCacheWrapper",
    "query_with_cache",
    "classify_ttl",
]
