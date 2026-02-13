"""
Correlation and Error Handling Utilities.

This module provides reusable decorators and context managers for:
- Automatic correlation ID generation and management
- Unified error handling with logging
- Cache operation error handling

These utilities help reduce code duplication across the codebase.

Usage:
    # Decorator for correlation ID
    @with_correlation("cache_get")
    async def get(self, key: str, correlation_id: str = None):
        # correlation_id is auto-managed
        ...
    
    # Context manager for error handling
    async with cache_error_handler("get", correlation_id):
        result = await self._internal_get(key)
"""

from __future__ import annotations

import functools
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, ParamSpec

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def generate_correlation_id(prefix: str = "") -> str:
    """
    Generate a unique correlation ID.
    
    Args:
        prefix: Optional prefix for the correlation ID
        
    Returns:
        str: Unique correlation ID
    """
    base_id = uuid.uuid4().hex[:12]
    timestamp = int(time.time() * 1000) % 1000000
    
    if prefix:
        return f"{prefix}_{base_id}_{timestamp}"
    return f"{base_id}_{timestamp}"


def with_correlation(
    operation_name: str,
    *,
    log_entry: bool = True,
    log_exit: bool = True,
    include_timing: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator that automatically manages correlation IDs for async methods.
    
    If the decorated function receives a correlation_id parameter that is None,
    it will automatically generate one. The correlation ID is then logged
    at entry and exit of the function.
    
    Args:
        operation_name: Name of the operation for logging
        log_entry: Whether to log on function entry
        log_exit: Whether to log on function exit
        include_timing: Whether to include execution timing
        
    Returns:
        Decorated function with correlation ID management
        
    Example:
        class MyCache:
            @with_correlation("cache_get")
            async def get(self, key: str, *, correlation_id: str = None) -> Any:
                # correlation_id is auto-generated if None
                return await self._internal_get(key)
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Check if correlation_id is in kwargs
            correlation_id = kwargs.get("correlation_id")
            
            # Generate correlation ID if not provided
            if correlation_id is None:
                correlation_id = generate_correlation_id(operation_name)
                kwargs["correlation_id"] = correlation_id
            
            # Bind correlation ID to logger context
            log = logger.bind(
                correlation_id=correlation_id,
                operation=operation_name,
            )
            
            start_time = time.perf_counter() if include_timing else None
            
            if log_entry:
                log.debug("Starting %s", operation_name)
            
            try:
                result = await func(*args, **kwargs)
                
                if log_exit:
                    if include_timing and start_time:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        log.debug(
                            f"Completed {operation_name}",
                            elapsed_ms=round(elapsed_ms, 2),
                        )
                    else:
                        log.debug("Completed %s", operation_name)
                
                return result
                
            except Exception as e:
                if include_timing and start_time:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    log.error(
                        f"Error in {operation_name}",
                        error=str(e),
                        error_type=type(e).__name__,
                        elapsed_ms=round(elapsed_ms, 2),
                    )
                else:
                    log.error(
                        f"Error in {operation_name}",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                raise
        
        return wrapper
    return decorator


def with_correlation_sync(
    operation_name: str,
    *,
    log_entry: bool = True,
    log_exit: bool = True,
    include_timing: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Synchronous version of with_correlation decorator.
    
    Args:
        operation_name: Name of the operation for logging
        log_entry: Whether to log on function entry
        log_exit: Whether to log on function exit
        include_timing: Whether to include execution timing
        
    Returns:
        Decorated function with correlation ID management
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            correlation_id = kwargs.get("correlation_id")
            
            if correlation_id is None:
                correlation_id = generate_correlation_id(operation_name)
                kwargs["correlation_id"] = correlation_id
            
            log = logger.bind(
                correlation_id=correlation_id,
                operation=operation_name,
            )
            
            start_time = time.perf_counter() if include_timing else None
            
            if log_entry:
                log.debug("Starting %s", operation_name)
            
            try:
                result = func(*args, **kwargs)
                
                if log_exit:
                    if include_timing and start_time:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        log.debug(
                            f"Completed {operation_name}",
                            elapsed_ms=round(elapsed_ms, 2),
                        )
                    else:
                        log.debug("Completed %s", operation_name)
                
                return result
                
            except Exception as e:
                log.error(
                    f"Error in {operation_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        return wrapper
    return decorator


@asynccontextmanager
async def cache_error_handler(
    operation: str,
    correlation_id: Optional[str] = None,
    *,
    default_value: Any = None,
    reraise: bool = False,
    log_level: str = "error",
    record_health: bool = True,
    health_service: Optional[Any] = None,
):
    """
    Async context manager for unified cache error handling.
    
    Provides consistent error handling, logging, and optional health recording
    for cache operations.
    
    Args:
        operation: Name of the cache operation
        correlation_id: Optional correlation ID for tracing
        default_value: Value to yield on error (if not reraising)
        reraise: Whether to reraise exceptions
        log_level: Logging level for errors ('error', 'warning', 'debug')
        record_health: Whether to record health check results
        health_service: Optional health service for recording
        
    Yields:
        ErrorContext: Context object with result tracking
        
    Example:
        async with cache_error_handler("get", correlation_id) as ctx:
            result = await self._internal_get(key)
            ctx.set_result(result)
        return ctx.result  # Returns default_value on error
    """
    correlation_id = correlation_id or generate_correlation_id(f"cache_{operation}")
    
    log = logger.bind(
        correlation_id=correlation_id,
        operation=f"cache_{operation}",
    )
    
    # Context for tracking results
    class ErrorContext:
        def __init__(self):
            self.result = default_value
            self.success = False
            self.error: Optional[Exception] = None
            self.start_time = time.perf_counter()
        
        def set_result(self, value: Any) -> None:
            self.result = value
            self.success = True
        
        @property
        def elapsed_ms(self) -> float:
            return (time.perf_counter() - self.start_time) * 1000
    
    ctx = ErrorContext()
    
    try:
        yield ctx
        ctx.success = True
        
        # Record health on success
        if record_health and health_service is not None:
            try:
                await health_service.record_health_check(
                    component=f"cache_{operation}",
                    status="healthy",
                    latency_ms=ctx.elapsed_ms,
                    correlation_id=correlation_id,
                )
            except Exception as exc:
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass
                
    except asyncio.CancelledError:
        # Don't catch cancellation
        raise
        
    except Exception as e:
        ctx.error = e
        ctx.success = False
        
        # Log the error
        log_func = getattr(log, log_level, log.error)
        log_func(
            f"Cache {operation} failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(ctx.elapsed_ms, 2),
        )
        
        # Record health on failure
        if record_health and health_service is not None:
            try:
                await health_service.record_health_check(
                    component=f"cache_{operation}",
                    status="unhealthy",
                    latency_ms=ctx.elapsed_ms,
                    error=str(e),
                    correlation_id=correlation_id,
                )
            except Exception as exc:
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass
        
        if reraise:
            raise


class OperationContext:
    """
    Context class for tracking operation results and errors.
    
    Used with cache_error_handler to track results across the context.
    """
    
    def __init__(self, default_value: Any = None):
        self.result = default_value
        self.success = False
        self.error: Optional[Exception] = None
        self.start_time = time.perf_counter()
        self.metadata: dict[str, Any] = {}
    
    def set_result(self, value: Any) -> None:
        """Set the operation result."""
        self.result = value
        self.success = True
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000


# Convenience functions for common patterns

def ensure_correlation_id(
    correlation_id: Optional[str],
    prefix: str = "",
) -> str:
    """
    Ensure a correlation ID exists, generating one if needed.
    
    Args:
        correlation_id: Existing correlation ID or None
        prefix: Prefix for generated ID
        
    Returns:
        str: Existing or newly generated correlation ID
    """
    return correlation_id or generate_correlation_id(prefix)


__all__ = [
    "generate_correlation_id",
    "with_correlation",
    "with_correlation_sync",
    "cache_error_handler",
    "OperationContext",
    "ensure_correlation_id",
]
