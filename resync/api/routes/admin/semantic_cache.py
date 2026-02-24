# ruff: noqa: E501
"""
Admin routes for Semantic Cache management.

v5.3.17 - Admin endpoints for:
- Cache statistics and metrics
- Threshold configuration
- Cache invalidation
- Health checks
- Cross-encoder reranking configuration (NEW)

Security: All endpoints require admin authentication.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from resync.api.auth import verify_admin_credentials
from resync.core.cache.embedding_model import get_model_info, preload_model
from resync.core.cache.redis_config import (
    RedisDatabase,
    check_redis_stack_available,
    redis_health_check,
)
from resync.core.cache.reranker import (
    get_reranker_info,
    preload_reranker,
    update_reranker_config,
)
from resync.core.cache.semantic_cache import get_semantic_cache

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/admin/semantic-cache",
    tags=["admin", "semantic-cache"],
    dependencies=[Depends(verify_admin_credentials)],
)


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    entries: int = Field(description="Number of cached entries")
    hits: int = Field(description="Total cache hits")
    misses: int = Field(description="Total cache misses")
    sets: int = Field(description="Total cache writes")
    errors: int = Field(description="Total errors")
    hit_rate_percent: float = Field(description="Cache hit rate percentage")
    avg_lookup_time_ms: float = Field(description="Average lookup time in ms")
    threshold: float = Field(description="Current similarity threshold")
    default_ttl: int = Field(description="Default TTL in seconds")
    max_entries: int = Field(description="Maximum entries limit")
    redis_stack_available: bool | None = Field(
        description="Whether Redis Stack is available"
    )
    used_memory_human: str = Field(description="Redis memory usage")


class ThresholdUpdateRequest(BaseModel):
    """Request model for updating cache threshold."""

    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="New similarity threshold (0-1, lower = stricter)",
        examples=[0.25, 0.3, 0.35],
    )


class ThresholdUpdateResponse(BaseModel):
    """Response model for threshold update."""

    old_threshold: float
    new_threshold: float
    message: str


class InvalidateRequest(BaseModel):
    """Request model for cache invalidation."""

    query: str | None = Field(default=None, description="Specific query to invalidate")
    pattern: str | None = Field(
        default=None, description="Pattern to match queries for invalidation"
    )
    all: bool = Field(
        default=False, description="Clear entire cache (use with caution!)"
    )


class InvalidateResponse(BaseModel):
    """Response model for cache invalidation."""

    invalidated_count: int
    message: str


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(description="Overall health status")
    redis_status: str
    redis_latency_ms: float | None = None
    redis_stack_available: bool
    embedding_model_status: str
    embedding_model_info: dict[str, Any]
    timestamp: datetime


class PreloadResponse(BaseModel):
    """Response model for model preload."""

    success: bool
    model_info: dict[str, Any]
    message: str


class RedisInfoResponse(BaseModel):
    """Response model for Redis server information."""

    host: str = Field(description="Redis host")
    port: int = Field(description="Redis port")
    database: int = Field(description="Redis database number")
    connected: bool = Field(description="Connection status")
    version: str | None = Field(description="Redis version")
    used_memory_human: str = Field(description="Used memory (human readable)")
    max_memory_human: str = Field(description="Max memory setting (human readable)")
    memory_usage_percent: float = Field(description="Memory usage percentage")
    maxmemory_policy: str = Field(description="Eviction policy")
    modules: dict[str, bool] = Field(description="Loaded modules status")
    databases: dict[str, dict[str, Any]] = Field(description="Database statistics")
    connected_clients: int = Field(description="Number of connected clients")
    ops_per_sec: int = Field(description="Operations per second")
    uptime_days: float = Field(description="Server uptime in days")


@router.get(
    "/stats",
    summary="Get cache statistics",
    description="Returns comprehensive statistics about semantic cache performance.",
)
async def get_cache_stats() -> CacheStatsResponse:
    """Get semantic cache statistics and metrics."""
    try:
        cache = await get_semantic_cache()
        stats = await cache.get_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get cache stats: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics. Check server logs for details.",
        ) from e


@router.get(
    "/health",
    summary="Health check",
    description="Check health of semantic cache components.",
)
async def health_check() -> HealthCheckResponse:
    """Perform comprehensive health check of semantic cache."""
    try:
        redis_health = await redis_health_check(RedisDatabase.SEMANTIC_CACHE)
        stack_info = await check_redis_stack_available()
        model_info = get_model_info()
        overall_status = "healthy"
        if redis_health["status"] != "healthy":
            overall_status = "degraded"
        if model_info["status"] == "fallback":
            overall_status = "degraded"
        if redis_health["status"] == "unhealthy":
            overall_status = "unhealthy"
        return HealthCheckResponse(
            status=overall_status,
            redis_status=redis_health["status"],
            redis_latency_ms=redis_health.get("latency_ms"),
            redis_stack_available=stack_info.get("search", False),
            embedding_model_status=model_info["status"],
            embedding_model_info=model_info,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Health check failed: %s", e)
        return HealthCheckResponse(
            status="unhealthy",
            redis_status="error",
            redis_latency_ms=None,
            redis_stack_available=False,
            embedding_model_status="error",
            embedding_model_info={"error": str(e)},
            timestamp=datetime.now(timezone.utc),
        )


@router.put(
    "/threshold",
    summary="Update similarity threshold",
    description="Update the similarity threshold for cache hits. Lower = stricter matching.",
)
async def update_threshold(request: ThresholdUpdateRequest) -> ThresholdUpdateResponse:
    """Update semantic cache similarity threshold."""
    try:
        cache = await get_semantic_cache()
        old_threshold = cache.threshold
        await cache.update_threshold(request.threshold)
        logger.info(
            "Cache threshold updated: %s -> %s", old_threshold, request.threshold
        )
        return ThresholdUpdateResponse(
            old_threshold=old_threshold,
            new_threshold=request.threshold,
            message=f"Threshold updated from {old_threshold} to {request.threshold}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Internal server error. Check server logs for details.",
        ) from e
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update threshold: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update threshold. Check server logs for details.",
        ) from e


@router.post(
    "/invalidate",
    summary="Invalidate cache entries",
    description="Invalidate specific cache entries or clear entire cache.",
)
async def invalidate_cache(request: InvalidateRequest) -> InvalidateResponse:
    """Invalidate semantic cache entries."""
    try:
        cache = await get_semantic_cache()
        if request.all:
            success = await cache.clear()
            if success:
                return InvalidateResponse(
                    invalidated_count=-1, message="Entire cache cleared successfully"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache",
            )
        if request.query:
            success = await cache.invalidate(request.query)
            return InvalidateResponse(
                invalidated_count=1 if success else 0,
                message=f"Query {('invalidated' if success else 'not found in cache')}",
            )
        if request.pattern:
            count = await cache.invalidate_pattern(request.pattern)
            return InvalidateResponse(
                invalidated_count=count,
                message=f"Invalidated {count} entries matching pattern '{request.pattern}'",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify query, pattern, or all=true",
        )
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to invalidate cache: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invalidate cache. Check server logs for details.",
        ) from e


@router.post(
    "/preload-model",
    summary="Preload embedding model",
    description="Preload the embedding model into memory to avoid cold-start latency.",
)
async def preload_embedding_model() -> PreloadResponse:
    """Preload embedding model into memory."""
    try:
        success = preload_model()
        model_info = get_model_info()
        if success:
            return PreloadResponse(
                success=True,
                model_info=model_info,
                message="Embedding model loaded successfully",
            )
        return PreloadResponse(
            success=False,
            model_info=model_info,
            message="Failed to load embedding model, using fallback",
        )
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to preload model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preload model. Check server logs for details.",
        ) from e


@router.get(
    "/test",
    summary="Test cache lookup",
    description="Test semantic cache with a query (does not call LLM).",
)
async def test_cache_lookup(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
) -> dict[str, Any]:
    """Test cache lookup without calling LLM."""
    try:
        cache = await get_semantic_cache()
        result = await cache.get(query)
        return {
            "query": query,
            "hit": result.hit,
            "response": result.response[:200] if result.response else None,
            "distance": result.distance,
            "lookup_time_ms": result.lookup_time_ms,
            "cached_query": result.entry.query if result.entry else None,
            "hit_count": result.entry.hit_count if result.entry else None,
        }
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Test lookup failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test failed. Check server logs for details.",
        ) from e


@router.post(
    "/test-store",
    summary="Test cache store",
    description="Store a test entry in cache (for testing purposes).",
)
async def test_cache_store(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    response: Annotated[str, Query(min_length=1, max_length=10000)],
    ttl: Annotated[int, Query(ge=60, le=604800)] = 3600,
) -> dict[str, Any]:
    """Store a test entry in cache."""
    try:
        cache = await get_semantic_cache()
        success = await cache.set(
            query=query,
            response=response,
            ttl=ttl,
            metadata={"test": True, "stored_via": "admin_endpoint"},
        )
        return {
            "success": success,
            "query": query,
            "response_length": len(response),
            "ttl": ttl,
            "message": "Entry stored successfully"
            if success
            else "Failed to store entry",
        }
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Test store failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test failed. Check server logs for details.",
        ) from e


@router.get(
    "/redis-info",
    summary="Get Redis server information",
    description="Returns detailed information about Redis server, modules, and databases.",
)
async def get_redis_info() -> RedisInfoResponse:
    """Get comprehensive Redis server information."""
    try:
        from resync.core.cache.redis_config import get_redis_client, get_redis_config

        config = get_redis_config()
        try:
            client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
            info = await client.info()
            connected = True
        except Exception as conn_err:
            logger.warning("Redis connection failed: %s", conn_err)
            connected = False
            info = {}
        used_memory = info.get("used_memory_human", "0B") if info else "N/A"
        max_memory = info.get("maxmemory_human", "0B") if info else "N/A"
        maxmemory = info.get("maxmemory", 0) if info else 0
        used_memory_bytes = info.get("used_memory", 0) if info else 0
        memory_percent = used_memory_bytes / maxmemory * 100 if maxmemory > 0 else 0.0
        modules_status = {
            "search": False,
            "json": False,
            "timeseries": False,
            "bloom": False,
        }
        if connected:
            try:
                stack_info = await check_redis_stack_available()
                modules_status = {
                    "search": stack_info.get("search", False),
                    "json": stack_info.get("ReJSON", False)
                    or stack_info.get("json", False),
                    "timeseries": stack_info.get("timeseries", False),
                    "bloom": stack_info.get("bf", False)
                    or stack_info.get("bloom", False),
                }
            except Exception as exc:
                logger.debug(
                    "suppressed_exception", exc_info=True, extra={"error": str(exc)}
                )
        databases = {}
        db_names = {
            0: "Connection Pools",
            1: "Sessions",
            2: "Cache Geral",
            3: "Semantic Cache",
            4: "Idempotency",
        }
        if connected:
            info.get("db0")
            for db_num in range(5):
                db_key = f"db{db_num}"
                db_info = info.get(db_key, {})
                if isinstance(db_info, dict):
                    keys = db_info.get("keys", 0)
                elif isinstance(db_info, str):
                    try:
                        parts = dict((p.split("=") for p in db_info.split(",")))
                        keys = int(parts.get("keys", 0))
                    except Exception:
                        keys = 0
                else:
                    keys = 0
                databases[db_key] = {
                    "name": db_names.get(db_num, f"DB {db_num}"),
                    "keys": keys,
                    "active": db_num == RedisDatabase.SEMANTIC_CACHE.value,
                }
        return RedisInfoResponse(
            host=config.host,
            port=config.port,
            database=RedisDatabase.SEMANTIC_CACHE.value,
            connected=connected,
            version=info.get("redis_version") if info else None,
            used_memory_human=used_memory,
            max_memory_human=max_memory if max_memory != "0B" else "Unlimited",
            memory_usage_percent=round(memory_percent, 1),
            maxmemory_policy=info.get("maxmemory_policy", "noeviction")
            if info
            else "unknown",
            modules=modules_status,
            databases=databases,
            connected_clients=info.get("connected_clients", 0) if info else 0,
            ops_per_sec=info.get("instantaneous_ops_per_sec", 0) if info else 0,
            uptime_days=round(info.get("uptime_in_seconds", 0) / 86400, 2)
            if info
            else 0,
        )
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get Redis info: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get Redis info. Check server logs for details.",
        ) from e


@router.post(
    "/redis-test",
    summary="Test Redis connection",
    description="Test Redis connection and return latency.",
)
async def test_redis_connection() -> dict[str, Any]:
    """Test Redis connection with ping."""
    try:
        import time

        from resync.core.cache.redis_config import get_redis_client

        start = time.perf_counter()
        client = get_redis_client(RedisDatabase.SEMANTIC_CACHE)
        pong = await client.ping()
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "success": pong,
            "latency_ms": round(latency_ms, 2),
            "message": "Redis connection successful" if pong else "Redis ping failed",
        }
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Redis connection test failed: %s", e)
        return {
            "success": False,
            "latency_ms": None,
            "message": f"Connection failed: {str(e)}",
            "error": str(e),
        }


class RerankerInfoResponse(BaseModel):
    """Response model for reranker information."""

    available: bool = Field(description="Whether reranker is available")
    enabled: bool = Field(description="Whether reranking is enabled for cache")
    model: str | None = Field(description="Reranker model name")
    loaded: bool = Field(description="Whether model is loaded in memory")
    threshold: float = Field(description="Cross-encoder similarity threshold")
    gray_zone_min: float = Field(description="Minimum distance for gray zone")
    gray_zone_max: float = Field(description="Maximum distance for gray zone")
    stats: dict[str, Any] = Field(description="Reranking statistics")


class RerankerConfigRequest(BaseModel):
    """Request model for updating reranker configuration."""

    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cross-encoder similarity threshold (0-1, higher = stricter)",
    )
    gray_zone_min: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum embedding distance for gray zone",
    )
    gray_zone_max: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum embedding distance for gray zone",
    )


@router.get(
    "/reranker",
    summary="Get reranker information",
    description="Returns information about the cross-encoder reranker configuration and stats.",
)
async def get_reranker_status() -> RerankerInfoResponse:
    """Get reranker status and configuration."""
    try:
        cache = await get_semantic_cache()
        info = get_reranker_info()
        stats = await cache.get_stats()
        return RerankerInfoResponse(
            available=info.get("available", False),
            enabled=cache.enable_reranking,
            model=info.get("model"),
            loaded=info.get("loaded", False),
            threshold=info.get("threshold", 0.5),
            gray_zone_min=info.get("gray_zone_min", 0.2),
            gray_zone_max=info.get("gray_zone_max", 0.35),
            stats={
                "reranks_total": stats.get("reranks_total", 0),
                "rerank_rejections": stats.get("rerank_rejections", 0),
                "rerank_rejection_rate_percent": stats.get(
                    "rerank_rejection_rate_percent", 0
                ),
            },
        )
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get reranker info: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reranker info. Check server logs for details.",
        ) from e


@router.put(
    "/reranker/enabled",
    summary="Toggle reranking on/off",
    description="Enable or disable cross-encoder reranking for gray zone queries.",
)
async def toggle_reranking(
    enabled: Annotated[bool, Query(description="Whether to enable reranking")],
) -> dict[str, Any]:
    """Enable or disable reranking."""
    try:
        cache = await get_semantic_cache()
        actual_state = cache.set_reranking_enabled(enabled)
        return {
            "requested": enabled,
            "actual": actual_state,
            "message": f"Reranking {('enabled' if actual_state else 'disabled')}"
            if actual_state == enabled
            else "Reranking could not be enabled (model not available)",
        }
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to toggle reranking: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle reranking. Check server logs for details.",
        ) from e


@router.put(
    "/reranker/config",
    summary="Update reranker configuration",
    description="Update cross-encoder threshold and gray zone boundaries.",
)
async def update_reranker_configuration(
    request: RerankerConfigRequest,
) -> dict[str, Any]:
    """Update reranker configuration."""
    try:
        new_config = update_reranker_config(
            threshold=request.threshold,
            gray_zone_min=request.gray_zone_min,
            gray_zone_max=request.gray_zone_max,
        )
        return {
            "success": True,
            "config": new_config,
            "message": "Reranker configuration updated",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Internal server error. Check server logs for details.",
        ) from e
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update reranker config: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update config. Check server logs for details.",
        ) from e


@router.post(
    "/reranker/preload",
    summary="Preload reranker model",
    description="Preload the cross-encoder model into memory to avoid cold-start latency.",
)
async def preload_reranker_model() -> dict[str, Any]:
    """Preload reranker model."""
    try:
        success = preload_reranker()
        info = get_reranker_info()
        return {
            "success": success,
            "model": info.get("model"),
            "loaded": info.get("loaded", False),
            "message": "Reranker model loaded successfully"
            if success
            else "Failed to load reranker model",
        }
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to preload reranker: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preload model. Check server logs for details.",
        ) from e


__all__ = ["router"]
