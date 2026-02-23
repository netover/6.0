"""
Performance Monitoring API Endpoints.

Provides REST API endpoints for monitoring and optimizing system performance.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from resync.core.performance_optimizer import get_performance_service
from resync.core.pools.pool_manager import get_connection_pool_manager
from resync.core.resource_manager import get_global_resource_pool

__all__ = [
    "logger",
    "performance_router",
    "router",
    "get_performance_report",
    "get_cache_metrics",
    "get_cache_recommendations",
    "get_pool_metrics",
    "get_pool_recommendations",
    "get_resource_stats",
    "detect_resource_leaks",
    "get_performance_health",
]


logger = logging.getLogger(__name__)

# Create router for performance endpoints
performance_router = APIRouter(prefix="/api/performance", tags=["performance"])
router = performance_router


@performance_router.get("/report")
async def get_performance_report() -> dict[str, Any]:
    """
    Get comprehensive system performance report.

    Returns:
        Dictionary containing:
        - Cache performance metrics and recommendations
        - Connection pool statistics and optimization suggestions
        - Resource usage and leak detection
        - Overall system health status
    """
    try:
        performance_service = get_performance_service()
        report = await performance_service.get_system_performance_report()

        return JSONResponse(status_code=status.HTTP_200_OK, content=report)
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error generating performance report: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate performance report. Check server logs for details.",
        ) from e


@performance_router.get("/cache/metrics")
async def get_cache_metrics() -> dict[str, Any]:
    """
    Get detailed cache performance metrics.

    Returns:
        Cache metrics including hit rates, eviction rates, and memory usage
    """
    try:
        performance_service = get_performance_service()

        cache_metrics = {}
        for cache_name, monitor in performance_service.cache_monitors.items():
            metrics = await monitor.get_current_metrics()
            cache_metrics[cache_name] = {
                "hit_rate": "{metrics.hit_rate:.2%}",
                "miss_rate": "{metrics.miss_rate:.2%}",
                "eviction_rate": "{metrics.eviction_rate:.2%}",
                "efficiency_score": "{metrics.calculate_efficiency_score():.1f}/100",
                "total_hits": metrics.total_hits,
                "total_misses": metrics.total_misses,
                "total_evictions": metrics.total_evictions,
                "current_size": metrics.current_size,
                "memory_usage_mb": "{metrics.memory_usage_mb:.2f}",
                "average_access_time_ms": "{metrics.average_access_time_ms:.2f}",
            }

        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"caches": cache_metrics}
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting cache metrics: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache metrics. Check server logs for details.",
        ) from e


@performance_router.get("/cache/recommendations")
async def get_cache_recommendations() -> dict[str, Any]:
    """
    Get optimization recommendations for all caches.

    Returns:
        Dictionary mapping cache names to lists of recommendations
    """
    try:
        performance_service = get_performance_service()

        recommendations = {}
        for cache_name, monitor in performance_service.cache_monitors.items():
            recommendations[
                cache_name
            ] = await monitor.get_optimization_recommendations()

        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"recommendations": recommendations}
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting cache recommendations: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache recommendations. Check server logs for details.",
        ) from e


@performance_router.get("/pools/metrics")
async def get_pool_metrics() -> dict[str, Any]:
    """
    Get detailed connection pool metrics.

    Returns:
        Connection pool statistics and performance metrics
    """
    try:
        pool_manager = await get_connection_pool_manager()
        report = await pool_manager.get_performance_report()

        return JSONResponse(status_code=status.HTTP_200_OK, content=report)
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting pool metrics: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get pool metrics. Check server logs for details.",
        ) from e


@performance_router.get("/pools/recommendations")
async def get_pool_recommendations() -> dict[str, Any]:
    """
    Get optimization recommendations for all connection pools.

    Returns:
        Dictionary mapping pool names to lists of recommendations
    """
    try:
        pool_manager = await get_connection_pool_manager()
        recommendations = await pool_manager.get_optimization_recommendations()

        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"recommendations": recommendations}
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting pool recommendations: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get pool recommendations. Check server logs for details.",
        ) from e


@performance_router.get("/resources/stats")
async def get_resource_stats() -> dict[str, Any]:
    """
    Get resource usage statistics.

    Returns:
        Resource pool statistics and active resource counts
    """
    try:
        resource_pool = get_global_resource_pool()
        stats = resource_pool.get_stats()

        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"resource_stats": stats}
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting resource stats: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get resource stats. Check server logs for details.",
        ) from e


@performance_router.get("/resources/leaks")
async def detect_resource_leaks(max_lifetime_seconds: int = 3600) -> dict[str, Any]:
    """
    Detect potential resource leaks.

    Args:
        max_lifetime_seconds: Maximum expected resource lifetime (default: 3600)

    Returns:
        List of potentially leaked resources
    """
    try:
        resource_pool = get_global_resource_pool()
        leaks = await resource_pool.detect_leaks(max_lifetime_seconds)

        leak_info = [
            {
                "resource_id": leak.resource_id,
                "resource_type": leak.resource_type,
                "lifetime_seconds": leak.get_lifetime_seconds(),
                "created_at": leak.created_at.isoformat(),
                "metadata": leak.metadata,
            }
            for leak in leaks
        ]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "leak_count": len(leaks),
                "leaks": leak_info,
                "max_lifetime_seconds": max_lifetime_seconds,
            },
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error detecting resource leaks: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect resource leaks. Check server logs for details.",
        ) from e


@performance_router.get("/health")
async def get_performance_health() -> dict[str, Any]:
    """
    Get overall performance health status.

    Returns:
        Health status with summary of all performance metrics
    """
    try:
        performance_service = get_performance_service()
        pool_manager = await get_connection_pool_manager()
        resource_pool = get_global_resource_pool()

        # Get cache health
        cache_health = "healthy"
        cache_issues = 0
        for _cache_name, monitor in performance_service.cache_monitors.items():
            metrics = await monitor.get_current_metrics()
            if metrics.hit_rate < 0.5 or metrics.calculate_efficiency_score() < 50:
                cache_health = "degraded"
                cache_issues += 1

        # Get pool health
        pool_health = "healthy"
        pool_issues = 0
        pool_stats = pool_manager.get_pool_stats()
        for _pool_name, stats in pool_stats.items():
            if (
                stats.get("connection_errors", 0) > 10
                or stats.get("pool_exhaustions", 0) > 0
            ):
                pool_health = "degraded"
                pool_issues += 1

        # Get resource health
        resource_stats = resource_pool.get_stats()
        resource_health = "healthy"
        if resource_stats["utilization"] > 90:
            resource_health = "degraded"

        # Determine overall health
        overall_health = "healthy"
        if (
            cache_health == "degraded"
            or pool_health == "degraded"
            or resource_health == "degraded"
        ):
            overall_health = "degraded"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "overall_health": overall_health,
                "cache_health": cache_health,
                "cache_issues": cache_issues,
                "pool_health": pool_health,
                "pool_issues": pool_issues,
                "resource_health": resource_health,
                "resource_utilization": "{resource_stats['utilization']:.1f}%",
            },
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting performance health: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance health. Check server logs for details.",
        ) from e
