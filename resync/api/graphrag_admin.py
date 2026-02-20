"""
GraphRAG Admin Endpoints

Administrative endpoints for managing GraphRAG features:
- View statistics
- Invalidate cache
- Configure discovery

Author: Resync Team
Version: 5.9.9
"""

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin/graphrag", tags=["graphrag-admin"])


class CacheInvalidationRequest(BaseModel):
    """Request to invalidate discovery cache."""

    job_name: str | None = None  # None = invalidate all


class DiscoveryTriggerRequest(BaseModel):
    """Request to manually trigger discovery for a job."""

    job_name: str
    force: bool = False  # Bypass filters


@router.get("/stats")
async def get_graphrag_stats():
    """
    Get GraphRAG statistics.

    Returns discovery counts, budget usage, cache info.
    """
    from resync.core.graphrag_integration import get_graphrag_integration

    graphrag = get_graphrag_integration()

    if not graphrag:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")

    try:
        return await graphrag.get_stats()
    except Exception as e:
        logger.error("Failed to get GraphRAG stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from None


@router.post("/cache/invalidate")
async def invalidate_discovery_cache(request: CacheInvalidationRequest):
    """
    Invalidate discovery cache.

    Use this after TWS plan changes (dependencies modified, jobs added/removed).

    Args:
        job_name: Specific job to invalidate (None = all jobs)

    Returns:
        Number of cache entries invalidated

    Example:
        POST /api/admin/graphrag/cache/invalidate
        {"job_name": "PAYROLL_NIGHTLY"}

        Or invalidate all:
        {"job_name": null}
    """
    from resync.core.graphrag_integration import get_graphrag_integration

    graphrag = get_graphrag_integration()

    if not graphrag or not graphrag.discovery_service:
        raise HTTPException(status_code=503, detail="Discovery service not available")

    try:
        deleted = await graphrag.discovery_service.invalidate_discovery_cache(
            request.job_name
        )

        return {
            "status": "success",
            "cache_entries_deleted": deleted,
            "job_name": request.job_name or "all",
        }

    except Exception as e:
        logger.error("Failed to invalidate cache: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from None


@router.post("/discover")
async def trigger_manual_discovery(request: DiscoveryTriggerRequest):
    """
    Manually trigger discovery for a job.

    Useful for testing or forcing immediate discovery without waiting for failure.

    Args:
        job_name: Name of job to discover
        force: Bypass filters (budget, cache, etc)

    Example:
        POST /api/admin/graphrag/discover
        {"job_name": "NEW_CRITICAL_JOB", "force": true}
    """
    from resync.core.graphrag_integration import get_graphrag_integration

    graphrag = get_graphrag_integration()

    if not graphrag or not graphrag.discovery_service:
        raise HTTPException(status_code=503, detail="Discovery service not available")

    try:
        # Create fake event to trigger discovery
        event_details = {
            "return_code": 12,
            "severity": "CRITICAL",
            "manual_trigger": True,
            "force": request.force,
        }

        if request.force:
            # Bypass filters - directly call _discover_and_store
            await create_tracked_task(
                graphrag.discovery_service._discover_and_store(
                    request.job_name, event_details
                )
            )

            return {
                "status": "triggered",
                "job_name": request.job_name,
                "message": "Discovery started in background (forced)",
            }
        # Use normal flow with filters
        await graphrag.discovery_service.on_job_failed(request.job_name, event_details)

        return {
            "status": "submitted",
            "job_name": request.job_name,
            "message": "Discovery submitted (subject to filters)",
        }

    except Exception as e:
        logger.error("Failed to trigger discovery: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from None


@router.post("/validation/reset-stats")
async def reset_validation_stats():
    """
    Reset cache validation statistics.

    Useful for daily/weekly resets or after configuration changes.
    """
    from resync.core.graphrag_integration import get_graphrag_integration

    graphrag = get_graphrag_integration()

    if not graphrag:
        raise HTTPException(status_code=503, detail="GraphRAG not initialized")

    try:
        graphrag.reset_validation_stats()

        return {
            "status": "success",
            "message": "Cache validation statistics reset successfully",
        }

    except Exception as e:
        logger.error("Failed to reset stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from None


@router.get("/config")
async def get_discovery_config():
    """
    Get current discovery configuration.

    Returns thresholds, budgets, critical jobs list, and validation settings.
    """
    from resync.core.event_driven_discovery import DiscoveryConfig
    from resync.core.smart_cache_validator import CacheValidationConfig

    return {
        "budget": {
            "max_discoveries_per_day": DiscoveryConfig.MAX_DISCOVERIES_PER_DAY,
            "max_discoveries_per_hour": DiscoveryConfig.MAX_DISCOVERIES_PER_HOUR,
        },
        "cache": {"ttl_days": DiscoveryConfig.DISCOVERY_CACHE_DAYS},
        "triggers": {
            "discover_on_new_error": DiscoveryConfig.DISCOVER_ON_NEW_ERROR,
            "discover_on_recurring_failure": DiscoveryConfig.DISCOVER_ON_RECURRING_FAILURE,
            "min_failures_to_trigger": DiscoveryConfig.MIN_FAILURES_TO_TRIGGER,
        },
        "validation": {
            "validate_on_abend": CacheValidationConfig.VALIDATE_ON_ABEND,
            "validate_on_failed": CacheValidationConfig.VALIDATE_ON_FAILED,
            "validate_on_late": CacheValidationConfig.VALIDATE_ON_LATE,
            "validate_on_stuck": CacheValidationConfig.VALIDATE_ON_STUCK,
            "trust_cache_days": CacheValidationConfig.TRUST_CACHE_DAYS,
            "auto_invalidate": CacheValidationConfig.AUTO_INVALIDATE,
            "auto_rediscover": CacheValidationConfig.AUTO_REDISCOVER,
        },
        "critical_jobs": list(DiscoveryConfig.CRITICAL_JOBS),
    }


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    max_discoveries_per_day: int | None = None
    max_discoveries_per_hour: int | None = None
    cache_ttl_days: int | None = None
    min_failures_to_trigger: int | None = None
    validate_on_abend: bool | None = None
    validate_on_failed: bool | None = None
    auto_invalidate: bool | None = None


@router.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """
    Update GraphRAG configuration and persist to file.

    Changes are applied immediately AND saved to config/graphrag.toml.
    Configuration persists across restarts.

    Example:
        POST /api/admin/graphrag/config/update
        {
            "max_discoveries_per_day": 10,
            "cache_ttl_days": 60,
            "validate_on_abend": true
        }
    """
    from pathlib import Path

    from resync.core.config_persistence import ConfigPersistenceManager
    from resync.core.event_driven_discovery import DiscoveryConfig
    from resync.core.smart_cache_validator import CacheValidationConfig

    updated = []

    try:
        # Prepare data to save
        graphrag_config = {}

        # Update budget settings
        if request.max_discoveries_per_day is not None:
            DiscoveryConfig.MAX_DISCOVERIES_PER_DAY = request.max_discoveries_per_day
            if "budget" not in graphrag_config:
                graphrag_config["budget"] = {}
            graphrag_config["budget"]["max_discoveries_per_day"] = (
                request.max_discoveries_per_day
            )
            updated.append("max_discoveries_per_day")

        if request.max_discoveries_per_hour is not None:
            DiscoveryConfig.MAX_DISCOVERIES_PER_HOUR = request.max_discoveries_per_hour
            if "budget" not in graphrag_config:
                graphrag_config["budget"] = {}
            graphrag_config["budget"]["max_discoveries_per_hour"] = (
                request.max_discoveries_per_hour
            )
            updated.append("max_discoveries_per_hour")

        if request.cache_ttl_days is not None:
            DiscoveryConfig.DISCOVERY_CACHE_DAYS = request.cache_ttl_days
            if "cache" not in graphrag_config:
                graphrag_config["cache"] = {}
            graphrag_config["cache"]["ttl_days"] = request.cache_ttl_days
            updated.append("cache_ttl_days")

        if request.min_failures_to_trigger is not None:
            DiscoveryConfig.MIN_FAILURES_TO_TRIGGER = request.min_failures_to_trigger
            if "triggers" not in graphrag_config:
                graphrag_config["triggers"] = {}
            graphrag_config["triggers"]["min_failures_to_trigger"] = (
                request.min_failures_to_trigger
            )
            updated.append("min_failures_to_trigger")

        # Update validation settings
        if request.validate_on_abend is not None:
            CacheValidationConfig.VALIDATE_ON_ABEND = request.validate_on_abend
            if "validation" not in graphrag_config:
                graphrag_config["validation"] = {}
            graphrag_config["validation"]["validate_on_abend"] = (
                request.validate_on_abend
            )
            updated.append("validate_on_abend")

        if request.validate_on_failed is not None:
            CacheValidationConfig.VALIDATE_ON_FAILED = request.validate_on_failed
            if "validation" not in graphrag_config:
                graphrag_config["validation"] = {}
            graphrag_config["validation"]["validate_on_failed"] = (
                request.validate_on_failed
            )
            updated.append("validate_on_failed")

        if request.auto_invalidate is not None:
            CacheValidationConfig.AUTO_INVALIDATE = request.auto_invalidate
            if "validation" not in graphrag_config:
                graphrag_config["validation"] = {}
            graphrag_config["validation"]["auto_invalidate"] = request.auto_invalidate
            updated.append("auto_invalidate")

        # ✅ PERSIST TO FILE
        if graphrag_config:
            config_file = (
                Path(__file__).parent.parent.parent / "config" / "graphrag.toml"
            )

            persistence = ConfigPersistenceManager(
                config_file=config_file, backup_dir=config_file.parent / "backups"
            )

            persistence.save_config(
                section="graphrag", data=graphrag_config, create_backup=True
            )

        logger.info("GraphRAG config updated and persisted: %s", updated)

        return {
            "status": "success",
            "updated_fields": updated,
            "message": "Configuration updated and saved to file (persists across restarts)",
        }

    except Exception as e:
        logger.error("Failed to update config: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from None
