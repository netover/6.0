"""
API Routes for Threshold Auto-Tuning.

Provides REST endpoints for managing Active Learning threshold auto-tuning:

Mode Management:
    GET  /api/v1/admin/threshold-tuning/status         - Get full status
    GET  /api/v1/admin/threshold-tuning/mode           - Get current mode
    PUT  /api/v1/admin/threshold-tuning/mode           - Set mode

Threshold Management:
    GET  /api/v1/admin/threshold-tuning/thresholds     - Get all thresholds
    PUT  /api/v1/admin/threshold-tuning/thresholds/{name} - Set threshold value
    POST /api/v1/admin/threshold-tuning/reset          - Reset to defaults
    POST /api/v1/admin/threshold-tuning/rollback       - Rollback to last good

Metrics:
    GET  /api/v1/admin/threshold-tuning/metrics        - Get metrics summary
    GET  /api/v1/admin/threshold-tuning/metrics/daily  - Get daily metrics

Recommendations:
    GET  /api/v1/admin/threshold-tuning/recommendations           - Get pending
    POST /api/v1/admin/threshold-tuning/recommendations/generate  - Generate new
    POST /api/v1/admin/threshold-tuning/recommendations/{id}/approve - Approve
    POST /api/v1/admin/threshold-tuning/recommendations/{id}/reject  - Reject

Auto-Adjustment:
    POST /api/v1/admin/threshold-tuning/auto-adjust    - Trigger adjustment cycle
    POST /api/v1/admin/threshold-tuning/circuit-breaker/reset - Reset circuit breaker

Audit:
    GET  /api/v1/admin/threshold-tuning/audit-log      - Get audit history
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# v5.9.5: Added authentication
router = APIRouter(
    prefix="/threshold-tuning",
    tags=["Threshold Tuning"],
    dependencies=[Depends(verify_admin_credentials)],
)


# =============================================================================
# Pydantic Models
# =============================================================================


class ModeRequest(BaseModel):
    """Request to set auto-tuning mode."""

    mode: str = Field(
        ...,
        description="Mode: 'off', 'low', 'mid', or 'high'",
        pattern="^(off|low|mid|high)$",
    )
    admin_user: str = Field(default="admin", description="User making the change")


class ThresholdRequest(BaseModel):
    """Request to set a threshold value."""

    value: float = Field(..., description="New threshold value")
    admin_user: str = Field(default="admin", description="User making the change")
    reason: str = Field(
        default="Manual adjustment via API", description="Reason for change"
    )


class ApprovalRequest(BaseModel):
    """Request to approve/reject a recommendation."""

    admin_user: str = Field(default="admin", description="User performing action")
    reason: str = Field(default="", description="Optional reason")


class ResetRequest(BaseModel):
    """Request for reset operations."""

    admin_user: str = Field(default="admin", description="User performing action")


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_manager():
    """Get the ThresholdTuningManager instance."""
    try:
        from resync.core.continual_learning.threshold_tuning import (
            get_threshold_tuning_manager,
        )

        return await get_threshold_tuning_manager()
    except ImportError as e:
        logger.error("Failed to import threshold_tuning module: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Threshold tuning module not available",
        ) from e


# =============================================================================
# Status Endpoints
# =============================================================================


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """
    Get full threshold tuning status.

    Returns complete status including:
    - Current mode and parameters
    - All thresholds with bounds
    - Metrics summary
    - Circuit breaker state
    - Pending recommendations
    - Recent audit log
    """
    manager = await _get_manager()
    status_data = await manager.get_full_status()
    return {"status": "success", "data": status_data}


# =============================================================================
# Mode Endpoints
# =============================================================================


@router.get("/mode")
async def get_mode() -> dict[str, Any]:
    """Get current auto-tuning mode."""
    manager = await _get_manager()
    mode = await manager.get_mode()
    return {
        "status": "success",
        "mode": mode.value,
        "params": manager.MODE_PARAMS[mode],
    }


@router.put("/mode")
async def set_mode(request: ModeRequest) -> dict[str, Any]:
    """
    Set auto-tuning mode.

    Modes:
    - **off**: Static thresholds (default behavior)
    - **low**: Metrics collection + recommendations (Phase 1-2)
    - **mid**: Conservative auto-adjustment (±5%, 24h intervals)
    - **high**: Aggressive auto-adjustment (±10%, 12h intervals)
    """
    try:
        from resync.core.continual_learning.threshold_tuning import AutoTuningMode

        mode = AutoTuningMode(request.mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode: {request.mode}. Must be one of: off, low, mid, high",
        ) from None

    manager = await _get_manager()
    return await manager.set_mode(mode, request.admin_user)


# =============================================================================
# Threshold Endpoints
# =============================================================================


@router.get("/thresholds")
async def get_thresholds() -> dict[str, Any]:
    """Get all threshold configurations."""
    manager = await _get_manager()
    thresholds = await manager.get_thresholds()
    return {"status": "success", "thresholds": thresholds}


@router.get("/thresholds/{name}")
async def get_threshold(name: str) -> dict[str, Any]:
    """Get a specific threshold configuration."""
    manager = await _get_manager()
    threshold = await manager.get_threshold(name)

    if not threshold:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Threshold not found: {name}"
        )

    return {"status": "success", "threshold": threshold.to_dict()}


@router.put("/thresholds/{name}")
async def set_threshold(name: str, request: ThresholdRequest) -> dict[str, Any]:
    """
    Set a threshold value.

    The value will be clamped to the threshold's min/max bounds.
    """
    manager = await _get_manager()
    result = await manager.set_threshold(
        name,
        request.value,
        request.admin_user,
        request.reason,
    )

    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
        )

    return result


@router.post("/reset")
async def reset_to_defaults(request: ResetRequest) -> dict[str, Any]:
    """Reset all thresholds to default values."""
    manager = await _get_manager()
    return await manager.reset_to_defaults(request.admin_user)


@router.post("/rollback")
async def rollback_to_last_good(request: ResetRequest) -> dict[str, Any]:
    """Rollback to last known good thresholds."""
    manager = await _get_manager()
    result = await manager.rollback_to_last_good(request.admin_user)

    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
        )

    return result


# =============================================================================
# Metrics Endpoints
# =============================================================================


@router.get("/metrics")
async def get_metrics(
    days: int = Query(
        default=30, ge=1, le=365, description="Number of days to aggregate"
    ),
) -> dict[str, Any]:
    """
    Get metrics summary for the specified period.

    Returns aggregated metrics including:
    - Total evaluations
    - True/False Positives/Negatives
    - Precision, Recall, F1 Score
    - Review and correction rates
    """
    manager = await _get_manager()
    metrics = await manager.get_metrics_summary(days)
    return {
        "status": "success",
        "period_days": days,
        "metrics": metrics.to_dict(),
    }


@router.get("/metrics/daily")
async def get_daily_metrics(
    days: int = Query(
        default=30, ge=1, le=365, description="Number of days to retrieve"
    ),
) -> dict[str, Any]:
    """
    Get daily metrics for charting.

    Returns per-day metrics for visualization.
    """
    manager = await _get_manager()
    daily = await manager.get_daily_metrics(days)
    return {
        "status": "success",
        "period_days": days,
        "daily_metrics": daily,
    }


# =============================================================================
# Recommendation Endpoints
# =============================================================================


@router.get("/recommendations")
async def get_recommendations() -> dict[str, Any]:
    """Get all pending recommendations."""
    manager = await _get_manager()
    recs = await manager.get_pending_recommendations()
    return {"status": "success", "recommendations": recs}


@router.post("/recommendations/generate")
async def generate_recommendations() -> dict[str, Any]:
    """
    Generate new threshold recommendations.

    Analyzes current metrics and generates recommendations
    for threshold adjustments.
    """
    manager = await _get_manager()
    recs = await manager.generate_recommendations()
    return {
        "status": "success",
        "generated": len(recs),
        "recommendations": [r.to_dict() for r in recs],
    }


@router.post("/recommendations/{recommendation_id}/approve")
async def approve_recommendation(
    recommendation_id: int,
    request: ApprovalRequest,
) -> dict[str, Any]:
    """
    Approve and apply a pending recommendation.

    This will update the threshold to the recommended value.
    """
    manager = await _get_manager()
    result = await manager.approve_recommendation(recommendation_id, request.admin_user)

    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
        )

    return result


@router.post("/recommendations/{recommendation_id}/reject")
async def reject_recommendation(
    recommendation_id: int,
    request: ApprovalRequest,
) -> dict[str, Any]:
    """Reject a pending recommendation."""
    manager = await _get_manager()
    result = await manager.reject_recommendation(
        recommendation_id,
        request.admin_user,
        request.reason,
    )

    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
        )

    return result


# =============================================================================
# Auto-Adjustment Endpoints
# =============================================================================


@router.post("/auto-adjust")
async def run_auto_adjustment() -> dict[str, Any]:
    """
    Manually trigger an auto-adjustment cycle.

    Only works in MID or HIGH mode. Respects cooldowns and
    minimum data point requirements.
    """
    manager = await _get_manager()
    return await manager.run_auto_adjustment_cycle()


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(request: ResetRequest) -> dict[str, Any]:
    """
    Reset the circuit breaker.

    Establishes current metrics as the new baseline.
    Only available after cooldown period.
    """
    manager = await _get_manager()
    result = await manager.reset_circuit_breaker(request.admin_user)

    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
        )

    return result


@router.get("/circuit-breaker/status")
async def get_circuit_breaker_status() -> dict[str, Any]:
    """Get circuit breaker status."""
    manager = await _get_manager()
    status_data = await manager.get_full_status()

    return {
        "status": "success",
        "circuit_breaker": {
            "active": status_data["circuit_breaker_active"],
            "activated_at": status_data["circuit_breaker_activated_at"],
            "baseline_f1": status_data["baseline_f1"],
            "cooldown_hours": manager.CIRCUIT_BREAKER_COOLDOWN_HOURS,
        },
    }


# =============================================================================
# Audit Log Endpoints
# =============================================================================


@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(
        default=50, ge=1, le=500, description="Maximum entries to return"
    ),
    threshold: str | None = Query(default=None, description="Filter by threshold name"),
) -> dict[str, Any]:
    """
    Get audit log entries.

    Returns history of all threshold changes and mode changes.
    """
    manager = await _get_manager()
    logs = await manager.get_audit_log(limit, threshold)
    return {"status": "success", "entries": logs}
