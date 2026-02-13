"""
Admin routes for RAG Reranker Gating configuration.

v5.9.9 - Admin endpoints for:
- View gating configuration and statistics
- Update gating thresholds
- Toggle gating on/off
- Monitor rerank activation rate

Security: All endpoints require admin authentication.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.auth import verify_admin_credentials
from resync.knowledge.config import CFG
from resync.knowledge.retrieval.reranker_interface import (
    IReranker,
    RerankGatingConfig,
    RerankGatingPolicy,
    create_reranker,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/rag-reranker",
    tags=["admin", "rag", "reranker"],
    dependencies=[Depends(verify_admin_credentials)],
)


# =============================================================================
# Singleton instances for runtime configuration
# =============================================================================

_gating_policy: RerankGatingPolicy | None = None
_reranker: IReranker | None = None


def get_gating_policy() -> RerankGatingPolicy:
    """Get or create the gating policy singleton."""
    global _gating_policy
    if _gating_policy is None:
        _gating_policy = RerankGatingPolicy(config=RerankGatingConfig.from_env())
    return _gating_policy


def get_reranker_instance() -> IReranker:
    """Get or create the reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = create_reranker()
    return _reranker


# =============================================================================
# Request/Response Models
# =============================================================================


class GatingConfigResponse(BaseModel):
    """Response model for gating configuration."""

    enabled: bool = Field(description="Whether gating is enabled")
    score_low_threshold: float = Field(description="Activate rerank if top1 < threshold")
    margin_threshold: float = Field(description="Activate rerank if top1-top2 < margin")
    max_candidates: int = Field(description="Maximum candidates to rerank")
    entropy_check_enabled: bool = Field(description="Whether entropy check is enabled")
    entropy_threshold: float = Field(description="Entropy threshold for activation")


class GatingStatsResponse(BaseModel):
    """Response model for gating statistics."""

    total_decisions: int = Field(description="Total gating decisions made")
    rerank_activated: int = Field(description="Times rerank was activated")
    rerank_rate: float = Field(description="Rerank activation rate (0-1)")
    reasons: dict[str, int] = Field(description="Breakdown by activation reason")
    config: GatingConfigResponse = Field(description="Current configuration")


class RerankerInfoResponse(BaseModel):
    """Response model for reranker information."""

    type: str = Field(description="Reranker type (noop, cross_encoder)")
    enabled: bool = Field(description="Whether reranking is enabled")
    available: bool = Field(description="Whether reranker is available")
    loaded: bool = Field(description="Whether model is loaded in memory")
    model: str | None = Field(description="Model name if applicable")
    threshold: float | None = Field(description="Score threshold")
    call_count: int = Field(description="Number of rerank calls")
    avg_latency_ms: float | None = Field(description="Average latency in ms")


class UpdateGatingConfigRequest(BaseModel):
    """Request model for updating gating configuration."""

    enabled: bool | None = Field(
        default=None,
        description="Enable/disable gating"
    )
    score_low_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="New score low threshold (0-1)"
    )
    margin_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="New margin threshold (0-1)"
    )
    max_candidates: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Max candidates to rerank (1-100)"
    )


class UpdateGatingConfigResponse(BaseModel):
    """Response model for config update."""

    message: str
    old_config: GatingConfigResponse
    new_config: GatingConfigResponse


class FullStatusResponse(BaseModel):
    """Combined status response."""

    reranker: RerankerInfoResponse
    gating: GatingStatsResponse
    rag_config: dict[str, Any]


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/status",
    response_model=FullStatusResponse,
    summary="Get full reranker and gating status",
    description="Returns complete information about reranker and gating configuration.",
)
async def get_full_status() -> FullStatusResponse:
    """Get full reranker status including gating stats."""
    try:
        policy = get_gating_policy()
        reranker = get_reranker_instance()

        reranker_info = reranker.get_info()
        gating_stats = policy.get_stats()

        return FullStatusResponse(
            reranker=RerankerInfoResponse(
                type=reranker_info.get("type", "unknown"),
                enabled=reranker_info.get("enabled", False),
                available=reranker_info.get("available", False),
                loaded=reranker_info.get("loaded", False),
                model=reranker_info.get("model"),
                threshold=reranker_info.get("threshold"),
                call_count=reranker_info.get("call_count", 0),
                avg_latency_ms=reranker_info.get("avg_latency_ms"),
            ),
            gating=GatingStatsResponse(
                total_decisions=gating_stats["total_decisions"],
                rerank_activated=gating_stats["rerank_activated"],
                rerank_rate=gating_stats["rerank_rate"],
                reasons=gating_stats["reasons"],
                config=GatingConfigResponse(
                    enabled=gating_stats["config"]["enabled"],
                    score_low_threshold=gating_stats["config"]["score_low_threshold"],
                    margin_threshold=gating_stats["config"]["margin_threshold"],
                    max_candidates=gating_stats["config"]["max_candidates"],
                    entropy_check_enabled=policy.config.enable_entropy_check,
                    entropy_threshold=policy.config.entropy_threshold,
                ),
            ),
            rag_config={
                "cross_encoder_enabled": CFG.enable_cross_encoder,
                "cross_encoder_model": CFG.cross_encoder_model,
                "cross_encoder_top_k": CFG.cross_encoder_top_k,
                "cross_encoder_threshold": CFG.cross_encoder_threshold,
                "rerank_gating_enabled": CFG.rerank_gating_enabled,
            },
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get reranker status: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get status. Check server logs for details.",
        ) from None


@router.get(
    "/gating/config",
    response_model=GatingConfigResponse,
    summary="Get gating configuration",
    description="Returns current gating thresholds and settings.",
)
async def get_gating_config() -> GatingConfigResponse:
    """Get current gating configuration."""
    policy = get_gating_policy()

    return GatingConfigResponse(
        enabled=policy.config.enabled,
        score_low_threshold=policy.config.score_low_threshold,
        margin_threshold=policy.config.margin_threshold,
        max_candidates=policy.config.max_candidates,
        entropy_check_enabled=policy.config.enable_entropy_check,
        entropy_threshold=policy.config.entropy_threshold,
    )


@router.get(
    "/gating/stats",
    response_model=GatingStatsResponse,
    summary="Get gating statistics",
    description="Returns gating decision statistics for monitoring.",
)
async def get_gating_stats() -> GatingStatsResponse:
    """Get gating statistics."""
    policy = get_gating_policy()
    stats = policy.get_stats()

    return GatingStatsResponse(
        total_decisions=stats["total_decisions"],
        rerank_activated=stats["rerank_activated"],
        rerank_rate=stats["rerank_rate"],
        reasons=stats["reasons"],
        config=GatingConfigResponse(
            enabled=stats["config"]["enabled"],
            score_low_threshold=stats["config"]["score_low_threshold"],
            margin_threshold=stats["config"]["margin_threshold"],
            max_candidates=stats["config"]["max_candidates"],
            entropy_check_enabled=policy.config.enable_entropy_check,
            entropy_threshold=policy.config.entropy_threshold,
        ),
    )


@router.put(
    "/gating/config",
    response_model=UpdateGatingConfigResponse,
    summary="Update gating configuration",
    description="Update gating thresholds at runtime. Changes are applied immediately.",
)
async def update_gating_config(
    request: UpdateGatingConfigRequest,
) -> UpdateGatingConfigResponse:
    """Update gating configuration at runtime."""
    policy = get_gating_policy()

    # Capture old config
    old_config = GatingConfigResponse(
        enabled=policy.config.enabled,
        score_low_threshold=policy.config.score_low_threshold,
        margin_threshold=policy.config.margin_threshold,
        max_candidates=policy.config.max_candidates,
        entropy_check_enabled=policy.config.enable_entropy_check,
        entropy_threshold=policy.config.entropy_threshold,
    )

    # Apply updates
    if request.enabled is not None:
        policy.config.enabled = request.enabled
        logger.info("Gating enabled set to: %s", request.enabled)

    if request.score_low_threshold is not None:
        policy.config.score_low_threshold = request.score_low_threshold
        logger.info("Score low threshold set to: %s", request.score_low_threshold)

    if request.margin_threshold is not None:
        policy.config.margin_threshold = request.margin_threshold
        logger.info("Margin threshold set to: %s", request.margin_threshold)

    if request.max_candidates is not None:
        policy.config.max_candidates = request.max_candidates
        logger.info("Max candidates set to: %s", request.max_candidates)

    # Capture new config
    new_config = GatingConfigResponse(
        enabled=policy.config.enabled,
        score_low_threshold=policy.config.score_low_threshold,
        margin_threshold=policy.config.margin_threshold,
        max_candidates=policy.config.max_candidates,
        entropy_check_enabled=policy.config.enable_entropy_check,
        entropy_threshold=policy.config.entropy_threshold,
    )

    return UpdateGatingConfigResponse(
        message="Gating configuration updated successfully",
        old_config=old_config,
        new_config=new_config,
    )


@router.post(
    "/gating/reset-stats",
    summary="Reset gating statistics",
    description="Reset all gating statistics counters to zero.",
)
async def reset_gating_stats() -> dict[str, str]:
    """Reset gating statistics."""
    policy = get_gating_policy()
    policy.reset_stats()

    logger.info("Gating statistics reset")
    return {"message": "Gating statistics reset successfully"}


@router.get(
    "/reranker/info",
    response_model=RerankerInfoResponse,
    summary="Get reranker information",
    description="Returns information about the current reranker implementation.",
)
async def get_reranker_info_endpoint() -> RerankerInfoResponse:
    """Get reranker information."""
    reranker = get_reranker_instance()
    info = reranker.get_info()

    return RerankerInfoResponse(
        type=info.get("type", "unknown"),
        enabled=info.get("enabled", False),
        available=info.get("available", False),
        loaded=info.get("loaded", False),
        model=info.get("model"),
        threshold=info.get("threshold"),
        call_count=info.get("call_count", 0),
        avg_latency_ms=info.get("avg_latency_ms"),
    )


@router.post(
    "/reranker/preload",
    summary="Preload reranker model",
    description="Preload the cross-encoder model into memory to avoid cold start.",
)
async def preload_reranker_model() -> dict[str, Any]:
    """Preload reranker model."""
    reranker = get_reranker_instance()

    # Check if it's a CrossEncoderReranker with preload method
    if hasattr(reranker, "preload"):
        success = reranker.preload()
        if success:
            return {
                "message": "Reranker model preloaded successfully",
                "status": "loaded",
            }
        return {
            "message": "Failed to preload reranker model",
            "status": "failed",
        }

    return {
        "message": "Reranker does not support preloading (NoOp or unavailable)",
        "status": "skipped",
    }


# =============================================================================
# Export router
# =============================================================================

__all__ = ["router"]
