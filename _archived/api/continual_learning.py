"""
Continual Learning API - Endpoints for managing the continual learning system.

Provides REST endpoints for:
- Recording feedback on responses
- Managing the human review queue
- Viewing system statistics
- Configuring continual learning settings
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/continual-learning", tags=["Continual Learning"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class FeedbackRequest(BaseModel):
    """Request model for recording feedback."""

    query: str = Field(..., description="The original query")
    doc_id: str = Field(..., description="Document ID that was retrieved")
    rating: int = Field(..., ge=-2, le=2, description="Rating from -2 (very bad) to +2 (very good)")
    user_id: str | None = Field(None, description="User identifier")
    response_text: str | None = Field(None, description="Generated response text")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Response model for feedback recording."""

    feedback_id: str
    message: str


class ReviewSubmission(BaseModel):
    """Request model for submitting a human review."""

    status: str = Field(..., description="New status: approved, corrected, rejected")
    reviewer_id: str = Field(..., description="ID of the reviewer")
    correction: str | None = Field(None, description="Corrected response (if status is corrected)")
    feedback: str | None = Field(None, description="Additional feedback from reviewer")


class ReviewItemResponse(BaseModel):
    """Response model for a review item."""

    id: str
    query: str
    response: str
    reasons: list[str]
    confidence_scores: dict[str, float]
    status: str
    created_at: str
    reviewed_at: str | None
    reviewed_by: str | None


class StatsResponse(BaseModel):
    """Response model for system statistics."""

    feedback: dict[str, Any]
    active_learning: dict[str, Any] | None
    config: dict[str, bool]


class EnrichmentRequest(BaseModel):
    """Request model for query enrichment."""

    query: str = Field(..., description="Query to enrich")
    instance_id: str | None = Field(None, description="TWS instance ID")


class EnrichmentResponse(BaseModel):
    """Response model for query enrichment."""

    original_query: str
    enriched_query: str
    context_added: list[str]
    entities_found: dict[str, list[str]]
    enrichment_source: str


# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """
    Record user feedback for a query-document pair.

    This feedback is used to improve RAG retrieval over time by:
    - Boosting documents with positive feedback
    - Penalizing documents with negative feedback
    """
    try:
        from resync.core.continual_learning import get_feedback_store

        store = get_feedback_store()
        feedback_id = await store.record_feedback(
            query=request.query,
            doc_id=request.doc_id,
            rating=request.rating,
            user_id=request.user_id,
            response_text=request.response_text,
            metadata=request.metadata,
        )

        return FeedbackResponse(feedback_id=feedback_id, message="Feedback recorded successfully")
    except Exception as e:
        logger.error("feedback_recording_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Get statistics about recorded feedback."""
    try:
        from resync.core.continual_learning import get_feedback_store

        store = get_feedback_store()
        return await store.get_feedback_stats()

    except Exception as e:
        logger.error("feedback_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/feedback/low-quality-documents")
async def get_low_quality_documents(
    threshold: float = Query(-0.5, description="Rating threshold"),
    min_feedback: int = Query(3, description="Minimum feedback count"),
):
    """Get documents with consistently negative feedback."""
    try:
        from resync.core.continual_learning import get_feedback_store

        store = get_feedback_store()
        docs = await store.get_low_quality_documents(
            threshold=threshold,
            min_feedback=min_feedback,
        )

        return [
            {
                "doc_id": d.doc_id,
                "total_feedback": d.total_feedback,
                "positive_count": d.positive_count,
                "negative_count": d.negative_count,
                "avg_rating": d.avg_rating,
                "feedback_weight": d.feedback_weight,
            }
            for d in docs
        ]
    except Exception as e:
        logger.error("low_quality_docs_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


# =============================================================================
# REVIEW QUEUE ENDPOINTS
# =============================================================================


@router.get("/review/pending", response_model=list[ReviewItemResponse])
async def get_pending_reviews(
    limit: int = Query(50, ge=1, le=200, description="Maximum items to return"),
    reason: str | None = Query(None, description="Filter by reason"),
):
    """Get pending items from the human review queue."""
    try:
        from resync.core.continual_learning import ReviewReason, get_active_learning_manager

        manager = get_active_learning_manager()

        reason_filter = None
        if reason:
            try:
                reason_filter = ReviewReason(reason)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid reason. Valid values: {[r.value for r in ReviewReason]}",
                ) from None

        items = await manager.get_pending_reviews(
            limit=limit,
            reason_filter=reason_filter,
        )

        return [
            ReviewItemResponse(
                id=item.id,
                query=item.query,
                response=item.response,
                reasons=[r.value for r in item.reasons],
                confidence_scores=item.confidence_scores,
                status=item.status.value,
                created_at=item.created_at.isoformat(),
                reviewed_at=item.reviewed_at.isoformat() if item.reviewed_at else None,
                reviewed_by=item.reviewed_by,
            )
            for item in items
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pending_reviews_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.post("/review/{review_id}")
async def submit_review(review_id: str, submission: ReviewSubmission):
    """Submit a human review for a queued item."""
    try:
        from resync.core.continual_learning import ReviewStatus, get_active_learning_manager

        manager = get_active_learning_manager()

        try:
            status = ReviewStatus(submission.status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {[s.value for s in ReviewStatus]}",
            ) from None

        success = await manager.submit_review(
            review_id=review_id,
            status=status,
            reviewer_id=submission.reviewer_id,
            correction=submission.correction,
            feedback=submission.feedback,
        )

        if success:
            return {"message": "Review submitted successfully", "review_id": review_id}
        raise HTTPException(status_code=404, detail="Review item not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("submit_review_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/review/stats")
async def get_review_stats():
    """Get statistics about the review queue."""
    try:
        from resync.core.continual_learning import get_active_learning_manager

        manager = get_active_learning_manager()
        return await manager.get_queue_stats()

    except Exception as e:
        logger.error("review_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.post("/review/expire-old")
async def expire_old_reviews():
    """Expire old pending reviews that haven't been processed."""
    try:
        from resync.core.continual_learning import get_active_learning_manager

        manager = get_active_learning_manager()
        expired_count = await manager.expire_old_reviews()

        return {"expired_count": expired_count, "message": f"Expired {expired_count} reviews"}
    except Exception as e:
        logger.error("expire_reviews_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


# =============================================================================
# CONTEXT ENRICHMENT ENDPOINTS
# =============================================================================


@router.post("/enrich", response_model=EnrichmentResponse)
async def enrich_query(request: EnrichmentRequest):
    """
    Enrich a query with learned context.

    Uses job patterns, error history, and dependencies to add
    relevant context to the query before RAG retrieval.
    """
    try:
        from resync.core.continual_learning import get_context_enricher

        enricher = get_context_enricher()
        result = await enricher.enrich_query(
            query=request.query,
            instance_id=request.instance_id,
        )

        return EnrichmentResponse(
            original_query=result.original_query,
            enriched_query=result.enriched_query,
            context_added=result.context_added,
            entities_found=result.entities_found,
            enrichment_source=result.enrichment_source,
        )
    except Exception as e:
        logger.error("enrichment_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================


@router.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get comprehensive statistics from all continual learning components."""
    try:
        from resync.core.continual_learning import get_continual_learning_orchestrator

        orchestrator = get_continual_learning_orchestrator()
        stats = await orchestrator.get_system_stats()

        return StatsResponse(**stats)
    except Exception as e:
        logger.error("system_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/health")
async def health_check():
    """Check health of continual learning components."""
    health = {
        "status": "healthy",
        "components": {},
    }

    try:
        from resync.core.continual_learning import get_feedback_store

        store = get_feedback_store()
        await store.initialize()
        health["components"]["feedback_store"] = "ok"
    except Exception as e:
        health["components"]["feedback_store"] = f"error: {str(e)}"
        health["status"] = "degraded"

    try:
        from resync.core.continual_learning import get_active_learning_manager

        manager = get_active_learning_manager()
        await manager.initialize()
        health["components"]["active_learning"] = "ok"
    except Exception as e:
        health["components"]["active_learning"] = f"error: {str(e)}"
        health["status"] = "degraded"

    try:
        from resync.core.continual_learning import get_context_enricher

        get_context_enricher()
        health["components"]["context_enricher"] = "ok"
    except Exception as e:
        health["components"]["context_enricher"] = f"error: {str(e)}"
        health["status"] = "degraded"

    return health


# =============================================================================
# AUDIT PIPELINE ENDPOINTS
# =============================================================================


@router.get("/audit/error-patterns")
async def get_error_patterns(
    entity: str | None = Query(None, description="Filter by entity"),
    min_confidence: float = Query(0.5, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=500),
):
    """Get known error patterns from the knowledge graph."""
    try:
        from resync.core.continual_learning import get_audit_to_kg_pipeline

        pipeline = get_audit_to_kg_pipeline()
        return await pipeline.get_error_patterns(
            entity=entity,
            min_confidence=min_confidence,
            limit=limit,
        )

    except Exception as e:
        logger.error("error_patterns_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e
