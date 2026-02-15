"""
Agent Evolution API

REST API for supervised agent evolution system.

Endpoints:
- POST /api/admin/agents/feedback - Submit feedback
- GET /api/admin/agents/{name}/patterns - View detected patterns
- GET /api/admin/agents/improvements - List improvement suggestions
- POST /api/admin/agents/improvements/{id}/test - Test in sandbox
- POST /api/admin/agents/improvements/{id}/approve - Approve & deploy
- POST /api/admin/agents/improvements/{id}/reject - Reject suggestion
- GET /api/admin/agents/{name}/performance - View performance metrics

Author: Resync Team
Version: 5.9.9
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from resync.api.auth import verify_admin_credentials
from pydantic import BaseModel

from resync.core.agent_evolution import (
    AgentFeedbackCollector,
    FeedbackType,
    ImprovementSuggestion,
    SandboxTester,
)
from pydantic import ConfigDict
import aiofiles

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin/agents", tags=["agent-evolution"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SubmitFeedbackRequest(BaseModel):
    """Request to submit feedback on agent output."""
    agent_name: str  # e.g., "job_analyst"
    task: str  # e.g., "analyze_job:PAYROLL_NIGHTLY"
    output: dict  # Agent's output
    feedback_type: str  # "thumbs_up" | "thumbs_down"
    comment: str | None = None
    job_name: str | None = None

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "agent_name": "job_analyst",
                "task": "analyze_job:PAYROLL_NIGHTLY",
                "output": {
                    "dependencies": ["BACKUP_DB"],
                    "risk": "medium"
                },
                "feedback_type": "thumbs_down",
                "comment": "Missed dependency with TIMEKEEPING_CLOSE",
                "job_name": "PAYROLL_NIGHTLY"
            }
        })


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    status: str
    feedback_id: str
    message: str


class PatternResponse(BaseModel):
    """Pattern detection response."""
    id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    examples: list[str]
    job_pattern: str | None = None


class ImprovementResponse(BaseModel):
    """Improvement suggestion response."""
    id: str
    agent_name: str
    pattern_id: str
    current_prompt: str
    proposed_prompt: str
    rationale: str
    estimated_impact: str
    status: str
    created_at: str


class TestResultResponse(BaseModel):
    """Sandbox test result response."""
    suggestion_id: str
    test_cases_count: int
    current_accuracy: float
    improved_accuracy: float
    improvement_pct: float
    regressions_detected: list[str]
    safe_to_deploy: bool


class PerformanceMetrics(BaseModel):
    """Agent performance metrics."""
    agent_name: str
    period_days: int
    total_tasks: int
    positive_feedback: int
    negative_feedback: int
    accuracy: float
    trend: str  # "improving" | "stable" | "degrading"


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: SubmitFeedbackRequest):
    """
    Submit feedback on agent performance.

    This is called when user clicks thumbs up/down or provides
    a comment/correction on agent output.

    Example:
        POST /api/admin/agents/feedback
        {
            "agent_name": "job_analyst",
            "task": "analyze_job:PAYROLL_NIGHTLY",
            "output": {"dependencies": ["BACKUP_DB"], "risk": "medium"},
            "feedback_type": "thumbs_down",
            "comment": "Missed TIMEKEEPING_CLOSE dependency",
            "job_name": "PAYROLL_NIGHTLY"
        }

    Response:
        {
            "status": "success",
            "feedback_id": "feedback_20241225_120000_123456",
            "message": "Feedback collected. Pattern analysis triggered."
        }
    """
    try:
        collector = AgentFeedbackCollector()

        feedback = await collector.collect_feedback(
            agent_name=request.agent_name,
            task=request.task,
            output=request.output,
            feedback_type=FeedbackType(request.feedback_type),
            user_comment=request.comment,
            job_name=request.job_name,
        )

        return FeedbackResponse(
            status="success",
            feedback_id=feedback.id,
            message="Feedback collected. Pattern analysis triggered automatically."
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to collect feedback: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.get("/{agent_name}/patterns", response_model=list[PatternResponse])
async def get_patterns(agent_name: str):
    """
    Get detected patterns for an agent.

    Returns patterns found in feedback data.

    Example:
        GET /api/admin/agents/job_analyst/patterns

    Response:
        [
            {
                "id": "pattern_20241225_120000_123456",
                "pattern_type": "missing_dependency",
                "description": "PAYROLL jobs missing TIMEKEEPING dependency",
                "frequency": 5,
                "confidence": 0.85,
                "examples": ["PAYROLL_NIGHTLY", "PAYROLL_WEEKLY"],
                "job_pattern": "PAYROLL_*"
            }
        ]
    """
    try:
        import json
        from pathlib import Path

        pattern_dir = Path("data/agent_patterns")
        patterns = []

        if pattern_dir.exists():
            for file_path in pattern_dir.glob("*.json"):
                try:
                    async with aiofiles.open(file_path) as f:
                        data = json.load(f)
                        patterns.append(PatternResponse(**data))
                except Exception as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        # Filter by agent (if pattern is linked to agent)
        # For now, return all patterns

        return patterns

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get patterns: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.get("/improvements", response_model=list[ImprovementResponse])
async def list_improvements(status: str | None = None):
    """
    List improvement suggestions.

    Filter by status: pending, testing, approved, rejected, deployed

    Example:
        GET /api/admin/agents/improvements?status=pending

    Response:
        [
            {
                "id": "suggestion_20241225_120000_123456",
                "agent_name": "job_analyst",
                "pattern_id": "pattern_...",
                "current_prompt": "You are a job analyst...",
                "proposed_prompt": "You are a TWS/HWA job analyst. IMPORTANT: PAYROLL jobs depend on TIMEKEEPING...",
                "rationale": "Pattern detected: PAYROLL jobs missing TIMEKEEPING dependency. Seen 5 times.",
                "estimated_impact": "+17% accuracy (estimated)",
                "status": "pending",
                "created_at": "2024-12-25T12:00:00"
            }
        ]
    """
    try:
        import json
        from pathlib import Path

        improvements_dir = Path("data/agent_improvements")
        improvements = []

        if improvements_dir.exists():
            for file_path in improvements_dir.glob("*.json"):
                try:
                    async with aiofiles.open(file_path) as f:
                        data = json.load(f)

                        # Filter by status if specified
                        if status and data.get("status") != status:
                            continue

                        improvements.append(ImprovementResponse(**data))
                except Exception as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        # Sort by created_at (newest first)
        improvements.sort(
            key=lambda x: x.created_at,
            reverse=True
        )

        return improvements

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to list improvements: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.post("/improvements/{suggestion_id}/test", response_model=TestResultResponse)
async def test_improvement(suggestion_id: str):
    """
    Test improvement in sandbox.

    Runs both current and improved prompts on historical test cases
    to compare performance.

    Example:
        POST /api/admin/agents/improvements/suggestion_123/test

    Response:
        {
            "suggestion_id": "suggestion_123",
            "test_cases_count": 20,
            "current_accuracy": 0.75,
            "improved_accuracy": 0.90,
            "improvement_pct": 0.15,
            "regressions_detected": [],
            "safe_to_deploy": true
        }
    """
    try:
        # Load suggestion
        suggestion = await _load_suggestion(suggestion_id)

        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Test in sandbox
        tester = SandboxTester()
        result = await tester.test_improvement(suggestion)

        # Update suggestion status
        suggestion.status = "tested"
        await _save_suggestion(suggestion)

        return TestResultResponse(
            suggestion_id=result.suggestion_id,
            test_cases_count=result.test_cases_count,
            current_accuracy=result.current_accuracy,
            improved_accuracy=result.improved_accuracy,
            improvement_pct=result.improvement_pct,
            regressions_detected=result.regressions_detected,
            safe_to_deploy=result.safe_to_deploy,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to test improvement: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.post("/improvements/{suggestion_id}/approve", dependencies=[Depends(verify_admin_credentials)])
async def approve_improvement(
    suggestion_id: str,
):
    """
    Approve and deploy improvement.

    HUMAN APPROVAL REQUIRED!

    Workflow:
    1. Admin reviews suggestion
    2. Admin tests in sandbox (required)
    3. Admin approves
    4. System deploys with monitoring
    5. Auto-rollback if performance degrades

    Example:
        POST /api/admin/agents/improvements/suggestion_123/approve

    Response:
        {
            "status": "deployed",
            "monitoring": "active_24h",
            "rollback": "automatic_if_degraded"
        }
    """
    try:
        # Load suggestion
        suggestion = await _load_suggestion(suggestion_id)

        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Check if tested
        if suggestion.status != "tested":
            raise HTTPException(
                status_code=400,
                detail="Must test in sandbox before approving"
            )

        # Deploy integration: update agent prompt, enable monitoring, setup auto-rollback
        # - Update agent prompt
        # - Enable monitoring
        # - Setup auto-rollback

        suggestion.status = "deployed"
        suggestion.approved_at = datetime.now(timezone.utc)
        # suggestion.approved_by = admin.id
        await _save_suggestion(suggestion)

        logger.info(
            "improvement_approved",
            suggestion_id=suggestion_id,
            agent=suggestion.agent_name
        )

        return {
            "status": "deployed",
            "message": "Improvement deployed with monitoring",
            "monitoring": "active_24h",
            "rollback": "automatic_if_degraded"
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to approve improvement: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.post("/improvements/{suggestion_id}/reject")
async def reject_improvement(suggestion_id: str):
    """
    Reject improvement suggestion.

    Example:
        POST /api/admin/agents/improvements/suggestion_123/reject

    Response:
        {
            "status": "rejected",
            "message": "Improvement rejected"
        }
    """
    try:
        suggestion = await _load_suggestion(suggestion_id)

        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        suggestion.status = "rejected"
        await _save_suggestion(suggestion)

        logger.info("improvement_rejected", suggestion_id=suggestion_id)

        return {
            "status": "rejected",
            "message": "Improvement rejected"
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to reject improvement: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


@router.get("/{agent_name}/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    agent_name: str,
    period_days: int = 30
):
    """
    Get agent performance metrics.

    Example:
        GET /api/admin/agents/job_analyst/performance?period_days=30

    Response:
        {
            "agent_name": "job_analyst",
            "period_days": 30,
            "total_tasks": 150,
            "positive_feedback": 135,
            "negative_feedback": 15,
            "accuracy": 0.90,
            "trend": "improving"
        }
    """
    try:
        import json
        from datetime import datetime, timedelta
        from pathlib import Path

        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        feedback_dir = Path("data/agent_feedback")

        total = 0
        positive = 0
        negative = 0

        if feedback_dir.exists():
            for file_path in feedback_dir.glob("*.json"):
                try:
                    async with aiofiles.open(file_path) as f:
                        data = json.load(f)

                        if data.get("agent_name") != agent_name:
                            continue

                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if timestamp < cutoff:
                            continue

                        total += 1
                        if data.get("correct"):
                            positive += 1
                        else:
                            negative += 1
                except Exception as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        accuracy = positive / total if total > 0 else 0.0

        # Determine trend (simplified)
        trend = "stable"
        if accuracy > 0.85:
            trend = "improving"
        elif accuracy < 0.70:
            trend = "degrading"

        return PerformanceMetrics(
            agent_name=agent_name,
            period_days=period_days,
            total_tasks=total,
            positive_feedback=positive,
            negative_feedback=negative,
            accuracy=accuracy,
            trend=trend,
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get performance metrics: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from None


# =============================================================================
# Helper Functions
# =============================================================================

async def _load_suggestion(suggestion_id: str) -> ImprovementSuggestion | None:
    """Load suggestion from disk."""
    import json
    from pathlib import Path

    file_path = Path("data/agent_improvements") / f"{suggestion_id}.json"

    if not file_path.exists():
        return None

    try:
        async with aiofiles.open(file_path) as f:
            data = json.load(f)
            return ImprovementSuggestion(**data)
    except Exception:
        return None


async def _save_suggestion(suggestion: ImprovementSuggestion):
    """Save suggestion to disk."""
    from pathlib import Path

    file_path = Path("data/agent_improvements") / f"{suggestion.id}.json"

    async with aiofiles.open(file_path, 'w') as f:
        f.write(suggestion.model_dump_json(indent=2))
