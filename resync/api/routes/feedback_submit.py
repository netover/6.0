"""Public feedback submission API (chat UI).

This router is intentionally *additive* and separate from admin curation routes.

Why this exists:
- The chat UI needs a simple endpoint to persist thumbs up/down + comments.
- Admin curation endpoints live under /api/v1/admin/feedback and are protected.

Data model:
- Persists to learning.feedback table via resync.core.database.models.stores.Feedback.

Security / privacy notes:
- The UI may optionally submit query_text/response_text for richer curation.
- These fields can include sensitive data; we keep them optional and bounded.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from resync.core.database import Feedback, get_session


router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback"])


class SubmitChatFeedbackRequest(BaseModel):
    """Payload for chat UI feedback."""

    trace_id: str = Field(..., min_length=1, max_length=255)
    rating: int = Field(..., ge=1, le=5)
    feedback_type: Literal["general", "accuracy", "helpfulness", "speed"] = "general"

    # Optional comment (modal)
    comment: str | None = Field(None, max_length=2000)

    # Optional context capture (bounded)
    query_text: str | None = Field(None, max_length=20000)
    response_text: str | None = Field(None, max_length=20000)

    # Optional stable message index (assistant message ordinal)
    message_index: int | None = Field(None, ge=0, le=100000)

    model_config = ConfigDict(extra="forbid")


class SubmitChatFeedbackResponse(BaseModel):
    status: Literal["ok"]
    feedback_id: int


@router.post("/submit", response_model=SubmitChatFeedbackResponse)
async def submit_chat_feedback(req: SubmitChatFeedbackRequest) -> SubmitChatFeedbackResponse:
    """Persist feedback from the chat UI."""

    trace_id = req.trace_id.strip()
    if not trace_id:
        raise HTTPException(status_code=422, detail="trace_id is required")

    feedback_text = req.comment.strip() if req.comment else None
    query_text = req.query_text.strip() if req.query_text else None
    response_text = req.response_text.strip() if req.response_text else None

    # Mirror existing conventions used elsewhere: positive >= 4, negative <= 2
    is_positive = True if req.rating >= 4 else False if req.rating <= 2 else None

    query_id = (
        f"chat:{trace_id}:{req.message_index}" if req.message_index is not None else f"chat:{trace_id}"
    )

    try:
        async with get_session() as session:
            row = Feedback(
                session_id=trace_id,
                query_id=query_id,
                query_text=query_text,
                response_text=response_text,
                rating=req.rating,
                feedback_type=req.feedback_type,
                feedback_text=feedback_text,
                is_positive=is_positive,
                metadata_={
                    "source": "chat-ui",
                    "trace_id": trace_id,
                    "message_index": req.message_index,
                    "has_context": bool(query_text or response_text),
                },
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)

        return SubmitChatFeedbackResponse(status="ok", feedback_id=int(row.id))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Keep response generic; rely on global handlers/logging for details
        raise HTTPException(status_code=500, detail="Failed to submit feedback") from exc
