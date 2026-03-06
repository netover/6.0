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

import asyncio
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from resync.core.async_utils import classify_exception, with_timeout
from resync.core.database import Feedback, get_session
from resync.core.logging_compat import log_event
from resync.core.security import verify_api_key
from resync.core.valkey_init import get_valkey_client
from resync.settings import get_settings

logger = structlog.get_logger(__name__)
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
async def submit_chat_feedback(
    req: SubmitChatFeedbackRequest,
    request: Request,
) -> SubmitChatFeedbackResponse:
    """Persist feedback from the chat UI."""

    settings = get_settings()

    # Optional shared-secret gating (off by default). Configure via Settings as
    # `feedback_submit_api_key_hash` to require a header `X-Feedback-Key`.
    required_hash = getattr(settings, "feedback_submit_api_key_hash", None)
    if required_hash:
        provided = (request.headers.get("x-feedback-key") or "").strip()
        if (not provided) or (not verify_api_key(provided, required_hash)):
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Best-effort Valkey-backed fixed-window rate limit (atomic, Lua).
    # Default: 30 requests per 60 seconds per client IP.
    # In production, fail-closed if rate limiter is unavailable (configurable by env).
    try:
        valkey = get_valkey_client()
        client_ip = (request.client.host if request.client else "unknown") or "unknown"
        key = f"rate:feedback_submit:{client_ip}"
        lua = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local ttl = tonumber(ARGV[2])
        local current = valkey.call('INCR', key)
        if current == 1 then
          valkey.call('EXPIRE', key, ttl)
        end
        if current > limit then
          return 0
        end
        return 1
        """
        settings = get_settings()
        try:
            ok = await with_timeout(
                valkey.eval(lua, 1, key, "30", "60"),
                getattr(settings, "valkey_health_timeout", 2.0),
                op="valkey.eval(rate_limit)",
            )
        except Exception as e:
            reason, status_code = classify_exception(e)
            logger.debug(
                "feedback rate-limit valkey eval failed (%s, %s): %s",
                reason,
                status_code,
                str(e),
                exc_info=True,
            )
            ok = bool(
                getattr(get_settings(), "rate_limit_fail_open_feedback", True)
            )  # degrade (configurável)
        if int(ok) != 1:
            log_event(logger, "warning", "feedback_rate_limited", client_ip=client_ip)
            raise HTTPException(status_code=429, detail="Too many requests")
    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, asyncio.CancelledError):
            raise
        env = str(getattr(settings, "environment", "")).lower()
        if env in {"prod", "production"}:
            raise HTTPException(
                status_code=503, detail="Rate limiter unavailable"
            ) from e
        # Non-prod: fail-open to avoid breaking chat UX during local/dev setups.

    trace_id = req.trace_id.strip()
    if not trace_id:
        raise HTTPException(status_code=422, detail="trace_id is required")

    # Defense-in-depth: hard caps on stored text (never trust client-only bounds).
    feedback_text = (req.comment.strip() if req.comment else None)
    query_text = (req.query_text.strip() if req.query_text else None)
    response_text = (req.response_text.strip() if req.response_text else None)
    if query_text is not None:
        query_text = query_text[:4000]
    if response_text is not None:
        response_text = response_text[:8000]
    if feedback_text is not None:
        feedback_text = feedback_text[:2000]

    # Mirror existing conventions used elsewhere: positive >= 4, negative <= 2
    is_positive = True if req.rating >= 4 else False if req.rating <= 2 else None

    query_id = (
        f"chat:{trace_id}:{req.message_index}"
        if req.message_index is not None
        else f"chat:{trace_id}"
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
