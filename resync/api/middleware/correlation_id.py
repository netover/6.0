# pylint: skip-file
# mypy: ignore-errors
"""Correlation ID middleware.

Adds a stable request correlation identifier for the lifetime of an HTTP request.

Responsibilities:
- Read/propagate `X-Correlation-ID` from the incoming request (or generate one)
- Generate a unique `X-Request-ID` for each request (always generated)
- Store both IDs in:
  - `scope['state']` (so downstream Starlette/FastAPI can access)
  - `resync.core.context` contextvars (for logging correlation in async tasks)
- Attach headers to the outgoing response

Notes on middleware ordering (Starlette/FastAPI):
- The last `app.add_middleware()` call is executed first.
- In production, this middleware should be outermost so correlation IDs are
  available for ALL downstream middleware logs.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

try:
    import structlog
except ImportError:
    structlog = None

from resync.core.context import (
    reset_correlation_id,
    reset_request_id,
    reset_trace_id,
    set_correlation_id,
    set_request_id,
    set_trace_id,
)
from resync.core.langfuse.trace_utils import hash_user_id, normalize_trace_id
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Try to import Langfuse context
try:
    from langfuse.decorators import langfuse_context

    LANGFUSE_AVAILABLE = True
except Exception as exc:
    LANGFUSE_AVAILABLE = False
    langfuse_context = None
    logger.warning("langfuse_context_unavailable reason=%s", type(exc).__name__)


class CorrelationIdMiddleware:
    """ASGI middleware that injects correlation and request IDs."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Correlation-ID"):
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)

        correlation_id = headers.get(self.header_name) or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Store on scope.state for request handlers / other middleware
        state: dict[str, Any] = scope.setdefault("state", {})  # type: ignore[assignment]
        state["correlation_id"] = correlation_id
        state["request_id"] = request_id
        state["start_time"] = time.time()

        # Store in contextvars for structured logging across awaits
        cid_token = set_correlation_id(correlation_id)
        rid_token = set_request_id(request_id)

        # Create a W3C-compatible trace ID from the correlation ID
        trace_id = normalize_trace_id(correlation_id)
        tid_token = set_trace_id(trace_id)

        # Update Langfuse context with trace ID and potential user ID
        if LANGFUSE_AVAILABLE and langfuse_context:
            try:
                # Set the trace ID for the current context
                langfuse_context.update_current_trace(
                    trace_id=trace_id,
                    metadata={
                        "request_id": request_id,
                        "correlation_id": correlation_id,
                        "source": "http_middleware",
                    },
                )

                # Try to capture user_id from Authorization header (best effort)
                auth_header = headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    try:
                        # Decode without verification first just to get 'sub' for logging/tracing
                        # We don't rely on this for auth (middleware does that), just for tracing context
                        import jwt

                        # Decode unverified to avoid dependency on secret keys/settings here
                        # This is safe because we only use it for tracing attribution, not access control
                        payload = jwt.decode(token, options={"verify_signature": False})
                        if "sub" in payload:
                            user_id = str(payload["sub"])
                            hashed_user = hash_user_id(user_id)
                            langfuse_context.update_current_trace(user_id=hashed_user)
                    except Exception:
                        # Ignore decoding errors in middleware - not our job to validate
                        pass
            except Exception as e:
                logger.debug("langfuse_context_update_failed", error=str(e))

        # Bind to structlog contextvars if available
        if structlog:
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(
                correlation_id=correlation_id,
                request_id=request_id,
                trace_id=trace_id,
            )

        async def send_wrapper(message: dict) -> None:
            if message.get("type") == "http.response.start":
                out_headers = MutableHeaders(raw=message["headers"])
                out_headers.setdefault(self.header_name, correlation_id)
                out_headers.setdefault("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Avoid context leakage across requests/tasks
            try:
                reset_request_id(rid_token)
            except Exception:
                logger.debug("failed_to_reset_request_id", exc_info=True)
            try:
                reset_trace_id(tid_token)
            except Exception:
                logger.debug("failed_to_reset_trace_id", exc_info=True)
            try:
                reset_correlation_id(cid_token)
            except Exception:
                logger.debug("failed_to_reset_correlation_id", exc_info=True)

            if structlog:
                structlog.contextvars.clear_contextvars()
