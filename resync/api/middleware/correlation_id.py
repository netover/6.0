# pylint: disable-all
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

Critical fixes applied:
- P0-10: Removed JWT decode without verification (security vulnerability)
- P1-20: Fixed scope['state'] mutation with defensive copy
- P2-38: Reduced exception logging verbosity in finally blocks
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
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
    import sys as _sys
    from resync.core.exception_guard import maybe_reraise_programming_error
    _exc_type, _exc, _tb = _sys.exc_info()
    maybe_reraise_programming_error(_exc, _tb)

    LANGFUSE_AVAILABLE = False
    langfuse_context = None
    logger.warning("langfuse_context_unavailable", reason=type(exc).__name__)

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

        # P1-20 fix: Use defensive copy to prevent state mutation across requests
        # In some deployment models (e.g., with connection pooling), scope objects
        # may be reused. Creating a new dict ensures isolation.
        existing_state = scope.get("state")
        if existing_state is None:
            state: dict[str, Any] = {}
        else:
            # Shallow copy to prevent mutation of pre-existing state
            state = existing_state.copy() if isinstance(existing_state, dict) else {}
        
        scope["state"] = state
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

                # P0-10 fix: Get user_id from request.state (already validated by auth)
                # REMOVED: jwt.decode without verification (security vulnerability)
                # Auth middleware validates JWT and stores user_id in state.
                # Using that value is both more secure and more efficient.
                user_id = state.get("user_id")
                if user_id:
                    hashed_user = hash_user_id(str(user_id))
                    langfuse_context.update_current_trace(user_id=hashed_user)
                    logger.debug(
                        "langfuse_user_id_set",
                        user_id_hash=hashed_user[:8],  # Log only prefix for privacy
                    )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                # Langfuse context updates are best-effort — log at debug level
                logger.debug("langfuse_context_update_failed", error=type(e).__name__)

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
            # P2-38 fix: Reduce logging verbosity — only log error type, not full stack
            # Context reset failures are almost always LookupError (already reset)
            # or programming errors. Full stack traces are not needed in production.
            try:
                reset_request_id(rid_token)
            except LookupError:
                # Already reset — this is expected in some scenarios
                pass
            except (TypeError, AttributeError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                # Programming error — log with stack trace for debugging
                logger.warning(
                    "request_id_reset_programming_error",
                    error_type=type(e).__name__,
                    exc_info=True,
                )
            
            try:
                reset_trace_id(tid_token)
            except LookupError:
                pass
            except (TypeError, AttributeError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning(
                    "trace_id_reset_programming_error",
                    error_type=type(e).__name__,
                    exc_info=True,
                )
            
            try:
                reset_correlation_id(cid_token)
            except LookupError:
                pass
            except (TypeError, AttributeError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning(
                    "correlation_id_reset_programming_error",
                    error_type=type(e).__name__,
                    exc_info=True,
                )

            if structlog:
                structlog.contextvars.clear_contextvars()
