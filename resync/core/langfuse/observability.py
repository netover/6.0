"""
LangFuse Observability Module.

Provides tracing and analytics for LLM calls:
- Request/response logging
- Latency tracking
- Token usage monitoring
- Cost estimation
- Error tracking

Usage:
    from resync.core.langfuse import trace_llm_call, get_tracer

    tracer = get_tracer()

    @trace_llm_call("chat_completion")
    async def my_llm_function():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)

# Try to import langfuse
try:
    from langfuse import Langfuse
    from langfuse.decorators import langfuse_context, observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LLMCallTrace:
    """Record of a single LLM call."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # Request
    model: str = ""
    prompt_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    input_tokens: int = 0

    # Response
    output: str = ""
    output_tokens: int = 0
    total_tokens: int = 0

    # Metadata
    temperature: float | None = None
    max_tokens: int | None = None
    user_id: str | None = None
    session_id: str | None = None

    # Status
    success: bool = True
    error: str | None = None
    error_type: str | None = None

    # Cost (estimated)
    estimated_cost_usd: float | None = None

    def complete(self, output: str = "", error: str | None = None) -> None:
        """Mark the trace as complete."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if error:
            self.success = False
            self.error = error
        else:
            self.output = output
            self.success = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error": self.error,
            "estimated_cost_usd": self.estimated_cost_usd,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class TraceSession:
    """A session containing multiple traces."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    traces: list[LLMCallTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_trace(self, trace: LLMCallTrace) -> None:
        """Add a trace to the session."""
        trace.session_id = self.session_id
        trace.user_id = self.user_id
        self.traces.append(trace)

    @property
    def total_duration_ms(self) -> float:
        """Get total duration of all traces."""
        return sum(t.duration_ms or 0 for t in self.traces)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used in session."""
        return sum(t.total_tokens for t in self.traces)

    @property
    def success_rate(self) -> float:
        """Get success rate of traces."""
        if not self.traces:
            return 1.0
        return sum(1 for t in self.traces if t.success) / len(self.traces)


# =============================================================================
# COST ESTIMATION
# =============================================================================

# Approximate costs per 1K tokens (USD) - Update periodically
MODEL_COSTS: dict[str, dict[str, float]] = {
    "meta/llama-3.1-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "meta/llama-3.1-8b-instruct": {"input": 0.0001, "output": 0.0001},
    "nvidia/llama-3.1-nemotron-70b-instruct": {"input": 0.00035, "output": 0.0004},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.0002, "output": 0.0002},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for an LLM call.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])

    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]

    return round(input_cost + output_cost, 6)


# =============================================================================
# TRACER
# =============================================================================


class LangFuseTracer:
    """
    LLM call tracer with LangFuse integration.

    Provides:
    - Automatic tracing of LLM calls
    - Session management
    - Cost tracking
    - Error monitoring

    Falls back to local logging if LangFuse is unavailable.
    """

    _instance: LangFuseTracer | None = None

    def __new__(cls) -> LangFuseTracer:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._client: Langfuse | None = None
        self._current_session: TraceSession | None = None
        self._traces: list[LLMCallTrace] = []
        self._lock = asyncio.Lock()

        # Initialize LangFuse if available
        if LANGFUSE_AVAILABLE and self._should_use_langfuse():
            try:
                self._client = Langfuse(
                    public_key=getattr(settings, "langfuse_public_key", None),
                    secret_key=getattr(settings, "langfuse_secret_key", None),
                    host=getattr(settings, "langfuse_host", "https://cloud.langfuse.com"),
                )
                logger.info("langfuse_tracer_initialized")
            except Exception as e:
                logger.warning("langfuse_tracer_init_failed", error=str(e))

        self._initialized = True

    def _should_use_langfuse(self) -> bool:
        """Check if LangFuse should be used."""
        return (
            getattr(settings, "langfuse_enabled", False)
            and getattr(settings, "langfuse_public_key", None)
            and getattr(settings, "langfuse_secret_key", None)
        )

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        model: str = "",
        prompt_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for tracing an LLM call.

        Usage:
            async with tracer.trace("chat_completion", model="llama-3.1-70b") as trace:
                result = await llm.generate(...)
                trace.output = result
                trace.output_tokens = 150

        Args:
            name: Name of the operation
            model: Model being used
            prompt_id: ID of prompt template if using one
            user_id: User identifier
            metadata: Additional metadata
        """
        trace = LLMCallTrace(
            name=name,
            model=model,
            prompt_id=prompt_id,
            user_id=user_id or (self._current_session.user_id if self._current_session else None),
            session_id=self._current_session.session_id if self._current_session else None,
        )

        try:
            yield trace
            trace.complete(output=trace.output)

            # Estimate cost
            if trace.input_tokens > 0 or trace.output_tokens > 0:
                trace.total_tokens = trace.input_tokens + trace.output_tokens
                trace.estimated_cost_usd = estimate_cost(
                    model, trace.input_tokens, trace.output_tokens
                )

        except Exception as e:
            trace.complete(error=str(e))
            trace.error_type = type(e).__name__
            raise
        finally:
            # Record trace
            async with self._lock:
                self._traces.append(trace)
                if self._current_session:
                    self._current_session.add_trace(trace)

            # Send to LangFuse
            if self._client:
                self._send_to_langfuse(trace, metadata)

            # Log locally
            logger.info(
                "llm_call_traced",
                trace_id=trace.trace_id,
                name=name,
                model=model,
                duration_ms=trace.duration_ms,
                tokens=trace.total_tokens,
                success=trace.success,
                cost_usd=trace.estimated_cost_usd,
            )

    def _send_to_langfuse(
        self, trace: LLMCallTrace, metadata: dict[str, Any] | None = None
    ) -> None:
        """Send trace to LangFuse."""
        if not self._client:
            return

        try:
            # Create trace in LangFuse using the official SDK
            langfuse_trace = self._client.trace(
                id=trace.trace_id,
                name=trace.name,
                user_id=trace.user_id,
                session_id=trace.session_id,
                metadata=metadata or {},
            )

            # Create a generation span for the LLM call
            langfuse_trace.generation(
                name=f"{trace.name}_generation",
                model=trace.model,
                prompt=trace.messages if trace.messages else None,
                completion=trace.output,
                usage={
                    "prompt_tokens": trace.input_tokens,
                    "completion_tokens": trace.output_tokens,
                    "total_tokens": trace.total_tokens,
                },
                level="ERROR" if not trace.success else "DEFAULT",
                status_message=trace.error if trace.error else None,
                metadata={
                    "temperature": trace.temperature,
                    "max_tokens": trace.max_tokens,
                    "prompt_id": trace.prompt_id,
                    "estimated_cost_usd": trace.estimated_cost_usd,
                    "duration_ms": trace.duration_ms,
                },
            )

            # Flush to ensure data is sent
            self._client.flush()

            logger.debug(
                "trace_sent_to_langfuse",
                trace_id=trace.trace_id,
                model=trace.model,
            )
        except Exception as e:
            logger.warning("langfuse_trace_failed", error=str(e), trace_id=trace.trace_id)

    def start_session(
        self, user_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> TraceSession:
        """
        Start a new trace session.

        Args:
            user_id: User identifier
            metadata: Session metadata

        Returns:
            New TraceSession
        """
        self._current_session = TraceSession(
            user_id=user_id,
            metadata=metadata or {},
        )

        logger.debug("trace_session_started", session_id=self._current_session.session_id)
        return self._current_session

    def end_session(self) -> TraceSession | None:
        """
        End the current session.

        Returns:
            The completed session, or None if no active session
        """
        session = self._current_session
        self._current_session = None

        if session:
            logger.debug(
                "trace_session_ended",
                session_id=session.session_id,
                trace_count=len(session.traces),
                total_tokens=session.total_tokens,
                success_rate=session.success_rate,
            )

        return session

    def get_recent_traces(self, limit: int = 100) -> list[LLMCallTrace]:
        """Get recent traces."""
        return self._traces[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get tracing statistics."""
        if not self._traces:
            return {
                "total_traces": 0,
                "success_rate": 1.0,
                "total_tokens": 0,
                "total_cost_usd": 0,
            }

        successful = sum(1 for t in self._traces if t.success)
        total_tokens = sum(t.total_tokens for t in self._traces)
        total_cost = sum(t.estimated_cost_usd or 0 for t in self._traces)
        avg_duration = sum(t.duration_ms or 0 for t in self._traces) / len(self._traces)

        return {
            "total_traces": len(self._traces),
            "success_rate": successful / len(self._traces),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_duration_ms": round(avg_duration, 2),
            "langfuse_enabled": self._client is not None,
        }


# =============================================================================
# DECORATOR
# =============================================================================


def trace_llm_call(
    name: str,
    model: str = "",
    prompt_id: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to trace LLM calls.

    Usage:
        @trace_llm_call("chat_completion", model="llama-3.1-70b")
        async def my_function(messages):
            return await client.chat.completions.create(...)

    Args:
        name: Name of the operation
        model: Model being used
        prompt_id: Prompt template ID if using one
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()

            async with tracer.trace(name=name, model=model, prompt_id=prompt_id) as trace:
                result = await func(*args, **kwargs)

                # Try to extract token counts from result
                if hasattr(result, "usage"):
                    usage = result.usage
                    trace.input_tokens = getattr(usage, "prompt_tokens", 0)
                    trace.output_tokens = getattr(usage, "completion_tokens", 0)

                # Try to extract output
                if hasattr(result, "choices") and result.choices:
                    choice = result.choices[0]
                    if hasattr(choice, "message"):
                        trace.output = getattr(choice.message, "content", "")

                return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_tracer: LangFuseTracer | None = None


def get_tracer() -> LangFuseTracer:
    """Get or create the singleton tracer."""
    global _tracer
    if _tracer is None:
        _tracer = LangFuseTracer()
    return _tracer
