"""
Distributed Tracing System with OpenTelemetry and Jaeger.
"""

from __future__ import annotations

import contextlib
import functools

# pylint
import inspect
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.sdk.trace.sampling import (  # type: ignore
        Decision,
        Sampler,
        SamplingResult,
    )
except ImportError:
    trace = None  # type: ignore
    TracerProvider = object  # type: ignore
    BatchSpanProcessor = object  # type: ignore

    class Decision:  # type: ignore
        RECORD_AND_SAMPLE = 1
        DROP = 0

    class Sampler:  # type: ignore
        pass

    class SamplingResult:  # type: ignore
        def __init__(self, decision: int) -> None:
            self.decision = decision

    class SpanKind:  # type: ignore
        SERVER = 0
        CLIENT = 1

    class Status:  # type: ignore
        def __init__(self, status_code: int, description: str = "") -> None:
            self.status_code = status_code
            self.description = description

    class StatusCode:  # type: ignore
        OK = 0
        ERROR = 1

try:
    from opentelemetry.trace.propagation.tracecontext import TraceContextPropagator
except ImportError:
    TraceContextPropagator = None

try:
    # opentelemetry-exporter-jaeger was deprecated and removed from OTel Python ≥ 1.20.
    # Modern deployments should use the OTLP exporter and configure a Jaeger/Tempo
    # collector that accepts OTLP over gRPC or HTTP.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    JAEGER_AVAILABLE = True  # Keep name for backward compat; exporter is now OTLP-based.
    JaegerExporter = OTLPSpanExporter  # type: ignore[assignment, misc]
except ImportError:
    JAEGER_AVAILABLE = False

    class JaegerExporter:  # type: ignore[no-redef]
        """Stub when opentelemetry-exporter-otlp-proto-grpc is not installed."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def shutdown(self, **kwargs: object) -> None:
            pass

try:
    from opentelemetry.sdk.trace.export import ConsoleSpanProcessor

    CONSOLE_AVAILABLE = True
except ImportError:
    CONSOLE_AVAILABLE = False

    class ConsoleSpanProcessor:
        def __init__(self, **kwargs):
            pass

from resync.core.structured_logger import get_logger

class _NoOpSpan:
    """Fallback span used when OpenTelemetry is disabled or uninitialized."""

    def get_span_context(self):
        return self

    @property
    def is_valid(self):
        return False

    def set_attribute(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None

    def set_status(self, *args, **kwargs):
        return None

logger = get_logger(__name__)

current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)

class IntelligentSampler(Sampler):
    """Adaptive sampling strategy."""

    def __init__(self, base_sample_rate: float = 0.1, max_sample_rate: float = 1.0):
        self.base_sample_rate = base_sample_rate
        self.max_sample_rate = max_sample_rate
        self.error_count = 0
        self.total_requests = 0
        self.latency_threshold = 1.0
        self.error_threshold = 0.05
        self._decisions: dict[str, SamplingResult] = {}

    def get_description(self) -> str:
        return (
            "IntelligentSampler("
            f"base_rate={self.base_sample_rate}, "
            f"max_rate={self.max_sample_rate}"
            ")"
        )

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        if parent_context and parent_context.trace_flags.sampled:
            return SamplingResult(Decision.RECORD_AND_SAMPLE)

        if attributes:
            error_code = attributes.get(
                "http.status_code", attributes.get("status_code")
            )
            if error_code and str(error_code).startswith(("4", "5")):
                return SamplingResult(Decision.RECORD_AND_SAMPLE)

        current_rate = self._calculate_adaptive_rate()
        should_sample = (
            (trace_id % int(1 / current_rate)) == 0 if current_rate > 0 else False
        )

        if should_sample:
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        return SamplingResult(Decision.DROP)

    def _calculate_adaptive_rate(self) -> float:
        if self.total_requests == 0:
            return self.base_sample_rate
        error_rate = self.error_count / self.total_requests
        if error_rate > self.error_threshold:
            return min(
                self.max_sample_rate,
                self.base_sample_rate * (error_rate / self.error_threshold),
            )
        return self.base_sample_rate

@dataclass
class TraceConfiguration:
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    jaeger_service_name: str = "hwa-new"
    jaeger_tags: dict[str, str] = field(
        default_factory=lambda: {"service.version": "1.0.0"}
    )
    sampling_rate: float = 0.1
    adaptive_sampling: bool = True
    max_sampling_rate: float = 1.0
    max_batch_size: int = 512
    export_timeout_seconds: int = 30
    max_queue_size: int = 2048
    auto_instrument_http: bool = True
    auto_instrument_db: bool = True
    auto_instrument_asyncio: bool = True
    auto_instrument_external_calls: bool = True
    custom_span_processors: list[Any] = field(default_factory=list)
    custom_instrumentations: list[Any] = field(default_factory=list)

class DistributedTracingManager:
    """Main distributed tracing manager."""

    def __init__(self, config: TraceConfiguration | None = None):
        self.config = config or TraceConfiguration()
        self.tracer_provider: TracerProvider | None = None
        self.tracer: trace.Tracer | None = None
        self.jaeger_exporter: JaegerExporter | None = None
        self._instrumented = False
        self._running = False
        self.trace_metrics: dict[str, Any] = {
            "traces_created": 0,
            "spans_created": 0,
            "export_errors": 0,
            "export_success": 0,
            "sampling_decisions": 0,
        }
        self.propagator = TraceContextPropagator() if TraceContextPropagator else None

    def _initialize_tracing(self) -> None:
        self.tracer_provider = TracerProvider()
        if self.config.adaptive_sampling:
            sampler = IntelligentSampler(
                self.config.sampling_rate, self.config.max_sampling_rate
            )
        else:
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler

            sampler = TraceIdRatioBasedSampler(self.config.sampling_rate)
        self.tracer_provider.sampler = sampler

        self.jaeger_exporter = JaegerExporter(
            agent_host_name=urlparse(self.config.jaeger_endpoint).hostname,
            agent_port=int(urlparse(self.config.jaeger_endpoint).port or 14268),
            collector_endpoint=self.config.jaeger_endpoint,
        )

        span_processor = BatchSpanProcessor(
            self.jaeger_exporter,
            max_export_batch_size=self.config.max_batch_size,
            export_timeout_millis=self.config.export_timeout_seconds * 1000,
        )
        self.tracer_provider.add_span_processor(span_processor)

        if CONSOLE_AVAILABLE:
            self.tracer_provider.add_span_processor(ConsoleSpanProcessor())

        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        logger.info("Distributed tracing initialized")

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._initialize_tracing()
        await self._setup_auto_instrumentation()
        logger.info("Distributed tracing system started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        logger.info("Distributed tracing system stopped")

    async def _setup_auto_instrumentation(self) -> None:
        if self._instrumented:
            return
        try:
            self._instrumented = True
            logger.info("Auto-instrumentation completed")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Failed to setup auto-instrumentation: %s", e)

    @contextlib.contextmanager
    def trace_context(self, operation_name: str, **attributes):
        """Context manager for creating trace spans."""
        start_time = time.perf_counter()
        if not getattr(self, "tracer", None):
            yield _NoOpSpan()
            return

        with self.tracer.start_as_current_span(
            operation_name, attributes=attributes
        ) as span:
            if span.get_span_context().is_valid:
                trace_id = format(span.get_span_context().trace_id, "032x")
                span_id = format(span.get_span_context().span_id, "016x")
                current_trace_id.set(trace_id)
                current_span_id.set(span_id)
                span.set_attribute("trace.id", trace_id)
                span.set_attribute("span.id", span_id)

            self.trace_metrics["spans_created"] += 1
            try:
                yield span
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                span.record_exception(e)  # Record exception here
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1e3
                span.set_attribute("performance.duration_ms", duration_ms)

    def trace_method(self, operation_name: str | None = None):
        """Decorator for tracing method calls."""

        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                method_name = operation_name or f"{func.__name__}"
                class_name = args[0].__class__.__name__ if args else "unknown"
                span_name = f"{class_name}.{method_name}"
                # FIX: trace_context already records exception;
                # redundant recording removed
                with self.trace_context(
                    span_name, operation_type="method_call", is_async=True
                ) as span:
                    result = await func(*args, **kwargs)
                    span.set_attribute("result.success", True)
                    return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                method_name = operation_name or f"{func.__name__}"
                class_name = args[0].__class__.__name__ if args else "unknown"
                span_name = f"{class_name}.{method_name}"
                # FIX: trace_context already records exception;
                # redundant recording removed
                with self.trace_context(
                    span_name, operation_type="method_call", is_async=False
                ) as span:
                    result = func(*args, **kwargs)
                    span.set_attribute("result.success", True)
                    return result

            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def create_child_span(self, parent_span: trace.Span, name: str, **attributes):
        child_span = self.tracer.start_span(name, attributes=attributes)
        self.trace_metrics["spans_created"] += 1
        return child_span

    def inject_context(self, carrier: dict[str, str]) -> None:
        if self.propagator:
            self.propagator.inject(carrier)

    def extract_context(self, carrier: dict[str, str]):
        if self.propagator:
            return self.propagator.extract(carrier)
        return None

    def record_exception(self, exception: Exception) -> None:
        span = trace.get_current_span()
        if span:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))

_distributed_tracing_manager: DistributedTracingManager | None = None

def _get_tracing_manager() -> DistributedTracingManager:
    global _distributed_tracing_manager
    if _distributed_tracing_manager is None:
        _distributed_tracing_manager = DistributedTracingManager()
    return _distributed_tracing_manager

async def get_distributed_tracing_manager() -> DistributedTracingManager:
    manager = _get_tracing_manager()
    if not manager._running:
        await manager.start()
    return manager

def trace_method(operation_name: str | None = None):
    return _get_tracing_manager().trace_method(operation_name)

def trace_context(operation_name: str, **attributes):
    return _get_tracing_manager().trace_context(operation_name, **attributes)

def record_exception(exception: Exception) -> None:
    _get_tracing_manager().record_exception(exception)

def traced(operation_name: str, **attributes):
    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                manager = await get_distributed_tracing_manager()
                with manager.trace_context(operation_name, **attributes):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # FIX: Use _get_tracing_manager() to avoid AttributeError
            manager = _get_tracing_manager()
            with manager.trace_context(operation_name, **attributes):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator
