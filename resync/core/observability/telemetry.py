"""
OpenTelemetry Configuration - Distributed tracing and metrics.

v5.6.0: Production-ready observability with OpenTelemetry.

Features:
- Distributed tracing with auto-instrumentation
- Metrics collection
- Log correlation
- Context propagation

Usage:
    from resync.core.observability.telemetry import setup_telemetry

    # In lifespan startup
    setup_telemetry(app)

Environment Variables:
    OTEL_ENABLED: Enable OpenTelemetry (default: true)
    OTEL_SERVICE_NAME: Service name (default: resync)
    OTEL_SERVICE_VERSION: Service version (default: 5.6.0)
    OTEL_ENVIRONMENT: Environment name (default: production)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
    OTEL_TRACES_SAMPLER: Sampler type (default: parentbased_traceidratio)
    OTEL_TRACES_SAMPLER_ARG: Sampling rate 0-1 (default: 0.1 = 10%)
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from fastapi import FastAPI


logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Local no‑op tracer/span implementations
# -----------------------------------------------------------------------------


class _NoOpSpan:
    """A minimal span that does nothing."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        def noop(*args: Any, **kwargs: Any) -> Any:
            return None

        return noop

    @property
    def get_span_context(self) -> Any:  # type: ignore
        class _Ctx:
            trace_id = 0
            span_id = 0
            is_valid = False

        return _Ctx()


class _NoOpTracer:
    """A tracer that produces no‑op spans."""

    def get_current_span(self) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, *args: Any, **kwargs: Any):  # type: ignore
        class _SpanContextManager:
            def __enter__(self) -> _NoOpSpan:
                return _NoOpSpan()

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                pass

        return _SpanContextManager()


def is_telemetry_enabled() -> bool:
    """Check if OpenTelemetry is enabled."""
    return os.getenv("OTEL_ENABLED", "true").lower() in ("true", "1", "yes")


def get_service_name() -> str:
    """Get service name for telemetry."""
    return os.getenv("OTEL_SERVICE_NAME", "resync")


def get_service_version() -> str:
    """Get service version for telemetry."""
    return os.getenv("OTEL_SERVICE_VERSION", "5.6.0")


def get_environment() -> str:
    """Get environment name."""
    return os.getenv("OTEL_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))


def get_hostname() -> str:
    """Get hostname for resource attributes."""
    return socket.gethostname()


def setup_telemetry(app: FastAPI | None = None) -> None:
    """
    Setup OpenTelemetry instrumentation.

    This configures:
    - Tracer provider with OTLP exporter
    - Auto-instrumentation for FastAPI, SQLAlchemy, Redis, httpx
    - Resource attributes for service identification

    Args:
        app: FastAPI application instance (optional for auto-instrumentation)
    """
    if not is_telemetry_enabled():
        logger.info("OpenTelemetry disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.semconv.resource import ResourceAttributes
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi"
        )
        return

    # Create resource with service attributes
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: get_service_name(),
            ResourceAttributes.SERVICE_VERSION: get_service_version(),
            ResourceAttributes.SERVICE_INSTANCE_ID: get_hostname(),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: get_environment(),
            ResourceAttributes.HOST_NAME: get_hostname(),
            "host.type": "vm",  # Indicate VM deployment
        }
    )

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Setup OTLP exporter
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
    except Exception as e:
        logger.warning("Failed to setup OTLP exporter: %s", e)

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument frameworks
    _instrument_frameworks(app)

    logger.info(
        "OpenTelemetry initialized",
        service_name=get_service_name(),
        endpoint=otlp_endpoint,
        environment=get_environment(),
    )


def _instrument_frameworks(app: FastAPI | None = None) -> None:
    """Setup auto-instrumentation for frameworks."""
    # FastAPI
    if app:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.debug("FastAPI instrumented")
        except ImportError:
            pass
        except Exception as e:
            logger.warning("FastAPI instrumentation failed: %s", e)

    # SQLAlchemy
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(enable_commenter=True)
        logger.debug("SQLAlchemy instrumented")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("SQLAlchemy instrumentation failed: %s", e)

    # Redis
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
        logger.debug("Redis instrumented")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Redis instrumentation failed: %s", e)

    # HTTPX
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.debug("HTTPX instrumented")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("HTTPX instrumentation failed: %s", e)


def get_tracer(name: str = __name__) -> Any:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually __name__)

    Returns:
        Tracer instance (NoOp if telemetry disabled)
    """
    # If telemetry is disabled, return a local no‑op tracer without importing
    # opentelemetry.  Avoid ImportError when opentelemetry is not installed.
    if not is_telemetry_enabled():
        return _NoOpTracer()
    try:
        from opentelemetry import trace  # type: ignore

        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


def get_current_span() -> Any:
    """Get the current active span."""
    if not is_telemetry_enabled():
        return _NoOpSpan()
    try:
        from opentelemetry import trace  # type: ignore

        return trace.get_current_span()
    except Exception:
        return _NoOpSpan()


def get_current_trace_id() -> str | None:
    """Get the current trace ID as a hex string."""
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Get the current span ID as a hex string."""
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None


@contextmanager
def create_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Create a new span context manager.

    Usage:
        with create_span("process_order", {"order_id": "123"}) as span:
            # Do work
            span.set_attribute("items_count", 5)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span."""
    span = get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(
    exception: Exception, attributes: dict[str, Any] | None = None
) -> None:
    """Record an exception on the current span."""
    span = get_current_span()
    if span:
        span.record_exception(exception, attributes=attributes or {})


def set_span_status_error(message: str) -> None:
    """Set the current span status to error."""
    # If OpenTelemetry is available, set error status on current span; otherwise no‑op
    try:
        from opentelemetry.trace import Status, StatusCode  # type: ignore

        span = get_current_span()
        if getattr(span, "set_status", None):
            span.set_status(Status(StatusCode.ERROR, message))
    except Exception as exc:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass


# =============================================================================
# Logging Integration
# =============================================================================


def inject_trace_context(
    logger: logging.Logger, method_name: str, event_dict: dict
) -> dict:
    """
    Structlog processor to inject trace context into logs.

    Add to structlog processors:
        structlog.configure(
            processors=[
                inject_trace_context,
                ...
            ]
        )
    """
    trace_id = get_current_trace_id()
    span_id = get_current_span_id()

    if trace_id:
        event_dict["trace_id"] = trace_id
    if span_id:
        event_dict["span_id"] = span_id

    return event_dict


# =============================================================================
# Prometheus Metrics
# =============================================================================


def setup_prometheus_metrics(app: FastAPI) -> None:
    """
    Setup Prometheus metrics for FastAPI.

    Exposes /metrics endpoint with:
    - HTTP request count, latency, size
    - Default process metrics
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=[
                "/health",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
            ],
            env_var_name="PROMETHEUS_ENABLED",
            inprogress_name="http_requests_inprogress",
            inprogress_labels=True,
        )

        # Add default metrics
        instrumentator.add(
            instrumentator.metrics.default(
                metric_namespace="resync",
                metric_subsystem="api",
            )
        )

        # Instrument and expose
        instrumentator.instrument(app).expose(app, include_in_schema=False)

        logger.info("Prometheus metrics enabled at /metrics")

    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator not installed. "
            "Install with: pip install prometheus-fastapi-instrumentator"
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to setup Prometheus metrics: %s", e)


# =============================================================================
# Shutdown
# =============================================================================


def shutdown_telemetry() -> None:
    """Shutdown telemetry providers gracefully."""
    if not is_telemetry_enabled():
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")
    except Exception as e:
        logger.error("Error during telemetry shutdown: %s", e)
