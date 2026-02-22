"""Global exception handler middleware for standardized error responses."""

import logging
import time
import asyncio
from collections.abc import Callable
from typing import Any
from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from resync.core.exceptions import ResyncException
from resync.core.utils.async_bridge import fire_and_forget
from resync.core.utils.error_utils import (
    ErrorResponseBuilder,
    create_error_response_from_exception,
    create_json_response_from_error,
    extract_validation_errors,
    generate_correlation_id,
    log_error_response,
)

logger = logging.getLogger(__name__)


class GlobalExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling all exceptions and returning standardized error responses."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Process the request and handle any exceptions that occur."""
        correlation_id = generate_correlation_id()
        start_time = time.time()
        try:
            request.state.correlation_id = correlation_id
            response = await call_next(request)
            processing_time = time.time() - start_time
            if processing_time > 1.0:
                logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} took {processing_time:.2f}s",
                    extra={"correlation_id": correlation_id},
                )
            return response
        except RequestValidationError as exc:
            response = await self._handle_validation_error(request, exc, correlation_id)
            self._dispatch_log_error_metrics(
                exc.__class__.__name__, time.time() - start_time
            )
            return response
        except ResyncException as exc:
            response = await self._handle_resync_exception(request, exc, correlation_id)
            self._dispatch_log_error_metrics(
                exc.__class__.__name__, time.time() - start_time
            )
            return response
        except Exception as exc:
            logger.error(
                "exception_caught",
                exc_info=True,
                extra={"error": str(exc), "correlation_id": correlation_id},
            )
            response = await self._handle_generic_exception(request, exc, correlation_id)
            self._dispatch_log_error_metrics(
                exc.__class__.__name__, time.time() - start_time
            )
            return response

    def _dispatch_log_error_metrics(
        self, error_type: str, processing_time: float
    ) -> None:
        """Delega a métrica síncrona para uma thread isolada evitando o bloqueio do Event Loop."""

        async def _async_record() -> None:
            await asyncio.to_thread(
                self._log_error_metrics_sync, error_type, processing_time
            )

        fire_and_forget(_async_record(), logger=logger, name=f"metric_{error_type}")

    def _log_error_metrics_sync(self, error_type: str, processing_time: float) -> None:
        """Execução real do registro síncrono da métrica."""
        try:
            from resync.core.metrics import runtime_metrics

            runtime_metrics.record_error(error_type, processing_time)
        except ImportError:
            logger.debug("Could not record error metrics for %s", error_type)

    async def _handle_validation_error(
        self, request: Request, exc: RequestValidationError, correlation_id: str
    ) -> JSONResponse:
        """Handle FastAPI validation errors."""
        logger.info(
            "Validation error for %s %s: %s", request.method, request.url.path, exc
        )
        builder = ErrorResponseBuilder()
        builder.with_correlation_id(correlation_id)
        builder.with_request_context(request)
        validation_errors = extract_validation_errors(exc)
        error_response = builder.build_validation_error(validation_errors)
        log_error_response(error_response, exc)
        return create_json_response_from_error(error_response)

    async def _handle_resync_exception(
        self, request: Request, exc: ResyncException, correlation_id: str
    ) -> JSONResponse:
        """Handle custom Resync exceptions."""
        logger.error(
            "Resync exception for %s %s: %s", request.method, request.url.path, exc
        )
        error_response = create_error_response_from_exception(
            exc, request, correlation_id
        )
        log_error_response(error_response, exc)
        return create_json_response_from_error(error_response)

    async def _handle_generic_exception(
        self, request: Request, exc: Exception, correlation_id: str
    ) -> JSONResponse:
        """Handle generic exceptions."""
        logger.error(
            f"Unhandled exception for {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        error_response = create_error_response_from_exception(
            exc, request, correlation_id
        )
        log_error_response(error_response, exc)
        return create_json_response_from_error(error_response)


def add_global_exception_handler(app: Any) -> None:
    """Add the global exception handler middleware to a FastAPI application."""
    app.add_middleware(GlobalExceptionHandlerMiddleware)
    logger.info("Global exception handler middleware added")


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Specific handler for validation errors."""
    correlation_id = getattr(request.state, "correlation_id", generate_correlation_id())
    logger.info("Validation error for %s %s: %s", request.method, request.url.path, exc)
    builder = ErrorResponseBuilder()
    builder.with_correlation_id(correlation_id)
    builder.with_request_context(request)
    validation_errors = extract_validation_errors(exc)
    error_response = builder.build_validation_error(validation_errors)
    log_error_response(error_response, exc)
    return create_json_response_from_error(error_response)


async def http_exception_handler(request: Request, exc: Any) -> JSONResponse:
    """Handler for HTTP exceptions from FastAPI."""
    from fastapi import HTTPException

    if not isinstance(exc, HTTPException):
        correlation_id = getattr(
            request.state, "correlation_id", generate_correlation_id()
        )
        error_response = create_error_response_from_exception(
            exc, request, correlation_id
        )
        return create_json_response_from_error(error_response)
    correlation_id = getattr(request.state, "correlation_id", generate_correlation_id())
    logger.info(
        f"HTTP exception {exc.status_code} for {request.method} {request.url.path}: {exc.detail}"
    )
    builder = ErrorResponseBuilder()
    builder.with_correlation_id(correlation_id)
    builder.with_request_context(request)
    if exc.status_code == 401:
        error_response = builder.build_authentication_error(
            "unauthorized", details={"detail": exc.detail}
        )
    elif exc.status_code == 403:
        error_response = builder.build_authorization_error(
            "forbidden", details={"detail": exc.detail}
        )
    elif exc.status_code == 404:
        error_response = builder.build_business_logic_error(
            "resource_not_found", resource="Resource"
        )
    elif exc.status_code == 429:
        error_response = builder.build_rate_limit_error(100, "minute")
    else:
        error_response = builder.build_system_error(
            "internal_server_error", details={"detail": exc.detail}
        )
    log_error_response(error_response, exc)
    return create_json_response_from_error(error_response)


def register_exception_handlers(app: Any) -> None:
    """Register all exception handlers with the FastAPI application."""
    from fastapi import HTTPException
    from resync.core.exceptions import ResyncException

    add_global_exception_handler(app)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ResyncException, resync_exception_handler)
    logger.info("All exception handlers registered")


async def resync_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle custom Resync exceptions."""
    correlation_id = getattr(request.state, "correlation_id", generate_correlation_id())
    logger.error(
        "Resync exception for %s %s: %s", request.method, request.url.path, exc
    )
    error_response = create_error_response_from_exception(exc, request, correlation_id)
    log_error_response(error_response, exc)
    return create_json_response_from_error(error_response)


class ErrorContext:
    """Context manager for managing error context including correlation IDs."""

    def __init__(self, request: Request | None = None):
        self.request = request
        self.correlation_id = None

    def __enter__(self):
        """Enter the context and set up correlation ID."""
        if self.request:
            self.correlation_id = getattr(
                self.request.state, "correlation_id", generate_correlation_id()
            )
        else:
            self.correlation_id = generate_correlation_id()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and handle any exceptions."""
        if exc_val:
            logger.error(
                f"Exception in context {self.correlation_id}: {exc_val}", exc_info=True
            )
        return False


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request, generating one if not present."""
    return getattr(request.state, "correlation_id", generate_correlation_id())
