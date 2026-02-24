# pylint: disable=all
"""Validation middleware for automatic request validation."""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .common import ValidationErrorResponse, ValidationSeverity

logger = structlog.get_logger(__name__)


class ValidationMiddleware:
    """Middleware for automatic request validation using Pydantic models."""

    def __init__(
        self,
        validation_models: dict[str, type[BaseModel]] | None = None,
        strict_mode: bool = True,
        sanitize_input: bool = True,
        enable_logging: bool = True,
        custom_validators: dict[str, Callable] | None = None,
        error_handler: Callable | None = None,
    ):
        """
        Initialize validation middleware.

        Args:
            validation_models: Mapping of endpoint paths to validation models
            strict_mode: Whether to enforce strict validation (reject unknown fields)
            sanitize_input: Whether to sanitize input data
            enable_logging: Whether to enable validation logging
            custom_validators: Custom validation functions
            error_handler: Custom error handler function
        """
        self.validation_models = validation_models or {}
        self.strict_mode = strict_mode
        self.sanitize_input = sanitize_input
        self.enable_logging = enable_logging
        self.custom_validators = custom_validators or {}
        self.error_handler = error_handler

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through validation middleware.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain

        Returns:
            Response object
        """
        try:
            # Skip validation for certain paths
            if self._should_skip_validation(request):
                return await call_next(request)

            # Get validation model for this endpoint
            validation_model = self._get_validation_model(request)
            if not validation_model:
                return await call_next(request)

            # Extract and validate request data
            validated_data = await self._validate_request(request, validation_model)
            if validated_data is not None:
                # Store validated data in request state for later use
                request.state.validated_data = validated_data

            # Process request
            response = await call_next(request)

            # Log validation success if enabled
            if self.enable_logging:
                self._log_validation_success(request, validated_data)

            return response

        except ValidationError as e:
            return await self._handle_validation_error(request, e)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("validation_middleware_error", error=str(e), exc_info=True)
            return await self._handle_internal_error(request, e)

    def _should_skip_validation(self, request: Request) -> bool:
        """
        Determine if validation should be skipped for this request.

        Args:
            request: FastAPI request object

        Returns:
            True if validation should be skipped
        """
        # Skip for health check endpoints
        if request.url.path in ["/health", "/health/"]:
            return True

        # Skip for static files
        if request.url.path.startswith(("/static", "/assets", "/favicon.ico")):
            return True

        # Skip for documentation endpoints
        if request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            return True

        # Skip for OPTIONS requests (CORS preflight)
        return request.method == "OPTIONS"

    def _get_validation_model(self, request: Request) -> type[BaseModel] | None:
        """
        Get validation model for the request endpoint.

        Args:
            request: FastAPI request object

        Returns:
            Validation model class or None
        """
        # Direct path matching
        direct_match = self.validation_models.get(request.url.path)
        if direct_match:
            return direct_match

        # Pattern matching for parameterized paths
        path = request.url.path
        for pattern, model in self.validation_models.items():
            if self._path_matches_pattern(path, pattern):
                return model

        return None

    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """
        Check if path matches a parameterized pattern.

        Args:
            path: Actual request path
            pattern: Pattern with parameters (e.g., /agents/{agent_id})

        Returns:
            True if path matches pattern
        """
        # Simple pattern matching - can be enhanced with regex
        pattern_parts = pattern.split("/")
        path_parts = path.split("/")

        if len(pattern_parts) != len(path_parts):
            return False

        for pattern_part, path_part in zip(pattern_parts, path_parts, strict=False):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                continue  # Parameter placeholder
            if pattern_part != path_part:
                return False

        return True

    async def _validate_request(
        self, request: Request, validation_model: type[BaseModel]
    ) -> dict[str, Any] | None:
        """
        Validate request data against the validation model.

        Args:
            request: FastAPI request object
            validation_model: Pydantic model class

        Returns:
            Validated data or None if no data to validate
        """
        # Determine validation approach based on request method and content type
        request.headers.get("content-type", "")

        if request.method in ["POST", "PUT", "PATCH"]:
            return await self._validate_body_data(request, validation_model)
        if request.method in ["GET", "DELETE"]:
            return self._validate_query_params(request, validation_model)
        return None

    async def _validate_body_data(
        self, request: Request, validation_model: type[BaseModel]
    ) -> dict[str, Any] | None:
        """
        Validate request body data.

        Args:
            request: FastAPI request object
            validation_model: Pydantic model class

        Returns:
            Validated data
        """
        try:
            # Read request body
            body = await request.body()
            if not body:
                return None

            # Parse based on content type
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                data = json.loads(body.decode("utf-8"))
            elif "application/x-www-form-urlencoded" in content_type:
                form_data = await request.form()
                data = dict(form_data)
            elif "multipart/form-data" in content_type:
                form_data = await request.form()
                data = dict(form_data)
            else:
                # Try to parse as JSON by default
                try:
                    data = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    return None

            # Apply custom validators if any
            data = self._apply_custom_validators(data, request)

            # Validate with Pydantic model
            validated_model = validation_model(**data)

            return validated_model.model_dump()

        except json.JSONDecodeError as e:
            raise ValidationError.from_exception_data(
                "body", cast(Any, [str(e)])
            ) from e
        except ValidationError:
            raise
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("body_validation_error", error=str(e), exc_info=True)
            raise ValidationError.from_exception_data(
                "body", cast(Any, [str(e)])
            ) from e

    def _validate_query_params(
        self, request: Request, validation_model: type[BaseModel]
    ) -> dict[str, Any] | None:
        """
        Validate query parameters.

        Args:
            request: FastAPI request object
            validation_model: Pydantic model class

        Returns:
            Validated data
        """
        try:
            # Extract query parameters
            query_params = dict(request.query_params)

            if not query_params:
                return None

            # Convert parameter values
            converted_params: dict[str, str | list[str]] = {}
            for key, value in query_params.items():
                # Handle list parameters
                if key.endswith("[]"):
                    key = key[:-2]
                    if key not in converted_params:
                        converted_params[key] = []
                    converted_params[key].append(value)
                else:
                    converted_params[key] = value

            # Apply custom validators if any
            converted_params = self._apply_custom_validators(converted_params, request)

            # Validate with Pydantic model
            validated_model = validation_model(**converted_params)

            return validated_model.model_dump()

        except ValidationError:
            raise
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "query_parameter_validation_error", error=str(e), exc_info=True
            )
            raise ValidationError.from_exception_data(
                "query_params", cast(Any, [str(e)])
            ) from e

    def _apply_custom_validators(
        self, data: dict[str, Any], request: Request
    ) -> dict[str, Any]:
        """
        Apply custom validators to the data.

        Args:
            data: Request data
            request: FastAPI request object

        Returns:
            Processed data
        """
        # Apply custom validators based on the endpoint
        for validator_name, validator_func in self.custom_validators.items():
            try:
                data = validator_func(data, request)
            except Exception as e:
                logger.warning(
                    "custom_validator_failed",
                    validator_name=validator_name,
                    error=str(e),
                )

        return data

    def _log_validation_success(
        self, request: Request, validated_data: dict[str, Any] | None
    ) -> None:
        """
        Log successful validation.

        Args:
            request: FastAPI request object
            validated_data: Validated request data
        """
        logger.info(
            "validation_successful",
            method=request.method,
            path=request.url.path,
            data_validated=validated_data is not None,
        )

    async def _handle_validation_error(
        self, request: Request, validation_error: ValidationError
    ) -> JSONResponse:
        """
        Handle validation errors.

        Args:
            request: FastAPI request object
            validation_error: Pydantic validation error

        Returns:
            JSON response with validation error details
        """
        # Extract error details
        error_details = []
        for error in validation_error.errors():
            error_detail = {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "severity": ValidationSeverity.ERROR.value,
            }
            error_details.append(error_detail)

        # Create error response
        error_response = ValidationErrorResponse(
            error="Validation failed",
            message="Request validation failed. Please check the provided data.",
            details=error_details,
            severity=ValidationSeverity.ERROR,
            timestamp=datetime.now(timezone.utc),
            path=request.url.path,
            method=request.method,
        )

        # Log validation failure
        if self.enable_logging:
            logger.warning(
                "validation_failed",
                method=request.method,
                path=request.url.path,
                error_count=len(error_details),
            )

        # Use custom error handler if provided
        if self.error_handler:
            try:
                return await self.error_handler(request, error_response)
            except Exception as e:
                logger.error("Custom error handler failed: %s", str(e), exc_info=True)

        # Return standard error response
        return JSONResponse(status_code=422, content=error_response.model_dump())

    async def _handle_internal_error(
        self, request: Request, error: Exception
    ) -> JSONResponse:
        """
        Handle internal validation errors.

        Args:
            request: FastAPI request object
            error: Internal error

        Returns:
            JSON response with error details
        """
        logger.error("Internal validation error: %s", str(error), exc_info=True)

        error_response = ValidationErrorResponse(
            error="Internal validation error",
            message="An internal error occurred during validation.",
            details=[
                {"message": str(error), "severity": ValidationSeverity.ERROR.value}
            ],
            severity=ValidationSeverity.ERROR,
            timestamp=datetime.now(timezone.utc),
            path=request.url.path,
            method=request.method,
        )

        return JSONResponse(status_code=500, content=error_response.model_dump())


class ValidationConfig:
    """Configuration for validation middleware."""

    def __init__(
        self,
        validation_models: dict[str, type[BaseModel]] | None = None,
        strict_mode: bool = True,
        sanitize_input: bool = True,
        enable_logging: bool = True,
        custom_validators: dict[str, Callable] | None = None,
        error_handler: Callable | None = None,
        skip_paths: list[str] | None = None,
        rate_limit_validation: bool = False,
        max_validation_errors: int = 50,
    ):
        """
        Initialize validation configuration.

        Args:
            validation_models: Mapping of endpoint paths to validation models
            strict_mode: Whether to enforce strict validation
            sanitize_input: Whether to sanitize input data
            enable_logging: Whether to enable validation logging
            custom_validators: Custom validation functions
            error_handler: Custom error handler function
            skip_paths: Paths to skip validation for
            rate_limit_validation: Whether to rate limit validation requests
            max_validation_errors: Maximum number of validation errors to return
        """
        self.validation_models = validation_models or {}
        self.strict_mode = strict_mode
        self.sanitize_input = sanitize_input
        self.enable_logging = enable_logging
        self.custom_validators = custom_validators or {}
        self.error_handler = error_handler
        self.skip_paths = skip_paths or []
        self.rate_limit_validation = rate_limit_validation
        self.max_validation_errors = max_validation_errors


def create_validation_middleware(config: ValidationConfig) -> ValidationMiddleware:
    """
    Create validation middleware from configuration.

    Args:
        config: Validation configuration

    Returns:
        ValidationMiddleware instance
    """
    return ValidationMiddleware(
        validation_models=config.validation_models,
        strict_mode=config.strict_mode,
        sanitize_input=config.sanitize_input,
        enable_logging=config.enable_logging,
        custom_validators=config.custom_validators,
        error_handler=config.error_handler,
    )


# Common validation utilities
async def validate_json_body(
    request: Request, model: type[BaseModel]
) -> dict[str, Any]:
    """
    Validate JSON request body against a Pydantic model.

    Args:
        request: FastAPI request object
        model: Pydantic model class

    Returns:
        Validated data

    Raises:
        ValidationError: If validation fails
    """
    try:
        body = await request.body()
        data = (
            json.loads(body.decode("utf-8"))
            if isinstance(body, bytes)
            else json.loads(body)
        )

        validated_model = model(**data)
        return validated_model.model_dump()
    except json.JSONDecodeError as e:
            raise ValidationError.from_exception_data(
                "body", cast(Any, [str(e)])
            ) from e
    except ValidationError:
        raise


def validate_query_params(request: Request, model: type[BaseModel]) -> dict[str, Any]:
    """
    Validate query parameters against a Pydantic model.

    Args:
        request: FastAPI request object
        model: Pydantic model class

    Returns:
        Validated data

    Raises:
        ValidationError: If validation fails
    """
    query_params = dict(request.query_params)

    # Convert parameter values
    converted_params: dict[str, str | list[str]] = {}
    for key, value in query_params.items():
        if key.endswith("[]"):
            key = key[:-2]
            if key not in converted_params:
                converted_params[key] = []
            converted_params[key].append(value)
        else:
            converted_params[key] = value

    validated_model = model(**converted_params)
    return validated_model.model_dump()


# Export for use in FastAPI applications
__all__ = [
    "ValidationMiddleware",
    "ValidationConfig",
    "create_validation_middleware",
    "validate_json_body",
    "validate_query_params",
]
