"""Exception handlers globais para FastAPI.

Este módulo implementa handlers para todas as exceções da aplicação,
convertendo-as em respostas HTTP padronizadas seguindo RFC 7807.
"""

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from resync.api.models.responses import (
    ValidationErrorDetail,
    create_problem_detail,
    create_validation_problem_detail,
)
from resync.core.context import get_correlation_id
from resync.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BaseAppException,
    InternalError,
    RateLimitError,
    ResourceConflictError,
    ResourceNotFoundError,
    ResyncException,
    ValidationError,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

async def base_app_exception_handler(
    request: Request, exc: BaseAppException
) -> JSONResponse:
    """Handler para exceções da aplicação (BaseAppException).

    Args:
        request: Requisição HTTP
        exc: Exceção da aplicação

    Returns:
        JSONResponse com problema detalhado
    """
    # Logar exceção
    logger.error(
        "application_exception",
        message=exc.message,
        error_code=exc.error_code.value,
        status_code=exc.status_code,
        correlation_id=exc.correlation_id,
        path=request.url.path,
        method=request.method,
        exc_info=exc.original_exception is not None,
    )

    # Criar problem detail
    problem = create_problem_detail(
        type_uri=f"https://api.example.com/errors/{exc.error_code.value.lower()}",
        title=exc.error_code.name.replace("_", " ").title(),
        status=exc.status_code,
        detail=exc.message,
        instance=str(request.url.path),
    )

    # Adicionar headers específicos
    headers = {}

    # Rate limiting - FIX: Safe access to details attribute
    details = getattr(exc, "details", None) or {}
    if isinstance(exc, RateLimitError) and details.get("retry_after"):
        headers["Retry-After"] = str(details["retry_after"])

    # Correlation ID
    if exc.correlation_id:
        headers["X-Correlation-ID"] = exc.correlation_id

    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        headers=headers,
    )

async def resync_exception_handler(
    request: Request, exc: ResyncException
) -> JSONResponse:
    """Handler para exceções ResyncException.

    Args:
        request: Requisição HTTP
        exc: Exceção ResyncException

    Returns:
        JSONResponse com problema detalhado
    """
    correlation_id = get_correlation_id()

    # Logar exceção
    logger.error(
        "resync_exception",
        message=exc.message,
        error_code=exc.error_code,
        severity=exc.severity,
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        exc_info=exc.original_exception is not None,
    )

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        severity_map = {
            "low": 400,
            "medium": 400,
            "high": 500,
            "critical": 500,
        }
        status_code = severity_map.get(exc.severity.lower() if exc.severity else "", 500)

    # Criar problem detail
    problem = create_problem_detail(
        type_uri=f"https://api.example.com/errors/{exc.error_code.lower()}",
        title=exc.error_code.replace("_", " ").title(),
        status=status_code,
        detail=getattr(exc, "user_friendly_message", None) or exc.message,
        instance=str(request.url.path),
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(exclude_none=True),
        headers=headers,
    )

async def validation_exception_handler(
    request: Request, exc: RequestValidationError | PydanticValidationError
) -> JSONResponse:
    """Handler para erros de validação do Pydantic/FastAPI.

    Args:
        request: Requisição HTTP
        exc: Exceção de validação

    Returns:
        JSONResponse com erros de validação detalhados
    """
    correlation_id = get_correlation_id()

    # Logar erro
    logger.warning(
        "Validation error",
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        errors=exc.errors(),
    )

    # Converter erros do Pydantic para nosso formato
    validation_errors = []

    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])

        validation_errors.append(
            ValidationErrorDetail(
                field=field,
                message=error["msg"],
                code=error["type"],
                value=error.get("input"),
            )
        )

    # Criar problem detail
    problem = create_validation_problem_detail(
        title="Validation Error",
        errors=validation_errors,
        detail=f"Validation failed with {len(validation_errors)} error(s)",
        instance=str(request.url.path),
    )

    headers = {}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=problem.model_dump(exclude_none=True),
        headers=headers,
    )

async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handler para exceções HTTP do Starlette.

    Args:
        request: Requisição HTTP
        exc: Exceção HTTP

    Returns:
        JSONResponse com problema detalhado
    """
    correlation_id = get_correlation_id()

    # Logar exceção
    logger.warning(
        f"HTTP exception: {exc.detail}",
        status_code=exc.status_code,
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    )

    # Map HTTP errors to Resync domain exceptions, and build the response
    # from the mapped instance (instead of constructing and discarding it).
    #
    # IMPORTANT: preserve client-facing detail for 4xx responses. For 5xx we
    # keep a generic message to avoid leaking internals.
    safe_message = (
        str(exc.detail)
        if exc.status_code < 500
        else "Internal server error. Check server logs for details."
    )

    mapped: BaseAppException
    if exc.status_code == 404:
        mapped = ResourceNotFoundError(
            message=safe_message, correlation_id=correlation_id
        )
    elif exc.status_code == 401:
        mapped = AuthenticationError(
            message=safe_message, correlation_id=correlation_id
        )
    elif exc.status_code == 403:
        mapped = AuthorizationError(message=safe_message, correlation_id=correlation_id)
    elif exc.status_code == 409:
        mapped = ResourceConflictError(
            message=safe_message, correlation_id=correlation_id
        )
    elif exc.status_code == 429:
        mapped = RateLimitError(message=safe_message, correlation_id=correlation_id)
    elif exc.status_code >= 500:
        mapped = InternalError(
            message=safe_message,
            correlation_id=correlation_id,
            details={
                "original_detail": str(exc.detail),
                "http_status": exc.status_code,
            },
            original_exception=exc,
        )
    else:
        mapped = ValidationError(message=safe_message, correlation_id=correlation_id)

    return await base_app_exception_handler(request, mapped)

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler para exceções não tratadas.

    Args:
        request: Requisição HTTP
        exc: Exceção não tratada

    Returns:
        JSONResponse com erro interno
    """
    correlation_id = get_correlation_id()

    # Logar exceção com stack trace
    logger.critical(
        "unhandled_exception",
        exception_msg=str(exc),
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        exception_type=type(exc).__name__,
        exc_info=True,
    )

    import os
    include_details = os.getenv("ENVIRONMENT", "development") != "production"
    details = (
        {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        if include_details
        else {"exception_type": type(exc).__name__}
    )

    internal = InternalError(
        message="An unexpected error occurred",
        details=details,
        correlation_id=correlation_id,
        original_exception=exc,
    )
    return await base_app_exception_handler(request, internal)

# ============================================================================
# REGISTRATION HELPER
# ============================================================================

def register_exception_handlers(app) -> None:
    """Registra todos os exception handlers na aplicação FastAPI.

    Args:
        app: Instância da aplicação FastAPI
    """
    # Exceções da aplicação (BaseAppException)
    app.add_exception_handler(BaseAppException, base_app_exception_handler)

    # Exceções ResyncException (enhanced exceptions)
    app.add_exception_handler(ResyncException, resync_exception_handler)

    # Exceções de validação
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, validation_exception_handler)

    # Exceções HTTP do Starlette
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Exceções não tratadas
    app.add_exception_handler(Exception, unhandled_exception_handler)

    logger.info("Exception handlers registered successfully")

__all__ = [
    "base_app_exception_handler",
    "resync_exception_handler",
    "validation_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",
    "register_exception_handlers",
]
