"""Dependências compartilhadas para endpoints FastAPI.

Este módulo fornece funções de dependência para injeção em endpoints,
incluindo gerenciamento de idempotência, autenticação, e obtenção de IDs de contexto.
"""

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from resync.core.exceptions import (
    AuthenticationError,
    ServiceUnavailableError,
    ValidationError,
)
from resync.core.idempotency.manager import IdempotencyManager
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Legacy fallback (non-HTTP contexts). The canonical HTTP path uses app.state.

# Rate limit configuration (module-level constants)
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

# ============================================================================
# IDEMPOTENCY DEPENDENCIES
# ============================================================================

def get_idempotency_manager(request: Request) -> IdempotencyManager:
    """Get IdempotencyManager from the canonical enterprise state.

    The HTTP path uses a single typed container stored on ``app.state.enterprise_state``.
    This dependency fails fast with 503 when Redis is unavailable, rather than returning
    a degraded object with a mismatched interface (which would crash later).
    """
    from resync.core.types.app_state import enterprise_state_from_request

    st = enterprise_state_from_request(request)
    if not getattr(st, "redis_available", False):
        raise ServiceUnavailableError(
            "Idempotency is unavailable because Redis is disabled or not connected."
        )
    return st.idempotency_manager

async def get_idempotency_key(
    x_idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
) -> str | None:
    """Extrai idempotency key do header.

    Args:
        x_idempotency_key: Header X-Idempotency-Key

    Returns:
        Idempotency key ou None
    """
    return x_idempotency_key

async def require_idempotency_key(
    x_idempotency_key: str = Header(..., alias="X-Idempotency-Key"),
) -> str:
    """Extrai e valida idempotency key obrigatória.

    Args:
        x_idempotency_key: Header X-Idempotency-Key

    Returns:
        Idempotency key

    Raises:
        ValidationError: Se key não foi fornecida ou é inválida
    """
    if not x_idempotency_key:
        raise ValidationError(
            message="Idempotency key is required for this operation",
            details={
                "header": "X-Idempotency-Key",
                "hint": "Include X-Idempotency-Key header with a unique UUID",
            },
        )

    # BUG FIX: Use native uuid.UUID() module instead of fragile regex
    # This accepts any valid UUID (v1, v4, v7, etc.) instead of just v4
    import uuid

    try:
        uuid_obj = uuid.UUID(x_idempotency_key)
    except ValueError as e:
        raise ValidationError(
            message="Invalid idempotency key format",
            details={
                "header": "X-Idempotency-Key",
                "expected": "Valid UUID (any version)",
                "received": x_idempotency_key,
            },
        ) from e

    return str(uuid_obj)

# ============================================================================
# CORRELATION ID DEPENDENCIES
# ============================================================================

async def get_correlation_id(
    x_correlation_id: str | None = Header(None, alias="X-Correlation-ID"),
) -> str:
    """Obtém ou gera correlation ID.

    Args:
        x_correlation_id: Header X-Correlation-ID

    Returns:
        Correlation ID
    """
    if x_correlation_id:
        return x_correlation_id

    # Tentar obter do contexto
    from resync.core.context import get_correlation_id as get_ctx_correlation_id

    ctx_id = get_ctx_correlation_id()
    if ctx_id:
        return ctx_id

    # Gerar novo
    import uuid

    return str(uuid.uuid4())

# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    """Obtém usuário atual a partir do token JWT.

    Args:
        credentials: Credenciais de autenticação injetadas pelo FastAPI.

    Returns:
        Um dicionário representando o usuário ou None se não autenticado.

    Raises:
        AuthenticationError: Se houver erro de infraestrutura (Redis, parsing, etc)
    """
    if not credentials:
        return None

    try:
        # Import security module for JWT validation
        from resync.api.core.security import verify_token_async

        token = credentials.credentials
        payload = await verify_token_async(token)

        if not payload:
            return None

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username", payload.get("sub")),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", []),
        }
    except AuthenticationError:
        # Re-raise auth errors - these are expected "not authenticated" cases
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # BUG FIX: Log and re-raise infrastructure errors instead of silently swallowing them
        # This prevents masking serious issues like Redis unavailability or token parsing errors
        logger.error("Authentication infrastructure error", error=str(e), exc_info=True)
        # Re-raise as authentication error to inform the client appropriately
        raise AuthenticationError(
            message="Authentication service unavailable",
            details={"code": "AUTH_INFRA_FAILURE"},
        ) from e

async def require_authentication(
    user: dict | None = Depends(get_current_user),
) -> dict:
    """Garante que um usuário esteja autenticado.

    Args:
        user: O usuário obtido da dependência `get_current_user`.

    Returns:
        Dados do usuário

    Raises:
        AuthenticationError: Se o usuário não estiver autenticado.
    """
    if not user:
        raise AuthenticationError(
            message="Authentication required",
            details={"headers": {"WWW-Authenticate": "Bearer"}},
        )

    return user
