"""Dependências compartilhadas para endpoints FastAPI.

Este módulo fornece funções de dependência para injeção em endpoints,
incluindo gerenciamento de idempotência, autenticação, e obtenção de IDs de contexto.
"""

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from resync.core.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from resync.core.idempotency.manager import IdempotencyManager
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Legacy fallback (non-HTTP contexts). The canonical HTTP path uses app.state.
_idempotency_manager: IdempotencyManager | None = None


# Rate limit configuration (module-level constants)
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

# ============================================================================
# IDEMPOTENCY DEPENDENCIES
# ============================================================================


def get_idempotency_manager(request: Request) -> IdempotencyManager:
    """Obtem IdempotencyManager do app.state.

    Este provider e *canonico* no caminho HTTP. O objeto e inicializado no lifespan
    e exposto via request.app.state.
    """
    from resync.core.wiring import STATE_IDEMPOTENCY_MANAGER
    try:
        manager: IdempotencyManager = getattr(request.app.state, STATE_IDEMPOTENCY_MANAGER)
        # If Redis is disabled/unavailable, the manager may be present but
        # non-functional. Fail with a clear error instead of a NoneType crash.
        if getattr(manager, "redis", None) is None:
            raise ServiceUnavailableError(
                "Idempotency is unavailable because Redis is disabled or not connected."
            )
        return manager
    except AttributeError as e:
        raise ServiceUnavailableError(
            "IdempotencyManager not initialized. Ensure lifespan init_domain_singletons ran."
        ) from e


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
        ValidationError: Se key não foi fornecida
    """
    if not x_idempotency_key:
        raise ValidationError(
            message="Idempotency key is required for this operation",
            details={
                "header": "X-Idempotency-Key",
                "hint": "Include X-Idempotency-Key header with a unique UUID",
            },
        )

    # Validar formato (deve ser UUID v4)
    import re

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    if not uuid_pattern.match(x_idempotency_key):
        raise ValidationError(
            message="Invalid idempotency key format",
            details={
                "header": "X-Idempotency-Key",
                "expected": "UUID v4 format",
                "received": x_idempotency_key,
            },
        )

    return x_idempotency_key


async def initialize_idempotency_manager(redis_client):
    """
    Initialize the global idempotency manager with Redis client.

    Args:
        redis_client: Redis async client for persistence
    """
    global _idempotency_manager
    try:
        # Initialize the global manager with the new refactored structure
        manager = IdempotencyManager(redis_client)
        # Store globally for dependency injection
        _idempotency_manager = manager

        logger.info("idempotency_manager_initialized", redis_available=True)

    except Exception as e:
        logger.error(
            "idempotency_manager_initialization_failed",
            error=str(e),
            redis_available=False,
        )
        # Create in-memory fallback
        # Note: In production, this should not be used
        # For now, we'll just log the error and continue


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
    """
    if not credentials:
        return None

    try:
        # Import security module for JWT validation
        from resync.api.core.security import verify_token

        token = credentials.credentials
        payload = verify_token(token)

        if not payload:
            return None

        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username", payload.get("sub")),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", []),
        }
    except Exception:
        return None


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


# ============================================================================
# RATE LIMITING DEPENDENCIES (v5.9.4 - Redis-based with TTL)
# ============================================================================

# Redis client for rate limiting (initialized lazily)
_rate_limit_redis: "Any | None" = None


async def _get_rate_limit_redis():
    """Get or create Redis client for rate limiting."""
    global _rate_limit_redis
    if _rate_limit_redis is None:
        try:
            import redis.asyncio as redis

            from resync.settings import settings

            _rate_limit_redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception as e:
            logger.warning("rate_limit_redis_unavailable", error=str(e))
            return None
    return _rate_limit_redis


async def check_rate_limit(request: Request) -> None:
    """Verifica rate limit usando Redis com sliding window e TTL automático.

    v5.9.4: Migrado de memória RAM para Redis para evitar memory leak
    em cenários de DDoS com IP spoofing. Chaves expiram automaticamente
    via TTL, eliminando necessidade de garbage collection manual.

    Args:
        request: Request object

    Raises:
        RateLimitError: Se o limite de taxa for excedido.
    """
    import time

    client_ip = request.client.host if request.client else "unknown"
    redis_client = await _get_rate_limit_redis()

    # Fallback para rate limiting básico se Redis indisponível
    if redis_client is None:
        await _check_rate_limit_fallback(request)
        return

    try:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        key = f"ratelimit:{client_ip}"

        # Usar pipeline para atomicidade
        async with redis_client.pipeline(transaction=True) as pipe:
            # Remove timestamps antigos (fora da janela)
            await pipe.zremrangebyscore(key, 0, window_start)
            # Conta requests na janela atual
            await pipe.zcard(key)
            # Adiciona timestamp atual
            await pipe.zadd(key, {str(now): now})
            # Define TTL para limpeza automática (evita memory leak)
            await pipe.expire(key, RATE_LIMIT_WINDOW + 10)
            results = await pipe.execute()

        request_count = results[1]  # Resultado do ZCARD

        if request_count >= RATE_LIMIT_REQUESTS:
            # Calcular tempo até próxima janela disponível
            async with redis_client.pipeline() as pipe:
                await pipe.zrange(key, 0, 0, withscores=True)
                oldest = await pipe.execute()

            retry_after = RATE_LIMIT_WINDOW
            if oldest and oldest[0]:
                oldest_ts = oldest[0][0][1]
                retry_after = max(1, int(oldest_ts + RATE_LIMIT_WINDOW - now))

            raise RateLimitError(
                message="Rate limit exceeded",
                details={
                    "retry_after": retry_after,
                    "limit": RATE_LIMIT_REQUESTS,
                    "window": RATE_LIMIT_WINDOW,
                },
            )

    except RateLimitError:
        raise
    except Exception as e:
        # Em caso de erro Redis, permite requisição (fail-open)
        # mas loga para monitoramento
        logger.warning("rate_limit_redis_error", error=str(e), client_ip=client_ip)


async def _check_rate_limit_fallback(request: Request) -> None:
    """Fallback rate limiting com LRU cache limitado (proteção contra OOM).

    Usado apenas quando Redis está indisponível. Implementa cache com
    tamanho máximo fixo para evitar memory leak.
    """
    from collections import OrderedDict
    from datetime import datetime, timedelta, timezone

    # LRU cache com tamanho máximo (evita OOM)
    MAX_TRACKED_IPS = 10000

    if not hasattr(_check_rate_limit_fallback, "_store"):
        _check_rate_limit_fallback._store = OrderedDict()

    store = _check_rate_limit_fallback._store
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

    # Limpar IPs mais antigos se exceder limite (LRU eviction)
    while len(store) >= MAX_TRACKED_IPS:
        store.popitem(last=False)

    # Limpar timestamps antigos do IP atual
    if client_ip in store:
        store[client_ip] = [ts for ts in store[client_ip] if ts > window_start]
        # Move para o final (mais recente)
        store.move_to_end(client_ip)
    else:
        store[client_ip] = []

    # Verificar limite
    if len(store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise RateLimitError(
            message="Rate limit exceeded",
            details={
                "retry_after": RATE_LIMIT_WINDOW,
                "limit": RATE_LIMIT_REQUESTS,
            },
        )

    # Registrar requisição
    store[client_ip].append(now)