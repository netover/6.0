"""
Middleware de Idempotency Keys para FastAPI

Este middleware implementa validação e processamento de chaves de idempotência
para endpoints FastAPI, garantindo que operações críticas não sejam executadas
múltiplas vezes.

Características:
- Validação automática de chaves de idempotência
- Bloqueio de processamento concorrente
- Cache automático de respostas
- Integração com sistema de logging estruturado
- Headers customizáveis

Author: Resync Team
Date: October 2025
"""

import json
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from resync.core.context import get_correlation_id
from resync.core.idempotency import IdempotencyConflictError, IdempotencyManager
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """
    Middleware FastAPI para processamento de idempotency keys

    Intercepta requisições HTTP, valida chaves de idempotência e
    gerencia cache de respostas para prevenir execução duplicada
    de operações críticas.
    """

    def __init__(
        self,
        app: Callable,
        idempotency_manager: IdempotencyManager,
        idempotency_header: str = "Idempotency-Key",
        exclude_paths: set[str] | None = None,
        exclude_methods: set[str] | None = None,
    ):
        """
        Inicializa middleware de idempotency

        Args:
            app: Aplicação FastAPI
            idempotency_manager: Gerenciador de idempotency
            idempotency_header: Nome do header para chave de idempotência
            exclude_paths: Caminhos excluídos do middleware
            exclude_methods: Métodos HTTP excluídos do middleware
        """
        super().__init__(app)
        self.idempotency_manager = idempotency_manager
        self.idempotency_header = idempotency_header
        self.exclude_paths = exclude_paths or {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        self.exclude_methods = exclude_methods or {"GET", "HEAD", "OPTIONS"}

        logger.info(
            "Idempotency middleware initialized",
            idempotency_header=idempotency_header,
            exclude_paths=list(self.exclude_paths),
            exclude_methods=list(self.exclude_methods),
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa requisição através do middleware

        Args:
            request: Requisição FastAPI
            call_next: Próximo handler na cadeia

        Returns:
            Resposta HTTP
        """
        # Verificar se deve aplicar idempotency
        if not self._should_apply_idempotency(request):
            return await call_next(request)

        # Extrair chave de idempotência
        idempotency_key = self._extract_idempotency_key(request)
        if not idempotency_key:
            # Se endpoint requer idempotency mas não foi fornecida, erro 400
            if self._requires_idempotency(request):
                raise HTTPException(
                    status_code=400,
                    detail="Idempotency-Key header required for this operation",
                )
            # Para endpoints opcionais, continua sem idempotency
            return await call_next(request)

        correlation_id = get_correlation_id()

        # Redis is required to enforce idempotency. When Redis is explicitly
        # disabled or unavailable, degrade gracefully:
        # - required-idempotency endpoints → 503
        # - optional endpoints → pass-through
        if getattr(self.idempotency_manager, "redis", None) is None:
            if self._requires_idempotency(request):
                raise HTTPException(
                    status_code=503,
                    detail="Idempotency requires Redis, but Redis is disabled or unavailable",
                )
            return await call_next(request)

        logger.debug(
            "Processing request with idempotency key",
            idempotency_key=idempotency_key,
            path=request.url.path,
            method=request.method,
            correlation_id=correlation_id,
        )

        try:
            # Compute request fingerprint once.
            # We must restore the body after reading it because BaseHTTPMiddleware
            # consumes the stream, which would make it unavailable to the endpoint.
            request_data = await self._extract_request_data(request)

            # Restore the body if it was read
            if "body" in request_data and isinstance(
                request_data.get("body"), (dict, str, bytes)
            ):
                body = await request.body()

                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

            cached_response = await self.idempotency_manager.get_cached_response(
                idempotency_key, request_data
            )

            if cached_response:
                logger.info(
                    "Returning cached response for idempotency key",
                    idempotency_key=idempotency_key,
                    status_code=cached_response["status_code"],
                    correlation_id=correlation_id,
                )

                # Retornar resposta cacheada
                return self._create_response_from_cache(cached_response)

            # Atomically mark as in-progress using SET NX EX.  If
            # ``mark_processing`` returns False the key was already held by
            # another concurrent request → 409 Conflict.  This replaces the
            # previous ``is_processing()`` + ``mark_processing()`` two-step
            # which had a TOCTOU race window.
            marked = await self.idempotency_manager.mark_processing(idempotency_key)
            if not marked:
                logger.warning(
                    "Operation already in progress for idempotency key",
                    idempotency_key=idempotency_key,
                    correlation_id=correlation_id,
                )
                raise HTTPException(
                    status_code=409, detail="Operation already in progress"
                )

            try:
                # Executar operação
                response = await call_next(request)

                # Cache da resposta se for bem-sucedida
                if self._should_cache_response(response):
                    await self._cache_response(
                        idempotency_key, response, request, request_data
                    )

                return response

            finally:
                # Sempre limpar marca de processamento
                await self.idempotency_manager.clear_processing(idempotency_key)

        except IdempotencyConflictError as e:
            # Same key, different request payload => 409 Conflict
            logger.warning(
                "idempotency_key_conflict",
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
            )
            raise HTTPException(
                status_code=409,
                detail="Idempotency key was already used with different request data",
            ) from e
        except HTTPException:
            # Re-lançar exceções HTTP sem modificação
            raise
        except Exception as e:
            # Para outras exceções, log e re-lança
            logger.error(
                "Unexpected error in idempotency middleware",
                idempotency_key=idempotency_key,
                error=str(e),
                correlation_id=correlation_id,
            )
            raise

    def _should_apply_idempotency(self, request: Request) -> bool:
        """
        Verifica se deve aplicar idempotency nesta requisição

        Args:
            request: Requisição HTTP

        Returns:
            True se deve aplicar idempotency
        """
        # Excluir caminhos específicos
        if request.url.path in self.exclude_paths:
            return False

        # Excluir métodos específicos
        return request.method not in self.exclude_methods

    def _requires_idempotency(self, request: Request) -> bool:
        """
        Verifica se endpoint requer idempotency obrigatoriamente

        Baseado no método HTTP - operações que modificam estado
        geralmente requerem idempotency.

        Args:
            request: Requisição HTTP

        Returns:
            True se requer idempotency
        """
        # Métodos que modificam estado requerem idempotency
        return request.method in {"POST", "PUT", "PATCH", "DELETE"}

    def _extract_idempotency_key(self, request: Request) -> str | None:
        """
        Extrai chave de idempotência do header

        Args:
            request: Requisição HTTP

        Returns:
            Chave de idempotência ou None
        """
        return request.headers.get(self.idempotency_header)

    async def _extract_request_data(self, request: Request) -> dict:
        """
        Extrai dados relevantes da requisição para hash

        Args:
            request: Requisição HTTP

        Returns:
            Dicionário com dados da requisição
        """
        try:
            # Para requisições com body, incluir no hash
            if request.method in {"POST", "PUT", "PATCH"}:
                # Check Content-Length to avoid reading huge streams into memory
                content_length = request.headers.get("content-length")
                if not content_length or int(content_length) > 1024 * 1024:  # 1MB limit
                    # Skip body for large requests or unknown size (streaming)
                    # This prevents consuming streams or OOM on large uploads
                    return {
                        "method": request.method,
                        "path": str(request.url.path),
                        "query_params": dict(request.query_params),
                        "body_skipped": "large_or_stream",
                    }

                body = await request.body()
                if body:
                    try:
                        json_data = json.loads(body)
                        return {
                            "method": request.method,
                            "path": str(request.url.path),
                            "query_params": dict(request.query_params),
                            "body": json_data,
                        }
                    except json.JSONDecodeError:
                        # Se não for JSON, usar body como string
                        return {
                            "method": request.method,
                            "path": str(request.url.path),
                            "query_params": dict(request.query_params),
                            "body": body.decode("utf-8", errors="ignore"),
                        }

            # Para outros métodos, usar apenas path e query
            return {
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning(
                "Failed to extract request data for idempotency", error=str(e)
            )
            # Em caso de erro, retornar dados mínimos
            return {"method": request.method, "path": str(request.url.path)}

    def _should_cache_response(self, response: Response) -> bool:
        """
        Verifica se deve cachear a resposta

        Args:
            response: Resposta HTTP

        Returns:
            True se deve cachear
        """
        # Só cachear respostas de sucesso
        return 200 <= response.status_code < 300

    async def _cache_response(
        self,
        idempotency_key: str,
        response: Response,
        request: Request,
        request_data: dict | None = None,
    ) -> None:
        """
        Cache da resposta para idempotency

        Args:
            idempotency_key: Chave de idempotência
            response: Resposta HTTP
            request: Requisição original
        """
        try:
            # Extrair dados da resposta
            response_data = await self._extract_response_data(response)

            # Metadata da requisição
            metadata = {
                "method": request.method,
                "path": str(request.url.path),
                "user_agent": request.headers.get("User-Agent"),
                "correlation_id": get_correlation_id(),
            }

            # Cache da resposta
            if request_data is None:
                request_data = await self._extract_request_data(request)

            success = await self.idempotency_manager.cache_response(
                idempotency_key=idempotency_key,
                response_data=response_data,
                status_code=response.status_code,
                request_data=request_data,
                metadata=metadata,
            )

            if success:
                logger.debug(
                    "Response cached for idempotency",
                    idempotency_key=idempotency_key,
                    status_code=response.status_code,
                )
            else:
                logger.warning(
                    "Failed to cache response for idempotency",
                    idempotency_key=idempotency_key,
                )

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "Error caching response for idempotency",
                idempotency_key=idempotency_key,
                error=str(e),
            )

    async def _extract_response_data(self, response: Response) -> dict:
        """
        Extrai dados da resposta para cache

        Args:
            response: Resposta HTTP

        Returns:
            Dados da resposta
        """
        try:
            # Para respostas JSON, extrair dados
            if hasattr(response, "body") and response.body:
                try:
                    # Handle different types of response body
                    if isinstance(response.body, (bytes, bytearray)):
                        return json.loads(response.body.decode("utf-8"))
                    if isinstance(response.body, str):
                        return json.loads(response.body)
                    if hasattr(response.body, "decode"):
                        return json.loads(response.body.decode("utf-8"))
                    # Fallback for other types
                    return json.loads(str(response.body))
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    AttributeError,
                ) as exc:
                    # Body not JSON-decodable (or not available). This is expected in some cases.
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)

            # Para outros tipos, retornar estrutura básica
            return {"message": "Response cached", "status_code": response.status_code}

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("Failed to extract response data", error=str(e))
            return {"cached": True, "status_code": response.status_code}

    def _create_response_from_cache(self, cached: dict) -> Response:
        """
        Cria resposta HTTP a partir do cache

        Args:
            cached: Dados cacheados

        Returns:
            Resposta HTTP
        """
        # Adicionar header indicando que veio do cache
        headers = {"X-Idempotency-Cache": "HIT"}

        # Criar resposta baseada no tipo de dados
        if isinstance(cached["data"], dict):
            return JSONResponse(
                content=cached["data"],
                status_code=cached["status_code"],
                headers=headers,
            )
        # Para outros tipos, criar resposta genérica
        return JSONResponse(
            content={"result": cached["data"]},
            status_code=cached["status_code"],
            headers=headers,
        )


class IdempotencyConfig:
    """
    Configuração para middleware de idempotency
    """

    def __init__(
        self,
        header_name: str = "Idempotency-Key",
        exclude_paths: list[str] | None = None,
        exclude_methods: list[str] | None = None,
        required_for_mutations: bool = True,
    ):
        self.header_name = header_name
        self.exclude_paths = set(
            exclude_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        )
        self.exclude_methods = set(exclude_methods or ["GET", "HEAD", "OPTIONS"])
        self.required_for_mutations = required_for_mutations


def create_idempotency_middleware(
    idempotency_manager: IdempotencyManager, config: IdempotencyConfig | None = None
) -> Callable:
    """
    Factory function para criar middleware de idempotency

    Args:
        idempotency_manager: Gerenciador de idempotency
        config: Configuração opcional

    Returns:
        Função middleware para FastAPI
    """
    config = config or IdempotencyConfig()

    # Create the middleware instance once (avoid per-request instantiation).
    async def _noop_app(scope, receive, send):  # pragma: no cover
        return

    middleware = IdempotencyMiddleware(
        app=_noop_app,
        idempotency_manager=idempotency_manager,
        idempotency_header=config.header_name,
        exclude_paths=config.exclude_paths,
        exclude_methods=config.exclude_methods,
    )

    async def idempotency_middleware(request: Request, call_next: Callable) -> Response:
        return await middleware.dispatch(request, call_next)

    return idempotency_middleware


# Funções utilitárias


def generate_idempotency_key() -> str:
    """
    Gera uma nova chave de idempotência

    Returns:
        Chave UUID4 como string
    """
    import uuid

    return str(uuid.uuid4())


def validate_idempotency_key(key: str) -> bool:
    """
    Valida formato da chave de idempotência

    Args:
        key: Chave a validar

    Returns:
        True se válida
    """
    import re

    # UUID v4 pattern
    uuid_pattern = (
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    return bool(re.match(uuid_pattern, key, re.IGNORECASE))
