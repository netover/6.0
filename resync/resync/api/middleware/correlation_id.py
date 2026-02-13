"""Middleware para gerenciamento de Correlation IDs.

Este middleware implementa o padrão de Correlation ID para rastreamento
distribuído de requisições através de múltiplos serviços e componentes.

O Correlation ID é:
- Gerado automaticamente se não fornecido
- Propagado através de headers HTTP
- Disponível em todo o contexto da requisição
- Incluído em logs e respostas de erro

PERFORMANCE: Pure ASGI implementation (1.5-2x faster than BaseHTTPMiddleware)
"""

import logging
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Nome do header para Correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
CORRELATION_ID_CTX_KEY = "correlation_id"


class CorrelationIdMiddleware:
    """Pure ASGI middleware para gerenciar Correlation IDs em requisições HTTP.

    Funcionalidades:
    - Extrai Correlation ID do header da requisição
    - Gera novo ID se não fornecido
    - Armazena no contexto da requisição (scope)
    - Adiciona ao header da resposta
    - Disponibiliza para logging

    PERFORMANCE: 1.5-2x faster than BaseHTTPMiddleware due to:
    - No coroutine overhead
    - Direct ASGI interface
    - Minimal memory allocations

    Attributes:
        header_name: Nome do header HTTP para o Correlation ID
        generate_if_missing: Se True, gera ID quando não fornecido
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = CORRELATION_ID_HEADER,
        generate_if_missing: bool = True,
    ):
        """Inicializa o middleware.

        Args:
            app: Aplicação ASGI
            header_name: Nome do header para Correlation ID
            generate_if_missing: Se deve gerar ID quando ausente
        """
        self.app = app
        self.header_name = header_name
        self.header_name_lower = header_name.lower()
        self.header_name_bytes = header_name.lower().encode()
        self.generate_if_missing = generate_if_missing

    @staticmethod
    def _generate_correlation_id() -> str:
        """Gera um novo Correlation ID único.

        Returns:
            String UUID v4 no formato sem hífens
        """
        return uuid.uuid4().hex

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI interface implementation.

        Args:
            scope: ASGI connection scope
            receive: ASGI receive channel
            send: ASGI send channel
        """
        if scope["type"] != "http":
            # Pass through non-HTTP requests
            await self.app(scope, receive, send)
            return

        # Extract correlation ID from request headers
        correlation_id = None
        for header_name, header_value in scope["headers"]:
            if header_name == self.header_name_bytes:
                correlation_id = header_value.decode()
                break

        # Generate if missing
        if not correlation_id and self.generate_if_missing:
            correlation_id = self._generate_correlation_id()
            logger.debug(
                "Generated new correlation ID",
                extra={"correlation_id": correlation_id},
            )
        elif correlation_id:
            logger.debug(
                "Using existing correlation ID",
                extra={"correlation_id": correlation_id},
            )

        # Store in scope for access by endpoints.
        # IMPORTANT: `scope` is a dict. Starlette/FastAPI may already have
        # `scope["state"]` populated by other middleware. Do NOT overwrite it.
        state = scope.setdefault("state", {})
        if correlation_id:
            state["correlation_id"] = correlation_id

        async def send_with_correlation_id(message: Message) -> None:
            """Intercept response and add correlation ID header."""
            if message["type"] == "http.response.start" and correlation_id:
                headers = MutableHeaders(scope=message)
                # Use assignment to avoid duplicating the header when upstream
                # middleware also sets it.
                headers[self.header_name] = correlation_id
            await send(message)

        await self.app(scope, receive, send_with_correlation_id)


def get_correlation_id_from_request(request) -> str | None:
    """Extract correlation ID from a request object.

    Args:
        request: Starlette/FastAPI Request object

    Returns:
        Correlation ID string or None if not found
    """
    # Try to get from request state first
    if hasattr(request, 'state') and hasattr(request.state, 'correlation_id'):
        return request.state.correlation_id

    # Try to get from scope
    if hasattr(request, 'scope') and 'state' in request.scope:
        return request.scope['state'].get('correlation_id')

    # Try to get from headers
    if hasattr(request, 'headers'):
        return request.headers.get(CORRELATION_ID_HEADER)

    return None
