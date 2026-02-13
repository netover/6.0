"""Gerenciamento de contexto global da aplicação.

Este módulo fornece acesso ao contexto da requisição atual usando contextvars,
permitindo que componentes em qualquer nível da aplicação acessem informações
como Correlation ID sem precisar passar explicitamente através de parâmetros.

Uso de contextvars garante isolamento entre requisições concorrentes.
"""

import uuid
from contextvars import ContextVar, Token

# Context variables para armazenar informações da requisição
_correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


# ============================================================================
# CORRELATION ID
# ============================================================================


def set_correlation_id(correlation_id: str) -> Token[str | None]:
    """Define o Correlation ID para o contexto atual.

    Args:
        correlation_id: ID de correlação

    Returns:
        Token para resetar o valor posteriormente
    """
    return _correlation_id_ctx.set(correlation_id)


def get_correlation_id() -> str | None:
    """Obtém o Correlation ID do contexto atual.

    Returns:
        Correlation ID se disponível, None caso contrário
    """
    return _correlation_id_ctx.get()


def get_or_create_correlation_id() -> str:
    """Obtém ou cria um novo Correlation ID.

    Se não houver Correlation ID no contexto, cria um novo.

    Returns:
        Correlation ID (existente ou novo)
    """
    correlation_id = get_correlation_id()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
    return correlation_id


def reset_correlation_id(token: Token[str | None]) -> None:
    """Reseta o Correlation ID para o valor anterior.

    Args:
        token: Token retornado por set_correlation_id
    """
    _correlation_id_ctx.reset(token)


def clear_correlation_id() -> None:
    """Limpa o Correlation ID do contexto."""
    _correlation_id_ctx.set(None)


# ============================================================================
# USER ID
# ============================================================================


def set_user_id(user_id: str) -> Token[str | None]:
    """Define o User ID para o contexto atual.

    Args:
        user_id: ID do usuário

    Returns:
        Token para resetar o valor posteriormente
    """
    return _user_id_ctx.set(user_id)


def get_user_id() -> str | None:
    """Obtém o User ID do contexto atual.

    Returns:
        User ID se disponível, None caso contrário
    """
    return _user_id_ctx.get()


def clear_user_id() -> None:
    """Limpa o User ID do contexto."""
    _user_id_ctx.set(None)


def reset_user_id(token: Token[str | None]) -> None:
    """Reseta o User ID para o valor anterior.

    Args:
        token: Token retornado por set_user_id
    """
    _user_id_ctx.reset(token)


# ============================================================================
# REQUEST ID
# ============================================================================


def set_request_id(request_id: str) -> Token[str | None]:
    """Define o Request ID para o contexto atual.

    Args:
        request_id: ID da requisição

    Returns:
        Token para resetar o valor posteriormente
    """
    return _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    """Obtém o Request ID do contexto atual.

    Returns:
        Request ID se disponível, None caso contrário
    """
    return _request_id_ctx.get()


def get_or_create_request_id() -> str:
    """Obtém ou cria um novo Request ID.

    Returns:
        Request ID (existente ou novo)
    """
    request_id = get_request_id()
    if not request_id:
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
    return request_id


def clear_request_id() -> None:
    """Limpa o Request ID do contexto."""
    _request_id_ctx.set(None)


def reset_request_id(token: Token[str | None]) -> None:
    """Reseta o Request ID para o valor anterior.

    Args:
        token: Token retornado por set_request_id
    """
    _request_id_ctx.reset(token)


# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================


def clear_context() -> None:
    """Limpa todo o contexto da requisição.

    Útil para testes ou limpeza entre requisições.
    """
    clear_correlation_id()
    clear_user_id()
    clear_request_id()


def get_context_dict() -> dict:
    """Retorna um dicionário com todo o contexto atual.

    Returns:
        Dicionário com correlation_id, user_id e request_id
    """
    return {
        "correlation_id": get_correlation_id(),
        "user_id": get_user_id(),
        "request_id": get_request_id(),
    }


class RequestContext:
    """Context manager para gerenciar contexto de requisição.

    Permite definir e limpar automaticamente o contexto usando 'with'.

    Example:
        ```python
        with RequestContext(correlation_id="123", user_id="user-456"):
            # Código com contexto definido
            print(get_correlation_id())  # "123"
            print(get_user_id())  # "user-456"
        # Contexto automaticamente limpo
        ```
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ):
        """Inicializa o context manager.

        Args:
            correlation_id: Correlation ID (opcional)
            user_id: User ID (opcional)
            request_id: Request ID (opcional)
        """
        self.correlation_id = correlation_id
        self.user_id = user_id
        self.request_id = request_id
        self.tokens: list[tuple[str, Token[str | None]]] = []

    def __enter__(self):
        """Entra no contexto, definindo os valores."""
        if self.correlation_id:
            self.tokens.append(("correlation_id", set_correlation_id(self.correlation_id)))
        if self.user_id:
            self.tokens.append(("user_id", set_user_id(self.user_id)))
        if self.request_id:
            self.tokens.append(("request_id", set_request_id(self.request_id)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sai do contexto, resetando os valores."""
        for ctx_name, token in reversed(self.tokens):
            if ctx_name == "correlation_id":
                reset_correlation_id(token)
            elif ctx_name == "user_id":
                reset_user_id(token)
            elif ctx_name == "request_id":
                reset_request_id(token)
        return False


__all__ = [
    # Correlation ID
    "set_correlation_id",
    "get_correlation_id",
    "get_or_create_correlation_id",
    "reset_correlation_id",
    "clear_correlation_id",
    # User ID
    "set_user_id",
    "get_user_id",
    "reset_user_id",
    "clear_user_id",
    # Request ID
    "set_request_id",
    "get_request_id",
    "get_or_create_request_id",
    "reset_request_id",
    "clear_request_id",
    # Context Management
    "clear_context",
    "get_context_dict",
    "RequestContext",
]
