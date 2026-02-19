"""Módulo para exceções customizadas da aplicação Resync.

Este módulo implementa uma hierarquia completa de exceções seguindo as melhores práticas:
- Códigos de erro padronizados
- Status HTTP semânticos
- Suporte a correlation IDs
- Contexto detalhado para debugging
- Separação entre erros de cliente (4xx) e servidor (5xx)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Códigos de erro padronizados da aplicação.

    Seguem o padrão de nomenclatura SCREAMING_SNAKE_CASE e são agrupados
    por categoria para facilitar identificação e tratamento.
    """

    # Erros de Validação (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"

    # Erros de Autenticação (401)
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"

    # Erros de Autorização (403)
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    ACCESS_DENIED = "ACCESS_DENIED"

    # Erros de Recurso (404)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"

    # Erros de Conflito (409)
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"

    # Erros de Negócio (422)
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"

    # Erros de Rate Limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Erros de Servidor (500)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNHANDLED_EXCEPTION = "UNHANDLED_EXCEPTION"

    # Erros de Integração (502)
    INTEGRATION_ERROR = "INTEGRATION_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    TWS_CONNECTION_ERROR = "TWS_CONNECTION_ERROR"
    # Erros de Autenticação TWS (401/502)
    TWS_AUTHENTICATION_ERROR = "TWS_AUTHENTICATION_ERROR"
    # Erros de Timeout TWS (504)
    TWS_TIMEOUT_ERROR = "TWS_TIMEOUT_ERROR"
    # Erros de Rate Limit TWS (429)
    TWS_RATE_LIMIT_ERROR = "TWS_RATE_LIMIT_ERROR"
    # Erros de Requisição Inválida TWS (400)
    TWS_BAD_REQUEST_ERROR = "TWS_BAD_REQUEST_ERROR"
    # Erros de Servidor TWS (5xx/502)
    TWS_SERVER_ERROR = "TWS_SERVER_ERROR"
    LLM_ERROR = "LLM_ERROR"
    # Erros de Autenticação LLM (401/502)
    LLM_AUTHENTICATION_ERROR = "LLM_AUTHENTICATION_ERROR"
    # Erros de Timeout LLM (504)
    LLM_TIMEOUT_ERROR = "LLM_TIMEOUT_ERROR"
    # Erros de Rate Limit LLM (429)
    LLM_RATE_LIMIT_ERROR = "LLM_RATE_LIMIT_ERROR"

    # Erros de Disponibilidade (503)
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"

    # Erros de Timeout (504)
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"

    # Erros de Banco de Dados
    DATABASE_ERROR = "DATABASE_ERROR"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_QUERY_ERROR = "DATABASE_QUERY_ERROR"

    # Erros de Cache
    CACHE_ERROR = "CACHE_ERROR"
    CACHE_MISS = "CACHE_MISS"
    POOL_EXHAUSTED = "POOL_EXHAUSTED"

    # Erros de Redis
    REDIS_ERROR = "REDIS_ERROR"
    REDIS_CONNECTION_ERROR = "REDIS_CONNECTION_ERROR"
    REDIS_AUTH_ERROR = "REDIS_AUTH_ERROR"
    REDIS_TIMEOUT_ERROR = "REDIS_TIMEOUT_ERROR"
    REDIS_INITIALIZATION_ERROR = "REDIS_INITIALIZATION_ERROR"

    # Erros de Configuração
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    MISSING_CONFIGURATION = "MISSING_CONFIGURATION"

    # Erros de Arquivo
    FILE_ERROR = "FILE_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_PROCESSING_ERROR = "FILE_PROCESSING_ERROR"

    # Erros de Rede
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"


class ErrorSeverity(str, Enum):
    """Níveis de severidade para erros."""

    CRITICAL = "critical"  # Sistema inoperante, requer ação imediata
    ERROR = "error"  # Funcionalidade comprometida
    WARNING = "warning"  # Problema potencial
    INFO = "info"  # Informação


class BaseAppException(Exception):
    """Exceção base para todas as exceções da aplicação.

    Fornece estrutura padronizada com:
    - Código de erro
    - Status HTTP
    - Correlation ID para rastreamento
    - Contexto adicional
    - Severidade
    - Timestamp

    Attributes:
        message: Mensagem descritiva do erro
        error_code: Código de erro padronizado
        status_code: Código de status HTTP
        details: Dicionário com contexto adicional
        correlation_id: ID para rastreamento distribuído
        severity: Nível de severidade do erro
        timestamp: Momento em que o erro ocorreu
        original_exception: Exceção original que causou este erro
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_exception: Exception | None = None,
    ):
        """Inicializa a exceção base.

        Args:
            message: Mensagem descritiva do erro
            error_code: Código de erro padronizado
            status_code: Código de status HTTP
            details: Contexto adicional (opcional)
            correlation_id: ID de correlação (opcional)
            severity: Nível de severidade
            original_exception: Exceção original (opcional)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.correlation_id = correlation_id
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)
        self.original_exception = original_exception

    def to_dict(self) -> dict[str, Any]:
        """Converte a exceção para dicionário.

        Returns:
            Dicionário com todos os atributos da exceção
        """
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "status_code": self.status_code,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        """Representação em string da exceção."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code={self.error_code.value}, "
            f"status_code={self.status_code}, "
            f"correlation_id={self.correlation_id})"
        )


# ============================================================================
# EXCEÇÕES DE CLIENTE (4xx) - Erros causados pelo cliente
# ============================================================================


class ValidationError(BaseAppException):
    """Erro de validação de dados de entrada.

    Usado quando os dados fornecidos pelo cliente não passam na validação.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class AuthenticationError(BaseAppException):
    """Erro de autenticação.

    Usado quando as credenciais são inválidas ou ausentes.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            status_code=401,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class AuthorizationError(BaseAppException):
    """Erro de autorização.

    Usado quando o usuário autenticado não tem permissão para a operação.
    """

    def __init__(
        self,
        message: str = "Authorization failed",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            status_code=403,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class ResourceNotFoundError(BaseAppException):
    """Erro quando recurso não é encontrado.

    Usado quando um recurso solicitado não existe.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=404,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.INFO,
            original_exception=original_exception,
        )


class ResourceConflictError(BaseAppException):
    """Erro de conflito de recurso.

    Usado quando há conflito com o estado atual do recurso.
    """

    def __init__(
        self,
        message: str = "Resource conflict",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_CONFLICT,
            status_code=409,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class RateLimitError(BaseAppException):
    """Erro de limite de taxa excedido.

    Usado quando o cliente excede o limite de requisições permitido.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


# ============================================================================
# EXCEÇÕES DE SERVIDOR (5xx) - Erros internos do servidor
# ============================================================================


class InternalError(BaseAppException):
    """Erro interno do servidor.

    Usado para erros inesperados que não se encaixam em outras categorias.
    """

    def __init__(
        self,
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class IntegrationError(BaseAppException):
    """Erro de integração com serviço externo.

    Usado quando há falha na comunicação com serviços externos.
    """

    def __init__(
        self,
        message: str = "External service integration error",
        service_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if service_name:
            details["service_name"] = service_name

        super().__init__(
            message=message,
            error_code=ErrorCode.INTEGRATION_ERROR,
            status_code=502,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class ServiceUnavailableError(BaseAppException):
    """Erro de serviço indisponível.

    Usado quando o serviço está temporariamente indisponível.
    """

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        # Performance: Use copy() to avoid mutating the original details dict
        # and use dict.get() pattern for cleaner optional value handling
        _details = details.copy() if details else {}
        
        # Only add retry_after if it's a positive value (edge case handling)
        if retry_after is not None and retry_after > 0:
            _details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=503,
            details=_details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class TWSError(BaseAppException):
    """Erro específico para falhas na integração com TWS.

    Usado para erros relacionados ao TWS Unified Client.
    """

    def __init__(
        self,
        message: str = "TWS integration error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTEGRATION_ERROR,
            status_code=502,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class LLMOperationError(BaseAppException):
    """Erro específico para falhas em operações de LLM.

    Mantido para compatibilidade com código legado; prefira ``LLMError``
    definido na seção de erros de integração.
    """

    def __init__(
        self,
        message: str = "LLM operation error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class CircuitBreakerError(BaseAppException):
    """Erro quando circuit breaker está aberto.

    Usado quando o circuit breaker impede chamadas a um serviço com falhas.
    """

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        service_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        # Use dict.copy() for better performance than creating new dict
        _details = details.copy() if details else {}
        
        if service_name:
            _details["service_name"] = service_name

        super().__init__(
            message=message,
            error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            status_code=503,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class TimeoutError(BaseAppException):
    """Erro de timeout.

    Usado quando uma operação excede o tempo limite.
    """

    def __init__(
        self,
        message: str = "Operation timeout",
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=message,
            error_code=ErrorCode.OPERATION_TIMEOUT,
            status_code=504,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


# ============================================================================
# EXCEÇÕES ESPECÍFICAS DO DOMÍNIO
# ============================================================================


class ConfigurationError(BaseAppException):
    """Erro de configuração."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.CRITICAL,
            original_exception=original_exception,
        )


class InvalidConfigError(ConfigurationError):
    """Exceção para erros de dados de configuração inválidos."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            config_key=config_key,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.INVALID_CONFIGURATION


class RedisError(BaseAppException):
    """Exceção base para erros relacionados ao Redis."""

    def __init__(
        self,
        message: str = "Redis error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.REDIS_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class RedisInitializationError(RedisError):
    """Erro ao inicializar Redis."""

    def __init__(
        self,
        message: str = "Redis initialization error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.REDIS_INITIALIZATION_ERROR
        self.severity = ErrorSeverity.CRITICAL


class RedisConnectionError(RedisInitializationError):
    """Erro de conexão ao Redis."""

    def __init__(
        self,
        message: str = "Redis connection error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.REDIS_CONNECTION_ERROR


class RedisAuthError(RedisInitializationError):
    """Erro de autenticação Redis."""

    def __init__(
        self,
        message: str = "Redis authentication error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.REDIS_AUTH_ERROR


class RedisTimeoutError(RedisInitializationError):
    """Timeout em operação Redis."""

    def __init__(
        self,
        message: str = "Redis timeout error",
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=message,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.REDIS_TIMEOUT_ERROR


class AgentError(BaseAppException):
    """Exceção para erros relacionados à criação ou gerenciamento de agentes."""

    def __init__(
        self,
        message: str = "Agent error",
        agent_id: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if agent_id:
            details["agent_id"] = agent_id

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class TWSConnectionError(IntegrationError):
    """Exceção para erros de conexão com a API do TWS."""

    def __init__(
        self,
        message: str = "TWS connection error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.TWS_CONNECTION_ERROR


# Additional TWS-specific exceptions
class TWSAuthenticationError(IntegrationError):
    """Exceção para falhas de autenticação com a API do TWS.

    Esta exceção é lançada quando uma chamada ao serviço TWS falha devido a
    problemas de autenticação, como credenciais inválidas ou ausentes. Ela
    estende :class:`IntegrationError` para preservar o contexto de serviço
    externo, mas define um código de erro específico e um status HTTP 401.
    """

    def __init__(
        self,
        message: str = "TWS authentication error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        # Indicar serviço para rastreamento
        details.setdefault("service_name", "TWS")
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        # Substituir código de erro e status HTTP
        self.error_code = ErrorCode.TWS_AUTHENTICATION_ERROR
        self.status_code = 401


class TWSTimeoutError(IntegrationError):
    """Exceção para timeouts em operações com o serviço TWS.

    Esta exceção indica que uma operação contra o TWS excedeu o tempo limite
    configurado. Ela inclui opcionalmente o número de segundos de timeout
    utilizado para auxiliar no diagnóstico.
    """

    def __init__(
        self,
        message: str = "TWS timeout error",
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        # Indicar serviço para rastreamento
        details.setdefault("service_name", "TWS")
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        # Atribuir código de erro e status de gateway timeout (504)
        self.error_code = ErrorCode.TWS_TIMEOUT_ERROR
        self.status_code = 504


class TWSRateLimitError(IntegrationError):
    """Exceção para rate limiting (429) retornado pelo TWS.

    Inclui opcionalmente o tempo sugerido de espera (retry_after_seconds) para
    implementar backoff respeitando o header `Retry-After`.
    """

    def __init__(
        self,
        message: str = "TWS rate limit exceeded",
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        details.setdefault("service_name", "TWS")
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.TWS_RATE_LIMIT_ERROR
        self.status_code = 429


class TWSBadRequestError(IntegrationError):
    """Exceção para erros de requisição inválida (400) retornados pelo TWS."""

    def __init__(
        self,
        message: str = "TWS bad request",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        details.setdefault("service_name", "TWS")
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.TWS_BAD_REQUEST_ERROR
        self.status_code = 400


class TWSServerError(IntegrationError):
    """Exceção para erros 5xx do TWS (incluindo 502/503/504 quando aplicável)."""

    def __init__(
        self,
        message: str = "TWS server error",
        status_code: int = 502,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        details.setdefault("service_name", "TWS")
        super().__init__(
            message=message,
            service_name="TWS",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.TWS_SERVER_ERROR
        self.status_code = status_code


class AgentExecutionError(BaseAppException):
    """Exceção para erros durante a execução de um agente de IA."""

    def __init__(
        self,
        message: str = "Agent execution error",
        agent_id: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if agent_id:
            details["agent_id"] = agent_id

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class ToolExecutionError(BaseAppException):
    """Exceção para erros durante a execução de uma ferramenta (tool)."""

    def __init__(
        self,
        message: str = "Tool execution error",
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if tool_name:
            details["tool_name"] = tool_name

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class ToolConnectionError(ToolExecutionError):
    """Exceção para erros de conexão dentro de uma ferramenta."""

    def __init__(
        self,
        message: str = "Tool connection error",
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            tool_name=tool_name,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.CONNECTION_ERROR


class ToolProcessingError(ToolExecutionError):
    """Exceção para erros de processamento de dados dentro de uma ferramenta."""


class KnowledgeGraphError(BaseAppException):
    """Exceção para erros relacionados ao Knowledge Graph (ex: Mem0)."""

    def __init__(
        self,
        message: str = "Knowledge graph error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class AuditError(BaseAppException):
    """Exceção para erros no sistema de auditoria (queue, lock, etc.)."""

    def __init__(
        self,
        message: str = "Audit system error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class FileProcessingError(BaseAppException):
    """Exceção para erros durante o processamento de arquivos."""

    def __init__(
        self,
        message: str = "File processing error",
        filename: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if filename:
            details["filename"] = filename

        super().__init__(
            message=message,
            error_code=ErrorCode.FILE_PROCESSING_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class LLMError(IntegrationError):
    """Exceção para erros na comunicação com o Large Language Model."""

    def __init__(
        self,
        message: str = "LLM error",
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name

        super().__init__(
            message=message,
            service_name="LLM",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.LLM_ERROR


# Additional LLM-specific exceptions
class LLMAuthenticationError(IntegrationError):
    """Exceção para falhas de autenticação com provedores de LLM.

    Esta exceção é usada quando uma requisição ao serviço de LLM falha por
    problemas de autenticação, como chaves de API inválidas. Ela estende
    :class:`IntegrationError` para manter o contexto de serviço externo, mas
    configura um código de erro e status HTTP apropriados (401).
    """

    def __init__(
        self,
        message: str = "LLM authentication error",
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        # Indicar serviço para rastreamento
        details.setdefault("service_name", "LLM")
        super().__init__(
            message=message,
            service_name="LLM",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.LLM_AUTHENTICATION_ERROR
        self.status_code = 401


class LLMTimeoutError(IntegrationError):
    """Exceção para timeouts em chamadas ao serviço LLM.

    Disparada quando uma chamada para o serviço de LLM excede o tempo limite
    configurado. Inclui opcionalmente o tempo limite para ajudar na análise.
    """

    def __init__(
        self,
        message: str = "LLM timeout error",
        timeout_seconds: float | None = None,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if model_name:
            details["model_name"] = model_name
        # Indicar serviço para rastreamento
        details.setdefault("service_name", "LLM")
        super().__init__(
            message=message,
            service_name="LLM",
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.LLM_TIMEOUT_ERROR
        self.status_code = 504


class LLMRateLimitError(RateLimitError):
    """Exceção para limites de taxa excedidos em provedores de LLM.

    Utilizada quando o serviço de LLM retorna um erro de rate limiting (por
    exemplo, HTTP 429). Herda de :class:`RateLimitError` para reutilizar a
    semântica de limite excedido, adicionando contexto de modelo e serviço.
    """

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        retry_after: int | None = None,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        # Indicar serviço para rastreamento
        details.setdefault("service_name", "LLM")
        super().__init__(
            message=message,
            retry_after=retry_after,
            details=details,
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.error_code = ErrorCode.LLM_RATE_LIMIT_ERROR


class ParsingError(BaseAppException):
    """Exceção para erros de parsing de dados (JSON, etc.)."""

    def __init__(
        self,
        message: str = "Parsing error",
        data_format: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if data_format:
            details["data_format"] = data_format

        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class NetworkError(BaseAppException):
    """Exceção para erros de rede genéricos."""

    def __init__(
        self,
        message: str = "Network error",
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.NETWORK_ERROR,
            status_code=502,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class DatabaseError(BaseAppException):
    """Exceção para erros de interação com o banco de dados."""

    def __init__(
        self,
        message: str = "Database error",
        query: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if query:
            # Não incluir query completa por segurança, apenas tipo
            details["query_type"] = query.split()[0] if query else None

        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class CacheError(BaseAppException):
    """Exceção para erros relacionados ao sistema de cache."""

    def __init__(
        self,
        message: str = "Cache error",
        cache_key: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if cache_key:
            details["cache_key"] = cache_key

        super().__init__(
            message=message,
            error_code=ErrorCode.CACHE_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class PoolExhaustedError(CacheError):
    """Exceção para quando o pool de conexões está esgotado."""

    def __init__(
        self,
        message: str = "Connection pool exhausted",
        pool_name: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if pool_name:
            details["pool_name"] = pool_name

        # Call BaseAppException directly to allow custom error_code/status_code
        BaseAppException.__init__(
            self,
            message=message,
            error_code=ErrorCode.POOL_EXHAUSTED,
            status_code=503,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class NotificationError(BaseAppException):
    """Exceção para erros durante o envio de notificações."""

    def __init__(
        self,
        message: str = "Notification error",
        notification_type: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if notification_type:
            details["notification_type"] = notification_type

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class NotFoundError(ResourceNotFoundError):
    """Exceção para quando um recurso não é encontrado.

    Alias para ResourceNotFoundError para compatibilidade.
    """


class PerformanceError(BaseAppException):
    """Exceção para erros relacionados à performance do sistema."""

    def __init__(
        self,
        message: str = "Performance degradation detected",
        metric_name: str | None = None,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if metric_name:
            details["metric_name"] = metric_name
        if threshold:
            details["threshold"] = threshold
        if actual_value:
            details["actual_value"] = actual_value

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            original_exception=original_exception,
        )


class HealthCheckError(BaseAppException):
    """Exceção para erros durante verificações de saúde do sistema."""

    def __init__(
        self,
        message: str = "Health check error",
        component: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        if details is None:
            details = {}
        if component:
            details["component"] = component

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details,
            correlation_id=correlation_id,
            severity=ErrorSeverity.ERROR,
            original_exception=original_exception,
        )


class CacheHealthCheckError(HealthCheckError):
    """Exceção quando health check do cache falha.

    Permite distinção de falhas de cache de outras falhas de health check.
    """

    def __init__(
        self,
        operation: str,
        details_info: str | None = None,
        message: str | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        self.operation = operation
        self.details_info = details_info

        if message is None:
            message = f"Cache health check failed: {operation}"
            if details_info:
                message += f" - {details_info}"

        super().__init__(
            message=message,
            component="cache",
            details={
                "operation": operation,
                "info": details_info,
            },
            correlation_id=correlation_id,
            original_exception=original_exception,
        )


# ============================================================================
# COMPATIBILIDADE COM CÓDIGO LEGADO
# ============================================================================

# Alias para manter compatibilidade com código existente
ResyncException = BaseAppException


class BusinessError(BaseAppException):
    """Exception for business rule violations."""

    def __init__(
        self,
        message: str = "Business rule violation",
        *,
        component: str = "business",
        details: dict | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_RULE_VIOLATION,
            status_code=400,
            details=details or {},
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.details.setdefault("component", component)


class WebSocketError(BaseAppException):
    """Exception for WebSocket communication errors."""

    def __init__(
        self,
        message: str = "WebSocket error",
        *,
        component: str = "websocket",
        details: dict | None = None,
        correlation_id: str | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.WEBSOCKET_ERROR,
            status_code=502,
            details=details or {},
            correlation_id=correlation_id,
            original_exception=original_exception,
        )
        self.details.setdefault("component", component)


# ============================================================================
# UTILITÁRIOS
# ============================================================================


def get_exception_by_error_code(error_code: ErrorCode) -> type[BaseAppException]:
    """Retorna a classe de exceção apropriada para um código de erro.

    Args:
        error_code: Código de erro

    Returns:
        Classe de exceção correspondente
    """
    mapping = {
        ErrorCode.VALIDATION_ERROR: ValidationError,
        ErrorCode.AUTHENTICATION_FAILED: AuthenticationError,
        ErrorCode.AUTHORIZATION_FAILED: AuthorizationError,
        ErrorCode.RESOURCE_NOT_FOUND: ResourceNotFoundError,
        ErrorCode.RESOURCE_CONFLICT: ResourceConflictError,
        ErrorCode.BUSINESS_RULE_VIOLATION: BusinessError,
        ErrorCode.RATE_LIMIT_EXCEEDED: RateLimitError,
        ErrorCode.INTEGRATION_ERROR: IntegrationError,
        ErrorCode.SERVICE_UNAVAILABLE: ServiceUnavailableError,
        ErrorCode.CIRCUIT_BREAKER_OPEN: CircuitBreakerError,
        ErrorCode.OPERATION_TIMEOUT: TimeoutError,
        ErrorCode.DATABASE_ERROR: DatabaseError,
        ErrorCode.CACHE_ERROR: CacheError,
        ErrorCode.CONFIGURATION_ERROR: ConfigurationError,
        ErrorCode.REDIS_ERROR: RedisError,
        ErrorCode.REDIS_CONNECTION_ERROR: RedisConnectionError,
        ErrorCode.REDIS_AUTH_ERROR: RedisAuthError,
        ErrorCode.REDIS_TIMEOUT_ERROR: RedisTimeoutError,
        ErrorCode.REDIS_INITIALIZATION_ERROR: RedisInitializationError,
        ErrorCode.TWS_CONNECTION_ERROR: TWSConnectionError,
        ErrorCode.LLM_ERROR: LLMError,
        ErrorCode.NETWORK_ERROR: NetworkError,
        ErrorCode.WEBSOCKET_ERROR: WebSocketError,
        # TWS specific errors
        ErrorCode.TWS_AUTHENTICATION_ERROR: TWSAuthenticationError,
        ErrorCode.TWS_TIMEOUT_ERROR: TWSTimeoutError,
        ErrorCode.TWS_RATE_LIMIT_ERROR: TWSRateLimitError,
        ErrorCode.TWS_BAD_REQUEST_ERROR: TWSBadRequestError,
        ErrorCode.TWS_SERVER_ERROR: TWSServerError,
        # LLM specific errors
        ErrorCode.LLM_AUTHENTICATION_ERROR: LLMAuthenticationError,
        ErrorCode.LLM_TIMEOUT_ERROR: LLMTimeoutError,
        ErrorCode.LLM_RATE_LIMIT_ERROR: LLMRateLimitError,
    }

    return mapping.get(error_code, InternalError)


# =============================================================================

__all__ = [
    # Enums
    "ErrorCode",
    "ErrorSeverity",
    # Base
    "BaseAppException",
    "ResyncException",  # Alias para compatibilidade
    # Client Errors (4xx)
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "ResourceConflictError",
    "RateLimitError",
    # Server Errors (5xx)
    "InternalError",
    "IntegrationError",
    "ServiceUnavailableError",
    "CircuitBreakerError",
    "TimeoutError",
    "PerformanceError",
    "HealthCheckError",
    "CacheHealthCheckError",
    "PoolExhaustedError",
    # Domain Specific
    "ConfigurationError",
    "InvalidConfigError",
    "RedisError",
    "RedisInitializationError",
    "RedisConnectionError",
    "RedisAuthError",
    "RedisTimeoutError",
    "AgentError",
    "TWSConnectionError",
    "TWSAuthenticationError",
    "TWSTimeoutError",
    "TWSRateLimitError",
    "TWSBadRequestError",
    "TWSServerError",
    "AgentExecutionError",
    "ToolExecutionError",
    "ToolConnectionError",
    "ToolProcessingError",
    "KnowledgeGraphError",
    "AuditError",
    "FileProcessingError",
    "LLMError",
    "LLMOperationError",
    "LLMAuthenticationError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "ParsingError",
    "NetworkError",
    "DatabaseError",
    "CacheError",
    "NotificationError",
    "NotFoundError",
    "PerformanceError",
    "HealthCheckError",
    # Utilities
    "get_exception_by_error_code",
]
