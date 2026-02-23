"""Sistema de logging estruturado usando structlog.

Este módulo configura logging estruturado em formato JSON para facilitar
análise automatizada, agregação e monitoramento de logs.

Características:
- Logs em formato JSON
- Correlation ID automático
- Contexto enriquecido
- Níveis de log configuráveis por ambiente
- Integração com sistemas de agregação (ELK, Loki, etc.)
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

try:
    import structlog  # type: ignore
    from structlog.types import EventDict, WrappedLogger  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore[assignment]
    EventDict = dict  # type: ignore[misc,assignment]
    WrappedLogger = Any  # type: ignore[misc,assignment]
from fastapi import Request

from .encoding_utils import can_encode

logger = logging.getLogger(__name__)


# Lazy import of settings to avoid circular dependencies
def _get_settings():
    """Lazy import of settings to avoid circular dependencies."""
    from resync.settings import settings

    return settings


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

# Store current request context for logging
_current_request_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_request_ctx", default=None
)


# ============================================================================
# CUSTOM PROCESSORS
# ============================================================================


def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona Correlation ID ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com correlation_id
    """
    from resync.core.context import get_correlation_id

    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_user_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona contexto do usuário ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com user_id
    """
    from resync.core.context import get_user_id

    user_id = get_user_id()
    if user_id:
        event_dict["user_id"] = user_id
    return event_dict


def add_request_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona contexto da requisição ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com request_id
    """
    from resync.core.context import get_request_id

    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_trace_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona Trace ID ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com trace_id
    """
    from resync.core.context import get_trace_id

    trace_id = get_trace_id()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict


def add_service_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona contexto do serviço ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com service_name e environment
    """
    settings = _get_settings()
    event_dict["service_name"] = settings.PROJECT_NAME
    event_dict["environment"] = settings.environment.value
    event_dict["version"] = settings.PROJECT_VERSION

    return event_dict


def add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona timestamp ISO 8601 ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com timestamp
    """
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
    return event_dict


def add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona nível de log padronizado.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com level
    """
    if method_name == "warn":
        method_name = "warning"

    event_dict["level"] = method_name.upper()
    return event_dict


def censor_sensitive_data(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Censura dados sensíveis nos logs.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com dados sensíveis censurados
    """
    sensitive_patterns = {
        # Exact key matches
        "password",
        "passwd",
        "pwd",
        "token",
        "secret",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "pin",
        "cvv",
        "ssn",
        "credit_card",
        "card_number",
        # Database connection strings
        "database_url",
        "db_url",
        "connection_string",
        "conn_str",
        "redis_url",
        "redis_password",
        # Additional sensitive fields
        "encryption_key",
        "signing_key",
        "jwt_secret",
        "session_secret",
        "cookie_secret",
        "admin_password",
        "tws_password",
    }

    import re

    # Pre-compile patterns once per call (not per log entry).
    _sensitive_value_patterns = [
        re.compile(r'(?:password|pwd|passwd)=["\']?[^"\'&\s]*["\']?', re.IGNORECASE),
        re.compile(r'(?:token|secret|key)=["\']?[^"\'&\s]*["\']?', re.IGNORECASE),
        re.compile(r"(?:authorization)[:\s]*bearer\s+[^\s]+", re.IGNORECASE),
        re.compile(r"(?:basic)\s+[a-zA-Z0-9+/=]+", re.IGNORECASE),
        re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}", re.IGNORECASE),
        re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b", re.IGNORECASE),
        re.compile(r"postgresql(?:\+asyncpg)?://[^@]+@[^\s]+", re.IGNORECASE),
        re.compile(r"redis://[^@]*@?[^\s]+", re.IGNORECASE),
        re.compile(r"mysql://[^@]+@[^\s]+", re.IGNORECASE),
        re.compile(r"mongodb://[^@]+@[^\s]+", re.IGNORECASE),
    ]

    def censor_dict(d: dict[str, Any] | Any) -> dict[str, Any]:  # type: ignore[assignment]
        """Censura recursivamente um dicionário."""
        result: dict[str, Any] = {}
        for key, value in d.items():
            key_lower = key.lower()

            # Verificar se a chave contém termo sensível
            if any(sensitive in key_lower for sensitive in sensitive_patterns):
                result[key] = "***REDACTED***"
            elif isinstance(value, dict):
                result[key] = censor_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    censor_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str):
                # Apply value pattern censoring
                censored_value = value
                for pattern in _sensitive_value_patterns:
                    # v6.2.1: Use pattern.sub and ensure case-insensitivity (Redaction bypass fix)
                    # The patterns are already compiled with re.IGNORECASE in _sensitive_value_patterns
                    censored_value = pattern.sub("***REDACTED***", censored_value)
                result[key] = censored_value
            else:
                result[key] = value  # type: ignore

        return result

    return censor_dict(event_dict)


def add_request_metadata(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Adiciona metadados da requisição ao log.

    Args:
        logger: Logger
        method_name: Nome do método de log
        event_dict: Dicionário do evento

    Returns:
        Event dict com metadados da requisição
    """
    request_ctx = _current_request_ctx.get()
    if request_ctx:
        event_dict.update(request_ctx)
    return event_dict


# ============================================================================
# CONFIGURATION
# ============================================================================


def protect_log_injection(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Prevent log injection by escaping newlines and control characters.

    OWASP Recommendation: Ensure user input cannot introduce false log entries.
    """
    for key, value in event_dict.items():
        if isinstance(value, str):
            # Replace newlines with escaped versions to keep log as single line per entry
            # and prevent forging new log entries.
            if "\n" in value or "\r" in value:
                event_dict[key] = value.replace("\n", "\\n").replace("\r", "\\r")
    return event_dict


def configure_structured_logging(
    log_level: str = "INFO", json_logs: bool = True, development_mode: bool = False
) -> None:
    """Configura logging estruturado unificado (stdlib + structlog) para a aplicação.

    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Se True, usa formato JSON; se False, formato legível
        development_mode: Se True, usa formato mais legível para desenvolvimento
    """
    level = getattr(logging, log_level.upper())

    # Processadores comuns utilizados por structlog e stdlib
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_log_level,
        add_correlation_id,
        add_trace_id,  # ← NOVO
        add_user_context,
        add_request_context,
        add_request_metadata,
        add_service_context,
        protect_log_injection,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        censor_sensitive_data,
    ]

    # Escolher renderer baseado no modo
    renderer: Any
    if development_mode or not json_logs:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    # 1. Configurar structlog
    structlog.configure(
        processors=shared_processors  # type: ignore[arg-type]
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 2. Configurar stdlib logging para usar os mesmos processadores
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,  # type: ignore[arg-type]
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Attach SecretRedactor to the handler (IO boundary security)
    from resync.core.logging_utils import SecretRedactor

    handler.addFilter(SecretRedactor())

    root_logger = logging.getLogger()

    # Limpar handlers existentes
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silenciar logs muito verbosos de bibliotecas
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None):
    """Get a structured logger if structlog is installed, else a safe stdlib fallback.

    The project uses structlog in production, but tests and lightweight environments
    may not have it installed. This fallback keeps the code importable and testable.
    """
    if structlog is None:  # pragma: no cover
        base = logging.getLogger(name or __name__)

        class _Shim:
            def __init__(self, logger: logging.Logger):
                self._l = logger

            def bind(self, **_kwargs):
                return self

            def debug(self, event: str, *args, **kwargs):
                self._l.debug(event, *args, extra=kwargs)

            def info(self, event: str, *args, **kwargs):
                self._l.info(event, *args, extra=kwargs)

            def warning(self, event: str, *args, **kwargs):
                self._l.warning(event, *args, extra=kwargs)

            def error(self, event: str, *args, **kwargs):
                self._l.error(event, *args, extra=kwargs)

            def critical(self, event: str, *args, **kwargs):
                self._l.critical(event, *args, extra=kwargs)

        return _Shim(base)

    return structlog.get_logger(name) if name else structlog.get_logger()


# ============================================================================
# LOGGING HELPERS
# ============================================================================


class LoggerAdapter:
    """Adapter para facilitar uso de logging estruturado.

    Fornece métodos convenientes para logging com contexto automático.
    """

    def __init__(self, logger: structlog.BoundLogger):
        """Inicializa o adapter.

        Args:
            logger: Logger estruturado
        """
        self.logger = logger

    def debug(self, event: str, **kwargs) -> None:
        """Log de debug."""
        self.logger.debug(event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        """Log de info."""
        self.logger.info(event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """Log de warning."""
        self.logger.warning(event, **kwargs)

    def error(self, event: str, exc_info: bool = False, **kwargs) -> None:
        """Log de error.

        Args:
            event: Mensagem do evento
            exc_info: Se True, inclui informações da exceção
            **kwargs: Contexto adicional
        """
        if exc_info:
            kwargs["exc_info"] = True

        self.logger.error(event, **kwargs)

    def critical(self, event: str, exc_info: bool = False, **kwargs) -> None:
        """Log de critical.

        Args:
            event: Mensagem do evento
            exc_info: Se True, inclui informações da exceção
            **kwargs: Contexto adicional
        """
        if exc_info:
            kwargs["exc_info"] = True

        self.logger.critical(event, **kwargs)

    def bind(self, **kwargs) -> LoggerAdapter:
        """Cria novo logger com contexto adicional.

        Args:
            **kwargs: Contexto a ser adicionado

        Returns:
            Novo LoggerAdapter com contexto
        """
        return LoggerAdapter(self.logger.bind(**kwargs))


def get_logger_adapter(name: str | None = None) -> LoggerAdapter:
    """Obtém um logger adapter.

    Args:
        name: Nome do logger

    Returns:
        LoggerAdapter configurado
    """
    return LoggerAdapter(get_logger(name))


# ============================================================================
# PERFORMANCE LOGGING
# ============================================================================


class PerformanceLogger:
    """Logger especializado para métricas de performance."""

    def __init__(self, logger: structlog.BoundLogger):
        """Inicializa o performance logger.

        Args:
            logger: Logger estruturado
        """
        self.logger = logger

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_size: int | None = None,
        response_size: int | None = None,
        user_agent: str | None = None,
        client_ip: str | None = None,
        **kwargs,
    ) -> None:
        """Loga informações de uma requisição HTTP.

        Args:
            method: Método HTTP
            path: Caminho da requisição
            status_code: Código de status da resposta
            duration_ms: Duração em milissegundos
            request_size: Tamanho da requisição em bytes
            response_size: Tamanho da resposta em bytes
            user_agent: User-Agent do cliente
            client_ip: IP do cliente
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "http_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if request_size is not None:
            log_data["request_size_bytes"] = request_size

        if response_size is not None:
            log_data["response_size_bytes"] = response_size

        if user_agent:
            log_data["user_agent"] = user_agent

        if client_ip:
            log_data["client_ip"] = client_ip

        log_data.update(kwargs)

        # Log as info for successful requests, warning for client errors, error for server errors
        if 200 <= status_code < 300:
            self.logger.info("http_request_processed", **log_data)
        elif 400 <= status_code < 500:
            self.logger.warning("http_client_error", **log_data)
        else:
            self.logger.error("http_server_error", **log_data)

    def log_database_query(
        self,
        query_type: str,
        duration_ms: float,
        rows_affected: int | None = None,
        query: str | None = None,
        **kwargs,
    ) -> None:
        """Loga informações de uma query de banco de dados.

        Args:
            query_type: Tipo de query (SELECT, INSERT, etc.)
            duration_ms: Duração em milissegundos
            rows_affected: Número de linhas afetadas
            query: Query SQL (censurada)
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "database_query",
            "query_type": query_type,
            "duration_ms": round(duration_ms, 2),
        }

        if rows_affected is not None:
            log_data["rows_affected"] = rows_affected

        if query:
            # Censor sensitive parts of the query
            import re

            censored_query = re.sub(
                r"(password|pwd|secret|token)\s*=\s*['\"][^'\"]*['\"]",
                r"\1=***REDACTED***",
                query,
                flags=re.IGNORECASE,
            )
            log_data["query"] = censored_query

        log_data.update(kwargs)

        # Warn for slow queries (> 1 second)
        if duration_ms > 1000:
            self.logger.warning("slow_database_query", **log_data)
        else:
            self.logger.info("database_query_executed", **log_data)

    def log_external_call(
        self,
        service_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        request_size: int | None = None,
        response_size: int | None = None,
        **kwargs,
    ) -> None:
        """Loga chamada a serviço externo.

        Args:
            service_name: Nome do serviço
            operation: Operação realizada
            duration_ms: Duração em milissegundos
            success: Se a chamada foi bem-sucedida
            request_size: Tamanho da requisição em bytes
            response_size: Tamanho da resposta em bytes
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "external_call",
            "service_name": service_name,
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "success": success,
        }

        if request_size is not None:
            log_data["request_size_bytes"] = request_size

        if response_size is not None:
            log_data["response_size_bytes"] = response_size

        log_data.update(kwargs)

        # Error for failures, warn for slow calls, info for normal
        if not success:
            self.logger.error("external_call_failed", **log_data)
        elif duration_ms > 5000:  # More than 5 seconds
            self.logger.warning("slow_external_call", **log_data)
        else:
            self.logger.info("external_call_completed", **log_data)

    def log_cache_operation(
        self,
        operation: str,
        key: str,
        hit: bool,
        duration_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Loga operações de cache.

        Args:
            operation: Tipo de operação (GET, SET, DELETE)
            key: Chave do cache
            hit: Se foi um hit ou miss
            duration_ms: Duração em milissegundos
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "cache_operation",
            "operation": operation,
            "key": key,
            "hit": hit,
        }

        if duration_ms is not None:
            log_data["duration_ms"] = round(duration_ms, 2)

        log_data.update(kwargs)

        self.logger.info("cache_operation_performed", **log_data)

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        source_ip: str | None = None,
        user_id: str | None = None,
        details: str | None = None,
        **kwargs,
    ) -> None:
        """Loga eventos de segurança.

        Args:
            event_type: Tipo de evento de segurança
            severity: Severidade (low, medium, high, critical)
            source_ip: IP de origem
            user_id: ID do usuário
            details: Detalhes do evento
            **kwargs: Contexto adicional
        """
        log_data = {
            "event_type": "security_event",
            "security_event_type": event_type,
            "severity": severity,
        }

        if source_ip:
            log_data["source_ip"] = source_ip

        if user_id:
            log_data["user_id"] = user_id

        if details:
            log_data["details"] = details

        log_data.update(kwargs)

        # Log with appropriate level based on severity
        if severity == "critical":
            self.logger.critical("security_event_detected", **log_data)
        elif severity == "high":
            self.logger.error("security_event_detected", **log_data)
        elif severity == "medium":
            self.logger.warning("security_event_detected", **log_data)
        else:
            self.logger.info("security_event_detected", **log_data)


def get_performance_logger(name: str | None = None) -> PerformanceLogger:
    """Obtém um performance logger.

    Args:
        name: Nome do logger

    Returns:
        PerformanceLogger configurado
    """
    return PerformanceLogger(get_logger(name))


# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================


def set_request_context(request: Request) -> None:
    """Define contexto da requisição para logging.

    Args:
        request: Requisição FastAPI
    """
    context = {
        "http_method": request.method,
        "http_path": request.url.path,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }

    # Add query parameters (filtered for sensitive data)
    if request.query_params:
        filtered_params = {}
        for key, value in request.query_params.items():
            if any(
                sensitive in key.lower()
                for sensitive in ["password", "token", "secret", "key", "auth"]
            ):
                filtered_params[key] = "***REDACTED***"
            else:
                filtered_params[key] = value
        context["query_params"] = filtered_params  # type: ignore[assignment]

    _current_request_ctx.set(context)


class SafeEncodingFormatter(logging.Formatter):
    """
    Logging formatter that prevents UnicodeEncodeError in non-UTF-8 streams.

    Applies fallback for unsupported characters, prioritizing readability.
    Replaces detectable emoji patterns with ASCII equivalents when needed.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Get base formatted message
        message = super().format(record)

        # Try to detect encoding from current context
        # This is a best-effort approach since logging doesn't always expose stream info
        encoding = None
        try:
            # Check if we can find a stream in the logging hierarchy
            import sys

            if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
                encoding = sys.stdout.encoding
        except Exception as _e:
            logger.debug("suppressed_exception", exc_info=True)  # was: pass

        if not can_encode(message, encoding=encoding):
            # Apply fallback: replace common emoji patterns
            message = (
                message.replace("✅", "[OK]")
                .replace("❌", "[ERR]")
                .replace("🚀", "[START]")
                .replace("🛑", "[STOP]")
            )
            # If still not encodable, use errors='replace' as last resort
            if not can_encode(message, encoding=encoding):
                try:
                    enc = encoding or "utf-8"
                    message = message.encode(enc, errors="replace").decode(enc)
                except Exception as _e:
                    # Can't use logger here (inside a formatter) - would cause recursion
                    # Just set a fallback message
                    message = "[ENCODING ERROR]"
        return message


class StructuredErrorLogger:
    """Structured error logger for consistent error logging."""

    @staticmethod
    def log_error(error: Exception, context: dict, level: str = "error") -> None:
        """
        Log error with structured context.

        Args:
            error: The exception to log
            context: Additional context information
            level: Log level (debug, info, warning, error, critical)
        """
        logger = get_logger(__name__)

        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context,
        }

        level_name = level.lower()
        if level_name == "debug":
            logger.debug("structured_error", **log_data)
        elif level_name == "info":
            logger.info("structured_error", **log_data)
        elif level_name == "warning":
            logger.warning("structured_error", **log_data)
        elif level_name == "critical":
            logger.critical("structured_error", **log_data)
        else:
            logger.error("structured_error", **log_data)


__all__ = [
    # Configuração
    "configure_structured_logging",
    "get_logger",
    "get_logger_adapter",
    "get_performance_logger",
    # Classes
    "LoggerAdapter",
    "PerformanceLogger",
    "StructuredErrorLogger",
    "SafeEncodingFormatter",
    # Processadores
    "add_correlation_id",
    "add_user_context",
    "add_request_context",
    "add_service_context",
    "add_timestamp",
    "add_log_level",
    "censor_sensitive_data",
    "add_request_metadata",
    # Context management
    "set_request_context",
]
