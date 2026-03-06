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
from pathlib import Path
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, cast

from fastapi import Request

try:
    import structlog  # type: ignore
    from structlog.types import EventDict, WrappedLogger  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore[assignment]
    EventDict = dict  # type: ignore[misc,assignment]
    WrappedLogger = Any  # type: ignore[misc,assignment]

from .encoding_utils import can_encode

logger = logging.getLogger(__name__)

StructlogProcessor = Callable[
    [Any, str, MutableMapping[str, Any]],
    Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...],
]

# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

_current_request_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_request_ctx", default=None
)

# ============================================================================
# CUSTOM PROCESSORS
# ============================================================================

def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    from resync.core.context import get_correlation_id

    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict

def add_user_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    from resync.core.context import get_user_id

    user_id = get_user_id()
    if user_id:
        event_dict["user_id"] = user_id
    return event_dict

def add_request_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    from resync.core.context import get_request_id

    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict

def add_trace_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    from resync.core.context import get_trace_id

    trace_id = get_trace_id()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict

def add_service_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    from resync.settings import settings

    event_dict["service_name"] = settings.PROJECT_NAME
    event_dict["environment"] = settings.environment.value
    event_dict["version"] = settings.PROJECT_VERSION
    return event_dict

def add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return event_dict

def add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict

def censor_sensitive_data(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    sensitive_patterns = {
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
        "database_url",
        "db_url",
        "connection_string",
        "conn_str",
        "valkey_url",
        "valkey_password",
        "encryption_key",
        "signing_key",
        "jwt_secret",
        "session_secret",
        "cookie_secret",
        "admin_password",
        "tws_password",
    }

    import re

    _sensitive_value_patterns = [
        re.compile(r'(?:password|pwd|passwd)=["\']?[^"\'&\s]*["\']?', re.IGNORECASE),
        re.compile(r'(?:token|secret|key)=["\']?[^"\'&\s]*["\']?', re.IGNORECASE),
        re.compile(r"(?:authorization)[:\s]*bearer\s+[^\s]+", re.IGNORECASE),
        re.compile(r"(?:basic)\s+[a-zA-Z0-9+/=]+", re.IGNORECASE),
        # JWT token pattern (eyJ...): JWT tokens are base64url encoded
        re.compile(r"eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*", re.IGNORECASE),
    ]

    def censor_dict(d: dict[str, Any] | Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in d.items():
            key_lower = key.lower()
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
                censored_value = value
                for pattern in _sensitive_value_patterns:
                    censored_value = pattern.sub("***REDACTED***", censored_value)
                result[key] = censored_value
            else:
                result[key] = value
        return result

    return censor_dict(event_dict)

def add_request_metadata(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    request_ctx = _current_request_ctx.get()
    if request_ctx:
        event_dict.update(request_ctx)
    return event_dict

def protect_log_injection(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    for key, value in event_dict.items():
        if isinstance(value, str):
            if "\n" in value or "\r" in value:
                event_dict[key] = value.replace("\n", "\\n").replace("\r", "\\r")
    return event_dict

# ============================================================================
# CONFIGURATION
# ============================================================================

def configure_structured_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    development_mode: bool = False,
    persist_logs_path: str | None = None,
) -> None:
    level = getattr(logging, log_level.upper())

    if structlog is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "structlog is required for Resync logging. Install it (see requirements.txt)."
        )

    shared_processors: list[StructlogProcessor] = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_log_level,
        add_correlation_id,
        add_trace_id,
        add_user_context,
        add_request_context,
        add_request_metadata,
        add_service_context,
        protect_log_injection,
        structlog.processors.StackInfoRenderer(),
        censor_sensitive_data,
    ]

    # Keep pretty exceptions in development console output.
    # format_exc_info is only needed for JSON pipelines and emits a warning with ConsoleRenderer.
    if not (development_mode or not json_logs):
        shared_processors.append(structlog.processors.format_exc_info)

    renderer: Any
    if development_mode or not json_logs:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    configure_processors: Iterable[StructlogProcessor] = shared_processors + [
        cast(
            StructlogProcessor,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ),
    ]

    structlog.configure(
        processors=configure_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=cast(Sequence[StructlogProcessor], shared_processors),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # P0 FIX: Always persist startup warnings/errors to logs/startup.log and logs/server_stdout.log.
    # This replaces the old hardcoded path approach but maintains log persistence.
    # The logs/ directory is configurable via LOG_DIR env var (defaults to ./logs).
    # This is intentional - we always want startup logs for debugging in all environments.
    import os as _os
    log_dir = Path(_os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        max_bytes = int(_os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
        backup_count = int(_os.getenv("LOG_BACKUP_COUNT", "5"))

        # Startup log - captures warnings/errors during boot (level=WARNING+)
        # Use mode="w" to truncate on each start, while still rotating by size.
        startup_handler = RotatingFileHandler(
            str(log_dir / "startup.log"),
            mode="w",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        startup_handler.setLevel(logging.WARNING)
        startup_handler.setFormatter(formatter)
        root_logger.addHandler(startup_handler)

        # Server stdout log - captures all application logs
        stdout_handler = RotatingFileHandler(
            str(log_dir / "server_stdout.log"),
            mode="w",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
    except OSError as e:
        # Don't crash the service if we can't persist logs.
        # Logging is already configured to stdout; keep running.
        root_logger.warning(
            "log_file_persistence_unavailable",
            extra={"log_dir": str(log_dir), "error": str(e)},
        )

    # Legacy: still support explicit persist_logs_path for specific use cases
    if persist_logs_path:
        try:
            p = Path(persist_logs_path)
            # Ensure parent directory exists to prevent FileNotFoundError on boot.
            if p.parent:
                p.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(p), mode="w")
        except OSError as e:
            # Don't crash the service if we can't persist logs.
            # Logging is already configured to stdout; keep running.
            root_logger.warning(
                "persist_log_file_unavailable",
                extra={"path": persist_logs_path, "error": str(e)},
            )
        else:
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

class FallbackLogger:
    """Structlog-compatible logger wrapper used when structlog isn't installed.

    Many parts of the codebase use structlog-style calls like:
        logger.info("event_name", key=value)

    The standard-library logging API does not accept arbitrary keyword args, so
    this wrapper serializes kwargs into `extra` and a readable message.
    """

    def __init__(self, base: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._base = base
        self._context = context or {}

    def bind(self, **kwargs: Any) -> "FallbackLogger":
        ctx = dict(self._context)
        ctx.update(kwargs)
        return FallbackLogger(self._base, ctx)

    def unbind(self, *keys: str) -> "FallbackLogger":
        ctx = dict(self._context)
        for k in keys:
            ctx.pop(k, None)
        return FallbackLogger(self._base, ctx)

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        payload = dict(self._context)
        payload.update(kwargs)
        # Keep message readable even without JSON formatting.
        if payload:
            msg = f"{event} | {payload}"
        else:
            msg = event
        self._base.log(level, msg)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)

    # Common structlog aliases used in some codebases
    warn = warning


def get_logger(name: str | None = None):
    """Return a structlog-compatible logger.

    If structlog is available, returns structlog.get_logger().
    Otherwise returns a FallbackLogger wrapper around stdlib logging.
    """
    if structlog is None:
        base = logging.getLogger(name or __name__)
        return FallbackLogger(base)
    return structlog.get_logger(name) if name else structlog.get_logger()
# ============================================================================
# LOGGING HELPERS
# ============================================================================

class LoggerAdapter:
    def __init__(self, logger: structlog.BoundLogger) -> None:
        self.logger = logger

    def debug(self, event: str, **kwargs) -> None:
        self.logger.debug(event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        self.logger.info(event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self.logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        self.logger.error(event, **kwargs)

    def critical(self, event: str, **kwargs) -> None:
        self.logger.critical(event, **kwargs)

    def exception(self, event: str, **kwargs) -> None:
        self.logger.exception(event, **kwargs)

    def bind(self, **kwargs) -> "LoggerAdapter":
        return LoggerAdapter(self.logger.bind(**kwargs))

def get_logger_adapter(name: str | None = None) -> LoggerAdapter:
    return LoggerAdapter(get_logger(name))

class PerformanceLogger:
    def __init__(self, logger: structlog.BoundLogger) -> None:
        self.logger = logger

    def log_request(
        self, method: str, path: str, status_code: int, duration_ms: float, **kwargs
    ) -> None:
        self.logger.info(
            "http_request",
            method=method,
            path=path,
            status=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )

def get_performance_logger(name: str | None = None) -> PerformanceLogger:
    return PerformanceLogger(get_logger(name))

def set_request_context(request: Request) -> None:
    context = {
        "http_method": request.method,
        "http_path": request.url.path,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    _current_request_ctx.set(context)

class SafeEncodingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not can_encode(message):
            return "[ENCODING ERROR]"
        return message

class StructuredErrorLogger:
    @staticmethod
    def log_error(error: Exception, context: dict, level: str = "error") -> None:
        logger = get_logger(__name__)
        getattr(logger, level.lower())("structured_error", error=str(error), **context)

__all__ = [
    "configure_structured_logging",
    "get_logger",
    "get_logger_adapter",
    "get_performance_logger",
    "LoggerAdapter",
    "PerformanceLogger",
    "StructuredErrorLogger",
    "SafeEncodingFormatter",
    "add_correlation_id",
    "add_user_context",
    "add_request_context",
    "add_service_context",
    "add_timestamp",
    "add_log_level",
    "censor_sensitive_data",
    "add_request_metadata",
    "set_request_context",
]
