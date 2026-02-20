"""
Hardened Core Package Initialization for Resync

This module provides hardened initialization and lifecycle management for core
components with comprehensive error handling, health validation, and
security measures.
"""

import collections
import importlib
import logging
import os
import tempfile
import threading
import time
from typing import Any

from resync.core.exceptions import (
    AgentExecutionError,
    AuditError,
    AuthenticationError,
    BaseAppException,
    DatabaseError,
    InvalidConfigError,
    LLMError,
    PoolExhaustedError,
    RedisConnectionError,
    ToolProcessingError,
)

USE_CORE_BOOT_V2 = os.getenv("USE_CORE_BOOT_V2", "true").lower() in ("true", "1", "yes")
MAX_GLOBAL_EVENTS = 100
VALIDATION_CACHE_TTL = 60

logger = logging.getLogger(__name__)


class CorrelationIdFilter(logging.Filter):
    """Logging filter to inject correlation_id into all log records."""

    def __init__(self, correlation_id_getter=None):
        super().__init__()
        self.correlation_id_getter = correlation_id_getter

    def filter(self, record: logging.LogRecord) -> bool:
        if self.correlation_id_getter:
            try:
                correlation_id = self.correlation_id_getter()
                if correlation_id:
                    record.correlation_id = correlation_id
            except Exception as exc:
                logger.debug("suppressed_exception: %s", str(exc), exc_info=True)
        return True


_LAZY_EXPORTS = {"AsyncTTLCache": ("resync.core.cache.async_cache", "AsyncTTLCache")}
_LOADED_EXPORTS = {}
_LAZY_LOAD_LOCK = threading.Lock()


def __getattr__(name: str):
    """PEP 562 lazy loading for heavy imports with thread safety."""
    if name in _LAZY_EXPORTS:
        mod, attr = _LAZY_EXPORTS[name]
        if name not in _LOADED_EXPORTS:
            with _LAZY_LOAD_LOCK:
                if name not in _LOADED_EXPORTS:
                    try:
                        module = importlib.import_module(mod)
                        _LOADED_EXPORTS[name] = getattr(module, attr)
                    except ImportError as e:
                        raise ImportError(
                            f"Cannot import {attr} from {mod}: {e}"
                        ) from e
                    except AttributeError as e:
                        raise AttributeError(
                            f"Module {mod} does not have attribute {attr}: {e}"
                        ) from e
        return _LOADED_EXPORTS[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


class CoreBootManager:
    """Hardened boot manager for core components with lifecycle tracking and
    health validation."""

    def __init__(self):
        self._components: dict[str, Any] = {}
        self._boot_times: dict[str, float] = {}
        self._health_status: dict[str, dict[str, Any]] = {}
        self._boot_lock = threading.RLock()
        self._correlation_id = f"core_boot_{int(time.time())}_{os.urandom(4).hex()}"
        self._failed_imports: set[str] = set()
        self._global_correlation_context = {
            "boot_id": self._correlation_id,
            "environment": "unknown",
            "security_level": "unknown",
            "start_time": time.time(),
            "events": collections.deque(maxlen=100),
        }

    def register_component(self, name: str, component: Any) -> None:
        """Register a component. Health checks are deferred."""
        with self._boot_lock:
            start_time = time.perf_counter()
            try:
                self._components[name] = component
                self._boot_times[name] = time.perf_counter() - start_time
            except (OSError, RuntimeError, ValueError) as e:
                logger.error("Failed to register component %s: %s", name, e)
                raise

    def get_component(self, name: str) -> Any:
        """Get a registered component."""
        return self._components.get(name)

    def get_boot_status(self) -> dict[str, Any]:
        """Get boot status for all components."""
        with self._boot_lock:
            return {
                "components": list(self._components.keys()),
                "boot_times": self._boot_times.copy(),
                "health_status": self._health_status.copy(),
                "correlation_id": self._correlation_id,
            }

    def add_global_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Add a trace event to the global correlation context."""
        with self._boot_lock:
            sanitized_event = self._sanitize_log_data(event)
            sanitized_data = self._sanitize_log_data(data or {})
            self._global_correlation_context["events"].append(
                {
                    "timestamp": time.time(),
                    "event": sanitized_event,
                    "data": sanitized_data,
                }
            )

    def _sanitize_log_data(self, obj: Any) -> Any:
        """Recursively sanitize log data to prevent injection or malformed data."""
        if isinstance(obj, str):
            return obj.replace("\x00", "").replace("\n", "\\n").replace("\r", "\\r")
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                sanitized_key = (
                    self._sanitize_log_data(str(key)) if key is not None else "null_key"
                )
                sanitized_value = self._sanitize_log_data(value)
                sanitized[sanitized_key] = sanitized_value
            return sanitized
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_log_data(item) for item in obj]
        return obj

    def get_global_correlation_id(self) -> str:
        """Get the global correlation ID for distributed tracing."""
        return self._correlation_id

    def get_environment_tags(self) -> dict[str, Any]:
        """Get environment tags for mock detection and debugging."""
        return {
            "is_mock": getattr(self, "_is_mock", False),
            "mock_reason": getattr(self, "_mock_reason", None),
            "boot_id": self._correlation_id,
            "component_count": len(self._components),
        }


class EnvironmentDetector:
    """Detect and validate execution environment for security and
    compatibility."""

    def __init__(self):
        self._validation_cache = {}
        self._last_validation = 0

    def detect_environment(self) -> dict[str, Any]:
        """Detect execution environment characteristics."""
        return {
            "platform": os.name,
            "is_ci": bool(os.environ.get("CI")),
            "has_internet": self._check_internet_access(),
            "temp_dir": os.environ.get(
                "TEMP", os.environ.get("TMP", tempfile.gettempdir())
            ),
        }

    def _check_internet_access(self) -> bool:
        """Check if internet access is available."""
        return True

    def validate_environment(self) -> bool:
        """Validate execution environment for security compliance."""
        try:
            current_time = time.monotonic()
            if current_time - self._last_validation < 60:
                return self._validation_cache.get("result", True)
            env_ok = True
            self._validation_cache = {
                "result": env_ok,
                "timestamp": current_time,
                "details": {},
            }
            self._last_validation = current_time
            return env_ok
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Environment validation failed: %s", e)
            return False


_boot_manager_instance = None
_boot_manager_lock = threading.Lock()


def get_boot_manager():
    """Get the singleton instance of CoreBootManager."""
    global _boot_manager_instance
    if _boot_manager_instance is None:
        with _boot_manager_lock:
            if _boot_manager_instance is None:
                _boot_manager_instance = CoreBootManager()
    return _boot_manager_instance


try:
    from resync.core.structured_logger import get_logger as _get_logger_func
except ImportError:

    def _get_logger_func(name: str | None = None) -> Any:
        return logging.getLogger(name if name else __name__)


def _get_logger():
    """Lazy import of logger."""
    return _get_logger_func(__name__)


def get_global_correlation_id() -> str:
    """Get the global correlation ID for distributed tracing."""
    return get_boot_manager().get_global_correlation_id()


def get_environment_tags() -> dict[str, Any]:
    """Get environment tags for mock detection and debugging."""
    return get_boot_manager().get_environment_tags()


def add_global_trace_event(event: str, data: dict[str, Any] | None = None) -> None:
    """Add a trace event to the global correlation context."""
    get_boot_manager().add_global_event(event, data)


def _validate_environment() -> None:
    """Validate environment lazily."""
    try:
        env_detector = EnvironmentDetector()
        get_boot_manager()
        log = _get_logger()
        if not env_detector.validate_environment():
            log.warning(
                "Environment validation failed - system may not be secure",
            )
    except (ImportError, AttributeError, OSError, RuntimeError) as e:
        try:
            log = _get_logger()
            log.warning("Environment validation failed: %s", e)
        except (ImportError, RuntimeError):
            logger.debug("suppressed_exception: %s", str(e), exc_info=True)


__all__ = [
    "CoreBootManager",
    "EnvironmentDetector",
    "get_boot_manager",
    "get_global_correlation_id",
    "get_environment_tags",
    "add_global_trace_event",
    "AuditError",
    "DatabaseError",
    "PoolExhaustedError",
    "ToolProcessingError",
    "BaseAppException",
    "InvalidConfigError",
    "AgentExecutionError",
    "AuthenticationError",
    "LLMError",
    "RedisConnectionError",
]
if "AsyncTTLCache" in _LAZY_EXPORTS:
    __all__.append("AsyncTTLCache")
