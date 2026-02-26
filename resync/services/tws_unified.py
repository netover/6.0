"""Unified TWS Client Access Module."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
import structlog
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from resync.core.exceptions import (
    CircuitBreakerError,
    TWSAuthenticationError,
    TWSConnectionError,
    TWSTimeoutError,
)
from resync.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    RetryWithBackoff,
    TimeoutManager,
)

logger = structlog.get_logger(__name__)

_UNRECOVERABLE_EXCEPTIONS = (
    SystemExit,
    KeyboardInterrupt,
    asyncio.CancelledError,
)
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError)

class TWSClientState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"

class TWSClientConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TWS_", env_file=".env", extra="ignore"
    )

    base_url: str = "http://localhost:31182"
    username: str = ""
    password: SecretStr = SecretStr("")
    engine_name: str = "DWC"
    engine_owner: str = ""

    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0

    @classmethod
    def from_settings(cls) -> TWSClientConfig:
        from resync.settings import settings

        password = getattr(settings, "tws_password", "")
        return cls(
            base_url=f"http://{settings.tws_host}:{settings.tws_port}",
            username=settings.tws_username,
            password=SecretStr(password),
            engine_name=settings.tws_engine_name,
            engine_owner=getattr(settings, "tws_engine_owner", ""),
            connect_timeout=getattr(settings, "tws_connect_timeout", 10.0),
            read_timeout=getattr(settings, "tws_request_timeout", 30.0),
        )

    def get_password(self) -> str:
        return self.password.get_secret_value()

@dataclass
class TWSClientMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_error: str | None = None

class OptimizedTWSClientAdapter:
    """Compatibility adapter over OptimizedTWSClient real API."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def close(self) -> None:
        await self._inner.close()

    async def get_engine_info(self) -> dict[str, Any]:
        data = await self._inner.get_engine_info()
        return data if isinstance(data, dict) else {"data": data}

    async def get_system_status(self) -> dict[str, Any]:
        info = await self.get_engine_info()
        return {"status": "OK", "engine": info}

    async def get_jobs(self, **params: Any) -> list[dict[str, Any]]:
        data = await self._inner.query_current_plan_jobs(**params)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ("jobs", "items", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return [d for d in value if isinstance(d, dict)]
        return []

    async def get_job(self, job_name: str) -> dict[str, Any]:
        jobs = await self.get_jobs(q=job_name, limit=1)
        return jobs[0] if jobs else {"job_name": job_name, "status": "UNKNOWN"}

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        job = await self.get_job(job_name)
        return {"job_name": job_name, "status": job.get("status", "UNKNOWN")}

    async def get_workstations(self) -> list[dict[str, Any]]:
        data = await self._inner.query_workstations(limit=200)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ("workstations", "items", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return [d for d in value if isinstance(d, dict)]
        return []

    async def get_plan(self) -> dict[str, Any]:
        streams = await self._inner.query_jobstreams(limit=50)
        return {"streams": streams}

class UnifiedTWSClient:
    def __init__(self, config: TWSClientConfig | None = None):
        self.config = config or TWSClientConfig.from_settings()
        self._client: OptimizedTWSClientAdapter | None = None
        self._state = TWSClientState.DISCONNECTED
        self._metrics = TWSClientMetrics()
        self._lock = asyncio.Lock()
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout,
                name="tws_client",
            )
        )
        self._retry_handler = RetryWithBackoff(
            RetryConfig(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                jitter=True,
                expected_exceptions=(TWSConnectionError, TWSTimeoutError),
            )
        )

    @property
    def state(self) -> TWSClientState:
        return (
            TWSClientState.CIRCUIT_OPEN
            if self._circuit_breaker.state.value == "open"
            else self._state
        )

    async def connect(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            self._state = TWSClientState.CONNECTING
            try:
                from resync.services.tws_service import OptimizedTWSClient

                raw = OptimizedTWSClient(
                    base_url=self.config.base_url,
                    username=self.config.username,
                    password=self.config.get_password(),
                    engine_name=self.config.engine_name,
                    engine_owner=self.config.engine_owner,
                )
                self._client = OptimizedTWSClientAdapter(raw)
                await self._verify_connection()
                self._state = TWSClientState.CONNECTED
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                self._client = None
                self._state = TWSClientState.ERROR
                self._metrics.last_error = str(e)
                if isinstance(e, _UNRECOVERABLE_EXCEPTIONS + _PROGRAMMING_ERRORS):
                    raise
                raise TWSConnectionError(f"Failed to connect to TWS: {e}") from e

    async def _verify_connection(self) -> None:
        if self._client is None:
            raise TWSConnectionError("Client not initialized")
        try:
            await TimeoutManager.with_timeout(
                self._client.get_engine_info(), self.config.connect_timeout
            )
        except asyncio.TimeoutError as e:
            raise TWSTimeoutError("Connection verification timed out") from e
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            if isinstance(e, _UNRECOVERABLE_EXCEPTIONS):
                raise
            if isinstance(e, httpx.HTTPStatusError):
                response = getattr(e, "response", None)
                if response is not None and response.status_code == 401:
                    raise TWSAuthenticationError("TWS authentication failed") from e
            if "401" in str(e) or "auth" in str(e).lower():
                raise TWSAuthenticationError("TWS authentication failed") from e
            raise

    async def disconnect(self) -> None:
        async with self._lock:
            if self._client is not None:
                try:
                    await self._client.close()
                finally:
                    self._client = None
                    self._state = TWSClientState.DISCONNECTED

    async def _execute_with_resilience(
        self, operation: str, func: Callable[[], Awaitable[Any]]
    ) -> Any:
        start_time = datetime.now(timezone.utc)
        self._metrics.total_requests += 1
        retries = 0

        async def _wrapped() -> Any:
            nonlocal retries
            if self._client is None:
                await self.connect()
            try:
                return await TimeoutManager.with_timeout(
                    func(), self.config.read_timeout
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
                retries += 1
                raise

        try:
            result = await self._circuit_breaker.call(
                self._retry_handler.execute, _wrapped
            )
            self._metrics.successful_requests += 1
            self._metrics.last_success = datetime.now(timezone.utc)
            self._metrics.total_latency_ms += (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            self._metrics.retries += max(0, retries - 1)
            return result
        except CircuitBreakerError:
            self._metrics.circuit_breaker_trips += 1
            self._state = TWSClientState.CIRCUIT_OPEN
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            if isinstance(e, _UNRECOVERABLE_EXCEPTIONS + _PROGRAMMING_ERRORS):
                raise
            self._metrics.failed_requests += 1
            self._metrics.last_failure = datetime.now(timezone.utc)
            self._metrics.last_error = str(e)
            raise

    async def get_system_status(self) -> dict[str, Any]:
        return await self._execute_with_resilience(
            "get_system_status", lambda: self._require_client().get_system_status()
        )

    async def get_engine_info(self) -> dict[str, Any]:
        return await self._execute_with_resilience(
            "get_engine_info", lambda: self._require_client().get_engine_info()
        )

    async def get_jobs(self, **params: Any) -> list[dict[str, Any]]:
        return await self._execute_with_resilience(
            "get_jobs", lambda: self._require_client().get_jobs(**params)
        )

    async def get_job(self, job_name: str) -> dict[str, Any]:
        return await self._execute_with_resilience(
            "get_job", lambda: self._require_client().get_job(job_name)
        )

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        return await self._execute_with_resilience(
            "get_job_status", lambda: self._require_client().get_job_status(job_name)
        )

    async def get_workstations(self) -> list[dict[str, Any]]:
        return await self._execute_with_resilience(
            "get_workstations", lambda: self._require_client().get_workstations()
        )

    async def get_plan(self) -> dict[str, Any]:
        return await self._execute_with_resilience(
            "get_plan", lambda: self._require_client().get_plan()
        )

    async def health_check(self) -> bool:
        try:
            await self.get_engine_info()
            return True
        except _PROGRAMMING_ERRORS:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            return False

    def _require_client(self) -> OptimizedTWSClientAdapter:
        if self._client is None:
            raise TWSConnectionError("TWS client is not connected")
        return self._client

    def get_metrics_summary(self) -> dict[str, Any]:
        success_rate = (
            self._metrics.successful_requests / self._metrics.total_requests
            if self._metrics.total_requests
            else 0.0
        )
        avg_latency = (
            self._metrics.total_latency_ms / self._metrics.successful_requests
            if self._metrics.successful_requests
            else 0.0
        )
        return {
            "state": self.state.value,
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "success_rate": success_rate,
            "retries": self._metrics.retries,
            "avg_latency_ms": avg_latency,
            "circuit_breaker_trips": self._metrics.circuit_breaker_trips,
            "last_success": self._metrics.last_success.isoformat()
            if self._metrics.last_success
            else None,
            "last_failure": self._metrics.last_failure.isoformat()
            if self._metrics.last_failure
            else None,
            "last_error": self._metrics.last_error,
        }

_tws_client_instance: UnifiedTWSClient | None = None
# Eagerly initialised — never None — eliminates TOCTOU race on first concurrent access
_tws_client_lock: asyncio.Lock = asyncio.Lock()

async def get_tws_client() -> UnifiedTWSClient:
    global _tws_client_instance
    if _tws_client_instance is None:
        async with _tws_client_lock:
            if _tws_client_instance is None:
                client = UnifiedTWSClient()
                await client.connect()
                _tws_client_instance = client
    return _tws_client_instance

async def reset_tws_client() -> None:
    global _tws_client_instance
    async with _tws_client_lock:
        if _tws_client_instance is not None:
            await _tws_client_instance.disconnect()
            _tws_client_instance = None

@asynccontextmanager
async def tws_client_context() -> AsyncIterator[UnifiedTWSClient]:
    yield await get_tws_client()

class MockTWSClient(UnifiedTWSClient):
    def __init__(self, responses: dict[str, Any] | None = None):
        self.config = TWSClientConfig()
        self._client = None
        self._state = TWSClientState.CONNECTED
        self._metrics = TWSClientMetrics()
        self._lock = asyncio.Lock()
        self._responses = responses or {}
        self._calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._circuit_breaker = CircuitBreaker(CircuitBreakerConfig(name="mock_tws"))
        self._retry_handler = RetryWithBackoff(
            RetryConfig(max_retries=1, base_delay=0.1)
        )

    async def connect(self) -> None:
        self._state = TWSClientState.CONNECTED

    async def disconnect(self) -> None:
        self._state = TWSClientState.DISCONNECTED

    def _mock_response(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        self._calls.append((operation, args, kwargs))
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1
        defaults: dict[str, Any] = {
            "get_system_status": {"status": "OK", "engine": "DWC"},
            "get_engine_info": {"name": "DWC", "version": "9.5"},
            "get_jobs": [],
            "get_job": {"name": "TEST_JOB", "status": "SUCC"},
            "get_job_status": {"status": "SUCC"},
            "get_workstations": [],
            "get_plan": {"streams": []},
        }
        return self._responses.get(operation, defaults.get(operation, {}))

    async def get_system_status(self) -> dict[str, Any]:
        return self._mock_response("get_system_status")

    async def get_engine_info(self) -> dict[str, Any]:
        return self._mock_response("get_engine_info")

    async def get_jobs(self, **params: Any) -> list[dict[str, Any]]:
        return self._mock_response("get_jobs", **params)

    async def get_job(self, job_name: str) -> dict[str, Any]:
        return self._mock_response("get_job", job_name)

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        return self._mock_response("get_job_status", job_name)

    async def get_workstations(self) -> list[dict[str, Any]]:
        return self._mock_response("get_workstations")

    async def get_plan(self) -> dict[str, Any]:
        return self._mock_response("get_plan")

    def get_calls(self) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
        return self._calls

async def use_mock_tws_client(responses: dict[str, Any] | None = None) -> None:
    global _tws_client_instance
    async with _tws_client_lock:
        _tws_client_instance = MockTWSClient(responses)

def get_mock_tws_client() -> MockTWSClient | None:
    if isinstance(_tws_client_instance, MockTWSClient):
        return _tws_client_instance
    return None

__all__ = [
    "UnifiedTWSClient",
    "TWSClientConfig",
    "TWSClientState",
    "TWSClientMetrics",
    "MockTWSClient",
    "get_tws_client",
    "reset_tws_client",
    "tws_client_context",
    "use_mock_tws_client",
    "get_mock_tws_client",
]
