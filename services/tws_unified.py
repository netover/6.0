"""
Unified TWS Client Access Module

v5.4.2: Consolidates all TWS client access into a single module with:
- Circuit breaker protection
- Retry with exponential backoff
- Health monitoring
- Singleton management
- Mock support for testing

This module replaces:
- Direct OptimizedTWSClient instantiation
- Legacy TWS client patterns
- Scattered TWS client factories

Usage:
    from resync.services.tws_unified import get_tws_client, tws_client_context

    # Simple access
    client = await get_tws_client()

    # Context manager with automatic cleanup
    async with tws_client_context() as client:
        status = await client.get_system_status()

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

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


# =============================================================================
# ENUMS & TYPES
# =============================================================================


class TWSClientState(str, Enum):
    """State of the TWS client."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class TWSClientConfig:
    """Configuration for TWS client."""

    # Connection settings
    base_url: str = "http://localhost:31182"
    username: str = ""
    password: str = ""
    engine_name: str = "DWC"
    engine_owner: str = ""

    # Timeout settings (seconds)
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0

    # Health check
    health_check_interval: int = 30

    @classmethod
    def from_settings(cls) -> TWSClientConfig:
        """Create config from application settings."""
        from resync.settings import settings

        return cls(
            base_url=f"http://{settings.tws_host}:{settings.tws_port}",
            username=settings.tws_username,
            password=settings.tws_password,
            engine_name=settings.tws_engine_name,
            engine_owner=getattr(settings, "tws_engine_owner", ""),
            connect_timeout=getattr(settings, "tws_connect_timeout", 10.0),
            read_timeout=getattr(settings, "tws_request_timeout", 30.0),
        )


@dataclass
class TWSClientMetrics:
    """Metrics for TWS client."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_error: str | None = None


# =============================================================================
# UNIFIED TWS CLIENT WRAPPER
# =============================================================================


class UnifiedTWSClient:
    """
    Unified TWS Client with built-in resilience patterns.

    Features:
    - Circuit breaker protection
    - Automatic retry with exponential backoff
    - Timeout management
    - Health monitoring
    - Metrics collection

    This class wraps OptimizedTWSClient and adds resilience.
    """

    def __init__(self, config: TWSClientConfig | None = None):
        """
        Initialize unified TWS client.

        Args:
            config: Client configuration. If None, loads from settings.
        """
        self.config = config or TWSClientConfig.from_settings()
        self._client: Any | None = None
        self._state = TWSClientState.DISCONNECTED
        self._metrics = TWSClientMetrics()
        self._lock = asyncio.Lock()

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout,
                name="tws_client",
            )
        )

        # Initialize retry handler
        self._retry_handler = RetryWithBackoff(
            RetryConfig(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                jitter=True,
                expected_exceptions=(TWSConnectionError, TWSTimeoutError),
            )
        )

        logger.info(
            "unified_tws_client_initialized",
            base_url=self.config.base_url,
            circuit_threshold=self.config.circuit_failure_threshold,
        )

    @property
    def state(self) -> TWSClientState:
        """Get current client state."""
        if self._circuit_breaker.state.value == "open":
            return TWSClientState.CIRCUIT_OPEN
        return self._state

    @property
    def metrics(self) -> TWSClientMetrics:
        """Get client metrics."""
        return self._metrics

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._state == TWSClientState.CONNECTED and self._client is not None

    async def connect(self) -> None:
        """
        Establish connection to TWS.

        Raises:
            TWSConnectionError: If connection fails
            TWSAuthenticationError: If authentication fails
        """
        async with self._lock:
            if self._client is not None:
                return

            self._state = TWSClientState.CONNECTING

            try:
                from resync.services.tws_service import OptimizedTWSClient

                self._client = OptimizedTWSClient(
                    base_url=self.config.base_url,
                    username=self.config.username,
                    password=self.config.password,
                    engine_name=self.config.engine_name,
                    engine_owner=self.config.engine_owner,
                )

                # Verify connection with a health check
                await self._verify_connection()

                self._state = TWSClientState.CONNECTED
                logger.info("tws_client_connected", base_url=self.config.base_url)

            except Exception as e:
                # Re-raise critical system exceptions and programming errors
                if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError, TypeError, AttributeError, NameError)):
                    raise
                self._state = TWSClientState.ERROR
                self._metrics.last_error = str(e)
                logger.error("tws_client_connection_failed", error=str(e))
                raise TWSConnectionError(f"Failed to connect to TWS: {e}") from e

    async def _verify_connection(self) -> None:
        """Verify connection is working."""
        if self._client is None:
            raise TWSConnectionError("Client not initialized")

        try:
            # Try to get engine info as a health check
            await TimeoutManager.with_timeout(
                self._client.get_engine_info(),
                self.config.connect_timeout,
            )
        except asyncio.TimeoutError as e:
            raise TWSTimeoutError("Connection verification timed out") from e
        except Exception as e:
            # Re-raise critical system exceptions
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                raise
            if "401" in str(e) or "auth" in str(e).lower():
                raise TWSAuthenticationError("TWS authentication failed") from e
            raise

    async def disconnect(self) -> None:
        """Close connection to TWS."""
        async with self._lock:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception as e:
                    # Re-raise critical system exceptions
                    if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)):
                        raise
                    logger.warning("tws_client_close_error", error=str(e))
                finally:
                    self._client = None
                    self._state = TWSClientState.DISCONNECTED
                    logger.info("tws_client_disconnected")

    async def _execute_with_resilience(self, operation: str, func, *args, **kwargs) -> Any:
        """
        Execute operation with circuit breaker and retry.

        Args:
            operation: Name of the operation (for logging)
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            CircuitBreakerError: If circuit is open
            TWSConnectionError: If operation fails after retries
        """
        start_time = datetime.now(timezone.utc)
        self._metrics.total_requests += 1

        async def _wrapped():
            if self._client is None:
                await self.connect()

            return await TimeoutManager.with_timeout(
                func(*args, **kwargs),
                self.config.read_timeout,
            )

        try:
            # Execute with circuit breaker and retry
            result = await self._circuit_breaker.call(
                self._retry_handler.execute,
                _wrapped,
            )

            # Update metrics on success
            self._metrics.successful_requests += 1
            self._metrics.last_success = datetime.now(timezone.utc)
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._metrics.total_latency_ms += latency

            logger.debug(
                "tws_operation_success",
                operation=operation,
                latency_ms=latency,
            )

            return result

        except CircuitBreakerError:
            self._metrics.circuit_breaker_trips += 1
            self._state = TWSClientState.CIRCUIT_OPEN
            logger.warning("tws_circuit_breaker_open", operation=operation)
            raise

        except Exception as e:
            # Re-raise critical system exceptions and programming errors
            if isinstance(e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError, TypeError, AttributeError, NameError)):
                raise
            
            self._metrics.failed_requests += 1
            self._metrics.last_failure = datetime.now(timezone.utc)
            self._metrics.last_error = str(e)

            logger.error(
                "tws_operation_failed",
                operation=operation,
                error=str(e),
            )
            raise

    # =========================================================================
    # TWS API Methods (Delegated to underlying client)
    # =========================================================================

    async def get_system_status(self) -> dict[str, Any]:
        """Get TWS system status."""
        return await self._execute_with_resilience(
            "get_system_status",
            self._client.get_system_status,
        )

    async def get_engine_info(self) -> dict[str, Any]:
        """Get TWS engine information."""
        return await self._execute_with_resilience(
            "get_engine_info",
            self._client.get_engine_info,
        )

    async def get_jobs(self, **params) -> list[dict[str, Any]]:
        """Get list of jobs."""
        return await self._execute_with_resilience(
            "get_jobs",
            self._client.get_jobs,
            **params,
        )

    async def get_job(self, job_name: str) -> dict[str, Any]:
        """Get specific job details."""
        return await self._execute_with_resilience(
            "get_job",
            self._client.get_job,
            job_name,
        )

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Get job status."""
        return await self._execute_with_resilience(
            "get_job_status",
            self._client.get_job_status,
            job_name,
        )

    async def get_workstations(self) -> list[dict[str, Any]]:
        """Get list of workstations."""
        return await self._execute_with_resilience(
            "get_workstations",
            self._client.get_workstations,
        )

    async def get_plan(self) -> dict[str, Any]:
        """Get current plan."""
        return await self._execute_with_resilience(
            "get_plan",
            self._client.get_plan,
        )

    async def health_check(self) -> bool:
        """
        Check if TWS is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.get_engine_info()
            return True
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return False

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of client metrics."""
        avg_latency = (
            self._metrics.total_latency_ms / self._metrics.successful_requests
            if self._metrics.successful_requests > 0
            else 0
        )

        return {
            "state": self.state.value,
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "success_rate": (
                self._metrics.successful_requests / self._metrics.total_requests
                if self._metrics.total_requests > 0
                else 0
            ),
            "avg_latency_ms": avg_latency,
            "circuit_breaker_trips": self._metrics.circuit_breaker_trips,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "last_success": (
                self._metrics.last_success.isoformat() if self._metrics.last_success else None
            ),
            "last_failure": (
                self._metrics.last_failure.isoformat() if self._metrics.last_failure else None
            ),
            "last_error": self._metrics.last_error,
        }


# =============================================================================
# SINGLETON & ACCESS FUNCTIONS
# =============================================================================


_tws_client_instance: UnifiedTWSClient | None = None
_tws_client_lock: asyncio.Lock | None = None
_tws_client_lock_loop: asyncio.AbstractEventLoop | None = None


async def get_tws_client() -> UnifiedTWSClient:
    """
    Get the singleton TWS client instance.

    Creates and connects the client on first call.

    Returns:
        UnifiedTWSClient instance
    """
    global _tws_client_instance

    if _tws_client_instance is None:
        async with _get_tws_client_lock():
            if _tws_client_instance is None:
                _tws_client_instance = UnifiedTWSClient()
                await _tws_client_instance.connect()

    return _tws_client_instance


async def reset_tws_client() -> None:
    """Reset the singleton TWS client (for testing or reconnection)."""
    global _tws_client_instance

    async with _get_tws_client_lock():
        if _tws_client_instance is not None:
            _tws_client_instance.disconnect()
            _tws_client_instance = None


@asynccontextmanager
async def tws_client_context() -> AsyncIterator[UnifiedTWSClient]:
    """
    Context manager for TWS client access.

    Example:
        async with tws_client_context() as client:
            status = await client.get_system_status()
    """
    client = await get_tws_client()
    try:
        yield client
    finally:
        # Don't disconnect - we're using singleton
        pass


# =============================================================================
# MOCK CLIENT FOR TESTING
# =============================================================================


class MockTWSClient(UnifiedTWSClient):
    """
    Mock TWS client for testing.

    Provides canned responses without actually connecting to TWS.
    """

    def __init__(self, responses: dict[str, Any] | None = None):
        """
        Initialize mock client.

        Args:
            responses: Dict mapping operation names to responses
        """
        # Don't call parent __init__ to avoid connecting
        self.config = TWSClientConfig()
        self._state = TWSClientState.CONNECTED
        self._metrics = TWSClientMetrics()
        self._responses = responses or {}
        self._calls: list[tuple[str, tuple, dict]] = []

    def connect(self) -> None:
        """Mock connect - always succeeds."""
        self._state = TWSClientState.CONNECTED

    def disconnect(self) -> None:
        """Mock disconnect."""
        self._state = TWSClientState.DISCONNECTED

    def _mock_response(self, operation: str, *args, **kwargs) -> Any:
        """Get mock response for operation."""
        self._calls.append((operation, args, kwargs))
        self._metrics.total_requests += 1
        self._metrics.successful_requests += 1

        if operation in self._responses:
            return self._responses[operation]

        # Default responses
        defaults = {
            "get_system_status": {"status": "OK", "engine": "DWC"},
            "get_engine_info": {"name": "DWC", "version": "9.5"},
            "get_jobs": [],
            "get_job": {"name": "TEST_JOB", "status": "SUCC"},
            "get_job_status": {"status": "SUCC"},
            "get_workstations": [],
            "get_plan": {"date": "2025-01-01"},
        }
        return defaults.get(operation, {})

    async def get_system_status(self) -> dict[str, Any]:
        return self._mock_response("get_system_status")

    async def get_engine_info(self) -> dict[str, Any]:
        return self._mock_response("get_engine_info")

    async def get_jobs(self, **params) -> list[dict[str, Any]]:
        return self._mock_response("get_jobs", **params)

    async def get_job(self, job_name: str) -> dict[str, Any]:
        return self._mock_response("get_job", job_name)

    async def get_job_status(self, job_name: str) -> dict[str, Any]:
        return self._mock_response("get_job_status", job_name)

    async def get_workstations(self) -> list[dict[str, Any]]:
        return self._mock_response("get_workstations")

    async def get_plan(self) -> dict[str, Any]:
        return self._mock_response("get_plan")

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        """Get list of calls made to the mock."""
        return self._calls


def use_mock_tws_client(responses: dict[str, Any] | None = None) -> None:
    """
    Configure the module to use a mock client.

    Call this in test setup.

    Args:
        responses: Dict mapping operation names to responses
    """
    global _tws_client_instance
    _tws_client_instance = MockTWSClient(responses)


def get_mock_tws_client() -> MockTWSClient | None:
    """Get the mock client if one is configured."""
    global _tws_client_instance
    if isinstance(_tws_client_instance, MockTWSClient):
        return _tws_client_instance
    return None