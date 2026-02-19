"""
Startup Smoke Tests

v5.4.2: Comprehensive smoke tests to verify critical components at startup.

These tests are designed to:
1. Run quickly (< 30 seconds total)
2. Verify critical dependencies
3. Catch configuration errors
4. Prevent broken deployments

Usage:
    # Run as standalone
    python -m resync.tests.smoke_tests

    # Run with pytest
    pytest resync/tests/smoke_tests.py -v

    # Run in CI/CD
    python -c "from resync.tests.smoke_tests import run_smoke_tests; exit(0 if run_smoke_tests() else 1)"

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import inspect
from collections.abc import Callable

# Add project root to path for standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pytest
import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# TYPES
# =============================================================================


class TestStatus(str, Enum):
    """Status of a smoke test."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class SmokeTestResult:
    """Result of a single smoke test."""

    name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeTestSuite:
    """Results of all smoke tests."""

    results: list[SmokeTestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.WARNING)

    @property
    def success(self) -> bool:
        return self.failed == 0

    def add_result(self, result: SmokeTestResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_tests": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "message": r.message,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# =============================================================================
# SMOKE TEST RUNNER
# =============================================================================


class SmokeTestRunner:
    """
    Runs smoke tests to verify system startup.

    Tests are grouped by category:
    - Configuration: Verify settings and environment
    - Dependencies: Verify external services
    - Core: Verify core modules
    - Integration: Verify component integration
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.suite = SmokeTestSuite()
        # Ensure critical env vars are set for tests
        os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
        os.environ.setdefault("RESYNC_REDIS_LAZY_INIT", "1")
        os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://resync:resync@localhost:5432/resync")

    async def _run_async_test(
        self,
        test_func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        timeout: float | None,
    ) -> Any:
        """Run async test with optional timeout."""
        if timeout:
            return await asyncio.wait_for(test_func(*args, **kwargs), timeout)
        return await test_func(*args, **kwargs)

    def _run_sync_test(
        self,
        test_func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        timeout: float | None,
    ) -> Any:
        """Run sync test."""
        # Sync tests don't support timeout easily in this runner
        return test_func(*args, **kwargs)

    async def run_test(
        self,
        name: str,
        test_func: Callable[..., Any],
        *args: Any,
        skip_condition: Callable[[], bool] | None = None,
        skip_reason: str = "",
        timeout: float | None = None,
        retries: int = 0,
        **kwargs: Any,
    ) -> SmokeTestResult:
        """
        Run a single smoke test with configurable execution options.

        Args:
            name: Test name (for logging and result tracking).
            test_func: Test function to execute (sync or async).
            *args: Positional arguments passed to the test function.
            skip_condition: Optional callable returning True to skip the test.
            skip_reason: Human-readable reason for skipping (shown in output).
            timeout: Optional timeout in seconds for test execution.
            retries: Number of retry attempts on failure (default: 0).
            **kwargs: Keyword arguments passed to the test function.

        Returns:
            SmokeTestResult containing test execution outcome and metrics.

        Raises:
            ValueError: If name is empty or test_func is not callable.
        """
        # Input validation for better error messages
        if not name or not name.strip():
            raise ValueError("Test name cannot be empty")
        if not callable(test_func):
            raise ValueError(f"test_func must be callable, got {type(test_func).__name__}")

        name = name.strip()
        # Check skip condition
        if skip_condition and skip_condition():
            result = SmokeTestResult(
                name=name,
                status=TestStatus.SKIPPED,
                duration_ms=0,
                message=skip_reason or "Skipped by condition",
            )
            self._log_result(result)
            return result

        # Execute test with optional retry logic
        last_error: BaseException | None = None
        for attempt in range(1, retries + 2):
            start = time.perf_counter()
            
            try:
                # Run test with timeout support (for async functions)
                if inspect.iscoroutinefunction(test_func):
                    await self._run_async_test(test_func, args, kwargs, timeout)
                else:
                    self._run_sync_test(test_func, args, kwargs, timeout)

                # Test passed
                duration_ms = (time.perf_counter() - start) * 1000
                result = SmokeTestResult(
                    name=name,
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message="OK",
                )
                break  # Exit retry loop on success

            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - start) * 1000
                last_error = TimeoutError(f"Test timed out after {timeout}s")
                result = SmokeTestResult(
                    name=name,
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message="Timeout",
                    error=str(last_error),
                )

            except AssertionError as e:
                duration_ms = (time.perf_counter() - start) * 1000
                last_error = e
                if attempt <= retries:
                    continue  # Retry on assertion failure
                result = SmokeTestResult(
                    name=name,
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message="Assertion failed",
                    error=str(e),
                )

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                last_error = e
                if attempt <= retries:
                    continue  # Retry on error
                result = SmokeTestResult(
                    name=name,
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                message=f"Exception: {type(e).__name__}",
                error=str(e),
            )

        self._log_result(result)
        self.suite.add_result(result)
        return result

    def _log_result(self, result: SmokeTestResult) -> None:
        """Log test result."""
        if not self.verbose:
            return

        status_icon = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.WARNING: "⚠️",
        }

        icon = status_icon.get(result.status, "?")
        print(f"  {icon} {result.name} ({result.duration_ms:.1f}ms) - {result.message}")

        if result.error and result.status == TestStatus.FAILED:
            print(f"     Error: {result.error}")

    async def run_all(self) -> SmokeTestSuite:
        """Run all smoke tests."""
        self.suite = SmokeTestSuite()
        self.suite.started_at = datetime.now(timezone.utc)

        start = time.perf_counter()

        if self.verbose:
            print("\n" + "=" * 60)
            print("🔥 SMOKE TESTS - Resync v5.4.2")
            print("=" * 60)

        # Configuration Tests
        if self.verbose:
            print("\n📋 Configuration Tests")
            print("-" * 40)

        await self.run_test(
            "Environment Variables",
            self._test_environment_variables,
        )
        await self.run_test(
            "Settings Load",
            self._test_settings_load,
        )
        await self.run_test(
            "Redis Strategy Config",
            self._test_redis_strategy_config,
        )
        await self.run_test(
            "LiteLLM Config (No Hardcoded Keys)",
            self._test_litellm_config_security,
        )

        # Dependency Tests
        if self.verbose:
            print("\n🔌 Dependency Tests")
            print("-" * 40)

        await self.run_test(
            "Redis Connection",
            self._test_redis_connection,
            skip_condition=lambda: os.getenv("SKIP_REDIS_TEST") == "1",
            skip_reason="SKIP_REDIS_TEST=1",
        )
        await self.run_test(
            "Database Connection",
            self._test_database_connection,
            skip_condition=lambda: os.getenv("SKIP_DB_TEST") == "1",
            skip_reason="SKIP_DB_TEST=1",
        )

        # Core Module Tests
        if self.verbose:
            print("\n🧩 Core Module Tests")
            print("-" * 40)

        await self.run_test(
            "Core Imports",
            self._test_core_imports,
        )
        await self.run_test(
            "Circuit Breaker",
            self._test_circuit_breaker,
        )
        await self.run_test(
            "Resilience Module",
            self._test_resilience_module,
        )
        await self.run_test(
            "Exception Hierarchy",
            self._test_exception_hierarchy,
        )

        # Service Tests
        if self.verbose:
            print("\n🛠️ Service Tests")
            print("-" * 40)

        await self.run_test(
            "TWS Unified Client",
            self._test_tws_unified_client,
        )
        await self.run_test(
            "LLM Fallback Service",
            self._test_llm_fallback_service,
        )
        await self.run_test(
            "RAG Chunking Module",
            self._test_rag_chunking,
        )

        # Integration Tests
        if self.verbose:
            print("\n🔗 Integration Tests")
            print("-" * 40)

        await self.run_test(
            "FastAPI App Creation",
            self._test_fastapi_app_creation,
        )
        await self.run_test(
            "Middleware Stack",
            self._test_middleware_stack,
        )
        await self.run_test(
            "Health Endpoint",
            self._test_health_endpoint,
        )

        # Complete
        self.suite.completed_at = datetime.now(timezone.utc)
        self.suite.total_duration_ms = (time.perf_counter() - start) * 1000

        if self.verbose:
            self._print_summary()

        return self.suite

    def _print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"  Total Tests: {len(self.suite.results)}")
        print(f"  ✅ Passed:   {self.suite.passed}")
        print(f"  ❌ Failed:   {self.suite.failed}")
        print(f"  ⏭️ Skipped:  {self.suite.skipped}")
        print(f"  ⚠️ Warnings: {self.suite.warnings}")
        print("  Duration:   {self.suite.total_duration_ms:.1f}ms")
        print()

        if self.suite.success:
            print("✅ ALL SMOKE TESTS PASSED")
        else:
            print("❌ SOME SMOKE TESTS FAILED")
            print("\nFailed tests:")
            for r in self.suite.results:
                if r.status == TestStatus.FAILED:
                    print(f"  - {r.name}: {r.error}")

        print()

    # =========================================================================
    # CONFIGURATION TESTS
    # =========================================================================

    def _test_environment_variables(self) -> None:
        """Test required environment variables."""
        required_vars = [
            "REDIS_URL",
        ]

        recommended_vars = [
            "DATABASE_URL",
            "LLM_API_KEY",
            "OPENROUTER_API_KEY",
        ]

        missing_required = []
        missing_recommended = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in recommended_vars:
            if not os.getenv(var):
                missing_recommended.append(var)

        if missing_required:
            raise AssertionError(f"Missing required env vars: {missing_required}")

        if missing_recommended:
            logger.warning(
                "Missing recommended env vars",
                vars=missing_recommended,
            )

    def _test_settings_load(self) -> None:
        """Test that settings load correctly."""
        from resync.settings import settings

        assert settings is not None, "Settings is None"
        assert hasattr(settings, "REDIS_URL"), "Missing REDIS_URL"
        assert settings.llm_timeout > 0, "Invalid LLM timeout"
        assert settings.llm_timeout <= 60, (
            f"LLM timeout too high: {settings.llm_timeout}s (max 60s)"
        )

    def _test_redis_strategy_config(self) -> None:
        """Test Redis strategy configuration."""
        from resync.core.redis_strategy import RedisTier, get_redis_strategy

        strategy = get_redis_strategy()

        # Test tier classification
        assert strategy.get_tier("GET", "/health") == RedisTier.READ_ONLY
        assert strategy.get_tier("POST", "/tws/execute/job") == RedisTier.CRITICAL

        # Test startup config
        startup = strategy.get_startup_config()
        assert "fail_fast" in startup
        assert "max_retries" in startup

    def _test_litellm_config_security(self) -> None:
        """Test that litellm config has no hardcoded API keys."""
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "core" / "litellm_config.yaml"

        if not config_path.exists():
            logger.warning("litellm_config.yaml not found, skipping")
            return

        with open(config_path) as f:
            content = f.read()

        # Check for hardcoded keys
        dangerous_patterns = [
            "sk-",  # OpenAI key prefix
            "sk-or-",  # OpenRouter key prefix
            "anthropic-",  # Anthropic key prefix
        ]

        for pattern in dangerous_patterns:
            if f'"{pattern}' in content or f"'{pattern}" in content:
                # Check if it's in a comment
                lines = content.split("\n")
                for line in lines:
                    if (
                        pattern in line
                        and not line.strip().startswith("#")
                        and "os.environ" not in line
                        and "getenv" not in line
                    ):
                        raise AssertionError(f"Potential hardcoded API key found: {pattern}...")

    # =========================================================================
    # DEPENDENCY TESTS
    # =========================================================================

    async def _test_redis_connection(self) -> None:
        """Test Redis connection."""
        from resync.core.cache import get_redis_client

        client = await get_redis_client()
        result = await client.ping()
        assert result is True, "Redis ping failed"

    async def _test_database_connection(self) -> None:
        """Test database connection."""
        from resync.settings import settings

        if not settings.database_url:
            raise AssertionError("database_url not configured")

        # Try to import and check engine
        try:
            from resync.core.database.engine import get_engine

            engine = get_engine()
            assert engine is not None
        except ImportError:
            logger.warning("Database engine not available")

    # =========================================================================
    # CORE MODULE TESTS
    # =========================================================================

    def _test_core_imports(self) -> None:
        """Test that core modules can be imported."""
        imports = [
            "resync.core.exceptions",
            "resync.core.resilience",
            "resync.core.redis_strategy",
            "resync.core.structured_logger",
            # "resync.core.circuit_breaker",
            "resync.core.health.unified_health_service",
        ]

        failed = []
        for module in imports:
            try:
                __import__(module)
            except ImportError as e:
                failed.append(f"{module}: {e}")

        if failed:
            raise AssertionError(f"Failed imports: {failed}")

    async def _test_circuit_breaker(self) -> None:
        """Test circuit breaker functionality."""
        from resync.core.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState

        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker(config)

        # Test successful call
        async def success():
            return "ok"

        result = await cb.call(success)
        assert result == "ok"
        assert cb.state == CircuitBreakerState.CLOSED

    async def _test_resilience_module(self) -> None:
        """Test resilience patterns."""
        from resync.core.resilience import (
            CircuitBreaker,
            CircuitBreakerConfig,
            RetryConfig,
            RetryWithBackoff,
        )

        # Test circuit breaker
        config = CircuitBreakerConfig(failure_threshold=3, name="test")
        cb = CircuitBreaker(config)
        assert cb.state.value == "closed"

        # Test retry config
        retry_config = RetryConfig(max_retries=3)
        retry = RetryWithBackoff(retry_config)
        assert retry is not None

    def _test_exception_hierarchy(self) -> None:
        """Test exception hierarchy."""
        from resync.core.exceptions import (
            BaseAppException,
            CircuitBreakerError,
            LLMError,
            TWSError,
        )

        # Test inheritance
        assert issubclass(TWSError, BaseAppException)
        assert issubclass(LLMError, BaseAppException)
        assert issubclass(CircuitBreakerError, BaseAppException)

        # Test instantiation
        e = TWSError("test")
        assert e.message == "test"
        assert "test" in str(e)

    # =========================================================================
    # SERVICE TESTS
    # =========================================================================

    def _test_tws_unified_client(self) -> None:
        """Test TWS unified client module."""
        from resync.services.tws_unified import (
            MockTWSClient,
            TWSClientConfig,
            TWSClientState,
        )

        # Test config
        config = TWSClientConfig()
        assert config.connect_timeout > 0
        assert config.circuit_failure_threshold > 0

        # Test mock client
        mock = MockTWSClient()
        assert mock.state == TWSClientState.CONNECTED

    def _test_llm_fallback_service(self) -> None:
        """Test LLM fallback service module."""
        from resync.services.llm_fallback import (
            LLMFallbackConfig,
            LLMService,
        )

        # Test config
        config = LLMFallbackConfig()
        assert config.default_timeout > 0
        assert config.default_timeout <= 60  # Should be realistic
        assert len(config.fallback_chain) > 0

        # Test service creation
        service = LLMService(config)
        assert service is not None

        # Test known models
        assert "gpt-4" in LLMService.KNOWN_MODELS
        assert "gpt-3.5-turbo" in LLMService.KNOWN_MODELS

    def _test_rag_chunking(self) -> None:
        """Test RAG chunking module."""
        from resync.knowledge.ingestion.advanced_chunking import (
            AdvancedChunker,
            ChunkingConfig,
            ChunkingStrategy,
        )

        # Test config
        config = ChunkingConfig()
        assert config.max_tokens > 0
        assert config.strategy == ChunkingStrategy.TWS_OPTIMIZED

        # Test chunker creation
        chunker = AdvancedChunker(config)
        assert chunker is not None

    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================

    def _test_fastapi_app_creation(self) -> None:
        """Test FastAPI app can be created."""
        from fastapi import FastAPI

        app = FastAPI(title="Smoke Test")
        assert app is not None
        assert app.title == "Smoke Test"

    def _test_middleware_stack(self) -> None:
        """Test middleware can be imported."""
        from resync.api.middleware import (
            CorrelationIdMiddleware,
            RedisValidationMiddleware,
        )

        assert CorrelationIdMiddleware is not None
        assert RedisValidationMiddleware is not None

    async def _test_health_endpoint(self) -> None:
        """Test health endpoint logic."""
        # Simple test - just verify health service exists
        from resync.core.health import get_unified_health_service

        # Should not raise
        health_service = await get_unified_health_service()
        status = await health_service.perform_comprehensive_health_check()
        assert status is not None


# =============================================================================
# ENTRY POINTS
# =============================================================================


def run_smoke_tests(verbose: bool = True) -> bool:
    """
    Run all smoke tests.

    Returns:
        True if all tests passed, False otherwise
    """
    runner = SmokeTestRunner(verbose=verbose)
    suite = asyncio.run(runner.run_all())
    return suite.success


async def run_smoke_tests_async(verbose: bool = True) -> SmokeTestSuite:
    """
    Run all smoke tests asynchronously.

    Returns:
        SmokeTestSuite with results
    """
    runner = SmokeTestRunner(verbose=verbose)
    return await runner.run_all()


# =============================================================================
# PYTEST INTEGRATION
# =============================================================================


@pytest.mark.smoke
class TestSmokeTests:
    """Pytest wrapper for smoke tests."""

    @pytest.mark.asyncio
    async def test_smoke_suite(self):
        """Run full smoke test suite."""
        suite = await run_smoke_tests_async(verbose=False)

        # Report failures
        if not suite.success:
            failures = [r for r in suite.results if r.status == TestStatus.FAILED]
            failure_msgs = [f"{r.name}: {r.error}" for r in failures]
            pytest.fail(f"Smoke tests failed: {failure_msgs}")


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    success = run_smoke_tests(verbose=True)
    sys.exit(0 if success else 1)
