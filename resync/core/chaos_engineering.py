"""
Chaos Engineering and Fuzzing Framework for Resync Core Components

This module provides automated chaos testing, fuzzing, and stress testing
capabilities to validate system resilience under adversarial conditions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections.abc import Callable

from resync.core.task_tracker import create_tracked_task
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from resync.core import get_environment_tags, get_global_correlation_id
from resync.core.agent_manager import AgentManager
from resync.core.audit_db import add_audit_records_batch
from resync.core.audit_log import get_audit_log_manager
from resync.core.cache import AsyncTTLCache
from resync.core.metrics import log_with_correlation, runtime_metrics

logger = logging.getLogger(__name__)


@dataclass
class ChaosTestResult:
    """Result of a chaos engineering test."""

    test_name: str
    component: str
    duration: float
    success: bool
    error_count: int = 0
    operations_performed: int = 0
    anomalies_detected: list[str] = field(default_factory=list)
    correlation_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FuzzingScenario:
    """Definition of a fuzzing scenario."""

    name: str
    description: str
    fuzz_function: Callable[[], Any]
    expected_failures: list[str] = field(default_factory=list)
    max_duration: float = 30.0


class ChaosEngineer:
    """
    Chaos Engineering orchestrator for systematic system testing.
    """

    def __init__(self) -> None:
        self.correlation_id = get_global_correlation_id()
        self.results: list[ChaosTestResult] = []
        self.active_tests: dict[str, asyncio.Task[Any]] = {}
        self._lock = threading.RLock()

    async def run_full_chaos_suite(
        self, duration_minutes: float = 5.0
    ) -> dict[str, Any]:
        """
        Run the complete chaos engineering test suite.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "chaos_engineering",
                "operation": "full_suite",
                "duration_minutes": duration_minutes,
                "global_correlation": self.correlation_id,
            }
        )

        start_time = time.time()
        logger.info(
            f"Starting chaos engineering suite for {duration_minutes} minutes",
            extra={"correlation_id": correlation_id},
        )

        try:
            # Run all tests concurrently
            test_tasks = [
                self._cache_race_condition_fuzzing(),
                self._agent_concurrent_initialization_chaos(),
                self._audit_db_failure_injection(),
                self._memory_pressure_simulation(),
                self._network_partition_simulation(),
                self._component_isolation_testing(),
            ]

            # Run tests with timeout
            timeout_seconds = max(1, int(duration_minutes * 60))
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*test_tasks, return_exceptions=True),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Mark timeout as an anomaly; individual tasks may still be
                # running but we return a clear degraded summary.
                logger.error(
                    "chaos_suite_timeout",
                    extra={
                        "timeout_seconds": timeout_seconds,
                        "correlation_id": correlation_id,
                    },
                )
                results = [
                    TimeoutError(f"Chaos suite timed out after {timeout_seconds}s")
                ]

            # Process results
            successful_tests = 0
            total_anomalies = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    test_name = f"test_{i}"
                    self.results.append(
                        ChaosTestResult(
                            test_name=test_name,
                            component="unknown",
                            duration=0,
                            success=False,
                            error_count=1,
                            anomalies_detected=[str(result)],
                            correlation_id=correlation_id,
                        )
                    )
                else:
                    if isinstance(result, ChaosTestResult):
                        self.results.append(result)
                        if result.success:
                            successful_tests += 1
                        total_anomalies += len(result.anomalies_detected)

            suite_duration = time.time() - start_time

            summary = {
                "correlation_id": correlation_id,
                "duration": suite_duration,
                "total_tests": len(test_tasks),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(test_tasks) if test_tasks else 0,
                "total_anomalies": total_anomalies,
                "test_results": [r.__dict__ for r in self.results],
                "environment": get_environment_tags(),
            }

            log_with_correlation(
                logging.INFO,
                f"Chaos suite completed: {successful_tests}/{len(test_tasks)} tests passed, "
                f"{total_anomalies} anomalies detected in {suite_duration:.1f}s",
                correlation_id,
            )

            return summary

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _cache_race_condition_fuzzing(self) -> ChaosTestResult:
        """
        Fuzzing test for cache race conditions under concurrent access.
        """
        test_name = "cache_race_condition_fuzzing"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        cache = AsyncTTLCache(ttl_seconds=10, num_shards=8)

        try:
            anomalies = []
            operations = 0
            errors = 0

            # Create multiple concurrent operations
            async def worker(worker_id: int) -> None:
                nonlocal operations, errors
                for i in range(100):
                    try:
                        # Random operations to create race conditions
                        op = random.choice(["set", "get", "delete"])
                        key = f"fuzz_key_{worker_id}_{i}_{random.randint(0, 10)}"
                        value = f"fuzz_value_{worker_id}_{i}"

                        if op == "set":
                            await cache.set(key, value, random.randint(1, 30))
                        elif op == "get":
                            await cache.get(key)
                        elif op == "delete":
                            await cache.delete(key)

                        operations += 1

                    except Exception as e:
                        logger.error("exception_caught", error=str(e), exc_info=True)
                        errors += 1
                        anomalies.append(f"Worker {worker_id} op {i}: {str(e)}")

            # Run 10 concurrent workers
            workers = [worker(i) for i in range(10)]
            await asyncio.gather(*workers, return_exceptions=True)

            # Check cache integrity
            metrics = cache.get_detailed_metrics()
            if metrics["size"] < 0 or metrics["hit_rate"] > 1.0:
                anomalies.append("Cache metrics corrupted")

            success = errors == 0 and len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="async_cache",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
                details={"cache_metrics": metrics},
            )

        finally:
            await cache.stop()
            runtime_metrics.close_correlation_id(correlation_id)

    async def _agent_concurrent_initialization_chaos(self) -> ChaosTestResult:
        """
        Chaos test for agent manager concurrent initialization.
        """
        test_name = "agent_concurrent_initialization_chaos"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        anomalies = []
        operations = 0
        errors = 0

        try:

            async def init_worker(worker_id: int) -> None:
                nonlocal operations, errors
                try:
                    # Try to initialize agent manager concurrently
                    manager = AgentManager()
                    operations += 1

                    # Try concurrent operations
                    tasks = []
                    for i in range(10):
                        task = await create_tracked_task(
                            self._simulate_agent_operation(manager, i),
                            name="simulate_agent_operation",
                        )
                        tasks.append(task)

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            errors += 1
                            anomalies.append(f"Agent operation failed: {str(result)}")

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    errors += 1
                    anomalies.append(f"Manager init {worker_id}: {str(e)}")

            # Run 5 concurrent initializations
            workers = [init_worker(i) for i in range(5)]
            await asyncio.gather(*workers, return_exceptions=True)

            success = errors == 0 and len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="agent_manager",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _simulate_agent_operation(
        self, manager: AgentManager, op_id: int
    ) -> None:
        """Simulate agent operations for chaos testing."""
        # Simulate getting non-existent agent
        try:
            await manager.get_agent(f"non_existent_agent_{op_id}")
        except ValueError as exc:
            logger.debug(
                "suppressed_exception", error=str(exc), exc_info=True
            )  # was: pass

        # Simulate getting agent details
        try:
            metrics = manager.get_detailed_metrics()
            if "total_agents" not in metrics:
                raise ValueError("Metrics missing total_agents")
        except Exception as e:
            raise RuntimeError(f"Metrics operation failed: {e}") from None

    async def _audit_db_failure_injection(self) -> ChaosTestResult:
        """
        Inject failures into audit database operations.
        """
        test_name = "audit_db_failure_injection"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        anomalies = []
        operations = 0
        errors = 0

        try:
            # Test batch operations with various failure scenarios
            test_memories = [
                {
                    "id": f"chaos_memory_{i}",
                    "user_query": f"Chaos query {i}",
                    "agent_response": f"Chaos response {i}",
                    "ia_audit_reason": "chaos_test",
                    "ia_audit_confidence": random.random(),
                }
                for i in range(50)
            ]

            # Inject some duplicates to test integrity
            test_memories.extend(test_memories[:10])  # Add duplicates

            # Test batch insert
            try:
                result = add_audit_records_batch(test_memories)
                operations += len(result)
                successful_inserts = sum(1 for r in result if r is not None)
                if successful_inserts < len(test_memories) - 10:  # Allow for duplicates
                    anomalies.append(
                        f"Unexpected batch insert failures: {len(result) - successful_inserts}"
                    )
            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                errors += 1
                anomalies.append(f"Batch insert failed: {str(e)}")

            # Test metrics retrieval
            try:
                audit_manager = get_audit_log_manager()
                metrics = audit_manager.get_audit_metrics()
                if "total_records" not in metrics:
                    anomalies.append("Audit metrics missing total_records")
                operations += 1
            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                errors += 1
                anomalies.append(f"Metrics retrieval failed: {str(e)}")

            # Test auto sweep
            try:
                sweep_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: __import__(
                        "resync.core.audit_db"
                    ).auto_sweep_pending_audits(1, 10),
                )
                operations += sweep_result.get("total_processed", 0)
            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                errors += 1
                anomalies.append(f"Auto sweep failed: {str(e)}")

            success = len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="audit_db",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _memory_pressure_simulation(self) -> ChaosTestResult:
        """
        Simulate memory pressure and resource exhaustion.
        """
        test_name = "memory_pressure_simulation"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        anomalies = []
        operations = 0
        errors = 0

        try:
            cache = AsyncTTLCache(ttl_seconds=1, num_shards=4)  # Short TTL

            # Simulate memory pressure with large objects
            large_objects = []
            for i in range(100):
                large_obj = {
                    "id": f"large_obj_{i}",
                    "data": "x" * 10000,  # 10KB per object
                    "metadata": {"size": 10000, "created": time.time()},
                }
                large_objects.append(large_obj)

                try:
                    await cache.set(f"large_key_{i}", large_obj, ttl_seconds=5)
                    operations += 1

                    # Periodic cleanup check
                    if i % 20 == 0:
                        await asyncio.sleep(0.1)  # Allow cleanup task to run
                        metrics = cache.get_detailed_metrics()
                        if metrics["size"] > 50:  # Too many items
                            anomalies.append(
                                f"Cache growing too large at iteration {i}: {metrics['size']}"
                            )

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    errors += 1
                    anomalies.append(f"Large object storage failed at {i}: {str(e)}")

            # Wait for cleanup
            await asyncio.sleep(2)
            final_metrics = cache.get_detailed_metrics()

            if final_metrics["size"] > 10:  # Should have cleaned up most items
                anomalies.append(
                    f"Cleanup ineffective: {final_metrics['size']} items remaining"
                )

            success = len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="memory_pressure",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
                details={"final_cache_metrics": final_metrics},
            )

        finally:
            await cache.stop()
            runtime_metrics.close_correlation_id(correlation_id)

    def _network_partition_simulation(self) -> ChaosTestResult:
        """
        Simulate network partitions and connectivity issues.
        """
        test_name = "network_partition_simulation"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        anomalies = []
        operations = 0
        errors = 0

        try:
            # Simulate network issues using mock patches
            original_import = __import__

            def failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
                # Randomly fail some imports to simulate network issues
                if (
                    random.random() < 0.1 and "agent" in name
                ):  # 10% failure rate for agent-related
                    raise ImportError(f"Simulated network failure for {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=failing_import):
                for i in range(20):
                    try:
                        # Try agent manager operations during simulated network issues
                        manager = AgentManager()
                        operations += 1

                        # Try to get metrics
                        metrics = manager.get_detailed_metrics()
                        if not isinstance(metrics, dict):
                            anomalies.append(f"Metrics not dict at iteration {i}")

                    except Exception as e:
                        logger.error("exception_caught", error=str(e), exc_info=True)
                        errors += 1
                        if "network failure" not in str(e):
                            anomalies.append(
                                f"Unexpected error during network chaos {i}: {str(e)}"
                            )

            success = len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="network_simulation",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _component_isolation_testing(self) -> ChaosTestResult:
        """
        Test component isolation and failure propagation.
        """
        test_name = "component_isolation_testing"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )

        start_time = time.time()
        anomalies = []
        operations = 0
        errors = 0

        try:
            # Test that components fail gracefully when isolated
            components_to_test = ["async_cache", "agent_manager", "audit_db"]

            for component in components_to_test:
                try:
                    # Simulate component isolation by patching dependencies
                    if component == "async_cache":
                        # Try cache operations with mocked failures
                        cache = AsyncTTLCache()
                        with patch.object(
                            cache, "set", side_effect=Exception("Simulated failure")
                        ):
                            try:
                                await cache.set("test_key", "test_value")
                                anomalies.append("Cache set should have failed")
                            except Exception as e:
                                # Expected failure in chaos test - cache should be broken
                                logger.debug(
                                    "Expected cache failure in chaos test: %s", e
                                )

                        await cache.stop()
                        operations += 1

                    elif component == "agent_manager":
                        # Test agent manager with missing dependencies
                        with patch(
                            "resync.core.agent_manager.Agent",
                            side_effect=ImportError("Simulated import failure"),
                        ):
                            try:
                                manager = AgentManager()
                                metrics = manager.get_detailed_metrics()
                                operations += 1
                            except Exception as e:
                                logger.error(
                                    "exception_caught", error=str(e), exc_info=True
                                )
                                if "import failure" not in str(e):
                                    anomalies.append(
                                        f"Agent manager unexpected error: {str(e)}"
                                    )

                    elif component == "audit_db":
                        # Test audit operations with DB failures
                        with patch(
                            "resync.core.audit_db.get_db_connection",
                            side_effect=Exception("Simulated DB failure"),
                        ):
                            try:
                                audit_manager = get_audit_log_manager()
                                metrics = audit_manager.get_audit_metrics()
                                if "error" not in metrics:
                                    anomalies.append(
                                        "Audit DB should have reported error"
                                    )
                            except Exception as e:
                                logger.error(
                                    "exception_caught", error=str(e), exc_info=True
                                )
                                errors += 1
                                if "DB failure" not in str(e):
                                    anomalies.append(
                                        f"Audit DB unexpected error: {str(e)}"
                                    )

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    errors += 1
                    anomalies.append(
                        f"Component {component} isolation test failed: {str(e)}"
                    )

            success = len(anomalies) == 0

            return ChaosTestResult(
                test_name=test_name,
                component="component_isolation",
                duration=time.time() - start_time,
                success=success,
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )

        finally:
            runtime_metrics.close_correlation_id(correlation_id)


class FuzzingEngine:
    """
    Automated fuzzing engine for input validation and edge case testing.
    """

    def __init__(self) -> None:
        self.correlation_id = get_global_correlation_id()
        self.scenarios: list[FuzzingScenario] = []
        self._setup_fuzzing_scenarios()

    def _setup_fuzzing_scenarios(self) -> None:
        """Setup fuzzing scenarios for different components."""

        # Cache fuzzing scenarios
        self.scenarios.extend(
            [
                FuzzingScenario(
                    name="cache_key_fuzzing",
                    description="Fuzz cache keys with edge cases",
                    fuzz_function=self._fuzz_cache_keys,
                    expected_failures=["TypeError", "ValueError"],
                ),
                FuzzingScenario(
                    name="cache_value_fuzzing",
                    description="Fuzz cache values with complex objects",
                    fuzz_function=self._fuzz_cache_values,
                    expected_failures=["TypeError", "RecursionError"],
                ),
                FuzzingScenario(
                    name="cache_ttl_fuzzing",
                    description="Fuzz TTL values with edge cases",
                    fuzz_function=self._fuzz_cache_ttl,
                    expected_failures=["ValueError", "OverflowError"],
                ),
            ]
        )

        # Agent fuzzing scenarios
        self.scenarios.extend(
            [
                FuzzingScenario(
                    name="agent_config_fuzzing",
                    description="Fuzz agent configurations",
                    fuzz_function=self._fuzz_agent_configs,
                    expected_failures=["ValidationError", "TypeError"],
                )
            ]
        )

        # Audit fuzzing scenarios
        self.scenarios.extend(
            [
                FuzzingScenario(
                    name="audit_record_fuzzing",
                    description="Fuzz audit record structures",
                    fuzz_function=self._fuzz_audit_records,
                    expected_failures=["TypeError", "ValueError"],
                )
            ]
        )

    async def run_fuzzing_campaign(self, max_duration: float = 60.0) -> dict[str, Any]:
        """
        Run a complete fuzzing campaign on all scenarios.
        """
        correlation_id = runtime_metrics.create_correlation_id(
            {
                "component": "fuzzing_engine",
                "operation": "campaign",
                "max_duration": max_duration,
                "scenarios": len(self.scenarios),
                "global_correlation": self.correlation_id,
            }
        )

        start_time = time.time()
        results = []

        try:
            for scenario in self.scenarios:
                scenario_start = time.time()

                try:
                    # Run scenario in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, scenario.fuzz_function),
                        timeout=scenario.max_duration,
                    )

                    duration = time.time() - scenario_start
                    results.append(
                        {
                            "scenario": scenario.name,
                            "description": scenario.description,
                            "duration": duration,
                            "success": True,
                            "result": result,
                        }
                    )

                except asyncio.TimeoutError:
                    results.append(
                        {
                            "scenario": scenario.name,
                            "description": scenario.description,
                            "duration": scenario.max_duration,
                            "success": False,
                            "error": "Timeout",
                        }
                    )

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    duration = time.time() - scenario_start
                    results.append(
                        {
                            "scenario": scenario.name,
                            "description": scenario.description,
                            "duration": duration,
                            "success": False,
                            "error": str(e),
                        }
                    )

            campaign_duration = time.time() - start_time
            successful_scenarios = sum(1 for r in results if r["success"])

            summary = {
                "correlation_id": correlation_id,
                "campaign_duration": campaign_duration,
                "total_scenarios": len(self.scenarios),
                "successful_scenarios": successful_scenarios,
                "success_rate": successful_scenarios / len(self.scenarios),
                "results": results,
                "environment": get_environment_tags(),
            }

            log_with_correlation(
                logging.INFO,
                f"Fuzzing campaign completed: {successful_scenarios}/{len(self.scenarios)} scenarios passed",
                correlation_id,
            )

            return summary

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    def _fuzz_cache_keys(self) -> dict[str, Any]:
        """Fuzz cache keys with various edge cases."""
        import asyncio

        cache = AsyncTTLCache()

        async def fuzz_keys() -> dict[str, Any]:
            test_cases = [
                # Valid keys
                "normal_key",
                "key_with_underscores",
                "key-with-dashes",
                "123_numeric_start",
                # Edge cases
                "",  # Empty string
                "a" * 1000,  # Very long key
                "key with spaces",
                "key\twith\ttabs",  # Tabs
                "key\nwith\nnewlines",  # Newlines
                "key\x00with\x00nulls",  # Null bytes
                "🧪_emoji_key",  # Unicode
                None,  # None key
                123,  # Integer key
                ["list", "key"],  # List key
                {"dict": "key"},  # Dict key
            ]

            results = {"passed": 0, "failed": 0, "errors": []}

            for i, key in enumerate(test_cases):
                try:
                    if key is None:
                        continue  # Skip None keys

                    await cache.set(key, f"value_{i}")
                    retrieved = await cache.get(key)

                    if retrieved == f"value_{i}":
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Key {repr(key)}: value mismatch")

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    results["failed"] += 1
                    results["errors"].append(f"Key {repr(key)}: {str(e)}")

            return results

        # Run async fuzzing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(fuzz_keys())
        finally:
            loop.close()

    def _fuzz_cache_values(self) -> dict[str, Any]:
        """Fuzz cache values with complex objects."""
        import asyncio

        cache = AsyncTTLCache()

        async def fuzz_values() -> dict[str, Any]:
            test_cases = [
                # Simple values
                "string_value",
                42,
                3.14,
                True,
                None,
                # Complex objects
                {"nested": {"dict": "value"}},
                ["list", "of", "items"],
                {"self_ref": None},  # Will be modified to self-reference
                # Large objects
                "x" * 100000,  # 100KB string
                list(range(10000)),  # Large list
                # Edge case objects
                object(),  # Generic object
            ]

            # Create self-referencing object
            self_ref = {"data": "value"}
            self_ref["self"] = self_ref
            test_cases[6] = self_ref

            results = {"passed": 0, "failed": 0, "errors": []}

            for i, value in enumerate(test_cases):
                try:
                    key = f"fuzz_value_key_{i}"
                    await cache.set(key, value)
                    retrieved = await cache.get(key)

                    # Basic equality check (may fail for complex objects)
                    try:
                        if retrieved == value or str(retrieved) == str(value):
                            results["passed"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Value {i}: mismatch")
                    except Exception as e:
                        logger.error("exception_caught", error=str(e), exc_info=True)
                        # For objects that can't be compared, just check if retrieval worked
                        if retrieved is not None:
                            results["passed"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Value {i}: retrieval failed")

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    results["failed"] += 1
                    results["errors"].append(f"Value {i}: {str(e)}")

            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(fuzz_values())
        finally:
            loop.close()

    def _fuzz_cache_ttl(self) -> dict[str, Any]:
        """Fuzz TTL values."""
        import asyncio

        cache = AsyncTTLCache()

        async def fuzz_ttl() -> dict[str, Any]:
            test_cases = [
                # Valid TTLs
                1,
                30,
                300,
                3600,
                # Edge cases
                0,  # Zero TTL
                -1,  # Negative TTL
                999999,  # Very large TTL
                float("inf"),  # Infinite TTL
                None,  # None TTL
                "30",  # String TTL
                [30],  # List TTL
            ]

            results = {"passed": 0, "failed": 0, "errors": []}

            for i, ttl in enumerate(test_cases):
                try:
                    key = f"fuzz_ttl_key_{i}"
                    await cache.set(key, f"value_{i}", ttl_seconds=ttl)
                    retrieved = await cache.get(key)

                    if retrieved is not None:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            f"TTL {repr(ttl)}: immediate expiration"
                        )

                except Exception as e:
                    logger.error("exception_caught", error=str(e), exc_info=True)
                    results["failed"] += 1
                    results["errors"].append(f"TTL {repr(ttl)}: {str(e)}")

            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(fuzz_ttl())
        finally:
            loop.close()

    def _fuzz_agent_configs(self) -> dict[str, Any]:
        """Fuzz agent configurations."""
        from resync.core.agent_manager import AgentConfig

        test_cases = [
            # Valid config
            {
                "id": "test_agent_1",
                "name": "Test Agent",
                "role": "Tester",
                "goal": "Test things",
                "backstory": "A testing agent",
                "tools": ["test_tool"],
                "model_name": "test_model",
            },
            # Edge cases
            {
                "id": "",  # Empty ID
                "name": "Test",
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": [],
                "model_name": "test",
            },
            {
                "id": "test_agent_2",
                "name": None,  # None name
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": ["test_tool"],
                "model_name": "test",
            },
            {
                "id": "test_agent_3",
                "name": "Test",
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": None,  # None tools
                "model_name": "test",
            },
            {
                "id": "test_agent_4",
                "name": "x" * 1000,  # Very long name
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": ["test_tool"],
                "model_name": "test",
            },
        ]

        results = {"passed": 0, "failed": 0, "errors": []}

        for i, config_data in enumerate(test_cases):
            try:
                config = AgentConfig(**config_data)
                results["passed"] += 1

                # Validation is handled by Pydantic in AgentConfig constructor
                # If we reach here, the config was created successfully
                # The hasattr checks below are redundant but kept for clarity in fuzzing context
                if not hasattr(config, "id") or not hasattr(config, "name"):
                    raise ValueError(
                        "Config missing required fields after construction"
                    )

            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                results["failed"] += 1
                results["errors"].append(f"Config {i}: {str(e)}")

        return results

    def _fuzz_audit_records(self) -> dict[str, Any]:
        """Fuzz audit record structures."""
        test_cases = [
            # Valid record
            {
                "id": "audit_1",
                "user_query": "Test query",
                "agent_response": "Test response",
                "ia_audit_reason": "test",
                "ia_audit_confidence": 0.8,
            },
            # Edge cases
            {"id": None, "user_query": "Test", "agent_response": "Test"},  # None ID
            {
                "id": "audit_2",
                "user_query": None,  # None query
                "agent_response": "Test",
            },
            {
                "id": "audit_3",
                "user_query": "Test",
                "agent_response": "x" * 100000,  # Very large response
            },
            {
                "id": "audit_4",
                "user_query": "Test",
                "agent_response": "Test",
                "ia_audit_reason": None,
                "ia_audit_confidence": "0.8",  # String instead of float
            },
        ]

        results = {"passed": 0, "failed": 0, "errors": []}

        for i, record in enumerate(test_cases):
            try:
                from resync.core.audit_db import add_audit_record

                result = add_audit_record(record)

                if (
                    result is not None or i > 0
                ):  # First record should succeed, others may fail
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Record {i}: unexpected failure")

            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                # Some failures are expected for malformed data
                if i == 0:  # First record should work
                    results["failed"] += 1
                    results["errors"].append(f"Record {i}: {str(e)}")
                else:
                    results["passed"] += 1  # Expected failure

        return results


# =============================================================================
# Lazy Initialization (v5.2.3.25)
# =============================================================================
# IMPORTANT: Instances are created on-demand to avoid resource allocation
# at import time. This prevents accidental initialization in production.
# =============================================================================

_chaos_engineer: ChaosEngineer | None = None
_fuzzing_engine: FuzzingEngine | None = None


def get_chaos_engineer() -> ChaosEngineer:
    """
    Get ChaosEngineer instance (lazy initialization).

    WARNING: This module should only be used in staging/test environments.
    Enabling chaos engineering in production can cause outages.

    Returns:
        ChaosEngineer instance
    """
    global _chaos_engineer
    if _chaos_engineer is None:
        logger.warning(
            "⚠️ ChaosEngineer initialized - ensure this is NOT production!",
            extra={"component": "chaos_engineering", "action": "initialization"},
        )
        _chaos_engineer = ChaosEngineer()
    return _chaos_engineer


def get_fuzzing_engine() -> FuzzingEngine:
    """
    Get FuzzingEngine instance (lazy initialization).

    WARNING: This module should only be used in staging/test environments.

    Returns:
        FuzzingEngine instance
    """
    global _fuzzing_engine
    if _fuzzing_engine is None:
        logger.warning(
            "⚠️ FuzzingEngine initialized - ensure this is NOT production!",
            extra={"component": "fuzzing_engine", "action": "initialization"},
        )
        _fuzzing_engine = FuzzingEngine()
    return _fuzzing_engine


async def run_chaos_engineering_suite(duration_minutes: float = 5.0) -> dict[str, Any]:
    """
    Convenience function to run chaos engineering suite.

    WARNING: Only use in staging/test environments!

    Args:
        duration_minutes: How long to run the chaos suite

    Returns:
        Summary dict with test results
    """
    return await get_chaos_engineer().run_full_chaos_suite(duration_minutes)


async def run_fuzzing_campaign(max_duration: float = 60.0) -> dict[str, Any]:
    """
    Convenience function to run fuzzing campaign.

    WARNING: Only use in staging/test environments!

    Args:
        max_duration: Maximum duration in seconds

    Returns:
        Summary dict with fuzzing results
    """
    return await get_fuzzing_engine().run_fuzzing_campaign(max_duration)
