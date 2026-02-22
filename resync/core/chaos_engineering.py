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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict
from unittest.mock import patch

from resync.core import get_environment_tags, get_global_correlation_id
from resync.core.agent_manager import AgentConfig, AgentManager
from resync.core.audit_db import add_audit_record, add_audit_records_batch
from resync.core.audit_log import get_audit_log_manager
from resync.core.cache import AsyncTTLCache
from resync.core.metrics import log_with_correlation, runtime_metrics
from resync.core.task_tracker import create_tracked_task

logger = logging.getLogger(__name__)


class FuzzStats(TypedDict):
    """Typing for fuzzing results to satisfy Mypy without type ignores."""

    passed: int
    failed: int
    errors: list[str]


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
    fuzz_function: Callable[[], Awaitable[dict[str, Any]]]
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
        """Run the complete chaos engineering test suite."""
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
            # PHASE 1: Parallel Execution
            timeout_seconds = max(1, int(duration_minutes * 60))
            tasks = []
            results = []

            try:
                async with asyncio.timeout(timeout_seconds):
                    try:
                        async with asyncio.TaskGroup() as tg:
                            tasks = [
                                tg.create_task(self._cache_race_condition_fuzzing(), name="cache_race"),
                                tg.create_task(self._agent_concurrent_initialization_chaos(), name="agent_chaos"),
                                tg.create_task(self._audit_db_failure_injection(), name="audit_chaos"),
                                tg.create_task(self._memory_pressure_simulation(), name="memory_chaos"),
                                tg.create_task(self._network_partition_simulation(), name="network_chaos"),
                                tg.create_task(self._component_isolation_testing(), name="isolation_chaos"),
                            ]
                    except* asyncio.CancelledError:
                        raise
                    except* Exception:
                        logger.warning("chaos_tasks_partial_failure", job_name="chaos_suite")
            except TimeoutError:
                logger.error("chaos_suite_timeout", timeout_seconds=timeout_seconds)
                # results will be partial, processed below

            # PHASE 2: Results Collection (Safe Extraction)
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        results.append(t.result())
                    except Exception as e:
                        if isinstance(e, asyncio.CancelledError):
                            raise
                        results.append(e)
                else:
                    # Task timed out or was cancelled
                    results.append(TimeoutError(f"Task {t.get_name()} failed or timed out"))

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
                elif isinstance(result, ChaosTestResult):
                    self.results.append(result)
                    if result.success:
                        successful_tests += 1
                    total_anomalies += len(result.anomalies_detected)

            suite_duration = time.time() - start_time
            summary = {
                "correlation_id": correlation_id,
                "duration": suite_duration,
                "total_tests": len(tasks),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(tasks) if tasks else 0,
                "total_anomalies": total_anomalies,
                "test_results": [r.__dict__ for r in self.results],
                "environment": get_environment_tags(),
            }

            log_with_correlation(
                logging.INFO,
                f"Chaos suite completed: {successful_tests}/{len(tasks)} tests passed, "
                f"{total_anomalies} anomalies detected in {suite_duration:.1f}s",
                correlation_id,
            )

            return summary

        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _cache_race_condition_fuzzing(self) -> ChaosTestResult:
        test_name = "cache_race_condition_fuzzing"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        cache = AsyncTTLCache(ttl_seconds=10, num_shards=8)

        try:
            anomalies: list[str] = []
            operations = 0
            errors = 0

            async def worker(worker_id: int) -> None:
                nonlocal operations, errors
                for i in range(100):
                    try:
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
                        if isinstance(e, asyncio.CancelledError):
                            raise
                        errors += 1
                        anomalies.append(f"Worker {worker_id} op {i}: {e!s}")

            async with asyncio.TaskGroup() as tg:
                for i in range(10):
                    tg.create_task(worker(i))
            # TaskGroup waits for all workers

            metrics = cache.get_detailed_metrics()
            if metrics["size"] < 0 or metrics["hit_rate"] > 1.0:
                anomalies.append("Cache metrics corrupted")

            return ChaosTestResult(
                test_name=test_name,
                component="async_cache",
                duration=time.time() - start_time,
                success=(errors == 0 and len(anomalies) == 0),
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
        test_name = "agent_concurrent_initialization_chaos"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        anomalies: list[str] = []
        operations = 0
        errors = 0

        try:

            async def init_worker(worker_id: int) -> None:
                nonlocal operations, errors
                try:
                    manager = AgentManager()
                    operations += 1
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
                            anomalies.append(f"Agent operation failed: {result!s}")

                except Exception as e:
                    errors += 1
                    anomalies.append(f"Manager init {worker_id}: {e!s}")

            async with asyncio.TaskGroup() as tg:
                for i in range(5):
                    tg.create_task(init_worker(i))
            # TaskGroup waits for all workers

            return ChaosTestResult(
                test_name=test_name,
                component="agent_manager",
                duration=time.time() - start_time,
                success=(errors == 0 and len(anomalies) == 0),
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
        try:
            await manager.get_agent(f"non_existent_agent_{op_id}")
        except ValueError:
            pass

        try:
            metrics = manager.get_detailed_metrics()
            if "total_agents" not in metrics:
                raise ValueError("Metrics missing total_agents")
        except Exception as e:
            raise RuntimeError(f"Metrics operation failed: {e}") from None

    async def _audit_db_failure_injection(self) -> ChaosTestResult:
        test_name = "audit_db_failure_injection"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        anomalies: list[str] = []
        operations = 0
        errors = 0

        try:
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
            test_memories.extend(test_memories[:10])

            try:
                result = add_audit_records_batch(test_memories)
                operations += len(result)
                successful_inserts = sum(1 for r in result if r is not None)
                if successful_inserts < len(test_memories) - 10:
                    anomalies.append(
                        f"Unexpected batch insert failures: {len(result) - successful_inserts}"
                    )
            except Exception as e:
                errors += 1
                anomalies.append(f"Batch insert failed: {e!s}")

            try:
                audit_manager = get_audit_log_manager()
                metrics = audit_manager.get_audit_metrics()
                if "total_records" not in metrics:
                    anomalies.append("Audit metrics missing total_records")
                operations += 1
            except Exception as e:
                errors += 1
                anomalies.append(f"Metrics retrieval failed: {e!s}")

            try:
                loop = asyncio.get_running_loop()
                sweep_result = await loop.run_in_executor(
                    None,
                    lambda: __import__(
                        "resync.core.audit_db"
                    ).auto_sweep_pending_audits(1, 10),
                )
                operations += sweep_result.get("total_processed", 0)
            except Exception as e:
                errors += 1
                anomalies.append(f"Auto sweep failed: {e!s}")

            return ChaosTestResult(
                test_name=test_name,
                component="audit_db",
                duration=time.time() - start_time,
                success=(len(anomalies) == 0),
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _memory_pressure_simulation(self) -> ChaosTestResult:
        test_name = "memory_pressure_simulation"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        anomalies: list[str] = []
        operations = 0
        errors = 0

        try:
            cache = AsyncTTLCache(ttl_seconds=1, num_shards=4)
            for i in range(100):
                large_obj = {
                    "id": f"large_obj_{i}",
                    "data": "x" * 10000,
                    "metadata": {"size": 10000, "created": time.time()},
                }
                try:
                    await cache.set(f"large_key_{i}", large_obj, ttl_seconds=5)
                    operations += 1

                    if i % 20 == 0:
                        await asyncio.sleep(0.1)
                        metrics = cache.get_detailed_metrics()
                        MAX_CACHE_SIZE = 50
                        if metrics["size"] > MAX_CACHE_SIZE:
                            anomalies.append(
                                f"Cache growing too large at {i}: {metrics['size']}"
                            )
                except Exception as e:
                    errors += 1
                    anomalies.append(f"Large object storage failed at {i}: {e!s}")

            await asyncio.sleep(2)
            final_metrics = cache.get_detailed_metrics()
            if final_metrics["size"] > 10:
                anomalies.append(
                    f"Cleanup ineffective: {final_metrics['size']} items remaining"
                )

            return ChaosTestResult(
                test_name=test_name,
                component="memory_pressure",
                duration=time.time() - start_time,
                success=(len(anomalies) == 0),
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
                details={"final_cache_metrics": final_metrics},
            )
        finally:
            await cache.stop()
            runtime_metrics.close_correlation_id(correlation_id)

    async def _network_partition_simulation(self) -> ChaosTestResult:
        test_name = "network_partition_simulation"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        anomalies: list[str] = []
        operations = 0
        errors = 0

        try:
            original_import = __import__

            def failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if random.random() < 0.1 and "agent" in name:
                    raise ImportError(f"Simulated network failure for {name}")
                return original_import(name, *args, **kwargs)

            # Using mock inside asyncio is tricky. We enforce it specifically around the synchronous AgentManager init.
            with patch("builtins.__import__", side_effect=failing_import):
                for i in range(20):
                    try:
                        manager = AgentManager()
                        operations += 1
                        metrics = manager.get_detailed_metrics()
                        if not isinstance(metrics, dict):
                            anomalies.append(f"Metrics not dict at iteration {i}")
                    except Exception as e:
                        errors += 1
                        if "network failure" not in str(e):
                            anomalies.append(
                                f"Unexpected error during network chaos {i}: {e!s}"
                            )
                    # Await to yield control and ensure safety
                    await asyncio.sleep(0)

            return ChaosTestResult(
                test_name=test_name,
                component="network_simulation",
                duration=time.time() - start_time,
                success=(len(anomalies) == 0),
                error_count=errors,
                operations_performed=operations,
                anomalies_detected=anomalies,
                correlation_id=correlation_id,
            )
        finally:
            runtime_metrics.close_correlation_id(correlation_id)

    async def _component_isolation_testing(self) -> ChaosTestResult:
        test_name = "component_isolation_testing"
        correlation_id = runtime_metrics.create_correlation_id(
            {"component": "chaos_engineering", "operation": test_name}
        )
        start_time = time.time()
        anomalies: list[str] = []
        operations = 0
        errors = 0

        try:
            components_to_test = ["async_cache", "agent_manager", "audit_db"]

            for component in components_to_test:
                try:
                    if component == "async_cache":
                        cache = AsyncTTLCache()
                        with patch.object(
                            cache, "set", side_effect=Exception("Simulated failure")
                        ):
                            try:
                                await cache.set("test_key", "test_value")
                                anomalies.append("Cache set should have failed")
                            except Exception:
                                pass
                        await cache.stop()
                        operations += 1

                    elif component == "agent_manager":
                        with patch(
                            "resync.core.agent_manager.Agent",
                            side_effect=ImportError("Simulated import failure"),
                        ):
                            try:
                                manager = AgentManager()
                                manager.get_detailed_metrics()
                                operations += 1
                            except Exception as e:
                                errors += 1
                                if "import failure" not in str(e):
                                    anomalies.append(
                                        f"Agent manager unexpected error: {e!s}"
                                    )

                    elif component == "audit_db":
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
                                errors += 1
                                if "DB failure" not in str(e):
                                    anomalies.append(
                                        f"Audit DB unexpected error: {e!s}"
                                    )

                except Exception as e:
                    errors += 1
                    anomalies.append(
                        f"Component {component} isolation test failed: {e!s}"
                    )

            return ChaosTestResult(
                test_name=test_name,
                component="component_isolation",
                duration=time.time() - start_time,
                success=(len(anomalies) == 0),
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
                FuzzingScenario(
                    name="agent_config_fuzzing",
                    description="Fuzz agent configurations",
                    fuzz_function=self._fuzz_agent_configs,
                    expected_failures=["ValidationError", "TypeError"],
                ),
                FuzzingScenario(
                    name="audit_record_fuzzing",
                    description="Fuzz audit record structures",
                    fuzz_function=self._fuzz_audit_records,
                    expected_failures=["TypeError", "ValueError"],
                ),
            ]
        )

    async def run_fuzzing_campaign(self, max_duration: float = 60.0) -> dict[str, Any]:
        """Run a complete fuzzing campaign on all scenarios."""
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
        results: list[dict[str, Any]] = []

        try:
            for scenario in self.scenarios:
                scenario_start = time.time()
                try:
                    # Direto no Event Loop atual (nativo async), removendo as threads bloqueantes
                    result = await asyncio.wait_for(
                        scenario.fuzz_function(),
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
                except TimeoutError:
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

    async def _fuzz_cache_keys(self) -> dict[str, Any]:
        """Fuzz cache keys with various edge cases."""
        cache = AsyncTTLCache()
        test_cases = [
            "normal_key",
            "key_with_underscores",
            "key-with-dashes",
            "123_numeric_start",
            "",
            "a" * 1000,
            "key with spaces",
            "key\twith\ttabs",
            "key\nwith\nnewlines",
            "key\x00with\x00nulls",
            "🧪_emoji_key",
            None,
            123,
            ["list", "key"],
            {"dict": "key"},
        ]

        results: FuzzStats = {"passed": 0, "failed": 0, "errors": []}

        try:
            for i, key in enumerate(test_cases):
                try:
                    if key is None:
                        continue
                    await cache.set(key, f"value_{i}")
                    retrieved = await cache.get(key)

                    if retrieved == f"value_{i}":
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Key {key!r}: value mismatch")

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Key {key!r}: {e!s}")
        finally:
            await cache.stop()

        return dict(results)

    async def _fuzz_cache_values(self) -> dict[str, Any]:
        """Fuzz cache values with complex objects."""
        cache = AsyncTTLCache()
        test_cases: list[Any] = [
            "string_value",
            42,
            3.14,
            True,
            None,
            {"nested": {"dict": "value"}},
            ["list", "of", "items"],
            {},  # Placeholder for self-ref
            "x" * 100000,
            list(range(10000)),
            object(),
        ]

        self_ref: dict[str, Any] = {"data": "value"}
        self_ref["self"] = self_ref
        test_cases[7] = self_ref

        results: FuzzStats = {"passed": 0, "failed": 0, "errors": []}

        try:
            for i, value in enumerate(test_cases):
                try:
                    key = f"fuzz_value_key_{i}"
                    await cache.set(key, value)
                    retrieved = await cache.get(key)

                    try:
                        if retrieved == value or str(retrieved) == str(value):
                            results["passed"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Value {i}: mismatch")
                    except Exception:
                        if retrieved is not None:
                            results["passed"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"Value {i}: retrieval failed")

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Value {i}: {e!s}")
        finally:
            await cache.stop()

        return dict(results)

    async def _fuzz_cache_ttl(self) -> dict[str, Any]:
        """Fuzz TTL values."""
        cache = AsyncTTLCache()
        test_cases = [1, 30, 300, 3600, 0, -1, 999999, float("inf"), None, "30", [30]]
        results: FuzzStats = {"passed": 0, "failed": 0, "errors": []}

        try:
            for i, ttl in enumerate(test_cases):
                try:
                    key = f"fuzz_ttl_key_{i}"
                    # Type ignore allowed here intentionally for fuzzing testing invalid inputs
                    await cache.set(key, f"value_{i}", ttl_seconds=ttl)  # type: ignore[arg-type]
                    retrieved = await cache.get(key)

                    if retrieved is not None:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            f"TTL {ttl!r}: immediate expiration"
                        )

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"TTL {ttl!r}: {e!s}")
        finally:
            await cache.stop()

        return dict(results)

    async def _fuzz_agent_configs(self) -> dict[str, Any]:
        """Fuzz agent configurations."""
        test_cases = [
            {
                "id": "test_agent_1",
                "name": "Test Agent",
                "role": "Tester",
                "goal": "Test things",
                "backstory": "A testing agent",
                "tools": ["test_tool"],
                "model_name": "test_model",
            },
            {
                "id": "",
                "name": "Test",
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": [],
                "model_name": "test",
            },
            {
                "id": "test_agent_2",
                "name": None,
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
                "tools": None,
                "model_name": "test",
            },
            {
                "id": "test_agent_4",
                "name": "x" * 1000,
                "role": "Tester",
                "goal": "Test",
                "backstory": "Test",
                "tools": ["test_tool"],
                "model_name": "test",
            },
        ]

        results: FuzzStats = {"passed": 0, "failed": 0, "errors": []}

        for i, config_data in enumerate(test_cases):
            try:
                AgentConfig(**config_data)  # type: ignore[arg-type]
                results["passed"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Config {i}: {e!s}")

        return dict(results)

    async def _fuzz_audit_records(self) -> dict[str, Any]:
        """Fuzz audit record structures."""
        test_cases = [
            {
                "id": "audit_1",
                "user_query": "Test query",
                "agent_response": "Test response",
                "ia_audit_reason": "test",
                "ia_audit_confidence": 0.8,
            },
            {"id": None, "user_query": "Test", "agent_response": "Test"},
            {"id": "audit_2", "user_query": None, "agent_response": "Test"},
            {"id": "audit_3", "user_query": "Test", "agent_response": "x" * 100000},
            {
                "id": "audit_4",
                "user_query": "Test",
                "agent_response": "Test",
                "ia_audit_reason": None,
                "ia_audit_confidence": "0.8",
            },
        ]

        results: FuzzStats = {"passed": 0, "failed": 0, "errors": []}

        for i, record in enumerate(test_cases):
            try:
                result = add_audit_record(record)
                if result is not None or i > 0:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Record {i}: unexpected failure")
            except Exception as e:
                if i == 0:
                    results["failed"] += 1
                    results["errors"].append(f"Record {i}: {e!s}")
                else:
                    results["passed"] += 1

        return dict(results)


# =============================================================================
# Lazy Initialization
# =============================================================================

_chaos_engineer: ChaosEngineer | None = None
_fuzzing_engine: FuzzingEngine | None = None


def get_chaos_engineer() -> ChaosEngineer:
    global _chaos_engineer
    if _chaos_engineer is None:
        logger.warning(
            "⚠️ ChaosEngineer initialized - ensure this is NOT production!",
            extra={"component": "chaos_engineering", "action": "initialization"},
        )
        _chaos_engineer = ChaosEngineer()
    return _chaos_engineer


def get_fuzzing_engine() -> FuzzingEngine:
    global _fuzzing_engine
    if _fuzzing_engine is None:
        logger.warning(
            "⚠️ FuzzingEngine initialized - ensure this is NOT production!",
            extra={"component": "fuzzing_engine", "action": "initialization"},
        )
        _fuzzing_engine = FuzzingEngine()
    return _fuzzing_engine


async def run_chaos_engineering_suite(duration_minutes: float = 5.0) -> dict[str, Any]:
    return await get_chaos_engineer().run_full_chaos_suite(duration_minutes)


async def run_fuzzing_campaign(max_duration: float = 60.0) -> dict[str, Any]:
    return await get_fuzzing_engine().run_fuzzing_campaign(max_duration)
