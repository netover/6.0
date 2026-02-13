"""
Performance benchmarking for Resync system
"""

import asyncio
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from resync.core.interfaces import IAgentManager, ITWSClient
from resync.core.metrics import runtime_metrics


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""

    name: str
    operation: str
    iterations: int
    total_time: float  # seconds
    avg_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    p95_time: float  # 95th percentile in seconds
    p99_time: float  # 99th percentile in seconds
    errors: int
    timestamp: datetime
    metadata: dict[str, Any] = None


class PerformanceBenchmark:
    """Performance benchmarking system for Resync"""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.logger = logging.getLogger(__name__)

    async def run_benchmark(
        self,
        name: str,
        operation: str,
        func: Callable,
        iterations: int = 100,
        warmup_rounds: int = 10,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a performance benchmark for a specific operation

        Args:
            name: Name of the benchmark
            operation: Description of the operation being benchmarked
            func: Function to benchmark
            iterations: Number of iterations to run
            warmup_rounds: Number of warmup rounds to run before timing
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        # Warmup rounds
        for _ in range(warmup_rounds):
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception as e:
                # Log warmup errors but don't fail - warmup is for system preparation
                self.logger.debug(f"Benchmark warmup error ignored: {e}", exc_info=True)

        times = []
        errors = 0

        # Actual benchmark rounds
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception as e:
                errors += 1
                self.logger.error("benchmark_error_in_iteration", iteration=i, error=str(e))
            finally:
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0

        # Calculate percentiles
        p95_time = 0
        p99_time = 0
        if times:
            sorted_times = sorted(times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
            p99_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

        result = BenchmarkResult(
            name=name,
            operation=operation,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p95_time=p95_time,
            p99_time=p99_time,
            errors=errors,
            timestamp=datetime.now(timezone.utc),
        )

        self.results.append(result)
        runtime_metrics.tws_status_requests_success.increment()

        return result

    def compare_with_baseline(
        self, result: BenchmarkResult, baseline: BenchmarkResult
    ) -> dict[str, float]:
        """
        Compare a benchmark result with a baseline

        Args:
            result: Current benchmark result
            baseline: Baseline benchmark result to compare against

        Returns:
            Dictionary with comparison metrics
        """
        return {
            "avg_time_improvement": ((baseline.avg_time - result.avg_time) / baseline.avg_time)
            * 100,
            "p95_time_improvement": ((baseline.p95_time - result.p95_time) / baseline.p95_time)
            * 100,
            "p99_time_improvement": ((baseline.p99_time - result.p99_time) / baseline.p99_time)
            * 100,
            "error_rate_improvement": ((baseline.errors - result.errors) / baseline.iterations)
            * 100,
        }

    def get_historical_performance(self, operation: str, days: int = 7) -> list[BenchmarkResult]:
        """
        Get historical performance data for an operation

        Args:
            operation: Operation to get history for
            days: Number of days to look back

        Returns:
            List of benchmark results from the specified time period
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            result
            for result in self.results
            if result.operation == operation and result.timestamp > cutoff_time
        ]

    def get_performance_trend(self, operation: str, days: int = 7) -> dict[str, list[float]]:
        """
        Get performance trend data for an operation

        Args:
            operation: Operation to get trend for
            days: Number of days to look back

        Returns:
            Dictionary with timeline and performance metrics
        """
        historical = self.get_historical_performance(operation, days)
        historical.sort(key=lambda x: x.timestamp)

        return {
            "timestamps": [result.timestamp for result in historical],
            "avg_times": [result.avg_time for result in historical],
            "p95_times": [result.p95_time for result in historical],
            "p99_times": [result.p99_time for result in historical],
            "error_counts": [result.errors for result in historical],
        }


class SystemBenchmarkRunner:
    """Runner for system-wide performance benchmarks"""

    def __init__(self, agent_manager: IAgentManager, tws_client: ITWSClient):
        self.benchmark = PerformanceBenchmark()
        self.agent_manager = agent_manager
        self.tws_client = tws_client
        self.logger = logging.getLogger(__name__)

    async def run_comprehensive_benchmark(self) -> dict[str, BenchmarkResult]:
        """
        Run comprehensive system benchmarks
        """
        results = {}

        # Benchmark agent operations
        results["agent_creation"] = await self.benchmark.run_benchmark(
            name="Agent Creation",
            operation="create_agent",
            func=self._benchmark_agent_creation,
        )

        # Benchmark TWS operations
        results["tws_status_check"] = await self.benchmark.run_benchmark(
            name="TWS Status Check",
            operation="tws_status",
            func=self._benchmark_tws_status,
        )

        # Benchmark AI operations
        results["ai_query"] = await self.benchmark.run_benchmark(
            name="AI Query",
            operation="ai_query",
            func=self._benchmark_ai_query,
            args=["What is the status of job XYZ?"],
        )

        # Benchmark cache operations
        results["cache_operations"] = await self.benchmark.run_benchmark(
            name="Cache Operations",
            operation="cache_set_get",
            func=self._benchmark_cache_operations,
        )

        return results

    async def _benchmark_agent_creation(self):
        """Benchmark agent creation operation"""
        # This is a placeholder - implement based on actual agent creation logic
        agents = await self.agent_manager.get_all_agents()
        return len(agents)

    async def _benchmark_tws_status(self):
        """Benchmark TWS status check operation"""
        # This is a placeholder - implement based on actual TWS client logic
        return await self.tws_client.get_system_status()

    async def _benchmark_ai_query(self, query: str):
        """Benchmark AI query operation"""
        # This is a placeholder - implement based on actual AI query logic
        # For now, we'll simulate with a simple response
        return f"Response to: {query}"

    async def _benchmark_cache_operations(self):
        """Benchmark cache operations"""
        # This is a placeholder - implement based on actual cache system
        # For now we'll just record metrics
        runtime_metrics.cache_sets.increment()
        runtime_metrics.cache_hits.increment()
        return True


# Create a global benchmark runner instance when needed
async def create_benchmark_runner(
    agent_manager: IAgentManager, tws_client: ITWSClient
) -> SystemBenchmarkRunner:
    """Create a system benchmark runner"""
    return SystemBenchmarkRunner(agent_manager, tws_client)
