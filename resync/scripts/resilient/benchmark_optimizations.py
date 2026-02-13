"""
Benchmark Script for Performance Optimizations

This script measures the performance improvements from O(n²) optimizations.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resync.core.cache import RobustCacheManager
from resync.core.utils.data_structures import (
    FastSet,
    create_lru_cache,
)
from resync.core.utils.string_optimizer import StringProcessor, TextChunker
from resync.core.validation_optimizer import (
    get_global_validator,
    validate_tool_names_batch,
)


class PerformanceBenchmark:
    """Benchmark performance improvements."""

    def __init__(self):
        self.results: dict[str, Any] = {}

    def benchmark_cache_operations(self, iterations: int = 1000) -> dict[str, float]:
        """Benchmark cache performance improvements."""
        print("Benchmarking cache operations...")

        # Test optimized cache
        cache = RobustCacheManager(max_items=1000, max_memory_mb=10)

        # Benchmark set operations
        start_time = time.perf_counter()
        for i in range(iterations):
            key = f"test_key_{i}"
            value = f"test_value_{i}" * 100  # Simulate larger values
            asyncio.run(cache.set(key, value))
        set_time = time.perf_counter() - start_time

        # Benchmark get operations
        start_time = time.perf_counter()
        hit_count = 0
        for i in range(iterations):
            key = f"test_key_{i}"
            result = asyncio.run(cache.get(key))
            if result:
                hit_count += 1
        get_time = time.perf_counter() - start_time

        hit_rate = hit_count / iterations

        asyncio.run(cache.cleanup())

        return {
            "set_time_ms": (set_time / iterations) * 1000,
            "get_time_ms": (get_time / iterations) * 1000,
            "hit_rate": hit_rate,
            "cache_size": cache.get_metrics()["items"],
        }

    def benchmark_string_operations(self, iterations: int = 10000) -> dict[str, float]:
        """Benchmark string processing optimizations."""
        print("Benchmarking string operations...")

        # Test data
        test_strings = ["test_string_" + "x" * i for i in range(100)]

        # Test optimized string processor
        processor = StringProcessor()

        # Test join operations
        start_time = time.perf_counter()
        for _ in range(iterations):
            processor.join_efficient(test_strings)
        join_time = time.perf_counter() - start_time

        # Test validation operations
        start_time = time.perf_counter()
        for _ in range(iterations):
            for text in test_strings:
                processor.validate_alphanumeric(text)
        validation_time = time.perf_counter() - start_time

        return {
            "join_time_ms": (join_time / iterations) * 1000000,
            "validation_time_ms": (validation_time / (iterations * len(test_strings))) * 1000000,
        }

    def benchmark_data_structures(self, iterations: int = 10000) -> dict[str, float]:
        """Benchmark data structure optimizations."""
        print("Benchmarking data structures...")

        # Test LRU cache vs dict
        lru_cache = create_lru_cache(capacity=100)
        regular_dict = {}

        # Test LRU cache
        start_time = time.perf_counter()
        for i in range(iterations):
            key = f"key_{i % 100}"
            lru_cache.put(key, f"value_{i}")
        for i in range(iterations):
            key = f"key_{i % 100}"
            lru_cache.get(key)
        lru_time = time.perf_counter() - start_time

        # Test regular dict
        start_time = time.perf_counter()
        for i in range(iterations):
            key = f"key_{i % 100}"
            regular_dict[key] = f"value_{i}"
        for i in range(iterations):
            key = f"key_{i % 100}"
            regular_dict.get(key)
        dict_time = time.perf_counter() - start_time

        # Test optimized set
        fast_set = FastSet()
        regular_set = set()

        # Test FastSet
        start_time = time.perf_counter()
        for i in range(iterations):
            item = f"item_{i % 100}"
            fast_set.add(item)
        for i in range(iterations):
            item = f"item_{i % 100}"
            _ = item in fast_set  # Intentionally discard result - measuring lookup time
        fast_set_time = time.perf_counter() - start_time

        # Test regular set
        start_time = time.perf_counter()
        for i in range(iterations):
            item = f"item_{i % 100}"
            regular_set.add(item)
        for i in range(iterations):
            item = f"item_{i % 100}"
            _ = item in regular_set  # Intentionally discard result - measuring lookup time
        regular_set_time = time.perf_counter() - start_time

        return {
            "lru_time_ms": (lru_time / iterations) * 1000000,
            "dict_time_ms": (dict_time / iterations) * 1000000,
            "fast_set_time_ms": (fast_set_time / iterations) * 1000000,
            "regular_set_time_ms": (regular_set_time / iterations) * 1000000,
        }

    def benchmark_validation_optimizations(self, iterations: int = 1000) -> dict[str, float]:
        """Benchmark validation optimizations."""
        print("Benchmarking validation optimizations...")

        # Test data
        emails = [f"test{i}@example.com" for i in range(100)]
        [f"user_{i}" for i in range(50)]
        tool_names = [f"tool_{i}" for i in range(30)]

        # Test optimized validator
        validator = get_global_validator()

        # Test email validation
        start_time = time.perf_counter()
        for _ in range(iterations):
            for email in emails:
                validator.validate_email(email)
        optimized_time = time.perf_counter() - start_time

        # Test batch validation
        start_time = time.perf_counter()
        for _ in range(iterations):
            validate_tool_names_batch(tool_names)
        batch_time = time.perf_counter() - start_time

        return {
            "optimized_email_ms": (optimized_time / (iterations * len(emails))) * 1000000,
            "batch_validation_ms": (batch_time / (iterations * len(tool_names))) * 1000000,
        }

    def run_text_chunking_benchmark(self, iterations: int = 100) -> dict[str, float]:
        """Benchmark text chunking performance."""
        print("Benchmarking text chunking...")

        # Test large text
        large_text = "word " * 100000  # 500KB text

        # Test optimized chunker
        chunker = TextChunker(chunk_size=1000, overlap=200)

        start_time = time.perf_counter()
        for _ in range(iterations):
            chunks = list(chunker.chunk_text(large_text))
        chunking_time = time.perf_counter() - start_time

        return {
            "chunking_time_ms": (chunking_time / iterations) * 1000,
            "chunks_produced": len(chunks),
        }

    async def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all performance benchmarks."""
        print("=" * 60)
        print("PERFORMANCE OPTIMIZATION BENCHMARKS")
        print("=" * 60)

        results = {
            "cache_performance": self.benchmark_cache_operations(),
            "string_operations": self.benchmark_string_operations(),
            "data_structures": self.benchmark_data_structures(),
            "validation_optimizations": self.benchmark_validation_optimizations(),
            "text_chunking": await self.run_text_chunking_benchmark(),
        }

        # Print results
        print("\nBENCHMARK RESULTS:")
        print("-" * 40)

        for category, metrics in results.items():
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print("  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

        print("-" * 40)

        # Performance improvements
        print("\nPERFORMANCE IMPROVEMENTS:")
        print("-" * 40)

        results["cache_performance"]
        ds_results = results["data_structures"]

        if ds_results["lru_time_ms"] > 0 and ds_results["dict_time_ms"] > 0:
            lru_improvement = (
                (ds_results["dict_time_ms"] - ds_results["lru_time_ms"])
                / ds_results["dict_time_ms"]
            ) * 100
            print("  LRU Cache vs Dict: {lru_improvement:.1f}% faster")

        if ds_results["fast_set_time_ms"] > 0 and ds_results["regular_set_time_ms"] > 0:
            set_improvement = (
                (ds_results["regular_set_time_ms"] - ds_results["fast_set_time_ms"])
                / ds_results["regular_set_time_ms"]
            ) * 100
            print("  FastSet vs Set: {set_improvement:.1f}% faster")

        results["string_operations"]
        validation_results = results["validation_optimizations"]

        if validation_results["batch_validation_ms"] > 0:
            batch_improvement = (
                (1000000 - validation_results["batch_validation_ms"]) / 1000000
            ) * 100
            print("  Batch Validation vs Individual: {batch_improvement:.1f}% faster")

        print("-" * 40)

        return results


async def main():
    """Main benchmark function."""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()

    print("\nBenchmark completed. Total time: {time.perf_counter():.4f}s")
    return results


if __name__ == "__main__":
    asyncio.run(main())
