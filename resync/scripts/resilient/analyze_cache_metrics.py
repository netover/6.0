#!/usr/bin/env python3
"""
Semantic Cache Metrics Analyzer.

v5.3.16 - Analyzes cache performance from logs and Redis stats.

Usage:
    python scripts/analyze_cache_metrics.py
    python scripts/analyze_cache_metrics.py --days 7
    python scripts/analyze_cache_metrics.py --json

Output:
    - Cache hit rate
    - Estimated cost savings
    - Latency statistics
    - Entry distribution
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CacheMetrics:
    """Container for cache metrics."""

    entries: int = 0
    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0
    hit_rate: float = 0.0
    avg_lookup_time_ms: float = 0.0
    estimated_savings_usd: float = 0.0
    threshold: float = 0.25
    redis_stack_available: bool = False
    memory_usage: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "entries": self.entries,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate_percent": round(self.hit_rate * 100, 2),
            "avg_lookup_time_ms": round(self.avg_lookup_time_ms, 2),
            "estimated_savings_usd": round(self.estimated_savings_usd, 2),
            "threshold": self.threshold,
            "redis_stack_available": self.redis_stack_available,
            "memory_usage": self.memory_usage,
        }


async def fetch_cache_stats() -> CacheMetrics:
    """Fetch current cache statistics from Redis."""
    try:
        from resync.core.cache import get_semantic_cache

        cache = await get_semantic_cache()
        stats = await cache.get_stats()

        total_requests = stats.get("hits", 0) + stats.get("misses", 0)
        hit_rate = stats.get("hits", 0) / total_requests if total_requests > 0 else 0

        # Cost per LLM call estimate (average across providers)
        cost_per_call = 0.02  # $0.02 per API call
        savings = stats.get("hits", 0) * cost_per_call

        return CacheMetrics(
            entries=stats.get("entries", 0),
            hits=stats.get("hits", 0),
            misses=stats.get("misses", 0),
            sets=stats.get("sets", 0),
            errors=stats.get("errors", 0),
            hit_rate=hit_rate,
            avg_lookup_time_ms=stats.get("avg_lookup_time_ms", 0),
            estimated_savings_usd=savings,
            threshold=stats.get("threshold", 0.25),
            redis_stack_available=stats.get("redis_stack_available", False),
            memory_usage=stats.get("used_memory_human", "unknown"),
        )

    except Exception as e:
        print(f"Error fetching cache stats: {e}", file=sys.stderr)
        return CacheMetrics()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return "{seconds:.1f}s"
    if seconds < 3600:
        return "{seconds / 60:.1f}m"
    if seconds < 86400:
        return "{seconds / 3600:.1f}h"
    return "{seconds / 86400:.1f}d"


def print_report(metrics: CacheMetrics, period_days: int = 1) -> None:
    """Print formatted metrics report."""
    print("\n" + "=" * 60)
    print("  SEMANTIC CACHE METRICS REPORT")
    print("  Resync v5.3.16")
    print("=" * 60)
    print(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n📊 CACHE PERFORMANCE")
    print("-" * 40)
    print("  Cache Entries:      {metrics.entries:,}")
    print("  Total Hits:         {metrics.hits:,}")
    print("  Total Misses:       {metrics.misses:,}")
    print("  Total Sets:         {metrics.sets:,}")
    print("  Errors:             {metrics.errors:,}")

    total = metrics.hits + metrics.misses
    if total > 0:
        print("\n  Hit Rate:           {metrics.hit_rate * 100:.1f}%")

        # Visual hit rate bar
        bar_width = 30
        filled = int(metrics.hit_rate * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  [{bar}]")

    print("\n💰 COST ANALYSIS")
    print("-" * 40)
    print("  Estimated Savings:  ${metrics.estimated_savings_usd:.2f}")
    print("  LLM Calls Avoided:  {metrics.hits:,}")
    print("  Cost per Call:      $0.02 (estimated)")

    if metrics.misses > 0:
        monthly_projection = (metrics.estimated_savings_usd / period_days) * 30
        print("\n  Projected Monthly:  ${monthly_projection:.2f}")

    print("\n⚡ LATENCY")
    print("-" * 40)
    print("  Avg Lookup Time:    {metrics.avg_lookup_time_ms:.1f}ms")
    print("  vs LLM Response:    ~2,000-5,000ms")

    if metrics.avg_lookup_time_ms > 0:
        speedup = 2500 / metrics.avg_lookup_time_ms  # Assume 2.5s average LLM
        print("  Speedup:            {speedup:.0f}x faster")

    print("\n⚙️ CONFIGURATION")
    print("-" * 40)
    print(f"  Threshold:          {metrics.threshold}")
    print(
        f"  Redis Stack:        {'✓ Available' if metrics.redis_stack_available else '✗ Fallback mode'}"
    )
    print(f"  Memory Usage:       {metrics.memory_usage}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)

    if metrics.hit_rate < 0.3:
        print("  ⚠️  Hit rate is low (<30%)")
        print("      Consider increasing threshold to 0.30-0.35")
    elif metrics.hit_rate < 0.5:
        print("  📈 Hit rate is moderate (30-50%)")
        print("      Monitor for 1-2 weeks before adjusting")
    elif metrics.hit_rate < 0.7:
        print("  ✅ Hit rate is good (50-70%)")
        print("      Current configuration is working well")
    else:
        print("  🎯 Hit rate is excellent (>70%)")
        print("      Consider verifying no false positives")

    if metrics.errors > metrics.hits * 0.05:
        print("  ⚠️  Error rate is high")
        print("      Check Redis connection and logs")

    if not metrics.redis_stack_available:
        print("  📝 Redis Stack not available")
        print("      Install for better vector search performance")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Semantic Cache metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_cache_metrics.py
  python scripts/analyze_cache_metrics.py --days 7
  python scripts/analyze_cache_metrics.py --json > metrics.json
        """,
    )
    parser.add_argument(
        "--days", "-d", type=int, default=1, help="Analysis period in days (default: 1)"
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only output numbers (for scripting)"
    )

    args = parser.parse_args()

    # Set up environment
    os.environ.setdefault("RESYNC_DISABLE_REDIS", "false")

    try:
        # Fetch metrics
        metrics = asyncio.run(fetch_cache_stats())

        if args.json:
            print(json.dumps(metrics.to_dict(), indent=2))
        elif args.quiet:
            print("hit_rate={metrics.hit_rate * 100:.1f}")
            print(f"entries={metrics.entries}")
            print("savings=${metrics.estimated_savings_usd:.2f}")
        else:
            print_report(metrics, args.days)

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
