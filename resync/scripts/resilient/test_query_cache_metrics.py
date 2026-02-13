#!/usr/bin/env python3
"""
Test script for Query Classification Cache and Metrics.

v5.2.3.24: Validates cache functionality, metrics collection, and performance.

Usage:
    python scripts/test_query_cache_metrics.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_query_classification_cache():
    """Test QueryClassificationCache functionality."""
    from resync.knowledge.retrieval.hybrid_retriever import (
        QueryClassificationCache,
        QueryClassificationResult,
        QueryType,
    )

    print("=" * 70)
    print("🧪 Query Classification Cache Test")
    print("=" * 70)

    cache = QueryClassificationCache(max_size=5, ttl_seconds=2)

    # Test 1: Basic put/get
    print("\n📋 Test 1: Basic put/get")
    result = QueryClassificationResult(
        query_type=QueryType.EXACT_MATCH,
        vector_weight=0.2,
        bm25_weight=0.8,
    )
    cache.put("status job AWSBH001", result)
    
    cached = cache.get("status job AWSBH001")
    assert cached is not None, "Cache should return stored value"
    assert cached.cached == True, "Cached flag should be True"
    assert cached.query_type == QueryType.EXACT_MATCH
    print("   ✅ Basic put/get works")

    # Test 2: Cache miss
    print("\n📋 Test 2: Cache miss")
    miss = cache.get("non-existent query")
    assert miss is None, "Should return None for missing key"
    print("   ✅ Cache miss returns None")

    # Test 3: LRU eviction
    print("\n📋 Test 3: LRU eviction (max_size=5)")
    for i in range(10):
        r = QueryClassificationResult(
            query_type=QueryType.DEFAULT,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        cache.put(f"query {i}", r)
    
    stats = cache.stats()
    assert stats["size"] <= 5, f"Cache size should be <= 5, got {stats['size']}"
    print(f"   ✅ Cache size after 10 inserts: {stats['size']}")

    # Test 4: TTL expiration
    print("\n📋 Test 4: TTL expiration (2 seconds)")
    cache.clear()
    cache.put("ttl test", result)
    
    cached_before = cache.get("ttl test")
    assert cached_before is not None, "Should exist before TTL"
    print("   Waiting 3 seconds for TTL expiration...")
    time.sleep(3)
    
    cached_after = cache.get("ttl test")
    assert cached_after is None, "Should be None after TTL"
    print("   ✅ TTL expiration works")

    # Test 5: Normalization
    print("\n📋 Test 5: Query normalization")
    cache.clear()
    cache.put("  Status  JOB  AWSBH001  ", result)
    
    # Should match with different spacing/case
    normalized = cache.get("status job awsbh001")
    assert normalized is not None, "Should match normalized query"
    print("   ✅ Query normalization works")

    return True


def test_query_metrics():
    """Test QueryMetrics functionality."""
    from resync.knowledge.retrieval.hybrid_retriever import QueryMetrics, QueryType

    print("\n" + "=" * 70)
    print("🧪 Query Metrics Test")
    print("=" * 70)

    metrics = QueryMetrics()

    # Record various queries
    print("\n📋 Recording sample queries...")
    
    # EXACT_MATCH queries
    for _ in range(5):
        metrics.record_query(QueryType.EXACT_MATCH, 50.0, 3, cached=False)
    for _ in range(3):
        metrics.record_query(QueryType.EXACT_MATCH, 5.0, 3, cached=True)

    # SEMANTIC queries
    for _ in range(4):
        metrics.record_query(QueryType.SEMANTIC, 80.0, 5, cached=False)
    for _ in range(2):
        metrics.record_query(QueryType.SEMANTIC, 8.0, 5, cached=True)

    # MIXED queries
    for _ in range(2):
        metrics.record_query(QueryType.MIXED, 60.0, 4, cached=False)

    stats = metrics.get_stats()

    print("\n📊 Aggregated Statistics:")
    print(f"   Total queries: {stats['total_queries']}")
    print("   Cache hit rate: {stats['cache']['hit_rate']:.1%}")
    
    print("\n   By type:")
    for qtype, data in stats["by_type"].items():
        print("   - {qtype}: {data['count']} queries, {data['avg_latency_ms']:.1f}ms avg, {data['avg_results']:.1f} results avg")

    # Validations
    assert stats["total_queries"] == 16, f"Expected 16 total queries, got {stats['total_queries']}"
    assert stats["cache"]["hits"] == 5, f"Expected 5 cache hits, got {stats['cache']['hits']}"
    assert stats["cache"]["misses"] == 11, f"Expected 11 cache misses, got {stats['cache']['misses']}"
    
    print("\n   ✅ Metrics collection works correctly")
    return True


def test_integration():
    """Test integration with HybridRetrieverConfig."""
    from resync.knowledge.retrieval.hybrid_retriever import (
        HybridRetrieverConfig,
        QueryClassificationCache,
        QueryMetrics,
    )

    print("\n" + "=" * 70)
    print("🧪 Integration Test")
    print("=" * 70)

    # Test config with cache enabled
    config = HybridRetrieverConfig(
        cache_enabled=True,
        cache_max_size=500,
        cache_ttl_seconds=1800,
        metrics_enabled=True,
    )

    print(f"\n   cache_enabled: {config.cache_enabled}")
    print(f"   cache_max_size: {config.cache_max_size}")
    print(f"   cache_ttl_seconds: {config.cache_ttl_seconds}")
    print(f"   metrics_enabled: {config.metrics_enabled}")

    # Verify cache and metrics can be created from config
    if config.cache_enabled:
        cache = QueryClassificationCache(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds,
        )
        print(f"   ✅ Cache created with max_size={cache._max_size}")

    if config.metrics_enabled:
        metrics = QueryMetrics()
        print("   ✅ Metrics collector created")

    return True


def test_query_type_enum():
    """Test QueryType enum values."""
    from resync.knowledge.retrieval.hybrid_retriever import QueryType

    print("\n" + "=" * 70)
    print("🧪 QueryType Enum Test")
    print("=" * 70)

    expected_types = ["exact_match", "semantic", "mixed", "default"]
    
    for qt in QueryType:
        assert qt.value in expected_types, f"Unexpected QueryType: {qt.value}"
        print(f"   ✅ {qt.name} = '{qt.value}'")

    return True


def main():
    """Run all tests."""
    results = []

    results.append(("Query Classification Cache", test_query_classification_cache()))
    results.append(("Query Metrics", test_query_metrics()))
    results.append(("Integration", test_integration()))
    results.append(("QueryType Enum", test_query_type_enum()))

    print("\n" + "=" * 70)
    print("📊 Final Results")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All Fase 3 tests passed!")
        print("   - Query classification cache: Working")
        print("   - Performance metrics: Working")
        print("   - TTL expiration: Working")
        print("   - LRU eviction: Working")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
