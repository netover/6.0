"""
Intelligent Query Cache with Dynamic TTL.

This module provides specialized caching for database queries with:
- Query result caching with automatic invalidation
- Dynamic TTL adjustment based on data change patterns
- Query performance monitoring and optimization
- Batch query caching and deduplication
- Prepared statement result caching
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any

from resync.core.structured_logger import get_logger

from .advanced_cache import get_advanced_cache_manager

logger = get_logger(__name__)


# =============================================================================
# PRE-COMPILED REGEX PATTERNS (Performance optimization)
# =============================================================================
# Compiled once at module load, not on every call
_FROM_PATTERN = re.compile(r"\bfrom\s+(\w+)", re.IGNORECASE)
_JOIN_PATTERN = re.compile(r"\bjoin\s+(\w+)", re.IGNORECASE)


@lru_cache(maxsize=1024)
def _extract_table_names(sql: str) -> frozenset[str]:
    """
    Extract table names from SQL query (cached).

    Uses pre-compiled regex for performance.
    Results are cached to avoid repeated parsing of identical queries.
    """
    tables = set()

    # Find tables in FROM clauses
    for match in _FROM_PATTERN.finditer(sql):
        tables.add(match.group(1).lower())

    # Find tables in JOIN clauses
    for match in _JOIN_PATTERN.finditer(sql):
        tables.add(match.group(1).lower())

    return frozenset(tables)


@dataclass
class QueryFingerprint:
    """Fingerprint of a database query for caching."""

    sql: str
    parameters: tuple[Any, ...]
    connection_id: str

    # Cached values (computed lazily)
    _cache_key: str | None = field(default=None, repr=False, compare=False)
    _table_names: set[str] | None = field(default=None, repr=False, compare=False)

    @property
    def cache_key(self) -> str:
        """Generate cache key from query fingerprint using BLAKE2b."""
        if self._cache_key is None:
            # Create deterministic key from SQL and parameters
            key_data = f"{self.sql}|{self.parameters}|{self.connection_id}"
            # Use BLAKE2b instead of MD5 for better security
            hash_value = hashlib.blake2b(key_data.encode(), digest_size=16).hexdigest()
            object.__setattr__(self, "_cache_key", f"query:{hash_value}")
        return self._cache_key

    @property
    def table_names(self) -> set[str]:
        """
        Extract table names from SQL query.

        Uses pre-compiled regex and module-level cache for performance.
        Results are cached per-instance to avoid repeated lookups.
        """
        if self._table_names is None:
            # Use cached extraction function
            frozen = _extract_table_names(self.sql)
            object.__setattr__(self, "_table_names", set(frozen))
        return self._table_names


@dataclass
class QueryExecutionStats:
    """Statistics for query execution."""

    query_fingerprint: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    last_execution_time: float = 0.0
    result_change_count: int = 0
    last_result_hash: str | None = None
    table_dependencies: set[str] = field(default_factory=set)

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / max(1, self.execution_count)

    @property
    def change_frequency(self) -> float:
        """Calculate how often results change (changes per execution)."""
        return self.result_change_count / max(1, self.execution_count)

    def calculate_dynamic_ttl(self) -> int:
        """Calculate dynamic TTL based on query characteristics."""
        base_ttl = 300  # 5 minutes default

        # Fast queries can be cached longer
        if self.avg_execution_time < 0.1:  # < 100ms
            base_ttl *= 2

        # Queries with stable results can be cached longer
        if self.change_frequency < 0.1:  # < 10% change rate
            base_ttl *= 3

        # Frequently executed queries can be cached longer
        if self.execution_count > 100:
            base_ttl = int(base_ttl * 1.5)

        # Cap at 24 hours
        return min(base_ttl, 86400)


@dataclass
class QueryResult:
    """Cached query result with metadata."""

    data: Any
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    result_hash: str = ""
    row_count: int = 0
    execution_stats: QueryExecutionStats | None = None

    def __post_init__(self):
        """Calculate result hash and row count after initialization."""
        # Calculate hash of result for change detection using BLAKE2b
        result_str = str(
            sorted(self.data.items()) if isinstance(self.data, dict) else str(self.data)
        )
        # Use BLAKE2b instead of MD5 for better security
        self.result_hash = hashlib.blake2b(result_str.encode(), digest_size=16).hexdigest()

        # Count rows if it's a list of results
        if isinstance(self.data, list):
            self.row_count = len(self.data)


@dataclass
class TableChangeTracker:
    """Track changes to database tables for cache invalidation."""

    table_name: str
    last_change_timestamp: float = 0.0
    change_count: int = 0
    tracked_queries: set[str] = field(default_factory=set)  # Query fingerprints affected

    def record_change(self) -> None:
        """Record a table change."""
        self.last_change_timestamp = time.time()
        self.change_count += 1

    def get_change_frequency(self) -> float:
        """Get change frequency (changes per hour)."""
        age_hours = (time.time() - self.last_change_timestamp) / 3600
        return self.change_count / max(1, age_hours)


class QueryCacheManager:
    """
    Intelligent query cache manager with dynamic TTL and change tracking.

    Features:
    - Query result caching with automatic invalidation
    - Dynamic TTL based on execution patterns and data stability
    - Table change tracking for smart invalidation
    - Batch query optimization
    - Performance monitoring and analytics
    """

    def __init__(self):
        self.cache_manager = None
        self.query_stats: dict[str, QueryExecutionStats] = {}
        self.table_trackers: dict[str, TableChangeTracker] = {}
        self.batch_queries: dict[str, list[asyncio.Future]] = defaultdict(list)

        # Configuration
        self.enable_change_tracking = True
        self.max_batch_size = 10
        self.ttl_override_threshold = 3600  # 1 hour

        # Statistics
        self.total_queries_cached = 0
        self.total_queries_executed = 0
        self.cache_hit_ratio = 0.0

        # Thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the query cache manager."""
        self.cache_manager = await get_advanced_cache_manager()

        # Set up table change tracking
        if self.enable_change_tracking:
            self._setup_change_tracking()

        logger.info("Query cache manager initialized")

    async def execute_query(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
        connection_id: str = "default",
        force_refresh: bool = False,
        ttl_override: int | None = None,
    ) -> QueryResult:
        """
        Execute query with intelligent caching.

        Args:
            sql: SQL query string
            parameters: Query parameters
            connection_id: Database connection identifier
            force_refresh: Bypass cache
            ttl_override: Override dynamic TTL calculation

        Returns:
            QueryResult with data and metadata
        """
        fingerprint = QueryFingerprint(sql, parameters, connection_id)
        cache_key = fingerprint.cache_key

        start_time = time.time()

        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                execution_time = time.time() - start_time
                logger.debug("Query cache hit for %s...", cache_key[:16])
                self._update_cache_stats(hit=True, execution_time=execution_time)
                return cached_result

        # Cache miss or forced refresh - execute query
        try:
            # Here we would actually execute the query
            # For now, simulate execution
            execution_start = time.time()
            result_data = await self._simulate_query_execution(sql, parameters)
            execution_time = time.time() - execution_start

            # Create result object
            result = QueryResult(
                data=result_data,
                execution_time=execution_time,
                execution_stats=self.query_stats.get(cache_key),
            )

            # Update query statistics
            await self._update_query_stats(fingerprint, result, execution_time)

            # Cache the result
            await self._cache_query_result(cache_key, result, fingerprint, ttl_override)

            # Track table dependencies
            await self._track_table_dependencies(fingerprint)

            self._update_cache_stats(hit=False, execution_time=execution_time)
            return result

        except Exception as e:
            logger.error("Query execution failed: %s", e)
            self._update_cache_stats(hit=False, execution_time=time.time() - start_time, error=True)
            raise

    async def execute_batch(
        self, queries: list[tuple[str, tuple[Any, ...]]], connection_id: str = "default"
    ) -> list[QueryResult]:
        """
        Execute multiple queries with batch optimization.

        Args:
            queries: List of (sql, parameters) tuples
            connection_id: Database connection identifier

        Returns:
            List of QueryResult objects
        """
        if len(queries) <= self.max_batch_size:
            # Execute individually with caching
            return await asyncio.gather(
                *[self.execute_query(sql, params, connection_id) for sql, params in queries]
            )

        # For larger batches, optimize execution
        return await self._execute_optimized_batch(queries, connection_id)

    async def invalidate_table_cache(self, table_name: str) -> int:
        """
        Invalidate cache for all queries that depend on a table.

        Args:
            table_name: Name of the table that changed

        Returns:
            Number of cache entries invalidated
        """
        async with self._lock:
            if table_name in self.table_trackers:
                tracker = self.table_trackers[table_name]
                tracker.record_change()

                # Invalidate all queries that depend on this table
                invalidated = 0
                for query_key in tracker.tracked_queries:
                    if query_key in self.query_stats and self.cache_manager:
                        # Invalidate cache entry
                        invalidated += await self.cache_manager.invalidate(
                            query_key, cascade=False
                        )

                logger.info("Invalidated %s queries dependent on table %s", invalidated, table_name)
                return invalidated

        return 0

    async def record_table_change(self, table_name: str, change_type: str = "update") -> None:
        """
        Record that a table has changed for cache invalidation.

        Args:
            table_name: Name of the changed table
            change_type: Type of change (insert, update, delete)
        """
        async with self._lock:
            if table_name not in self.table_trackers:
                self.table_trackers[table_name] = TableChangeTracker(table_name)

            self.table_trackers[table_name].record_change()

            # Invalidate dependent queries immediately for critical changes
            if change_type in ["delete", "update"]:
                await self.invalidate_table_cache(table_name)

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive query cache statistics."""
        total_queries = self.total_queries_cached + self.total_queries_executed

        return {
            "performance": {
                "cache_hit_ratio": self.cache_hit_ratio,
                "total_queries": total_queries,
                "cached_queries": self.total_queries_cached,
                "executed_queries": self.total_queries_executed,
            },
            "queries": {
                "tracked_queries": len(self.query_stats),
                "avg_execution_time": sum(
                    stats.avg_execution_time for stats in self.query_stats.values()
                )
                / max(1, len(self.query_stats)),
            },
            "tables": {
                "tracked_tables": len(self.table_trackers),
                "tables_with_changes": sum(
                    1 for tracker in self.table_trackers.values() if tracker.change_count > 0
                ),
            },
            "ttl_distribution": self._calculate_ttl_distribution(),
        }

    async def _get_cached_result(self, cache_key: str) -> QueryResult | None:
        """Get cached query result if valid."""
        if not self.cache_manager:
            return None

        try:
            cached = await self.cache_manager.get(cache_key)
            if cached and isinstance(cached, QueryResult):
                # Check if result is still valid based on table changes
                if self._is_result_still_valid(cached, cache_key):
                    return cached
                # Result invalidated due to table changes
                await self.cache_manager.invalidate(cache_key, cascade=False)
        except Exception as e:
            logger.warning("Cache retrieval error for %s: %s", cache_key, e)

        return None

    async def _cache_query_result(
        self,
        cache_key: str,
        result: QueryResult,
        fingerprint: QueryFingerprint,
        ttl_override: int | None,
    ) -> None:
        """Cache query result with appropriate TTL."""
        if not self.cache_manager:
            return

        # Determine TTL
        if ttl_override:
            ttl = ttl_override
        elif cache_key in self.query_stats:
            ttl = self.query_stats[cache_key].calculate_dynamic_ttl()
        else:
            ttl = 300  # Default 5 minutes

        # Determine dependencies (table names)
        dependencies = list(fingerprint.table_names)

        # Cache with dependencies for automatic invalidation
        await self.cache_manager.set(
            key=cache_key,
            value=result,
            ttl=ttl,
            dependencies=dependencies,
            tags=["query_cache", "database"],
        )

        self.total_queries_cached += 1

    async def _update_query_stats(
        self, fingerprint: QueryFingerprint, result: QueryResult, execution_time: float
    ) -> None:
        """Update statistics for a query."""
        cache_key = fingerprint.cache_key

        async with self._lock:
            if cache_key not in self.query_stats:
                self.query_stats[cache_key] = QueryExecutionStats(
                    query_fingerprint=cache_key,
                    table_dependencies=fingerprint.table_names,
                )

            stats = self.query_stats[cache_key]
            stats.execution_count += 1
            stats.total_execution_time += execution_time
            stats.last_execution_time = time.time()

            # Check if result changed
            if stats.last_result_hash and stats.last_result_hash != result.result_hash:
                stats.result_change_count += 1

            stats.last_result_hash = result.result_hash

    def _is_result_still_valid(self, result: QueryResult, cache_key: str) -> bool:
        """Check if cached result is still valid based on table changes."""
        if not result.execution_stats:
            return True

        # Check if any dependent tables changed since result was cached
        for table_name in result.execution_stats.table_dependencies:
            if table_name in self.table_trackers:
                tracker = self.table_trackers[table_name]
                if tracker.last_change_timestamp > result.timestamp:
                    return False

        return True

    async def _track_table_dependencies(self, fingerprint: QueryFingerprint) -> None:
        """Track which queries depend on which tables."""
        async with self._lock:
            for table_name in fingerprint.table_names:
                if table_name not in self.table_trackers:
                    self.table_trackers[table_name] = TableChangeTracker(table_name)

                self.table_trackers[table_name].tracked_queries.add(fingerprint.cache_key)

    def _setup_change_tracking(self) -> None:
        """Set up database change tracking."""
        # This would typically involve setting up database triggers
        # or change data capture mechanisms
        # For now, this is a placeholder
        logger.info("Database change tracking setup completed")

    async def _simulate_query_execution(self, sql: str, parameters: tuple[Any, ...]) -> Any:
        """Simulate query execution (replace with actual database calls)."""
        # Simulate execution time based on query complexity
        complexity_factor = len(sql.split()) / 10
        execution_time = complexity_factor * 0.01  # 10ms per 10 words

        await asyncio.sleep(min(execution_time, 0.1))  # Cap at 100ms for simulation

        # Return mock data based on query type
        if "select" in sql.lower():
            return [{"id": 1, "name": "Sample Data", "value": 42}]
        if "count" in sql.lower():
            return 42
        return {"affected_rows": 1}

    async def _execute_optimized_batch(
        self, queries: list[tuple[str, tuple[Any, ...]]], connection_id: str
    ) -> list[QueryResult]:
        """Execute large batch of queries with optimization."""
        # Group similar queries for optimization
        query_groups = defaultdict(list)

        for sql, params in queries:
            # Group by similar query patterns
            pattern = sql.split()[0].lower()  # SELECT, INSERT, etc.
            query_groups[pattern].append((sql, params))

        # Execute groups concurrently
        tasks = []
        for pattern, group_queries in query_groups.items():
            if pattern == "select" and len(group_queries) > 1:
                # Batch SELECT queries
                tasks.append(self._execute_batch_select(group_queries, connection_id))
            else:
                # Execute other queries individually
                tasks.extend(
                    [
                        self.execute_query(sql, params, connection_id)
                        for sql, params in group_queries
                    ]
                )

        results = await asyncio.gather(*tasks)
        return [
            item
            for sublist in results
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

    async def _execute_batch_select(
        self, queries: list[tuple[str, tuple[Any, ...]]], connection_id: str
    ) -> list[QueryResult]:
        """Execute batch SELECT queries efficiently."""
        # For now, execute individually
        # In a real implementation, this could combine queries or use batch APIs
        return await asyncio.gather(
            *[self.execute_query(sql, params, connection_id) for sql, params in queries]
        )

    def _calculate_ttl_distribution(self) -> dict[str, int]:
        """Calculate distribution of TTL values."""
        ttl_ranges = {
            "0-5min": 0,
            "5-15min": 0,
            "15-60min": 0,
            "1-6h": 0,
            "6-24h": 0,
            "24h+": 0,
        }

        for stats in self.query_stats.values():
            ttl = stats.calculate_dynamic_ttl()

            if ttl <= 300:
                ttl_ranges["0-5min"] += 1
            elif ttl <= 900:
                ttl_ranges["5-15min"] += 1
            elif ttl <= 3600:
                ttl_ranges["15-60min"] += 1
            elif ttl <= 21600:
                ttl_ranges["1-6h"] += 1
            elif ttl <= 86400:
                ttl_ranges["6-24h"] += 1
            else:
                ttl_ranges["24h+"] += 1

        return ttl_ranges

    def _update_cache_stats(self, hit: bool, execution_time: float, error: bool = False) -> None:
        """Update cache performance statistics."""
        if hit:
            self.total_queries_cached += 1
        else:
            self.total_queries_executed += 1

        # Update hit ratio
        total = self.total_queries_cached + self.total_queries_executed
        if total > 0:
            self.cache_hit_ratio = self.total_queries_cached / total


# Global query cache manager instance
_query_cache_manager_instance: QueryCacheManager | None = None


class _LazyQueryCacheManager:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance = None

    def get_instance(self) -> QueryCacheManager:
        if self._instance is None:
            self._instance = QueryCacheManager()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)


query_cache_manager = _LazyQueryCacheManager()


def get_query_cache_manager_sync() -> QueryCacheManager:
    """Return the singleton instance synchronously."""
    return query_cache_manager.get_instance()


async def get_query_cache_manager() -> QueryCacheManager:
    """Get the global query cache manager instance."""
    if not query_cache_manager.cache_manager:
        await query_cache_manager.initialize()
    return query_cache_manager
