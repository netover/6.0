"""
Intelligent Database Query Optimization and Batching.

This module provides advanced database optimization features including:
- Intelligent query batching and grouping
- Query execution optimization
- Connection multiplexing support
- Performance monitoring and analytics
- Automatic query rewriting for better performance
"""

import asyncio
import contextlib
import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryBatch:
    """Represents a batch of queries for optimized execution."""

    queries: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    batch_id: str = ""
    created_at: float = field(default_factory=time.time)
    priority: int = 1  # 1=low, 5=high
    timeout: float = 30.0
    max_queries: int = 50

    def __post_init__(self):
        """Generate batch ID after initialization."""
        if not self.batch_id and self.queries:
            # Create deterministic batch ID based on first query
            sql, params = self.queries[0]
            batch_content = f"{sql}|{params}|{self.created_at}"
            # Use BLAKE2b instead of MD5 for better security
            hash_value = hashlib.blake2b(batch_content.encode(), digest_size=4).hexdigest()
            self.batch_id = f"batch_{hash_value}"

    @property
    def is_full(self) -> bool:
        """Check if batch is at capacity."""
        return len(self.queries) >= self.max_queries

    @property
    def is_expired(self) -> bool:
        """Check if batch has expired."""
        return time.time() - self.created_at > self.timeout

    def add_query(self, sql: str, params: tuple[Any, ...]) -> bool:
        """Add query to batch. Returns True if added, False if batch is full."""
        if self.is_full:
            return False

        self.queries.append((sql, params))
        return True

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics for this batch."""
        query_types = defaultdict(int)
        for sql, _ in self.queries:
            # Simple classification based on SQL keywords
            sql_lower = sql.lower().strip()
            if sql_lower.startswith("select"):
                query_types["select"] += 1
            elif sql_lower.startswith("insert"):
                query_types["insert"] += 1
            elif sql_lower.startswith("update"):
                query_types["update"] += 1
            elif sql_lower.startswith("delete"):
                query_types["delete"] += 1
            else:
                query_types["other"] += 1

        return {
            "batch_id": self.batch_id,
            "query_count": len(self.queries),
            "query_types": dict(query_types),
            "age": time.time() - self.created_at,
            "priority": self.priority,
        }


@dataclass
class QueryOptimizationRule:
    """Rule for optimizing database queries."""

    pattern: str  # Regex pattern to match
    optimization: str  # Description of optimization
    priority: int = 1  # Priority for application (1-10)
    applies_to: set[str] = field(default_factory=lambda: {"select", "insert", "update", "delete"})

    def matches(self, sql: str) -> bool:
        """Check if this rule matches the SQL query."""
        return bool(re.search(self.pattern, sql, re.IGNORECASE))

    def optimize(self, sql: str, params: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        """
        Apply optimization to the query.

        This is a base implementation. Subclasses should override for specific optimizations.
        """
        return sql, params


@dataclass
class IndexOptimizationRule(QueryOptimizationRule):
    """Rule for index usage optimization."""

    def optimize(self, sql: str, params: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        """Add index hints where beneficial."""
        # This is a simplified example. Real implementation would analyze query patterns
        # and suggest index usage based on table statistics
        return sql, params


@dataclass
class JoinOptimizationRule(QueryOptimizationRule):
    """Rule for JOIN optimization."""

    def optimize(self, sql: str, params: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        """Optimize JOIN operations."""
        # Analyze JOIN patterns and suggest improvements
        return sql, params


@dataclass
class BatchOptimizationResult:
    """Result of batch optimization."""

    original_queries: int
    optimized_queries: int
    batches_created: int
    estimated_savings_ms: float
    execution_plan: dict[str, Any]


class DatabaseOptimizer:
    """
    Intelligent database query optimizer with batching capabilities.

    Features:
    - Automatic query batching for similar operations
    - Query optimization rules engine
    - Performance monitoring and analytics
    - Connection multiplexing support
    - Intelligent query rewriting
    """

    def __init__(self):
        self.batches: dict[str, QueryBatch] = {}
        self.optimization_rules: list[QueryOptimizationRule] = []

        # Configuration
        self.max_batch_size = 50
        self.batch_timeout = 10.0  # 10ms window for batching
        self.max_concurrent_batches = 10

        # Statistics
        self.total_queries_processed = 0
        self.total_batches_created = 0
        self.total_optimization_savings = 0.0

        # Background tasks
        self._batch_processor_task: asyncio.Task | None = None
        self._running = False

        # Optimization rules
        self._setup_optimization_rules()

    async def start(self) -> None:
        """Start the database optimizer."""
        if self._running:
            return

        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("Database optimizer started")

    async def stop(self) -> None:
        """Stop the database optimizer."""
        if not self._running:
            return

        self._running = False
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_processor_task

        logger.info("Database optimizer stopped")

    def add_optimization_rule(self, rule: QueryOptimizationRule) -> None:
        """Add a new optimization rule."""
        self.optimization_rules.append(rule)
        # Sort by priority (higher priority first)
        self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)

    async def optimize_query(
        self, sql: str, params: tuple[Any, ...], enable_batching: bool = True
    ) -> tuple[str, tuple[Any, ...], str | None]:
        """
        Optimize a single query and optionally batch it.

        Returns:
            Tuple of (optimized_sql, optimized_params, batch_id)
            batch_id is None if not batched
        """
        self.total_queries_processed += 1

        # Apply optimization rules
        optimized_sql, optimized_params = self._apply_optimization_rules(sql, params)

        # Check if query can be batched
        if enable_batching:
            batch_id = await self._try_add_to_batch(optimized_sql, optimized_params)
            if batch_id:
                return optimized_sql, optimized_params, batch_id

        return optimized_sql, optimized_params, None

    async def optimize_batch(
        self, queries: list[tuple[str, tuple[Any, ...]]], batch_strategy: str = "smart"
    ) -> BatchOptimizationResult:
        """
        Optimize a batch of queries.

        Args:
            queries: List of (sql, params) tuples
            batch_strategy: Strategy for batching ("smart", "type", "none")

        Returns:
            Optimization result with execution plan
        """
        if batch_strategy == "none":
            # Apply individual optimizations only
            optimized_queries = []
            for sql, params in queries:
                opt_sql, opt_params = self._apply_optimization_rules(sql, params)
                optimized_queries.append((opt_sql, opt_params))

            return BatchOptimizationResult(
                original_queries=len(queries),
                optimized_queries=len(optimized_queries),
                batches_created=0,
                estimated_savings_ms=0.0,
                execution_plan={"strategy": "individual", "queries": optimized_queries},
            )

        # Group queries for batching
        if batch_strategy == "smart":
            batches = self._create_smart_batches(queries)
        elif batch_strategy == "type":
            batches = self._create_type_based_batches(queries)
        else:
            batches = [QueryBatch(queries)]

        # Estimate savings
        total_round_trips_original = len(queries)
        total_round_trips_optimized = len(batches)
        round_trip_savings = total_round_trips_original - total_round_trips_optimized

        # Estimate time savings (rough approximation)
        avg_round_trip_time = 5.0  # 5ms per round trip
        estimated_savings_ms = round_trip_savings * avg_round_trip_time

        self.total_optimization_savings += estimated_savings_ms
        self.total_batches_created += len(batches)

        return BatchOptimizationResult(
            original_queries=len(queries),
            optimized_queries=len(queries),  # All queries preserved
            batches_created=len(batches),
            estimated_savings_ms=estimated_savings_ms,
            execution_plan={
                "strategy": batch_strategy,
                "batches": [batch.get_execution_stats() for batch in batches],
                "total_round_trips_saved": round_trip_savings,
            },
        )

    async def execute_optimized_batch(self, batch: QueryBatch, executor: callable) -> list[Any]:
        """
        Execute an optimized batch using the provided executor.

        Args:
            batch: QueryBatch to execute
            executor: Async function that takes (sql, params) and returns result

        Returns:
            List of execution results
        """
        start_time = time.time()

        try:
            # For now, execute queries individually
            # In a real implementation, this could combine queries where possible
            tasks = []
            for sql, params in batch.queries:
                task = asyncio.create_task(executor(sql, params))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            execution_time = time.time() - start_time

            # Log batch execution
            logger.info(
                "batch_executed",
                batch_id=batch.batch_id,
                query_count=len(batch.queries),
                execution_time_ms=execution_time * 1000,
                estimated_savings=batch.get_execution_stats(),
            )

            return results

        except Exception as e:
            logger.error(
                "batch_execution_failed",
                batch_id=batch.batch_id,
                error=str(e),
                query_count=len(batch.queries),
            )
            raise

    def get_optimizer_stats(self) -> dict[str, Any]:
        """Get comprehensive optimizer statistics."""
        active_batches = sum(1 for batch in self.batches.values() if not batch.is_expired)

        return {
            "queries_processed": self.total_queries_processed,
            "batches_created": self.total_batches_created,
            "active_batches": active_batches,
            "optimization_savings_ms": self.total_optimization_savings,
            "rules_applied": len(self.optimization_rules),
            "avg_batch_size": self._calculate_avg_batch_size(),
            "batching_efficiency": self._calculate_batching_efficiency(),
        }

    async def _batch_processor(self) -> None:
        """Background processor for query batches."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Process every second

                # Clean expired batches
                expired_batches = []
                for batch_id, batch in self.batches.items():
                    if batch.is_expired:
                        expired_batches.append(batch_id)

                for batch_id in expired_batches:
                    del self.batches[batch_id]

                # Process ready batches (this would trigger execution in real implementation)
                ready_batches = [
                    batch
                    for batch in self.batches.values()
                    if batch.is_full or (len(batch.queries) > 0 and batch.is_expired)
                ]

                for batch in ready_batches:
                    # Mark for execution (in real implementation, this would queue for execution)
                    logger.debug(
                        "batch_ready_for_execution",
                        batch_id=batch.batch_id,
                        query_count=len(batch.queries),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    def _apply_optimization_rules(
        self, sql: str, params: tuple[Any, ...]
    ) -> tuple[str, tuple[Any, ...]]:
        """Apply all applicable optimization rules to a query."""
        optimized_sql = sql
        optimized_params = params

        for rule in self.optimization_rules:
            if rule.matches(optimized_sql):
                try:
                    optimized_sql, optimized_params = rule.optimize(optimized_sql, optimized_params)
                    logger.debug(
                        "optimization_rule_applied",
                        rule=rule.optimization,
                        sql_preview=optimized_sql[:50],
                    )
                except Exception as e:
                    logger.warning("optimization_rule_failed", rule=rule.optimization, error=str(e))

        return optimized_sql, optimized_params

    async def _try_add_to_batch(self, sql: str, params: tuple[Any, ...]) -> str | None:
        """Try to add query to an existing batch. Returns batch_id if successful."""
        # Simple batching strategy - group by query type
        query_type = self._classify_query_type(sql)

        # Look for existing batch of same type
        for batch_id, batch in self.batches.items():
            if (
                batch_id.startswith(f"{query_type}_")
                and not batch.is_full
                and not batch.is_expired
                and batch.add_query(sql, params)
            ):
                return batch_id

        # Create new batch if none available
        if len(self.batches) < self.max_concurrent_batches:
            batch = QueryBatch()
            batch.add_query(sql, params)
            self.batches[batch.batch_id] = batch
            return batch.batch_id

        return None

    def _classify_query_type(self, sql: str) -> str:
        """Classify query type for batching."""
        sql_lower = sql.lower().strip()

        if sql_lower.startswith("select"):
            return "select"
        if sql_lower.startswith("insert"):
            return "insert"
        if sql_lower.startswith("update"):
            return "update"
        if sql_lower.startswith("delete"):
            return "delete"
        return "other"

    def _create_smart_batches(self, queries: list[tuple[str, tuple[Any, ...]]]) -> list[QueryBatch]:
        """Create batches using smart grouping strategy."""
        # Group by query type and similar structure
        type_groups = defaultdict(list)

        for sql, params in queries:
            query_type = self._classify_query_type(sql)
            # Create a simplified fingerprint for grouping
            fingerprint = self._create_simple_fingerprint(sql)
            group_key = f"{query_type}_{fingerprint}"
            type_groups[group_key].append((sql, params))

        # Create batches for each group
        batches = []
        for group_queries in type_groups.values():
            if len(group_queries) <= self.max_batch_size:
                batch = QueryBatch(queries=group_queries)
                batches.append(batch)
            else:
                # Split large groups into multiple batches
                for i in range(0, len(group_queries), self.max_batch_size):
                    chunk = group_queries[i : i + self.max_batch_size]
                    batch = QueryBatch(queries=chunk)
                    batches.append(batch)

        return batches

    def _create_type_based_batches(
        self, queries: list[tuple[str, tuple[Any, ...]]]
    ) -> list[QueryBatch]:
        """Create batches grouped by query type only."""
        type_groups = defaultdict(list)

        for sql, params in queries:
            query_type = self._classify_query_type(sql)
            type_groups[query_type].append((sql, params))

        batches = []
        for group_queries in type_groups.values():
            if len(group_queries) <= self.max_batch_size:
                batch = QueryBatch(queries=group_queries)
                batches.append(batch)
            else:
                # Split large groups
                for i in range(0, len(group_queries), self.max_batch_size):
                    chunk = group_queries[i : i + self.max_batch_size]
                    batch = QueryBatch(queries=chunk)
                    batches.append(batch)

        return batches

    def _create_simple_fingerprint(self, sql: str) -> str:
        """Create a simple fingerprint for query grouping using BLAKE2b."""
        # Remove literals and create pattern
        pattern = re.sub(r"'[^']*'", "?", sql)  # Replace string literals
        pattern = re.sub(r"\d+", "N", pattern)  # Replace numbers
        # Use BLAKE2b instead of MD5 for better security
        return hashlib.blake2b(pattern.encode(), digest_size=4).hexdigest()

    def _calculate_avg_batch_size(self) -> float:
        """Calculate average batch size."""
        if not self.batches:
            return 0.0

        total_queries = sum(len(batch.queries) for batch in self.batches.values())
        return total_queries / len(self.batches)

    def _calculate_batching_efficiency(self) -> float:
        """Calculate batching efficiency ratio."""
        if self.total_queries_processed == 0:
            return 0.0

        # Efficiency = queries processed / (batches created + individual queries)
        individual_queries = self.total_queries_processed - (
            self.total_batches_created * self._calculate_avg_batch_size()
        )

        total_round_trips = self.total_batches_created + individual_queries
        if total_round_trips == 0:
            return 0.0

        return self.total_queries_processed / total_round_trips

    def _setup_optimization_rules(self) -> None:
        """Set up default optimization rules."""
        # Index optimization rule
        self.add_optimization_rule(
            IndexOptimizationRule(
                pattern=r"SELECT.*WHERE.*=.*",
                optimization="Consider adding index on WHERE column",
                priority=8,
            )
        )

        # JOIN optimization rule
        self.add_optimization_rule(
            JoinOptimizationRule(
                pattern=r"JOIN.*ON.*=.*",
                optimization="Review JOIN conditions for optimization",
                priority=7,
            )
        )

        # Subquery optimization
        self.add_optimization_rule(
            QueryOptimizationRule(
                pattern=r"SELECT.*IN\s*\(",
                optimization="Consider JOIN instead of IN subquery",
                priority=6,
                applies_to={"select"},
            )
        )

        # LIKE optimization
        self.add_optimization_rule(
            QueryOptimizationRule(
                pattern=r"LIKE\s*'%.*%'",
                optimization="Leading wildcard in LIKE prevents index usage",
                priority=9,
                applies_to={"select"},
            )
        )


# Global optimizer instance
database_optimizer = DatabaseOptimizer()


async def get_database_optimizer() -> DatabaseOptimizer:
    """Get the global database optimizer instance."""
    if not database_optimizer._running:
        await database_optimizer.start()
    return database_optimizer
