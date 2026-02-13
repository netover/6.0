import time
from dataclasses import dataclass
from datetime import datetime

from resync.core.health.health_models import ComponentHealth, ComponentType, HealthStatus
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations."""

    # Query performance
    avg_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0
    queries_per_second: float = 0.0

    # Connection metrics
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    connection_utilization_percent: float = 0.0

    # Cache metrics
    query_cache_hit_rate: float = 0.0
    prepared_statement_cache_hit_rate: float = 0.0

    # Error metrics
    error_rate: float = 0.0
    total_errors: int = 0
    total_queries: int = 0

    # Throughput metrics
    rows_read_per_second: float = 0.0
    rows_written_per_second: float = 0.0

    # Timestamp
    timestamp: float = 0.0

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ReplicationStatus:
    """Replication status information for database clusters."""

    # Replication state
    is_enabled: bool = False
    is_running: bool = False
    replication_lag_seconds: float = 0.0

    # Replication nodes
    master_node: str | None = None
    replica_nodes: list[str] = None

    # Replication metrics
    replication_throughput_bytes_per_sec: float = 0.0
    replication_delay_ms: float = 0.0

    # Health indicators
    replication_health: HealthStatus = HealthStatus.UNKNOWN
    last_sync_timestamp: datetime | None = None

    # Error tracking
    replication_errors: int = 0
    last_error_message: str | None = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.replica_nodes is None:
            self.replica_nodes = []


class DatabaseHealthMonitor:
    """
    Monitor for database health, performance, and replication status.

    This class provides comprehensive database monitoring capabilities including:
    - Connection health checks
    - Query performance analysis
    - Replication status monitoring
    - Performance metrics collection
    """

    def __init__(self, connection_pool_manager=None):
        """
        Initialize the database health monitor.

        Args:
            connection_pool_manager: Optional connection pool manager instance
        """
        self.connection_pool_manager = connection_pool_manager
        self._monitoring_active = False
        self._metrics_history: list[PerformanceMetrics] = []
        self._max_history_size = 100

    async def check_connection_health(self) -> ComponentHealth:
        """
        Check the health of database connections.

        Returns:
            ComponentHealth: Health status of database connections
        """
        start_time = time.time()

        try:
            # Get connection pool manager if not provided
            if self.connection_pool_manager is None:
                from resync.core.connection_pool_manager import (
                    get_connection_pool_manager,
                )

                self.connection_pool_manager = await get_connection_pool_manager()

            if not self.connection_pool_manager:
                return ComponentHealth(
                    name="database_connections",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNKNOWN,
                    message="Connection pool manager not available",
                    last_check=datetime.now(),
                )

            # Test database connectivity
            pool_stats = self.connection_pool_manager.get_pool_stats()

            if not pool_stats:
                return ComponentHealth(
                    name="database_connections",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="No connection pool statistics available",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(),
                )

            # Get database pool specific stats
            db_pool_stats = pool_stats.get("database")
            if db_pool_stats is None:
                return ComponentHealth(
                    name="database_connections",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="Database pool not found in statistics",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(),
                )

            # Calculate connection utilization
            total_connections = getattr(db_pool_stats, "total_connections", 0)
            active_connections = getattr(db_pool_stats, "active_connections", 0)

            if total_connections == 0:
                return ComponentHealth(
                    name="database_connections",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="No database connections configured",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_check=datetime.now(),
                )

            connection_utilization = (active_connections / total_connections) * 100

            # Determine health status based on utilization
            if connection_utilization > 95:
                status = HealthStatus.UNHEALTHY
                message = (
                    "Database connection pool critically utilized: {connection_utilization:.1f}%"
                )
            elif connection_utilization > 85:
                status = HealthStatus.DEGRADED
                message = "Database connection pool highly utilized: {connection_utilization:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = (
                    "Database connection pool healthy: {connection_utilization:.1f}% utilized"
                )

            response_time_ms = (time.time() - start_time) * 1000

            # Create metadata with connection details
            metadata = {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "idle_connections": getattr(db_pool_stats, "idle_connections", 0),
                "connection_utilization_percent": round(connection_utilization, 1),
                "connection_errors": getattr(db_pool_stats, "connection_errors", 0),
                "pool_hits": getattr(db_pool_stats, "pool_hits", 0),
                "pool_misses": getattr(db_pool_stats, "pool_misses", 0),
                "waiting_connections": getattr(db_pool_stats, "waiting_connections", 0),
            }

            return ComponentHealth(
                name="database_connections",
                component_type=ComponentType.DATABASE,
                status=status,
                message=message,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                metadata=metadata,
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error("database_connection_health_check_failed", error=str(e))

            return ComponentHealth(
                name="database_connections",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection check failed: {str(e)}",
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                error_count=1,
            )

    async def check_query_performance(self) -> PerformanceMetrics:
        """
        Check database query performance metrics.

        Returns:
            PerformanceMetrics: Current database performance metrics
        """
        try:
            # Get connection pool manager if not provided
            if self.connection_pool_manager is None:
                from resync.core.connection_pool_manager import (
                    get_connection_pool_manager,
                )

                self.connection_pool_manager = await get_connection_pool_manager()

            metrics = PerformanceMetrics()

            if not self.connection_pool_manager:
                logger.warning("No connection pool manager available for performance metrics")
                return metrics

            # Get pool statistics for performance analysis
            pool_stats = self.connection_pool_manager.get_pool_stats()

            if pool_stats:
                db_pool_stats = pool_stats.get("database")
                if db_pool_stats:
                    # Extract performance metrics from pool stats
                    total_connections = getattr(db_pool_stats, "total_connections", 0)
                    active_connections = getattr(db_pool_stats, "active_connections", 0)

                    metrics.total_connections = total_connections
                    metrics.active_connections = active_connections
                    metrics.idle_connections = getattr(db_pool_stats, "idle_connections", 0)
                    metrics.connection_utilization_percent = (
                        active_connections / max(1, total_connections)
                    ) * 100

                    # Calculate error rate if we have error data
                    total_errors = getattr(db_pool_stats, "connection_errors", 0)
                    total_operations = getattr(db_pool_stats, "pool_hits", 0) + getattr(
                        db_pool_stats, "pool_misses", 0
                    )

                    if total_operations > 0:
                        metrics.error_rate = total_errors / total_operations
                        metrics.total_errors = total_errors
                        metrics.total_queries = total_operations

            # Add to history for trend analysis
            self._add_metrics_to_history(metrics)

            return metrics

        except Exception as e:
            logger.error("database_performance_check_failed", error=str(e))
            return PerformanceMetrics()

    async def check_replication_status(self) -> ReplicationStatus:
        """
        Check database replication status.

        Returns:
            ReplicationStatus: Current replication status
        """
        try:
            # For now, return a basic replication status
            # In a real implementation, this would query the database for replication info
            replication_status = ReplicationStatus(
                is_enabled=False,  # Default to False until replication is configured
                is_running=False,
                replication_health=HealthStatus.UNKNOWN,
                replication_errors=0,
            )

            # Try to get replication information from database
            # This is a placeholder - real implementation would query actual replication status
            try:
                if self.connection_pool_manager:
                    pool_stats = self.connection_pool_manager.get_pool_stats()
                    if pool_stats:
                        # Check if we have multiple database nodes indicating replication
                        db_pools = [k for k in pool_stats if "database" in k.lower()]
                        if len(db_pools) > 1:
                            replication_status.is_enabled = True
                            replication_status.replica_nodes = db_pools
                            replication_status.replication_health = HealthStatus.HEALTHY

            except Exception as e:
                logger.warning("replication_status_check_error", error=str(e))
                replication_status.last_error_message = str(e)
                replication_status.replication_errors += 1

            return replication_status

        except Exception as e:
            logger.error("replication_status_check_failed", error=str(e))
            return ReplicationStatus(
                is_enabled=False,
                is_running=False,
                replication_health=HealthStatus.UNHEALTHY,
                replication_errors=1,
                last_error_message=str(e),
            )

    def _add_metrics_to_history(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics to history for trend analysis."""
        self._metrics_history.append(metrics)

        # Maintain history size limit
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history = self._metrics_history[-self._max_history_size :]

    def get_metrics_history(self, limit: int | None = None) -> list[PerformanceMetrics]:
        """
        Get historical performance metrics.

        Args:
            limit: Maximum number of records to return (None for all)

        Returns:
            List of performance metrics in chronological order
        """
        if limit is None:
            return self._metrics_history.copy()
        return self._metrics_history[-limit:].copy()

    def clear_metrics_history(self) -> None:
        """Clear the metrics history."""
        self._metrics_history.clear()

    async def start_monitoring(self) -> None:
        """Start continuous monitoring (placeholder for future enhancement)."""
        self._monitoring_active = True
        logger.info("database_health_monitoring_started")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring_active = False
        logger.info("database_health_monitoring_stopped")
