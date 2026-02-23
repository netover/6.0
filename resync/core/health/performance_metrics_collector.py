# pylint: disable=all
# mypy: no-rerun
"""
Performance Metrics Collector

This module provides utilities for collecting and managing performance metrics
across different system components for health monitoring purposes.
"""

import time
from datetime import datetime, timezone
from typing import Any

import psutil
import structlog

from resync.core.connection_pool_manager import get_advanced_connection_pool_manager

logger = structlog.get_logger(__name__)


class PerformanceMetricsCollector:
    """
    Collects and manages performance metrics for health monitoring.

    This class provides methods to collect system performance metrics,
    connection pool statistics, and other performance-related data
    needed for comprehensive health monitoring.
    """

    def __init__(self):
        """Initialize the performance metrics collector."""
        self._last_collection_time: datetime | None = None
        self._cached_metrics: dict[str, Any] | None = None

    def get_system_performance_metrics(self) -> dict[str, Any]:
        """
        Get current system performance metrics.

        Returns:
            Dictionary containing system performance metrics
        """
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Get process-specific metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "process_memory_mb": process_memory_mb,
                "timestamp": time.time(),
                "collection_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("failed_to_get_system_performance_metrics", error=str(e))
            return {
                "error": str(e),
                "timestamp": time.time(),
                "collection_time": datetime.now(timezone.utc).isoformat(),
            }

    async def get_connection_pool_metrics(self) -> dict[str, Any]:
        """
        Get connection pool performance metrics.

        Returns:
            Dictionary containing connection pool metrics
        """
        try:
            pool_manager = get_advanced_connection_pool_manager()
            if pool_manager:
                return await pool_manager.get_performance_metrics()
            return {"error": "Advanced connection pool manager not available"}

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("failed_to_get_connection_pool_metrics", error=str(e))
            return {"error": str(e)}

    async def get_comprehensive_performance_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive performance metrics including system and connection pools.

        Returns:
            Dictionary containing all performance metrics
        """
        try:
            # Collect system metrics
            system_metrics = self.get_system_performance_metrics()

            # Collect connection pool metrics
            pool_metrics = await self.get_connection_pool_metrics()

            # Combine metrics
            comprehensive_metrics = {
                "system": system_metrics,
                "connection_pools": pool_metrics,
                "collection_metadata": {
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "metrics_version": "1.0",
                    "components_included": ["system", "connection_pools"],
                },
            }

            # Cache the results
            self._last_collection_time = datetime.now(timezone.utc)
            self._cached_metrics = comprehensive_metrics

            return comprehensive_metrics

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "failed_to_get_comprehensive_performance_metrics", error=str(e)
            )
            return {
                "error": str(e),
                "timestamp": time.time(),
                "collection_time": datetime.now(timezone.utc).isoformat(),
            }

    def get_cached_metrics(self) -> dict[str, Any] | None:
        """
        Get cached performance metrics if available and recent.

        Returns:
            Cached metrics or None if cache is stale or unavailable
        """
        if self._cached_metrics is None:
            return None

        # Consider cache stale after 30 seconds
        if self._last_collection_time:
            age_seconds = (
                datetime.now(timezone.utc) - self._last_collection_time
            ).total_seconds()
            if age_seconds > 30:
                return None

        return self._cached_metrics

    def clear_cache(self) -> None:
        """Clear cached performance metrics."""
        self._cached_metrics = None
        self._last_collection_time = None

    async def get_performance_summary(self) -> dict[str, Any]:
        """
        Get a summary of current performance status.

        Returns:
            Dictionary with performance summary and recommendations
        """
        try:
            metrics = await self.get_comprehensive_performance_metrics()

            if "error" in metrics:
                return {
                    "status": "error",
                    "message": "Failed to collect performance metrics",
                    "error": metrics["error"],
                }

            summary = {
                "status": "healthy",
                "timestamp": metrics["collection_metadata"]["collected_at"],
                "recommendations": [],
                "warnings": [],
            }

            # Analyze system metrics
            system_metrics = metrics.get("system", {})

            if "error" not in system_metrics:
                cpu_percent = system_metrics.get("cpu_percent", 0)
                memory_percent = system_metrics.get("memory_percent", 0)

                if cpu_percent > 90:
                    summary["warnings"].append(f"High CPU usage: {cpu_percent}%")
                    summary["status"] = "degraded"
                elif cpu_percent > 75:
                    summary["warnings"].append(f"Elevated CPU usage: {cpu_percent}%")

                if memory_percent > 90:
                    summary["warnings"].append(f"High memory usage: {memory_percent}%")
                    summary["status"] = "degraded"
                elif memory_percent > 80:
                    summary["warnings"].append(
                        f"Elevated memory usage: {memory_percent}%"
                    )

            # Analyze connection pool metrics
            pool_metrics = metrics.get("connection_pools", {})

            if "error" not in pool_metrics and "auto_scaling" in pool_metrics:
                # Check for pool-specific issues
                auto_scaling = pool_metrics["auto_scaling"]
                utilization = auto_scaling.get("load_score", 0)

                if utilization > 0.9:
                    summary["warnings"].append(
                        "High connection pool utilization: {utilization:.1%}"
                    )
                    summary["status"] = "degraded"
                elif utilization > 0.8:
                    summary["warnings"].append(
                        "Elevated connection pool utilization: {utilization:.1%}"
                    )

            # Generate recommendations
            if summary["status"] == "degraded":
                summary["recommendations"].append(
                    "Monitor system resources and consider scaling"
                )
            elif summary["warnings"]:
                summary["recommendations"].append(
                    "Monitor identified warning conditions"
                )

            return summary

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_get_performance_summary", error=str(e))
            return {
                "status": "error",
                "message": f"Failed to generate performance summary: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
