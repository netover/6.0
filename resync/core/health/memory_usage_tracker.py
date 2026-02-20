"""
Memory Usage Tracker

This module provides comprehensive memory usage tracking and monitoring
functionality for health checks and system monitoring.
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import gc
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil
import structlog

logger = structlog.get_logger(__name__)


class MemoryUsageTracker:
    """
    Tracks and monitors memory usage across the application.

    This class provides functionality for:
    - Real-time memory usage monitoring
    - Memory usage history and trends
    - Memory leak detection
    - Memory usage alerts and warnings
    - Garbage collection monitoring
    """

    def __init__(
        self,
        history_size: int = 1000,
        alert_threshold_mb: float = 500.0,
        warning_threshold_mb: float = 300.0,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize the memory usage tracker.

        Args:
            history_size: Maximum number of memory readings to keep in history
            alert_threshold_mb: Memory usage threshold for alerts (MB)
            warning_threshold_mb: Memory usage threshold for warnings (MB)
            check_interval_seconds: Interval for automatic memory checks
        """
        self.history_size = history_size
        self.alert_threshold_mb = alert_threshold_mb
        self.warning_threshold_mb = warning_threshold_mb
        self.check_interval_seconds = check_interval_seconds

        self.memory_history: list[dict[str, Any]] = []
        self._monitoring_active = False
        self._monitoring_task: asyncio.Task | None = None
        self._last_gc_collection = 0

    def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_task = track_task(
            self._monitoring_loop(), name="monitoring_loop"
        )
        logger.info("memory_usage_monitoring_started")

    async def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None
        logger.info("memory_usage_monitoring_stopped")

    async def _monitoring_loop(self) -> None:
        """Continuous memory monitoring loop."""
        while self._monitoring_active:
            try:
                await self.record_memory_usage()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("error_in_memory_monitoring_loop", error=str(e))
                await asyncio.sleep(10)  # Brief pause on error

    async def record_memory_usage(self) -> dict[str, Any]:
        """
        Record current memory usage statistics.

        Returns:
            Dictionary with current memory usage data
        """
        try:
            # Delegate to the synchronous version to avoid calling asyncio.run() from sync contexts.
            memory_data = self.record_memory_usage_sync()
            # Check for memory issues asynchronously
            self._check_memory_alerts(memory_data)
            return memory_data

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_record_memory_usage", error=str(e))
            return {"timestamp": datetime.now(timezone.utc), "error": str(e)}

    def record_memory_usage_sync(self) -> dict[str, Any]:
        """
        Record current memory usage statistics synchronously.

        This method mirrors ``record_memory_usage`` but is safe to call in a
        synchronous context. It does not perform any asynchronous operations.
        """
        try:
            # Get system memory information
            memory = psutil.virtual_memory()

            # Get process memory information
            process = psutil.Process()
            process_memory = process.memory_info()

            # Calculate memory usage
            memory_data = {
                "timestamp": datetime.now(timezone.utc),
                "system_memory_percent": memory.percent,
                "system_memory_used_gb": memory.used / (1024**3),
                "system_memory_available_gb": memory.available / (1024**3),
                "system_memory_total_gb": memory.total / (1024**3),
                "process_memory_rss_mb": process_memory.rss / (1024**2),
                "process_memory_vms_mb": process_memory.vms / (1024**2),
                "process_memory_percent": process.memory_percent(),
            }

            # Add garbage collection information
            gc_stats = gc.get_stats()
            memory_data["gc_collections"] = {
                "generation_0": gc_stats[0]["collections"],
                "generation_1": gc_stats[1]["collections"],
                "generation_2": gc_stats[2]["collections"],
            }

            # Add to history
            self.memory_history.append(memory_data)

            # Maintain history size
            if len(self.memory_history) > self.history_size:
                self.memory_history = self.memory_history[-self.history_size :]

            return memory_data

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_record_memory_usage_sync", error=str(e))
            return {"timestamp": datetime.now(timezone.utc), "error": str(e)}

    def _check_memory_alerts(self, memory_data: dict[str, Any]) -> None:
        """Check for memory usage alerts and warnings."""
        try:
            process_memory_mb = memory_data.get("process_memory_rss_mb", 0)
            system_memory_percent = memory_data.get("system_memory_percent", 0)

            # Check process memory thresholds
            if process_memory_mb > self.alert_threshold_mb:
                logger.warning(
                    "process_memory_usage_alert",
                    memory_mb=process_memory_mb,
                    threshold_mb=self.alert_threshold_mb,
                    memory_percent=system_memory_percent,
                )
            elif process_memory_mb > self.warning_threshold_mb:
                logger.info(
                    "process_memory_usage_warning",
                    memory_mb=process_memory_mb,
                    threshold_mb=self.warning_threshold_mb,
                    memory_percent=system_memory_percent,
                )

            # Check system memory thresholds
            if system_memory_percent > 95:
                logger.warning(
                    "system_memory_critical", memory_percent=system_memory_percent
                )
            elif system_memory_percent > 85:
                logger.info("system_memory_high", memory_percent=system_memory_percent)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_check_memory_alerts", error=str(e))

    def get_current_memory_usage(self) -> dict[str, Any]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with current memory usage information
        """
        if not self.memory_history:
            # Return current memory usage if no history
            return self.record_memory_usage_sync()

        return self.memory_history[-1].copy()

    def get_memory_history(
        self, hours: int = 24, max_entries: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get memory usage history for the specified time period.

        Args:
            hours: Number of hours to look back
            max_entries: Maximum number of entries to return

        Returns:
            List of memory usage records
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter by time
        filtered_history = [
            entry for entry in self.memory_history if entry["timestamp"] >= cutoff_time
        ]

        # Apply entry limit if specified
        if max_entries is not None and len(filtered_history) > max_entries:
            filtered_history = filtered_history[-max_entries:]

        return filtered_history

    def get_memory_trends(self) -> dict[str, Any]:
        """
        Analyze memory usage trends.

        Returns:
            Dictionary with memory trend analysis
        """
        if len(self.memory_history) < 2:
            return {
                "trend": "insufficient_data",
                "message": "Not enough data for trend analysis",
            }

        # Get recent data for trend analysis
        recent_count = min(100, len(self.memory_history))
        recent_data = self.memory_history[-recent_count:]

        # Calculate trends
        process_memory_values = [
            entry.get("process_memory_rss_mb", 0) for entry in recent_data
        ]
        system_memory_values = [
            entry.get("system_memory_percent", 0) for entry in recent_data
        ]

        if len(process_memory_values) < 2:
            return {
                "trend": "insufficient_data",
                "message": "Not enough process memory data for trend analysis",
            }

        # Simple linear trend calculation
        first_process = process_memory_values[0]
        last_process = process_memory_values[-1]
        process_trend = "stable"

        if last_process > first_process * 1.1:  # 10% increase
            process_trend = "increasing"
        elif last_process < first_process * 0.9:  # 10% decrease
            process_trend = "decreasing"

        # System memory trend
        first_system = system_memory_values[0]
        last_system = system_memory_values[-1]
        system_trend = "stable"

        if last_system > first_system * 1.05:  # 5% increase
            system_trend = "increasing"
        elif last_system < first_system * 0.95:  # 5% decrease
            system_trend = "decreasing"

        return {
            "process_memory_trend": process_trend,
            "system_memory_trend": system_trend,
            "current_process_mb": last_process,
            "current_system_percent": last_system,
            "analysis_period_hours": recent_count * self.check_interval_seconds / 3600,
            "data_points": len(recent_data),
        }

    def detect_memory_leaks(self) -> dict[str, Any]:
        """
        Detect potential memory leaks.

        Returns:
            Dictionary with memory leak analysis
        """
        if len(self.memory_history) < 10:
            return {
                "leak_detected": False,
                "confidence": "low",
                "message": "Insufficient data for leak detection",
            }

        # Analyze memory growth over time
        recent_count = min(50, len(self.memory_history))
        recent_data = self.memory_history[-recent_count:]

        process_memory_values = [
            entry.get("process_memory_rss_mb", 0) for entry in recent_data
        ]

        if len(process_memory_values) < 10:
            return {
                "leak_detected": False,
                "confidence": "low",
                "message": "Not enough data points for leak detection",
            }

        # Calculate memory growth rate
        time_deltas = []
        memory_deltas = []

        for i in range(1, len(recent_data)):
            current = recent_data[i]
            previous = recent_data[i - 1]

            time_diff = (current["timestamp"] - previous["timestamp"]).total_seconds()
            memory_diff = process_memory_values[i] - process_memory_values[i - 1]

            if time_diff > 0:
                time_deltas.append(time_diff)
                memory_deltas.append(memory_diff)

        if not memory_deltas:
            return {
                "leak_detected": False,
                "confidence": "low",
                "message": "Unable to calculate memory deltas",
            }

        # Calculate average memory growth per second
        avg_growth_per_second = sum(memory_deltas) / sum(time_deltas)

        # Detect potential leaks (steady growth > 1MB per minute)
        leak_threshold = 1.0 / 60  # 1MB per minute

        leak_detected = avg_growth_per_second > leak_threshold

        return {
            "leak_detected": leak_detected,
            "confidence": "medium",
            "avg_growth_mb_per_second": round(avg_growth_per_second, 4),
            "threshold_mb_per_second": leak_threshold,
            "analysis_period_minutes": sum(time_deltas) / 60,
            "data_points": len(memory_deltas),
        }

    def force_garbage_collection(self) -> dict[str, Any]:
        """
        Force garbage collection and return statistics.

        Returns:
            Dictionary with garbage collection results
        """
        try:
            # Record memory before GC
            before_memory = self.get_current_memory_usage()

            # Force garbage collection
            gc.collect()

            # Record memory after GC
            after_memory = self.get_current_memory_usage()

            # Calculate memory freed
            memory_freed_mb = before_memory.get(
                "process_memory_rss_mb", 0
            ) - after_memory.get("process_memory_rss_mb", 0)

            result = {
                "gc_performed": True,
                "memory_before_mb": before_memory.get("process_memory_rss_mb", 0),
                "memory_after_mb": after_memory.get("process_memory_rss_mb", 0),
                "memory_freed_mb": memory_freed_mb,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "forced_garbage_collection", memory_freed_mb=round(memory_freed_mb, 2)
            )

            return result

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_force_garbage_collection", error=str(e))
            return {
                "gc_performed": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_memory_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive memory usage summary.

        Returns:
            Dictionary with memory summary and recommendations
        """
        try:
            current_usage = self.get_current_memory_usage()
            trends = self.get_memory_trends()
            leak_analysis = self.detect_memory_leaks()

            summary = {
                "current_usage": current_usage,
                "trends": trends,
                "leak_analysis": leak_analysis,
                "history_stats": {
                    "total_records": len(self.memory_history),
                    "history_size_limit": self.history_size,
                    "monitoring_active": self._monitoring_active,
                },
                "thresholds": {
                    "alert_mb": self.alert_threshold_mb,
                    "warning_mb": self.warning_threshold_mb,
                },
                "recommendations": [],
            }

            # Generate recommendations
            if leak_analysis.get("leak_detected", False):
                summary["recommendations"].append(
                    "Potential memory leak detected - investigate application"
                )

            process_mb = current_usage.get("process_memory_rss_mb", 0)
            if process_mb > self.alert_threshold_mb:
                summary["recommendations"].append(
                    "Memory usage is critically high - consider scaling or optimization"
                )
            elif process_mb > self.warning_threshold_mb:
                summary["recommendations"].append(
                    "Memory usage is elevated - monitor closely"
                )

            if trends.get("process_memory_trend") == "increasing":
                summary["recommendations"].append(
                    "Memory usage is trending upward - investigate cause"
                )

            return summary

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_get_memory_summary", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
