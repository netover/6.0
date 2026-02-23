# pylint: disable=all
# mypy: no-rerun
"""
Health History Manager

This module provides comprehensive health history management functionality
including storage, cleanup, and retrieval of health check history data.
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from resync.core.health.health_models import (
    ComponentHealth,
    HealthCheckResult,
    HealthStatus,
    HealthStatusHistory,
)

logger = structlog.get_logger(__name__)


class HealthHistoryManager:
    """
    Manages health check history with efficient storage and cleanup.

    This class provides functionality for:
    - Storing health check results over time
    - Automatic cleanup based on size and age limits
    - Memory usage tracking and optimization
    - Retrieval of historical health data
    """

    def __init__(
        self,
        max_history_entries: int = 10000,
        history_retention_days: int = 30,
        memory_usage_threshold_mb: float = 100.0,
        cleanup_batch_size: int = 100,
        history_cleanup_threshold: float = 0.8,
    ):
        """
        Initialize the health history manager.

        Args:
            max_history_entries: Maximum number of history entries to keep
            history_retention_days: Number of days to retain history
            memory_usage_threshold_mb: Memory usage threshold in MB before cleanup
            cleanup_batch_size: Number of entries to remove in each cleanup batch
            history_cleanup_threshold: Threshold percentage for triggering cleanup
        """
        self.max_history_entries = max_history_entries
        self.history_retention_days = history_retention_days
        self.memory_usage_threshold_mb = memory_usage_threshold_mb
        self.cleanup_batch_size = cleanup_batch_size
        self.history_cleanup_threshold = history_cleanup_threshold

        self.health_history: list[HealthStatusHistory] = []
        self._memory_usage_mb: float = 0.0
        self._cleanup_lock = asyncio.Lock()

    async def add_health_result(self, result: HealthCheckResult) -> None:
        """
        Add a health check result to history.

        Args:
            result: Health check result to add
        """
        # Create history entry
        component_changes = self._get_component_changes(result.components)
        history_entry = HealthStatusHistory(
            timestamp=result.timestamp,
            overall_status=result.overall_status,
            component_changes=component_changes,
        )

        # Add to history
        self.health_history.append(history_entry)

        # Perform cleanup if needed
        await create_tracked_task(
            self._cleanup_health_history(), name="cleanup_health_history"
        )

        # Update memory usage tracking
        if self.memory_usage_threshold_mb > 0:
            await create_tracked_task(
                self._update_memory_usage(), name="update_memory_usage"
            )

    def _get_component_changes(
        self, components: dict[str, ComponentHealth]
    ) -> dict[str, HealthStatus]:
        """
        Track component status changes for history.

        Args:
            components: Current component health results

        Returns:
            Dictionary of component status changes
        """
        changes = {}

        # For now, record all current statuses as changes
        # In a more sophisticated implementation, this would compare with previous states
        for name, component in components.items():
            changes[name] = component.status

        return changes

    async def _cleanup_health_history(self) -> None:
        """
        Perform efficient cleanup of health history based on multiple criteria.
        """
        async with self._cleanup_lock:
            try:
                current_size = len(self.health_history)
                int(self.max_history_entries * self.history_cleanup_threshold)

                # Check if cleanup is needed based on size
                if current_size > self.max_history_entries:
                    entries_to_remove = (
                        current_size
                        - self.max_history_entries
                        + self.cleanup_batch_size
                    )
                    self.health_history = self.health_history[entries_to_remove:]
                    logger.debug(
                        "cleaned_up_health_history_entries_size_based",
                        entries_to_remove=entries_to_remove,
                        current_size=len(self.health_history),
                    )

                # Check if cleanup is needed based on age
                cutoff_date = datetime.now(timezone.utc) - timedelta(
                    days=self.history_retention_days
                )
                original_size = len(self.health_history)
                self.health_history = [
                    entry
                    for entry in self.health_history
                    if entry.timestamp >= cutoff_date
                ]
                removed_by_age = original_size - len(self.health_history)
                if removed_by_age > 0:
                    logger.debug(
                        "cleaned_up_health_history_entries_age_based",
                        removed_by_age=removed_by_age,
                        current_size=len(self.health_history),
                    )

                # Ensure we don't go below minimum required entries
                min_entries = max(10, self.cleanup_batch_size)
                if (
                    len(self.health_history) < min_entries
                    and original_size >= min_entries
                ):
                    # Keep at least some recent history
                    self.health_history = self.health_history[-min_entries:]

            except Exception as e:
                logger.error("error_during_health_history_cleanup", error=str(e))

    async def _update_memory_usage(self) -> None:
        """Update memory usage tracking for health history."""
        try:
            # Estimate memory usage of health history
            history_size = len(self.health_history)
            if history_size > 0:
                # Estimate: each entry ~2KB (accounting for metadata and component changes)
                estimated_size_bytes = history_size * 2048
                self._memory_usage_mb = estimated_size_bytes / (1024 * 1024)

                # Alert if memory usage exceeds threshold
                if self._memory_usage_mb > self.memory_usage_threshold_mb:
                    logger.warning(
                        "health_history_memory_usage_exceeds_threshold",
                        current_usage_mb=round(self._memory_usage_mb, 2),
                        threshold_mb=self.memory_usage_threshold_mb,
                    )
            else:
                self._memory_usage_mb = 0.0

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("error_updating_memory_usage", error=str(e))

    def get_memory_usage(self) -> dict[str, Any]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        return {
            "history_entries": len(self.health_history),
            "memory_usage_mb": round(self._memory_usage_mb, 2),
            "max_entries": self.max_history_entries,
            "retention_days": self.history_retention_days,
            "cleanup_threshold_percent": self.history_cleanup_threshold * 100,
            "memory_threshold_mb": self.memory_usage_threshold_mb,
            "cleanup_batch_size": self.cleanup_batch_size,
        }

    async def force_cleanup(self) -> dict[str, Any]:
        """
        Force immediate cleanup of health history.

        Returns:
            Dictionary with cleanup results
        """
        original_size = len(self.health_history)
        await self._cleanup_health_history()
        new_size = len(self.health_history)

        return {
            "original_entries": original_size,
            "cleaned_entries": original_size - new_size,
            "current_entries": new_size,
            "memory_usage_mb": round(self._memory_usage_mb, 2),
        }

    def get_health_history(
        self,
        hours: int = 24,
        max_entries: int | None = None,
        component_filter: str | None = None,
    ) -> list[HealthStatusHistory]:
        """
        Get health history for the specified time period.

        Args:
            hours: Number of hours to look back
            max_entries: Maximum number of entries to return
            component_filter: Optional component name to filter by

        Returns:
            List of health status history entries
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter by time
        filtered_history = [
            entry for entry in self.health_history if entry.timestamp >= cutoff_time
        ]

        # Filter by component if specified
        if component_filter:
            filtered_history = [
                entry
                for entry in filtered_history
                if component_filter in entry.component_changes
            ]

        # Apply entry limit if specified
        if max_entries is not None and len(filtered_history) > max_entries:
            # Return most recent entries
            filtered_history = filtered_history[-max_entries:]

        return filtered_history

    def get_component_status_history(
        self, component_name: str, hours: int = 24, max_entries: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get status history for a specific component.

        Args:
            component_name: Name of the component
            hours: Number of hours to look back
            max_entries: Maximum number of entries to return

        Returns:
            List of status changes for the component
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter entries that contain the component
        component_history = []
        for entry in self.health_history:
            if (
                entry.timestamp >= cutoff_time
                and component_name in entry.component_changes
            ):
                component_history.append(
                    {
                        "timestamp": entry.timestamp,
                        "status": entry.component_changes[component_name],
                        "overall_status": entry.overall_status,
                    }
                )

        # Apply entry limit if specified
        if max_entries is not None and len(component_history) > max_entries:
            component_history = component_history[-max_entries:]

        return component_history

    def get_history_stats(self) -> dict[str, Any]:
        """
        Get statistics about the health history.

        Returns:
            Dictionary with history statistics
        """
        if not self.health_history:
            return {
                "total_entries": 0,
                "oldest_entry": None,
                "newest_entry": None,
                "memory_usage_mb": 0.0,
            }

        timestamps = [entry.timestamp for entry in self.health_history]

        return {
            "total_entries": len(self.health_history),
            "oldest_entry": min(timestamps).isoformat(),
            "newest_entry": max(timestamps).isoformat(),
            "memory_usage_mb": round(self._memory_usage_mb, 2),
            "max_entries": self.max_history_entries,
            "retention_days": self.history_retention_days,
        }

    def clear_history(self) -> None:
        """Clear all health history."""
        self.health_history.clear()
        self._memory_usage_mb = 0.0
        logger.info("health_history_cleared")
