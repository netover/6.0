# ruff: noqa: E501
"""
Health Alerting and Reporting Components

This module provides comprehensive alerting and reporting functionality for
health monitoring, including alert generation, summary creation, and health
status reporting.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from resync.core.health.health_models import ComponentHealth, HealthStatus

logger = structlog.get_logger(__name__)

class HealthAlerting:
    """
    Manages health alerts and notifications.

    This class provides functionality for:
    - Generating alerts based on component health status
    - Managing alert thresholds and conditions
    - Tracking alert history and patterns
    """

    def __init__(self, alert_enabled: bool = True):
        """
        Initialize the health alerting system.

        Args:
            alert_enabled: Whether alerting is enabled
        """
        self.alert_enabled = alert_enabled
        self.alert_history: list[dict[str, Any]] = []
        self.max_alert_history = 1000

    def check_alerts(self, components: dict[str, ComponentHealth]) -> list[str]:
        """
        Check for alerts based on component health status.

        Args:
            components: Dictionary of component health results

        Returns:
            List of alert messages
        """
        if not self.alert_enabled:
            return []

        alerts: list[str] = []

        for name, comp in components.items():
            if comp.status == HealthStatus.UNHEALTHY:
                alerts.append(f"{name} is unhealthy")
            elif comp.status == HealthStatus.DEGRADED:
                # Include specific threshold breach information in alerts
                if name == "database" and "connection_usage_percent" in comp.metadata:
                    comp.metadata.get("threshold_percent", 80)
                    comp.metadata["connection_usage_percent"]
                    alerts.append(
                        "Database connection pool usage at {usage:.1f}% (threshold: {threshold}%)"
                    )
                else:
                    alerts.append(f"{name} is degraded")

        # Store alerts in history
        if alerts:
            self._add_to_alert_history(alerts, components)

        return alerts

    def _add_to_alert_history(
        self, alerts: list[str], components: dict[str, ComponentHealth]
    ) -> None:
        """Add alerts to history for tracking and analysis."""
        alert_entry = {
            "timestamp": datetime.now(timezone.utc),
            "alerts": alerts.copy(),
            "component_count": len(components),
            "unhealthy_count": sum(
                1 for c in components.values() if c.status == HealthStatus.UNHEALTHY
            ),
            "degraded_count": sum(
                1 for c in components.values() if c.status == HealthStatus.DEGRADED
            ),
        }

        self.alert_history.append(alert_entry)

        # Cleanup old entries if needed
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history :]

    def get_alert_history(
        self, hours: int = 24, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get alert history for the specified time period.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of entries to return

        Returns:
            List of alert history entries
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        filtered_history = [
            entry for entry in self.alert_history if entry["timestamp"] >= cutoff_time
        ]

        # Sort by timestamp (most recent first)
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply limit if specified
        if limit:
            filtered_history = filtered_history[:limit]

        return filtered_history

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics and patterns."""
        if not self.alert_history:
            return {"total_alerts": 0, "alert_rate": 0.0}

        total_alerts = sum(len(entry["alerts"]) for entry in self.alert_history)

        # Calculate alerts per hour over the last 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_alerts = [
            entry for entry in self.alert_history if entry["timestamp"] >= cutoff_time
        ]

        recent_total = sum(len(entry["alerts"]) for entry in recent_alerts)
        alert_rate = recent_total / 24.0  # alerts per hour

        return {
            "total_alerts": total_alerts,
            "alert_rate_per_hour": round(alert_rate, 2),
            "history_entries": len(self.alert_history),
            "recent_alerts": recent_total,
        }

class HealthReporting:
    """
    Manages health reporting and summary generation.

    This class provides functionality for:
    - Generating health status summaries
    - Creating detailed health reports
    - Formatting health data for different outputs
    """

    def __init__(self):
        """Initialize the health reporting system."""
        self._report_cache: dict[str, Any] | None = None
        self._last_report_time: datetime | None = None
        self.cache_duration_seconds = 30  # Cache reports for 30 seconds

    def generate_summary(
        self, components: dict[str, ComponentHealth]
    ) -> dict[str, int]:
        """
        Generate a summary of health status counts.

        Args:
            components: Dictionary of component health results

        Returns:
            Dictionary with health status counts
        """
        summary: dict[str, int] = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
        }

        for comp in components.values():
            if comp.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif comp.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif comp.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            else:
                summary["unknown"] += 1

        return summary

    def generate_detailed_report(
        self,
        components: dict[str, ComponentHealth],
        overall_status: HealthStatus,
        timestamp: float,
    ) -> dict[str, Any]:
        """
        Generate a detailed health report.

        Args:
            components: Dictionary of component health results
            overall_status: Overall system health status
            timestamp: Report timestamp

        Returns:
            Detailed health report dictionary
        """
        summary = self.generate_summary(components)

        # Calculate additional metrics
        total_components = len(components)
        healthy_percentage = (
            (summary["healthy"] / total_components * 100) if total_components > 0 else 0
        )

        # Get component details for report
        component_details = []
        for name, comp in components.items():
            component_details.append(
                {
                    "name": name,
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "last_check": (
                        comp.last_check.isoformat() if comp.last_check else None
                    ),
                    "error_count": getattr(comp, "error_count", 0),
                }
            )

        report = {
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "overall_status": overall_status.value,
            "summary": summary,
            "total_components": total_components,
            "healthy_percentage": round(healthy_percentage, 1),
            "components": component_details,
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0",
            },
        }

        # Cache the report
        self._report_cache = report
        self._last_report_time = datetime.now(timezone.utc)

        return report

    def get_cached_report(self) -> dict[str, Any] | None:
        """
        Get cached report if available and recent.

        Returns:
            Cached report or None if cache is stale or unavailable
        """
        if self._report_cache is None or self._last_report_time is None:
            return None

        age_seconds = (
            datetime.now(timezone.utc) - self._last_report_time
        ).total_seconds()
        if age_seconds > self.cache_duration_seconds:
            return None

        return self._report_cache

    def clear_report_cache(self) -> None:
        """Clear the cached report."""
        self._report_cache = None
        self._last_report_time = None

    def format_report_for_console(self, report: dict[str, Any]) -> str:
        """
        Format a health report for console output.

        Args:
            report: Health report dictionary

        Returns:
            Formatted string for console display
        """
        lines = [
            "=" * 60,
            "HEALTH REPORT",
            "=" * 60,
            f"Timestamp: {report['timestamp']}",
            f"Overall Status: {report['overall_status'].upper()}",
            f"Total Components: {report['total_components']}",
            f"Healthy: {report['summary']['healthy']} ({report['healthy_percentage']}%)",
            f"Degraded: {report['summary']['degraded']}",
            f"Unhealthy: {report['summary']['unhealthy']}",
            f"Unknown: {report['summary']['unknown']}",
            "",
            "COMPONENT DETAILS:",
            "-" * 40,
        ]

        for comp in report["components"]:
            status_icon = {
                "healthy": "✓",
                "degraded": "⚠",
                "unhealthy": "✗",
                "unknown": "?",
            }.get(comp["status"], "?")

            lines.append(f"{status_icon} {comp['name']}: {comp['status'].upper()}")
            if comp["message"]:
                lines.append(f"  Message: {comp['message']}")
            if comp["response_time_ms"]:
                lines.append("  Response Time: {comp['response_time_ms']:.1f}ms")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

class HealthStatusAggregator:
    """
    Aggregates health status information from multiple sources.

    This class provides functionality for:
    - Combining health data from different monitoring systems
    - Calculating overall health scores and trends
    - Identifying patterns and correlations in health data
    """

    def __init__(self):
        """Initialize the health status aggregator."""
        self._aggregation_cache: dict[str, any] | None = None
        self._last_aggregation: datetime | None = None

    def aggregate_health_status(
        self, health_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Aggregate health status from multiple health check results.

        Args:
            health_results: List of health check result dictionaries

        Returns:
            Aggregated health status information
        """
        if not health_results:
            return {
                "overall_status": "unknown",
                "total_checks": 0,
                "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Combine all components from all results
        all_components: dict[str, Any] = {}
        for result in health_results:
            components = result.get("components", {})
            all_components.update(components)

        # Calculate overall status
        overall_status = self._calculate_aggregated_status(all_components)

        # Calculate trends and patterns
        trends = self._calculate_health_trends(all_components)

        aggregation = {
            "overall_status": overall_status,
            "total_checks": len(health_results),
            "total_components": len(all_components),
            "components": all_components,
            "trends": trends,
            "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregation_metadata": {
                "sources": len(health_results),
                "aggregation_version": "1.0",
            },
        }

        # Cache the aggregation
        self._aggregation_cache = aggregation
        self._last_aggregation = datetime.now(timezone.utc)

        return aggregation

    def _calculate_aggregated_status(
        self, components: dict[str, ComponentHealth]
    ) -> str:
        """Calculate overall status from component health results."""
        if not components:
            return "unknown"

        status_priority = {
            HealthStatus.UNHEALTHY: 4,
            HealthStatus.DEGRADED: 3,
            HealthStatus.UNKNOWN: 2,
            HealthStatus.HEALTHY: 1,
        }

        # Find the worst status
        worst_status = HealthStatus.HEALTHY
        for comp in components.values():
            if status_priority.get(comp.status, 0) > status_priority.get(
                worst_status, 0
            ):
                worst_status = comp.status

        return worst_status.value

    def _calculate_health_trends(
        self, components: dict[str, ComponentHealth]
    ) -> dict[str, Any]:
        """Calculate health trends and patterns."""
        return {
            "improving": 0,
            "degrading": 0,
            "stable": 0,
            "new_issues": 0,
        }

        # This would implement trend analysis based on historical data
        # For now, return basic structure

    def get_aggregation_cache(self) -> dict[str, Any] | None:
        """Get cached aggregation if available."""
        if self._aggregation_cache is None or self._last_aggregation is None:
            return None

        # Consider cache stale after 60 seconds
        age_seconds = (
            datetime.now(timezone.utc) - self._last_aggregation
        ).total_seconds()
        if age_seconds > 60:
            return None

        return self._aggregation_cache
