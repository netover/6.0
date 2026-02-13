from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from resync.core.health.health_models import (
    ComponentHealth,
    ComponentType,
    HealthCheckResult,
    HealthStatus,
)
from resync.core.health.health_service import HealthCheckService


@dataclass
class ComponentHealthSummary:
    """Summary of health status for a specific component type."""

    component_type: ComponentType
    total_components: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    unknown_count: int
    average_response_time_ms: float | None = None
    components: list[ComponentHealth] = field(default_factory=list)

    @property
    def health_percentage(self) -> float:
        """Calculate the percentage of healthy components."""
        if self.total_components == 0:
            return 0.0
        return (self.healthy_count / self.total_components) * 100.0

    @property
    def status(self) -> HealthStatus:
        """Determine overall status for this component type."""
        if self.unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        if self.degraded_count > 0:
            return HealthStatus.DEGRADED
        if self.unknown_count > 0:
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY


@dataclass
class OverallHealthStatus:
    """Overall health status of the entire system."""

    status: HealthStatus
    timestamp: datetime
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    unknown_components: int
    overall_health_percentage: float
    component_summaries: dict[ComponentType, ComponentHealthSummary] = field(default_factory=dict)
    critical_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_system_healthy(self) -> bool:
        """Check if the overall system is considered healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Comprehensive health report containing all health check data."""

    timestamp: datetime
    overall_status: OverallHealthStatus
    component_health: dict[str, ComponentHealth] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    trends: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthMonitoringAggregator:
    """
    Aggregates health checks from multiple components and provides comprehensive health monitoring.

    This class works with the existing HealthCheckService to collect, analyze, and summarize
    health data across all system components. It provides methods for aggregating component
    health, generating overall system status, and identifying trends and issues.
    """

    def __init__(self, health_service: HealthCheckService | None = None):
        """
        Initialize the HealthMonitoringAggregator.

        Args:
            health_service: Optional HealthCheckService instance. If not provided,
                          a new instance will be created when needed.
        """
        self.health_service = health_service
        self._last_collection_time: datetime | None = None
        self._cached_report: HealthReport | None = None

    async def get_health_service(self) -> HealthCheckService:
        """Get or create the health check service instance."""
        if self.health_service is None:
            from resync.core.health.health_service import get_health_check_service

            self.health_service = await get_health_check_service()
        return self.health_service

    async def collect_all_health_checks(self) -> HealthReport:
        """
        Collect all health checks from the health service.

        Returns:
            HealthReport: Comprehensive health report containing all component health data
        """
        health_service = await self.get_health_service()

        # Perform comprehensive health check
        health_result = await health_service.perform_comprehensive_health_check()

        # Create overall health status
        overall_status = await self.generate_overall_health_status(health_result)

        # Generate the complete report
        report = HealthReport(
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            component_health=health_result.components,
            performance_metrics=health_result.performance_metrics,
            alerts=health_result.alerts,
            metadata={
                "correlation_id": health_result.correlation_id,
                "check_duration_ms": health_result.performance_metrics.get(
                    "total_check_time_ms", 0
                ),
                "components_checked": len(health_result.components),
            },
        )

        # Cache the report for a short time to avoid excessive checks
        self._last_collection_time = datetime.now(timezone.utc)
        self._cached_report = report

        return report

    async def aggregate_component_health(
        self,
    ) -> dict[ComponentType, ComponentHealthSummary]:
        """
        Aggregate health data by component type.

        Returns:
            Dict[ComponentType, ComponentHealthSummary]: Summary for each component type
        """
        # Get fresh health data if cache is stale
        if self._should_refresh_cache():
            await self.collect_all_health_checks()

        if not self._cached_report:
            # Fallback: collect fresh data
            await self.collect_all_health_checks()

        # Group components by type
        components_by_type: dict[ComponentType, list[ComponentHealth]] = defaultdict(list)

        for component in self._cached_report.component_health.values():
            components_by_type[component.component_type].append(component)

        # Create summaries for each component type
        summaries = {}
        for component_type, components in components_by_type.items():
            summary = self._create_component_summary(component_type, components)
            summaries[component_type] = summary

        return summaries

    async def generate_overall_health_status(
        self, health_result: HealthCheckResult | None = None
    ) -> OverallHealthStatus:
        """
        Generate overall health status for the system.

        Args:
            health_result: Optional HealthCheckResult. If not provided, fresh data will be collected.

        Returns:
            OverallHealthStatus: Overall system health status
        """
        if health_result is None:
            # Collect fresh health data
            report = await self.collect_all_health_checks()
            health_result = HealthCheckResult(
                overall_status=report.overall_status.status,
                timestamp=report.timestamp,
                components=report.component_health,
                summary={},
                alerts=report.alerts,
                performance_metrics=report.performance_metrics,
            )

        # Count components by status
        status_counts = Counter(component.status for component in health_result.components.values())

        total_components = len(health_result.components)
        healthy_count = status_counts.get(HealthStatus.HEALTHY, 0)
        degraded_count = status_counts.get(HealthStatus.DEGRADED, 0)
        unhealthy_count = status_counts.get(HealthStatus.UNHEALTHY, 0)
        unknown_count = status_counts.get(HealthStatus.UNKNOWN, 0)

        # Calculate overall health percentage
        overall_health_percentage = (
            (healthy_count / total_components * 100) if total_components > 0 else 0.0
        )

        # Determine overall status (worst status wins)
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif unknown_count > 0:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        # Generate component summaries
        component_summaries = await self.aggregate_component_health()

        # Generate critical issues and recommendations
        critical_issues = self._identify_critical_issues(health_result.components)
        recommendations = self._generate_recommendations(
            health_result.components, component_summaries
        )

        return OverallHealthStatus(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            total_components=total_components,
            healthy_components=healthy_count,
            degraded_components=degraded_count,
            unhealthy_components=unhealthy_count,
            unknown_components=unknown_count,
            overall_health_percentage=round(overall_health_percentage, 2),
            component_summaries=component_summaries,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )

    def _create_component_summary(
        self, component_type: ComponentType, components: list[ComponentHealth]
    ) -> ComponentHealthSummary:
        """Create a summary for a specific component type."""
        status_counts = Counter(component.status for component in components)

        # Calculate average response time
        response_times = [c.response_time_ms for c in components if c.response_time_ms is not None]
        average_response_time = (
            sum(response_times) / len(response_times) if response_times else None
        )

        return ComponentHealthSummary(
            component_type=component_type,
            total_components=len(components),
            healthy_count=status_counts.get(HealthStatus.HEALTHY, 0),
            degraded_count=status_counts.get(HealthStatus.DEGRADED, 0),
            unhealthy_count=status_counts.get(HealthStatus.UNHEALTHY, 0),
            unknown_count=status_counts.get(HealthStatus.UNKNOWN, 0),
            average_response_time_ms=(
                round(average_response_time, 2) if average_response_time else None
            ),
            components=components,
        )

    def _identify_critical_issues(self, components: dict[str, ComponentHealth]) -> list[str]:
        """Identify critical issues from component health data."""
        issues = []

        for name, component in components.items():
            if component.status == HealthStatus.UNHEALTHY:
                if component.message:
                    issues.append(f"{name}: {component.message}")
                else:
                    issues.append(f"{name} is unhealthy")

            # Check for specific critical conditions
            if component.component_type == ComponentType.DATABASE:
                usage_percent = component.metadata.get("connection_usage_percent")
                if usage_percent and usage_percent > 95:
                    issues.append(f"Database connection pool critically high: {usage_percent}%")

            elif component.component_type == ComponentType.MEMORY:
                usage_percent = component.metadata.get("memory_usage_percent")
                if usage_percent and usage_percent > 95:
                    issues.append(f"Memory usage critically high: {usage_percent}%")

            elif component.component_type == ComponentType.FILE_SYSTEM:
                usage_percent = component.metadata.get("disk_usage_percent")
                if usage_percent and usage_percent > 95:
                    issues.append(f"Disk usage critically high: {usage_percent}%")

        return issues

    def _generate_recommendations(
        self,
        components: dict[str, ComponentHealth],
        summaries: dict[ComponentType, ComponentHealthSummary],
    ) -> list[str]:
        """Generate recommendations based on component health."""
        recommendations = []

        # Check for degraded components that might need attention
        for component_type, summary in summaries.items():
            if summary.status == HealthStatus.DEGRADED:
                if component_type == ComponentType.DATABASE:
                    recommendations.append(
                        "Monitor database connection pool usage and consider scaling"
                    )
                elif component_type == ComponentType.MEMORY:
                    recommendations.append(
                        "Monitor memory usage and consider optimizing memory-intensive processes"
                    )
                elif component_type == ComponentType.CACHE:
                    recommendations.append(
                        "Review cache performance and consider cache optimization"
                    )

        # Check for high response times
        for component_type, summary in summaries.items():
            if summary.average_response_time and summary.average_response_time > 1000:  # > 1 second
                recommendations.append(
                    f"High response times detected for {component_type.value} components"
                )

        # General recommendations based on overall health
        unhealthy_types = [
            ct for ct, summary in summaries.items() if summary.status == HealthStatus.UNHEALTHY
        ]
        if unhealthy_types:
            recommendations.append(
                f"Immediate attention required for: {', '.join(ct.value for ct in unhealthy_types)}"
            )

        return recommendations

    def _should_refresh_cache(self, max_age_seconds: int = 30) -> bool:
        """Check if cached data should be refreshed."""
        if self._last_collection_time is None:
            return True

        age = datetime.now(timezone.utc) - self._last_collection_time
        return age.total_seconds() > max_age_seconds

    def get_cached_report(self) -> HealthReport | None:
        """Get cached health report if available and fresh."""
        if self._should_refresh_cache():
            return None
        return self._cached_report

    def clear_cache(self) -> None:
        """Clear the cached health report."""
        self._cached_report = None
        self._last_collection_time = None
