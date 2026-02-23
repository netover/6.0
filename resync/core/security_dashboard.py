# pylint: skip-file
# mypy: ignore-errors
"""
Security Metrics Dashboard and Monitoring System.

This module provides comprehensive security monitoring and dashboard capabilities including:
- Real-time security metrics collection and aggregation
- Interactive dashboards with customizable widgets
- Automated report generation and compliance tracking
- KPI monitoring and alerting system
- Integration with monitoring and visualization tools
- Historical trend analysis and forecasting
- Role-based access control for dashboard views
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of security metrics."""

    COUNT = "count"  # Raw counts (events, incidents, etc.)
    RATE = "rate"  # Rates per time period
    PERCENTAGE = "percentage"  # Percentage values
    DURATION = "duration"  # Time durations
    SCORE = "score"  # Calculated scores
    TREND = "trend"  # Trend indicators


class MetricCategory(Enum):
    """Security metric categories."""

    THREAT_DETECTION = "threat_detection"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE = "compliance"
    ACCESS_CONTROL = "access_control"
    SYSTEM_HEALTH = "system_health"
    USER_BEHAVIOR = "user_behavior"
    NETWORK_SECURITY = "network_security"
    DATA_PROTECTION = "data_protection"


@dataclass
class SecurityMetric:
    """Individual security metric."""

    metric_id: str
    name: str
    description: str
    category: MetricCategory
    metric_type: MetricType
    unit: str = ""  # events, %, seconds, etc.

    # Value tracking
    current_value: float = 0.0
    previous_value: float = 0.0
    baseline_value: float = 0.0

    # Metadata
    last_updated: float = field(default_factory=time.time)
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Thresholds and alerting
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    alert_enabled: bool = True

    # Historical data
    historical_values: deque = field(
        default_factory=lambda: deque(maxlen=10080)
    )  # 1 week at 1-minute intervals

    @property
    def trend(self) -> str:
        """Calculate trend direction."""
        if len(self.data_points) < 2:
            return "stable"

        recent = list(self.data_points)[
            -10:
        ]  # Last 10 points (timestamp, value tuples)
        if len(recent) < 2:
            return "stable"

        # Extract values from tuples
        values = [point[1] for point in recent]

        # Simple linear trend
        first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
        second_half = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)

        if second_half > first_half * 1.1:
            return "increasing"
        if second_half < first_half * 0.9:
            return "decreasing"
        return "stable"

    @property
    def status(self) -> str:
        """Get metric status based on thresholds."""
        if self.critical_threshold and self.current_value >= self.critical_threshold:
            return "critical"
        if self.warning_threshold and self.current_value >= self.warning_threshold:
            return "warning"
        return "normal"

    @property
    def percentage_change(self) -> float:
        """Calculate percentage change from previous value."""
        if self.previous_value == 0:
            return 0.0 if self.current_value == 0 else float("inf")

        return ((self.current_value - self.previous_value) / self.previous_value) * 100

    def update_value(self, new_value: float, timestamp: float | None = None) -> None:
        """Update metric value."""
        self.previous_value = self.current_value
        self.current_value = new_value
        self.last_updated = timestamp or time.time()

        # Add to data points
        self.data_points.append((self.last_updated, new_value))
        self.historical_values.append((self.last_updated, new_value))

    def get_summary(self) -> dict[str, Any]:
        """Get metric summary."""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "category": self.category.value,
            "type": self.metric_type.value,
            "unit": self.unit,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "baseline_value": self.baseline_value,
            "percentage_change": self.percentage_change,
            "trend": self.trend,
            "status": self.status,
            "last_updated": self.last_updated,
            "data_points_count": len(self.data_points),
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""

    widget_id: str
    title: str
    widget_type: str  # chart, gauge, table, alert, etc.
    metrics: list[str]  # Metric IDs to display
    position: dict[str, int] = field(default_factory=dict)  # x, y, width, height
    config: dict[str, Any] = field(default_factory=dict)  # Widget-specific config
    refresh_interval: int = 30  # seconds
    roles: set[str] = field(
        default_factory=lambda: {"admin", "security"}
    )  # Access control

    def can_access(self, user_roles: set[str]) -> bool:
        """Check if user can access this widget."""
        return bool(self.roles.intersection(user_roles))


@dataclass
class Dashboard:
    """Security dashboard configuration."""

    dashboard_id: str
    name: str
    description: str
    widgets: list[DashboardWidget] = field(default_factory=list)
    roles: set[str] = field(default_factory=lambda: {"admin", "security"})
    is_default: bool = False
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    def can_access(self, user_roles: set[str]) -> bool:
        """Check if user can access this dashboard."""
        return bool(self.roles.intersection(user_roles))

    def get_visible_widgets(self, user_roles: set[str]) -> list[DashboardWidget]:
        """Get widgets visible to user."""
        return [w for w in self.widgets if w.can_access(user_roles)]


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    name: str
    description: str
    metric_id: str
    condition: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: str  # low, medium, high, critical
    enabled: bool = True
    cooldown_minutes: int = 15
    notification_channels: list[str] = field(default_factory=lambda: ["email"])
    last_triggered: float | None = None

    def should_trigger(self, metric: SecurityMetric) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            cooldown_seconds = self.cooldown_minutes * 60
            if time.time() - self.last_triggered < cooldown_seconds:
                return False

        # Evaluate condition
        value = metric.current_value

        if self.condition == ">":
            return value > self.threshold
        if self.condition == "<":
            return value < self.threshold
        if self.condition == ">=":
            return value >= self.threshold
        if self.condition == "<=":
            return value <= self.threshold
        if self.condition == "==":
            return value == self.threshold
        if self.condition == "!=":
            return value != self.threshold

        return False

    def trigger(self) -> None:
        """Mark alert as triggered."""
        self.last_triggered = time.time()


@dataclass
class ComplianceReport:
    """Compliance report data."""

    report_id: str
    report_type: str  # daily, weekly, monthly, quarterly
    period_start: float
    period_end: float
    generated_at: float = field(default_factory=time.time)

    # Report sections
    executive_summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    compliance_score: float = 0.0

    # Metadata
    author: str = "Security Dashboard"
    reviewers: list[str] = field(default_factory=list)
    status: str = "draft"  # draft, reviewed, approved, published


@dataclass
class SecurityDashboardConfig:
    """Configuration for security dashboard system."""

    # Metric collection
    collection_interval_seconds: int = 60
    retention_days: int = 90
    max_data_points_per_metric: int = 1000

    # Dashboard settings
    default_refresh_interval: int = 30
    max_widgets_per_dashboard: int = 20
    enable_real_time_updates: bool = True

    # Alerting
    alert_check_interval_seconds: int = 30
    max_alerts_per_hour: int = 100
    alert_aggregation_window: int = 300  # 5 minutes

    # Reporting
    auto_generate_reports: bool = True
    report_schedule: dict[str, str] = field(
        default_factory=lambda: {
            "daily": "08:00",
            "weekly": "Monday 09:00",
            "monthly": "1st 10:00",
        }
    )

    # Access control
    enable_role_based_access: bool = True
    audit_dashboard_access: bool = True

    # Performance
    max_concurrent_users: int = 50
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


class MetricCollector:
    """Component for collecting security metrics from various sources."""

    def __init__(self, config: SecurityDashboardConfig):
        self.config = config
        self.metrics: dict[str, SecurityMetric] = {}

        # Initialize standard security metrics
        self._initialize_standard_metrics()

    def _initialize_standard_metrics(self) -> None:
        """Initialize standard security metrics."""
        standard_metrics = [
            # Threat Detection
            SecurityMetric(
                metric_id="threats_detected",
                name="Threats Detected",
                description="Number of security threats detected",
                category=MetricCategory.THREAT_DETECTION,
                metric_type=MetricType.COUNT,
                unit="threats",
                warning_threshold=10,
                critical_threshold=50,
            ),
            SecurityMetric(
                metric_id="anomalies_per_hour",
                name="Anomalies per Hour",
                description="Rate of behavioral anomalies detected",
                category=MetricCategory.THREAT_DETECTION,
                metric_type=MetricType.RATE,
                unit="anomalies/hour",
                warning_threshold=5.0,
                critical_threshold=15.0,
            ),
            # Incident Response
            SecurityMetric(
                metric_id="active_incidents",
                name="Active Incidents",
                description="Number of currently active security incidents",
                category=MetricCategory.INCIDENT_RESPONSE,
                metric_type=MetricType.COUNT,
                unit="incidents",
                warning_threshold=2,
                critical_threshold=5,
            ),
            SecurityMetric(
                metric_id="avg_response_time",
                name="Average Response Time",
                description="Average time to respond to security incidents",
                category=MetricCategory.INCIDENT_RESPONSE,
                metric_type=MetricType.DURATION,
                unit="minutes",
                warning_threshold=30.0,
                critical_threshold=120.0,
            ),
            # Compliance
            SecurityMetric(
                metric_id="compliance_score",
                name="Compliance Score",
                description="Overall security compliance score",
                category=MetricCategory.COMPLIANCE,
                metric_type=MetricType.SCORE,
                unit="%",
                warning_threshold=85.0,
                critical_threshold=70.0,
            ),
            SecurityMetric(
                metric_id="audit_findings",
                name="Audit Findings",
                description="Number of open audit findings",
                category=MetricCategory.COMPLIANCE,
                metric_type=MetricType.COUNT,
                unit="findings",
                warning_threshold=5,
                critical_threshold=15,
            ),
            # Access Control
            SecurityMetric(
                metric_id="failed_logins",
                name="Failed Login Attempts",
                description="Number of failed authentication attempts",
                category=MetricCategory.ACCESS_CONTROL,
                metric_type=MetricType.COUNT,
                unit="attempts",
                warning_threshold=50,
                critical_threshold=200,
            ),
            SecurityMetric(
                metric_id="privileged_access",
                name="Privileged Access Events",
                description="Number of privileged access operations",
                category=MetricCategory.ACCESS_CONTROL,
                metric_type=MetricType.COUNT,
                unit="events",
                warning_threshold=10,
                critical_threshold=25,
            ),
            # System Health
            SecurityMetric(
                metric_id="system_uptime",
                name="System Uptime",
                description="Overall system uptime percentage",
                category=MetricCategory.SYSTEM_HEALTH,
                metric_type=MetricType.PERCENTAGE,
                unit="%",
                warning_threshold=99.0,
                critical_threshold=95.0,
            ),
            SecurityMetric(
                metric_id="error_rate",
                name="System Error Rate",
                description="Rate of system errors",
                category=MetricCategory.SYSTEM_HEALTH,
                metric_type=MetricType.PERCENTAGE,
                unit="%",
                warning_threshold=1.0,
                critical_threshold=5.0,
            ),
            # User Behavior
            SecurityMetric(
                metric_id="suspicious_users",
                name="Suspicious Users",
                description="Number of users flagged for suspicious behavior",
                category=MetricCategory.USER_BEHAVIOR,
                metric_type=MetricType.COUNT,
                unit="users",
                warning_threshold=3,
                critical_threshold=10,
            ),
            # Network Security
            SecurityMetric(
                metric_id="blocked_connections",
                name="Blocked Network Connections",
                description="Number of blocked network connections",
                category=MetricCategory.NETWORK_SECURITY,
                metric_type=MetricType.COUNT,
                unit="connections",
                warning_threshold=100,
                critical_threshold=500,
            ),
            # Data Protection
            SecurityMetric(
                metric_id="encryption_coverage",
                name="Data Encryption Coverage",
                description="Percentage of data properly encrypted",
                category=MetricCategory.DATA_PROTECTION,
                metric_type=MetricType.PERCENTAGE,
                unit="%",
                warning_threshold=95.0,
                critical_threshold=85.0,
            ),
        ]

        for metric in standard_metrics:
            self.metrics[metric.metric_id] = metric

    def update_metric(
        self, metric_id: str, value: float, timestamp: float | None = None
    ) -> None:
        """Update a metric value."""
        if metric_id in self.metrics:
            self.metrics[metric_id].update_value(value, timestamp)

    def get_metric(self, metric_id: str) -> SecurityMetric | None:
        """Get a specific metric."""
        return self.metrics.get(metric_id)

    def get_metrics_by_category(self, category: MetricCategory) -> list[SecurityMetric]:
        """Get all metrics in a category."""
        return [m for m in self.metrics.values() if m.category == category]

    def get_all_metrics(self) -> list[SecurityMetric]:
        """Get all metrics."""
        return list(self.metrics.values())

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "total_metrics": len(self.metrics),
            "by_category": defaultdict(int),
            "by_status": defaultdict(int),
            "alerts_active": 0,
            "last_updated": 0,
        }

        for metric in self.metrics.values():
            summary["by_category"][metric.category.value] += 1
            summary["by_status"][metric.status] += 1
            summary["last_updated"] = max(summary["last_updated"], metric.last_updated)

            if metric.status in ["warning", "critical"]:
                summary["alerts_active"] += 1

        return dict(summary)


class AlertManager:
    """Component for managing security alerts."""

    def __init__(self, config: SecurityDashboardConfig):
        self.config = config
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=10000)

        # Initialize standard alert rules
        self._initialize_alert_rules()

    def _initialize_alert_rules(self) -> None:
        """Initialize standard alert rules."""
        rules = [
            AlertRule(
                rule_id="high_threat_rate",
                name="High Threat Detection Rate",
                description="Alert when threat detection rate is too high",
                metric_id="threats_detected",
                condition=">",
                threshold=25.0,
                severity="high",
                cooldown_minutes=30,
            ),
            AlertRule(
                rule_id="low_compliance_score",
                name="Low Compliance Score",
                description="Alert when compliance score drops too low",
                metric_id="compliance_score",
                condition="<",
                threshold=80.0,
                severity="high",
                cooldown_minutes=60,
            ),
            AlertRule(
                rule_id="multiple_failed_logins",
                name="Multiple Failed Logins",
                description="Alert on high rate of failed login attempts",
                metric_id="failed_logins",
                condition=">",
                threshold=100.0,
                severity="medium",
                cooldown_minutes=15,
            ),
            AlertRule(
                rule_id="system_downtime",
                name="System Downtime Alert",
                description="Alert when system uptime drops",
                metric_id="system_uptime",
                condition="<",
                threshold=98.0,
                severity="critical",
                cooldown_minutes=5,
            ),
        ]

        for rule in rules:
            self.alert_rules[rule.rule_id] = rule

    def check_alerts(self, metrics: dict[str, SecurityMetric]) -> list[dict[str, Any]]:
        """Check all alert rules and return triggered alerts."""
        triggered_alerts = []

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            metric = metrics.get(rule.metric_id)
            if not metric:
                continue

            if rule.should_trigger(metric):
                alert = {
                    "alert_id": f"alert_{int(time.time())}_{rule.rule_id}",
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "metric_id": rule.metric_id,
                    "metric_value": metric.current_value,
                    "threshold": rule.threshold,
                    "condition": rule.condition,
                    "timestamp": time.time(),
                    "description": f"{rule.description} - Current value: {metric.current_value}",
                }

                # Mark rule as triggered
                rule.trigger()

                # Add to active alerts
                self.active_alerts[alert["alert_id"]] = alert
                self.alert_history.append(alert)

                triggered_alerts.append(alert)

                logger.warning(
                    "security_alert_triggered",
                    alert_id=alert["alert_id"],
                    rule_name=rule.name,
                    severity=rule.severity,
                    metric_value=metric.current_value,
                )

        return triggered_alerts

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["acknowledged_by"] = user_id
            self.active_alerts[alert_id]["acknowledged_at"] = time.time()
            return True
        return False

    def resolve_alert(self, alert_id: str, resolution: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["resolved"] = True
            self.active_alerts[alert_id]["resolution"] = resolution
            self.active_alerts[alert_id]["resolved_at"] = time.time()

            # Remove from active alerts
            del self.active_alerts[alert_id]
            return True
        return False

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(
        self, limit: int = 100, severity_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Get alert history with optional filtering."""
        alerts = list(self.alert_history)[-limit:]

        if severity_filter:
            alerts = [a for a in alerts if a.get("severity") == severity_filter]

        return alerts


class ReportGenerator:
    """Component for generating security reports."""

    def __init__(self, config: SecurityDashboardConfig):
        self.config = config

    async def generate_daily_report(self) -> ComplianceReport:
        """Generate daily security report."""
        return await self._generate_report("daily")

    async def generate_weekly_report(self) -> ComplianceReport:
        """Generate weekly security report."""
        return await self._generate_report("weekly")

    async def generate_monthly_report(self) -> ComplianceReport:
        """Generate monthly security report."""
        return await self._generate_report("monthly")

    async def _generate_report(self, report_type: str) -> ComplianceReport:
        """Generate a security report."""
        # Calculate time range
        now = time.time()
        if report_type == "daily":
            period_start = now - 86400
        elif report_type == "weekly":
            period_start = now - 604800
        else:  # monthly
            period_start = now - 2592000

        report = ComplianceReport(
            report_id=f"report_{report_type}_{int(now)}",
            report_type=report_type,
            period_start=period_start,
            period_end=now,
        )

        # Generate report content
        self._populate_report_content(report)

        return report

    def _populate_report_content(self, report: ComplianceReport) -> None:
        """Populate report with actual content."""
        # This would integrate with the actual metric collectors
        # For now, provide sample content

        report.executive_summary = f"""
        Security Report for {report.report_type} period ending {datetime.fromtimestamp(report.period_end, tz=timezone.utc).strftime("%Y-%m-%d")}.

        This report provides an overview of security metrics, incidents, and compliance status for the reporting period.
        """

        report.findings = [
            {
                "severity": "medium",
                "category": "threat_detection",
                "title": "Increased Anomalous Activity",
                "description": "Detected 15% increase in anomalous user behavior patterns",
                "recommendation": "Review user behavior monitoring rules",
            },
            {
                "severity": "low",
                "category": "compliance",
                "title": "Minor Audit Findings",
                "description": "3 minor audit findings related to documentation",
                "recommendation": "Update security documentation",
            },
        ]

        report.recommendations = [
            "Enhance user behavior analytics",
            "Implement additional automated testing",
            "Review and update incident response procedures",
            "Strengthen network segmentation controls",
        ]

        report.metrics_summary = {
            "threats_detected": 45,
            "incidents_resolved": 12,
            "compliance_score": 87.5,
            "system_uptime": 99.8,
            "false_positives": 8,
        }

        report.compliance_score = 87.5


class SecurityDashboard:
    """
    Main security dashboard and monitoring system.

    Features:
    - Real-time security metrics collection and visualization
    - Customizable dashboards with role-based access
    - Automated alerting and notification system
    - Compliance reporting and historical analysis
    - KPI monitoring and trend analysis
    - Integration APIs for external tools
    """

    def __init__(self, config: SecurityDashboardConfig | None = None):
        self.config = config or SecurityDashboardConfig()

        # Core components
        self.metric_collector = MetricCollector(self.config)
        self.alert_manager = AlertManager(self.config)
        self.report_generator = ReportGenerator(self.config)

        # Dashboard management
        self.dashboards: dict[str, Dashboard] = {}
        self.default_dashboard: Dashboard | None = None

        # Background tasks
        self._collection_task: asyncio.Task | None = None
        self._alert_task: asyncio.Task | None = None
        self._reporting_task: asyncio.Task | None = None
        self._running = False

        # Initialize default dashboard
        self._initialize_default_dashboard()

    def _initialize_default_dashboard(self) -> None:
        """Initialize the default security dashboard."""
        widgets = [
            DashboardWidget(
                widget_id="threat_overview",
                title="Threat Overview",
                widget_type="chart",
                metrics=["threats_detected", "anomalies_per_hour"],
                position={"x": 0, "y": 0, "width": 6, "height": 4},
                config={"chart_type": "line", "time_range": "24h"},
            ),
            DashboardWidget(
                widget_id="incident_status",
                title="Incident Status",
                widget_type="gauge",
                metrics=["active_incidents"],
                position={"x": 6, "y": 0, "width": 3, "height": 4},
                config={"max_value": 10, "thresholds": {"warning": 2, "critical": 5}},
            ),
            DashboardWidget(
                widget_id="compliance_score",
                title="Compliance Score",
                widget_type="gauge",
                metrics=["compliance_score"],
                position={"x": 9, "y": 0, "width": 3, "height": 4},
                config={
                    "max_value": 100,
                    "thresholds": {"warning": 85, "critical": 70},
                },
            ),
            DashboardWidget(
                widget_id="access_control",
                title="Access Control",
                widget_type="table",
                metrics=["failed_logins", "privileged_access"],
                position={"x": 0, "y": 4, "width": 6, "height": 4},
                config={"show_trend": True},
            ),
            DashboardWidget(
                widget_id="system_health",
                title="System Health",
                widget_type="chart",
                metrics=["system_uptime", "error_rate"],
                position={"x": 6, "y": 4, "width": 6, "height": 4},
                config={"chart_type": "area", "dual_axis": True},
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Alerts",
                widget_type="alert",
                metrics=[],  # Special widget for alerts
                position={"x": 0, "y": 8, "width": 12, "height": 2},
                config={"max_alerts": 5, "auto_acknowledge": False},
            ),
        ]

        dashboard = Dashboard(
            dashboard_id="default_security",
            name="Security Overview",
            description="Default security monitoring dashboard",
            widgets=widgets,
            is_default=True,
        )

        self.dashboards[dashboard.dashboard_id] = dashboard
        self.default_dashboard = dashboard

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the security dashboard system.

        Args:
            tg: Optional TaskGroup to run the background tasks in
        """
        if self._running:
            return

        self._running = True
        if tg:
            self._collection_task = tg.create_task(
                self._metric_collection_worker(), name="metric_collection_worker"
            )
            self._alert_task = tg.create_task(
                self._alert_checking_worker(), name="alert_checking_worker"
            )
            self._reporting_task = tg.create_task(
                self._report_generation_worker(), name="report_generation_worker"
            )
        else:
            self._collection_task = track_task(
                self._metric_collection_worker(), name="metric_collection_worker"
            )
            self._alert_task = track_task(
                self._alert_checking_worker(), name="alert_checking_worker"
            )
            self._reporting_task = track_task(
                self._report_generation_worker(), name="report_generation_worker"
            )

        logger.info(
            "Security dashboard system started",
            method="task_group" if tg else "track_task",
        )

    async def stop(self) -> None:
        """Stop the security dashboard system."""
        if not self._running:
            return

        self._running = False

        for task in [self._collection_task, self._alert_task, self._reporting_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("Security dashboard system stopped")

    def update_metric(self, metric_id: str, value: float) -> None:
        """Update a metric value."""
        self.metric_collector.update_metric(metric_id, value)

    def get_dashboard(
        self, dashboard_id: str, user_roles: set[str]
    ) -> Dashboard | None:
        """Get a dashboard if user has access."""
        dashboard = self.dashboards.get(dashboard_id)
        if dashboard and dashboard.can_access(user_roles):
            return dashboard
        return None

    def get_default_dashboard(self, user_roles: set[str]) -> Dashboard | None:
        """Get the default dashboard for user."""
        if self.default_dashboard and self.default_dashboard.can_access(user_roles):
            return self.default_dashboard
        return None

    def get_dashboard_data(
        self, dashboard: Dashboard, user_roles: set[str], time_range: str = "24h"
    ) -> dict[str, Any]:
        """Get data for a dashboard."""
        visible_widgets = dashboard.get_visible_widgets(user_roles)

        dashboard_data = {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "widgets": [],
        }

        for widget in visible_widgets:
            widget_data = {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "type": widget.widget_type,
                "position": widget.position,
                "data": {},
            }

            # Get metric data for widget
            for metric_id in widget.metrics:
                metric = self.metric_collector.get_metric(metric_id)
                if metric:
                    widget_data["data"][metric_id] = metric.get_summary()

            # Special handling for alert widgets
            if widget.widget_type == "alert":
                widget_data["data"]["active_alerts"] = (
                    self.alert_manager.get_active_alerts()[
                        : widget.config.get("max_alerts", 5)
                    ]
                )

            dashboard_data["widgets"].append(widget_data)

        return dashboard_data

    def get_metrics_data(
        self,
        metric_ids: list[str] | None = None,
        category: MetricCategory | None = None,
        time_range: str = "24h",
    ) -> dict[str, Any]:
        """Get metrics data for API consumption."""
        if metric_ids:
            metrics = [self.metric_collector.get_metric(mid) for mid in metric_ids]
            metrics = [m for m in metrics if m is not None]
        elif category:
            metrics = self.metric_collector.get_metrics_by_category(category)
        else:
            metrics = self.metric_collector.get_all_metrics()

        return {
            "metrics": [m.get_summary() for m in metrics],
            "summary": self.metric_collector.get_metrics_summary(),
            "timestamp": time.time(),
        }

    def get_alerts_data(
        self, include_history: bool = False, limit: int = 50
    ) -> dict[str, Any]:
        """Get alerts data."""
        data = {
            "active_alerts": self.alert_manager.get_active_alerts(),
            "alert_rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "metric_id": rule.metric_id,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "enabled": rule.enabled,
                    "last_triggered": rule.last_triggered,
                }
                for rule in self.alert_manager.alert_rules.values()
            ],
        }

        if include_history:
            data["alert_history"] = self.alert_manager.get_alert_history(limit=limit)

        return data

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id, user_id)

    def resolve_alert(self, alert_id: str, resolution: str) -> bool:
        """Resolve an alert."""
        return self.alert_manager.resolve_alert(alert_id, resolution)

    async def generate_report(self, report_type: str) -> ComplianceReport | None:
        """Generate a compliance report."""
        if report_type == "daily":
            return await self.report_generator.generate_daily_report()
        if report_type == "weekly":
            return await self.report_generator.generate_weekly_report()
        if report_type == "monthly":
            return await self.report_generator.generate_monthly_report()
        return None

    def create_custom_dashboard(
        self,
        name: str,
        description: str,
        widgets: list[DashboardWidget],
        roles: set[str],
    ) -> str:
        """Create a custom dashboard."""
        dashboard_id = f"dashboard_{int(time.time())}_{hash(name) % 10000}"

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            widgets=widgets,
            roles=roles,
        )

        self.dashboards[dashboard_id] = dashboard
        return dashboard_id

    async def _metric_collection_worker(self) -> None:
        """Background worker for metric collection."""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval_seconds)

                # Simulate metric collection from various sources
                # In real implementation, this would integrate with actual systems
                self._collect_system_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metric collection worker error: %s", e)

    def _collect_system_metrics(self) -> None:
        """Collect metrics from system components."""
        # This would integrate with actual metric sources
        # For demonstration, simulate some metric updates

        import random

        # Simulate some realistic metric values
        self.metric_collector.update_metric(
            "system_uptime", 99.7 + random.uniform(-0.5, 0.5)
        )
        self.metric_collector.update_metric("threats_detected", random.randint(0, 10))
        self.metric_collector.update_metric("active_incidents", random.randint(0, 3))
        self.metric_collector.update_metric(
            "compliance_score", 85.0 + random.uniform(-5, 5)
        )

    async def _alert_checking_worker(self) -> None:
        """Background worker for alert checking."""
        while self._running:
            try:
                await asyncio.sleep(self.config.alert_check_interval_seconds)

                # Check for triggered alerts
                metrics = {
                    m.metric_id: m for m in self.metric_collector.get_all_metrics()
                }
                triggered_alerts = self.alert_manager.check_alerts(metrics)

                # Handle triggered alerts (send notifications, etc.)
                for alert in triggered_alerts:
                    self._handle_triggered_alert(alert)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert checking worker error: %s", e)

    def _handle_triggered_alert(self, alert: dict[str, Any]) -> None:
        """Handle a triggered alert."""
        # This would send notifications via email, Slack, etc.
        logger.warning(
            "alert_notification",
            alert_id=alert["alert_id"],
            severity=alert["severity"],
            message=alert["description"],
        )

    async def _report_generation_worker(self) -> None:
        """Background worker for automated report generation."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check every hour

                if not self.config.auto_generate_reports:
                    continue

                # Check if it's time to generate reports
                current_time = datetime.now(timezone.utc)

                # Generate daily report at configured time
                daily_time = self.config.report_schedule.get("daily", "08:00")
                if current_time.strftime("%H:%M") == daily_time:
                    report = await self.report_generator.generate_daily_report()
                    logger.info("Generated daily security report: %s", report.report_id)

                # Generate weekly report on configured day/time
                weekly_schedule = self.config.report_schedule.get(
                    "weekly", "Monday 09:00"
                )
                if (
                    weekly_schedule
                    and current_time.strftime("%A %H:%M") == weekly_schedule
                ):
                    report = await self.report_generator.generate_weekly_report()
                    logger.info(
                        "Generated weekly security report: %s", report.report_id
                    )

                # Generate monthly report on configured day/time
                monthly_schedule = self.config.report_schedule.get(
                    "monthly", "1st 10:00"
                )
                if (
                    monthly_schedule
                    and current_time.day == 1
                    and current_time.strftime("%H:%M") == monthly_schedule.split()[1]
                ):
                    report = await self.report_generator.generate_monthly_report()
                    logger.info(
                        "Generated monthly security report: %s", report.report_id
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Report generation worker error: %s", e)


# Global security dashboard instance
_security_dashboard_instance: SecurityDashboard | None = None


class _LazySecurityDashboard:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance = None

    def get_instance(self) -> SecurityDashboard:
        if self._instance is None:
            self._instance = SecurityDashboard()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)


security_dashboard = _LazySecurityDashboard()


async def get_security_dashboard(
    tg: asyncio.TaskGroup | None = None,
) -> SecurityDashboard:
    """
    Get the global security dashboard instance (lazy initialized).

    Args:
        tg: Optional TaskGroup to start the dashboard in
    """
    instance = security_dashboard.get_instance()
    if not instance._running:
        instance.start(tg=tg)
    return instance
