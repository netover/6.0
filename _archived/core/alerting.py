"""
Alerting system for Resync with multiple notification channels
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

import structlog

# Configure alerting logger
alerting_logger = structlog.get_logger("resync.alerting")

from resync.config.slo import KPI_DEFINITIONS, validate_slo_compliance
from resync.core.metrics import runtime_metrics
from resync.core.teams_integration import TeamsNotification, get_teams_integration


# Initialize default SLO alert rules at module level
def _initialize_slo_alerts(alerting_system):
    """Initialize SLO-based alert rules"""
    # Add SLO-specific alert rules if they don't already exist
    existing_names = [rule.name for rule in alerting_system.rules]

    # API response time SLO
    if "slo_api_response_time" not in existing_names:
        alerting_system.rules.append(
            AlertRule(
                name="slo_api_response_time",
                description="API response time violates SLO",
                metric_name="api_response_time",
                condition="gt",
                threshold=KPI_DEFINITIONS["api_response_time"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
            )
        )

    # Error rate SLO
    if "slo_api_error_rate" not in existing_names:
        alerting_system.rules.append(
            AlertRule(
                name="slo_api_error_rate",
                description="API error rate violates SLO",
                metric_name="api_error_rate",
                condition="gt",
                threshold=KPI_DEFINITIONS["api_error_rate"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
            )
        )

    # Availability SLO
    if "slo_availability" not in existing_names:
        alerting_system.rules.append(
            AlertRule(
                name="slo_availability",
                description="System availability violates SLO",
                metric_name="availability",
                condition="lt",
                threshold=KPI_DEFINITIONS["availability"]["critical_threshold"],
                severity=AlertSeverity.EMERGENCY,
                duration=timedelta(minutes=1),
            )
        )

    # Cache hit ratio SLO
    if "slo_cache_hit_ratio" not in existing_names:
        alerting_system.rules.append(
            AlertRule(
                name="slo_cache_hit_ratio",
                description="Cache hit ratio violates SLO",
                metric_name="cache_hit_ratio",
                condition="lt",
                threshold=KPI_DEFINITIONS["cache_hit_ratio"]["critical_threshold"],
                severity=AlertSeverity.WARNING,
                duration=timedelta(minutes=5),
            )
        )

    # TWS connection success rate SLO
    if "slo_tws_connection_success_rate" not in existing_names:
        alerting_system.rules.append(
            AlertRule(
                name="slo_tws_connection_success_rate",
                description="TWS connection success rate violates SLO",
                metric_name="tws_connection_success_rate",
                condition="lt",
                threshold=KPI_DEFINITIONS["tws_connection_success_rate"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
            )
        )


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Definition of an alert rule"""

    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "ge", "le"
    threshold: float
    severity: AlertSeverity
    duration: timedelta = timedelta(minutes=1)  # Time period over which threshold is checked
    enabled: bool = True


@dataclass
class Alert:
    """An instance of an alert"""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    value: float
    threshold: float
    metric_name: str
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None


class AlertingSystem:
    """
    Comprehensive alerting system with multiple notification channels
    """

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.active_alerts: list[Alert] = []
        self.escalation_policies: dict[str, list[Callable]] = {}
        self.alert_history: list[Alert] = []
        self.metrics_snapshot: dict[str, Any] = {}

        # Initialize with default alert rules
        self._initialize_default_rules()

        # Initialize SLO-based alert rules
        _initialize_slo_alerts(self)

    def _initialize_default_rules(self):
        """Initialize default alert rules based on SLO definitions"""
        # API response time alert
        self.rules.append(
            AlertRule(
                name="api_response_time_high",
                description="API response time exceeds threshold",
                metric_name="api_response_time",
                condition="gt",
                threshold=KPI_DEFINITIONS["api_response_time"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(seconds=30),
            )
        )

        # Error rate alert
        self.rules.append(
            AlertRule(
                name="error_rate_high",
                description="API error rate exceeds threshold",
                metric_name="api_error_rate",
                condition="gt",
                threshold=KPI_DEFINITIONS["api_error_rate"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
            )
        )

        # Availability alert
        self.rules.append(
            AlertRule(
                name="availability_low",
                description="System availability below threshold",
                metric_name="availability",
                condition="lt",
                threshold=KPI_DEFINITIONS["availability"]["critical_threshold"],
                severity=AlertSeverity.EMERGENCY,
                duration=timedelta(minutes=1),
            )
        )

        # Cache hit ratio alert
        self.rules.append(
            AlertRule(
                name="cache_hit_ratio_low",
                description="Cache hit ratio below threshold",
                metric_name="cache_hit_ratio",
                condition="lt",
                threshold=KPI_DEFINITIONS["cache_hit_ratio"]["critical_threshold"],
                severity=AlertSeverity.WARNING,
                duration=timedelta(minutes=5),
            )
        )

        # TWS connection success rate alert
        self.rules.append(
            AlertRule(
                name="tws_connection_failure",
                description="TWS connection success rate below threshold",
                metric_name="tws_connection_success_rate",
                condition="lt",
                threshold=KPI_DEFINITIONS["tws_connection_success_rate"]["critical_threshold"],
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
            )
        )

    async def evaluate_rules(self, metrics: dict[str, Any]) -> list[Alert]:
        """
        Evaluate all alert rules against current metrics and generate alerts
        """
        self.metrics_snapshot = metrics
        new_alerts = []

        # First, check SLO compliance
        slo_compliance = validate_slo_compliance(metrics)

        # Process SLO compliance alerts
        for kpi_name, is_compliant in slo_compliance.items():
            if not is_compliant:
                # Find corresponding alert rule, or create one if needed
                rule = next(
                    (
                        r
                        for r in self.rules
                        if r.metric_name == kpi_name and r.name.startswith("slo_")
                    ),
                    None,
                )
                if rule:
                    metric_value = metrics.get(kpi_name, 0)
                    existing_alert = self._find_active_alert(rule.name, rule.metric_name)

                    if not existing_alert:
                        alert = self._create_alert(rule, metric_value, metrics.get("timestamp"))
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)

                        # Log the alert
                        runtime_metrics.tws_status_requests_failed.increment()

        # Process existing alert rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.metric_name in metrics:
                metric_value = metrics[rule.metric_name]

                # Check if condition is met
                condition_met = self._check_condition(metric_value, rule.condition, rule.threshold)

                if condition_met:
                    # Check if this alert is already active
                    existing_alert = self._find_active_alert(rule.name, rule.metric_name)

                    # If no active alert for this rule, create a new one
                    if not existing_alert:
                        alert = self._create_alert(rule, metric_value, metrics.get("timestamp"))
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)

                        # Log the alert
                        runtime_metrics.tws_status_requests_failed.increment()

        return new_alerts

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if a condition is met"""
        if condition == "gt":
            return value > threshold
        if condition == "lt":
            return value < threshold
        if condition == "eq":
            return value == threshold
        if condition == "ne":
            return value != threshold
        if condition == "ge":
            return value >= threshold
        if condition == "le":
            return value <= threshold
        return False

    def _find_active_alert(self, rule_name: str, metric_name: str) -> Alert | None:
        """Find an active alert for a specific rule and metric"""
        for alert in self.active_alerts:
            if (
                alert.rule_name == rule_name
                and alert.metric_name == metric_name
                and not alert.acknowledged
            ):
                return alert
        return None

    def _create_alert(
        self, rule: AlertRule, value: float, timestamp: datetime | None = None
    ) -> Alert:
        """Create a new alert from a rule and metric value"""
        import uuid

        alert_id = str(uuid.uuid4())
        timestamp = timestamp or datetime.now(timezone.utc)

        message = (
            f"Alert '{rule.name}' triggered: {rule.description}. "
            f"Value: {value}, Threshold: {rule.threshold}"
        )

        return Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=timestamp,
            value=value,
            threshold=rule.threshold,
            metric_name=rule.metric_name,
        )

    async def process_alerts(self, new_alerts: list[Alert]):
        """
        Process new alerts - send notifications and apply escalation policies
        """
        for alert in new_alerts:
            # Send notifications based on severity
            await self._notify_alert(alert)

            # Apply escalation policies
            await self._apply_escalation_policy(alert)

    async def _notify_alert(self, alert: Alert):
        """Send notifications for an alert"""
        # For now, send to Teams - in a real implementation would have multiple channels
        try:
            teams_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": (
                    "ff0000"
                    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    else "ffa500"
                ),
                "summary": f"Alert: {alert.severity.value.upper()}",
                "sections": [
                    {
                        "activityTitle": f"Alert: {alert.rule_name}",
                        "activitySubtitle": f"Severity: {alert.severity.value}",
                        "facts": [
                            {"name": "Message", "value": alert.message},
                            {"name": "Timestamp", "value": alert.timestamp.isoformat()},
                            {"name": "Current Value", "value": str(alert.value)},
                            {"name": "Threshold", "value": str(alert.threshold)},
                        ],
                        "markdown": True,
                    }
                ],
            }

            teams_integration = get_teams_integration()
            notification = TeamsNotification(
                title="Alerting System",
                message=json.dumps(teams_message),
                severity="warning",
            )
            await teams_integration.send_notification(notification)
        except Exception as e:
            alerting_logger.error(
                "teams_notification_failed",
                alert_id=alert.id,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _apply_escalation_policy(self, alert: Alert):
        """Apply escalation policy for an alert"""
        policy_name = f"{alert.severity.value}_escalation"
        if policy_name in self.escalation_policies:
            for handler in self.escalation_policies[policy_name]:
                try:
                    await handler(alert)
                except Exception as e:
                    alerting_logger.error(
                        "escalation_handler_failed",
                        alert_id=alert.id,
                        escalation_policy=policy_name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)

                # Move to history
                self.active_alerts.remove(alert)
                self.alert_history.append(alert)
                return True
        return False

    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str):
        """Remove an alert rule by name"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]

    def get_active_alerts(self) -> list[Alert]:
        """Get all active (non-acknowledged) alerts"""
        return [alert for alert in self.active_alerts if not alert.acknowledged]

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history, limited to the specified number"""
        return self.alert_history[-limit:]

    def add_escalation_policy(self, severity: AlertSeverity, handler: Callable):
        """Add an escalation policy for a specific severity level"""
        policy_name = f"{severity.value}_escalation"
        if policy_name not in self.escalation_policies:
            self.escalation_policies[policy_name] = []
        self.escalation_policies[policy_name].append(handler)


# Global alerting system instance
alerting_system = AlertingSystem()


async def get_alerting_system() -> AlertingSystem:
    """
    Get the global alerting system instance
    """
    return alerting_system
