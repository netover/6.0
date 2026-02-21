"""
Automated Incident Response System.

This module provides intelligent incident response capabilities including:
- Automatic incident detection and classification
- Severity assessment and impact analysis
- Automated response actions (containment, eradication, recovery)
- Response playbooks and escalation procedures
- Integration with notification systems
- Post-incident analysis and reporting
- Continuous learning and improvement
"""

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Numeric severity level for correct ordering comparisons."""
        _levels = {
            "informational": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return _levels[self.value]

    def __ge__(self, other: "IncidentSeverity") -> bool:
        if not isinstance(other, IncidentSeverity):
            return NotImplemented
        return self.level >= other.level

    def __gt__(self, other: "IncidentSeverity") -> bool:
        if not isinstance(other, IncidentSeverity):
            return NotImplemented
        return self.level > other.level

    def __le__(self, other: "IncidentSeverity") -> bool:
        if not isinstance(other, IncidentSeverity):
            return NotImplemented
        return self.level <= other.level

    def __lt__(self, other: "IncidentSeverity") -> bool:
        if not isinstance(other, IncidentSeverity):
            return NotImplemented
        return self.level < other.level


class IncidentStatus(Enum):
    """Incident lifecycle status."""

    DETECTED = "detected"
    TRIAGED = "triaged"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    LESSONS_LEARNED = "lessons_learned"
    CLOSED = "closed"


class IncidentCategory(Enum):
    """Incident categories for classification."""

    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"
    CONFIGURATION_ERROR = "configuration_error"
    THIRD_PARTY_COMPROMISE = "third_party_compromise"
    PHYSICAL_SECURITY = "physical_security"
    OTHER = "other"


@dataclass
class Incident:
    """Security incident record."""

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    category: IncidentCategory
    status: IncidentStatus = IncidentStatus.DETECTED

    # Timeline
    detected_at: float = field(default_factory=time.time)
    triaged_at: float | None = None
    contained_at: float | None = None
    eradicated_at: float | None = None
    recovered_at: float | None = None
    closed_at: float | None = None

    # Impact assessment
    affected_users: int = 0
    affected_systems: int = 0
    data_exposed: str = ""  # Description of exposed data
    business_impact: str = ""  # Financial, operational impact

    # Response tracking
    assigned_to: str | None = None
    response_actions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    # Communication
    notifications_sent: list[dict[str, Any]] = field(default_factory=list)
    stakeholders_notified: set[str] = field(default_factory=set)

    # Metadata
    detection_source: str = ""
    correlation_id: str | None = None
    tags: set[str] = field(default_factory=set)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get incident duration in seconds."""
        end_time = self.closed_at or time.time()
        return end_time - self.detected_at

    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.status not in [
            IncidentStatus.CLOSED,
            IncidentStatus.LESSONS_LEARNED,
        ]

    @property
    def response_effectiveness(self) -> float:
        """Calculate response effectiveness score (0-100)."""
        if not self.response_actions:
            return 0.0

        # Score based on actions taken and timeline
        base_score = 50.0

        # Bonus for quick response
        if (
            self.contained_at and (self.contained_at - self.detected_at) < 3600
        ):  # < 1 hour
            base_score += 20

        # Bonus for comprehensive response
        action_types = {action["type"] for action in self.response_actions}
        if "isolate" in action_types:
            base_score += 10
        if "block" in action_types:
            base_score += 10
        if "notify" in action_types:
            base_score += 10

        # Penalty for high impact
        if self.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            base_score -= 10

        return max(0.0, min(100.0, base_score))

    def to_dict(self) -> dict[str, Any]:
        """Convert incident to dictionary."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "status": self.status.value,
            "detected_at": self.detected_at,
            "triaged_at": self.triaged_at,
            "contained_at": self.contained_at,
            "eradicated_at": self.eradicated_at,
            "recovered_at": self.recovered_at,
            "closed_at": self.closed_at,
            "affected_users": self.affected_users,
            "affected_systems": self.affected_systems,
            "data_exposed": self.data_exposed,
            "business_impact": self.business_impact,
            "assigned_to": self.assigned_to,
            "response_actions": self.response_actions,
            "evidence": self.evidence,
            "notifications_sent": self.notifications_sent,
            "stakeholders_notified": list(self.stakeholders_notified),
            "detection_source": self.detection_source,
            "correlation_id": self.correlation_id,
            "tags": list(self.tags),
            "custom_fields": self.custom_fields,
        }


@dataclass
class ResponseAction:
    """Automated response action."""

    action_id: str
    name: str
    description: str
    action_type: str  # isolate, block, notify, remediate, etc.
    severity_threshold: IncidentSeverity
    automated: bool = True
    requires_approval: bool = False

    # Execution details
    target_systems: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""
    rollback_procedure: str = ""

    def can_execute(self, incident: Incident) -> bool:
        """Check if action can be executed for incident."""
        return incident.severity >= self.severity_threshold


@dataclass
class ResponsePlaybook:
    """Incident response playbook."""

    playbook_id: str
    name: str
    description: str
    category: IncidentCategory
    severity_range: tuple[IncidentSeverity, IncidentSeverity]

    # Playbook steps
    triage_steps: list[str] = field(default_factory=list)
    containment_steps: list[str] = field(default_factory=list)
    eradication_steps: list[str] = field(default_factory=list)
    recovery_steps: list[str] = field(default_factory=list)
    lessons_learned_steps: list[str] = field(default_factory=list)

    # Automated actions
    automated_actions: list[ResponseAction] = field(default_factory=list)

    # Escalation rules
    escalation_threshold: timedelta = field(default_factory=lambda: timedelta(hours=1))
    notify_stakeholders: list[str] = field(default_factory=list)

    def applies_to(self, incident: Incident) -> bool:
        """Check if playbook applies to incident."""
        severity_ok = (
            self.severity_range[0] <= incident.severity <= self.severity_range[1]
        )
        category_ok = self.category == incident.category
        return severity_ok and category_ok

    def get_next_steps(self, incident: Incident) -> list[str]:
        """Get next steps based on incident status."""
        if incident.status == IncidentStatus.DETECTED:
            return self.triage_steps
        if incident.status == IncidentStatus.TRIAGED:
            return self.containment_steps
        if incident.status == IncidentStatus.CONTAINMENT:
            return self.eradication_steps
        if incident.status == IncidentStatus.ERADICATION:
            return self.recovery_steps
        if incident.status == IncidentStatus.RECOVERY:
            return self.lessons_learned_steps
        return []


@dataclass
class IncidentResponseConfig:
    """Configuration for incident response system."""

    # Response timing
    triage_timeout_minutes: int = 30
    containment_timeout_hours: int = 4
    eradication_timeout_hours: int = 24
    recovery_timeout_days: int = 7

    # Automation settings
    enable_automated_response: bool = True
    require_manual_approval: bool = False
    max_automated_actions: int = 5

    # Notification settings
    notify_on_detection: bool = True
    notify_on_containment: bool = True
    escalation_intervals_minutes: list[int] = field(
        default_factory=lambda: [60, 240, 1440]
    )

    # Stakeholder management
    critical_stakeholders: list[str] = field(default_factory=list)
    notification_channels: list[str] = field(default_factory=lambda: ["email", "slack"])

    # Learning and improvement
    enable_learning: bool = True
    review_incidents_after_days: int = 30


class IncidentDetector:
    """Component for detecting security incidents."""

    def __init__(self, config: IncidentResponseConfig):
        self.config = config
        self.detection_rules: list[dict[str, Any]] = []

        # Initialize detection rules
        self._initialize_detection_rules()

    def _initialize_detection_rules(self) -> None:
        """Initialize incident detection rules."""
        self.detection_rules = [
            {
                "name": "multiple_failed_logins",
                "category": IncidentCategory.UNAUTHORIZED_ACCESS,
                "severity": IncidentSeverity.MEDIUM,
                "condition": lambda events: (
                    len([e for e in events if e.get("event_type") == "failed_login"])
                    >= 10
                    and len(
                        set(e.get("ip_address") for e in events if e.get("ip_address"))
                    )
                    <= 3
                ),
                "time_window": 300,  # 5 minutes
            },
            {
                "name": "anomaly_detected",
                "category": IncidentCategory.UNAUTHORIZED_ACCESS,
                "severity": IncidentSeverity.HIGH,
                "condition": lambda events: (
                    any(
                        e.get("event_type") == "anomaly" and e.get("severity") == "high"
                        for e in events
                    )
                ),
                "time_window": 60,  # 1 minute
            },
            {
                "name": "data_breach_suspected",
                "category": IncidentCategory.DATA_BREACH,
                "severity": IncidentSeverity.CRITICAL,
                "condition": lambda events: (
                    any(
                        e.get("event_type") == "unauthorized_data_access"
                        for e in events
                    )
                    and any(e.get("severity") == "critical" for e in events)
                ),
                "time_window": 120,  # 2 minutes
            },
            {
                "name": "dos_attack",
                "category": IncidentCategory.DENIAL_OF_SERVICE,
                "severity": IncidentSeverity.HIGH,
                "condition": lambda events: (
                    len(
                        [
                            e
                            for e in events
                            if e.get("event_type") == "rate_limit_exceeded"
                        ]
                    )
                    >= 50
                    and len(
                        set(e.get("ip_address") for e in events if e.get("ip_address"))
                    )
                    <= 5
                ),
                "time_window": 300,  # 5 minutes
            },
        ]

    def detect_incident(self, events: list[dict[str, Any]]) -> Incident | None:
        """Detect incidents from security events."""
        for rule in self.detection_rules:
            if rule["condition"](events):
                # Create incident
                incident_id = f"inc_{int(time.time())}_{rule['name']}_{hash(str(events)[:50]) % 10000}"

                incident = Incident(
                    incident_id=incident_id,
                    title=f"Security Incident: {rule['name'].replace('_', ' ').title()}",
                    description=f"Detected {rule['name']} based on security event patterns",
                    severity=rule["severity"],
                    category=rule["category"],
                    detection_source="automated_detection",
                    evidence=[{"events": events, "rule": rule["name"]}],
                    tags={"automated", "detected"},
                )

                logger.warning(
                    "incident_detected",
                    incident_id=incident_id,
                    severity=incident.severity.value,
                    category=incident.category.value,
                )

                return incident

        return None


class IncidentResponder:
    """Component for executing automated incident response."""

    def __init__(self, config: IncidentResponseConfig):
        self.config = config
        self.response_actions: dict[str, ResponseAction] = {}

        # Initialize response actions
        self._initialize_response_actions()

    def _initialize_response_actions(self) -> None:
        """Initialize automated response actions."""
        actions = [
            ResponseAction(
                action_id="isolate_ip",
                name="Isolate Suspicious IP",
                description="Block IP address at network level",
                action_type="isolate",
                severity_threshold=IncidentSeverity.HIGH,
                automated=True,
                target_systems=["firewall", "load_balancer"],
                parameters={"block_duration_hours": 24},
                success_criteria="IP successfully blocked",
                rollback_procedure="Remove IP from block list",
            ),
            ResponseAction(
                action_id="disable_user",
                name="Disable User Account",
                description="Temporarily disable compromised user account",
                action_type="block",
                severity_threshold=IncidentSeverity.CRITICAL,
                automated=False,  # Requires approval
                requires_approval=True,
                target_systems=["identity_provider", "database"],
                parameters={"disable_duration_hours": 24},
                success_criteria="User account disabled",
                rollback_procedure="Re-enable user account",
            ),
            ResponseAction(
                action_id="notify_security_team",
                name="Notify Security Team",
                description="Send immediate notification to security team",
                action_type="notify",
                severity_threshold=IncidentSeverity.MEDIUM,
                automated=True,
                target_systems=["email", "slack", "sms"],
                parameters={
                    "urgency": "high",
                    "channels": ["security_team", "management"],
                },
                success_criteria="Notifications sent successfully",
            ),
            ResponseAction(
                action_id="enable_enhanced_monitoring",
                name="Enable Enhanced Monitoring",
                description="Increase monitoring level for affected systems",
                action_type="monitor",
                severity_threshold=IncidentSeverity.MEDIUM,
                automated=True,
                target_systems=["monitoring_system"],
                parameters={"monitoring_level": "enhanced", "duration_hours": 24},
                success_criteria="Monitoring level increased",
                rollback_procedure="Return to normal monitoring level",
            ),
            ResponseAction(
                action_id="backup_critical_data",
                name="Backup Critical Data",
                description="Create emergency backup of critical data",
                action_type="protect",
                severity_threshold=IncidentSeverity.HIGH,
                automated=True,
                target_systems=["backup_system"],
                parameters={"backup_type": "emergency", "include_audit_logs": True},
                success_criteria="Backup completed successfully",
            ),
        ]

        for action in actions:
            self.response_actions[action.action_id] = action

    async def execute_response(self, incident: Incident) -> list[dict[str, Any]]:
        """Execute automated response actions for incident."""
        executed_actions = []

        if not self.config.enable_automated_response:
            return executed_actions

        # Find applicable actions
        applicable_actions = [
            action
            for action in self.response_actions.values()
            if action.can_execute(incident) and action.automated
        ]

        # Limit number of automated actions
        applicable_actions = applicable_actions[: self.config.max_automated_actions]

        for action in applicable_actions:
            # Check if approval is required
            if action.requires_approval and self.config.require_manual_approval:
                # Skip automated execution, mark for manual review
                executed_actions.append(
                    {
                        "action_id": action.action_id,
                        "status": "pending_approval",
                        "timestamp": time.time(),
                        "reason": "Requires manual approval",
                    }
                )
                continue

            # Execute action
            try:
                result = self._execute_action(action, incident)

                executed_actions.append(
                    {
                        "action_id": action.action_id,
                        "status": "executed",
                        "timestamp": time.time(),
                        "result": result,
                    }
                )

                # Update incident
                incident.response_actions.append(
                    {
                        "action_id": action.action_id,
                        "timestamp": time.time(),
                        "result": result,
                        "automated": True,
                    }
                )

            except Exception as e:
                logger.error("Failed to execute action %s: %s", action.action_id, e)
                executed_actions.append(
                    {
                        "action_id": action.action_id,
                        "status": "failed",
                        "timestamp": time.time(),
                        "error": str(e),
                    }
                )

        return executed_actions

    def _execute_action(
        self, action: ResponseAction, incident: Incident
    ) -> dict[str, Any]:
        """Execute a specific response action."""
        # This is where you'd integrate with actual systems
        # For now, simulate execution

        if action.action_id == "notify_security_team":
            # Simulate notification
            result = {
                "notifications_sent": len(action.parameters.get("channels", [])),
                "channels": action.parameters.get("channels", []),
            }

        elif action.action_id == "isolate_ip":
            # Simulate IP blocking
            result = {
                "ips_blocked": 1,  # Would get from incident evidence
                "block_duration_hours": action.parameters.get("block_duration_hours"),
            }

        elif action.action_id == "enable_enhanced_monitoring":
            # Simulate monitoring increase
            result = {
                "monitoring_level": action.parameters.get("monitoring_level"),
                "duration_hours": action.parameters.get("duration_hours"),
            }

        else:
            # Generic success
            result = {"status": "simulated_execution"}

        logger.info(
            "response_action_executed",
            action_id=action.action_id,
            incident_id=incident.incident_id,
            result=result,
        )

        return result


class NotificationManager:
    """Component for managing incident notifications."""

    def __init__(self, config: IncidentResponseConfig):
        self.config = config

    async def notify_incident_detected(self, incident: Incident) -> None:
        """Send notifications for incident detection."""
        if not self.config.notify_on_detection:
            return

        message = self._format_incident_notification(incident, "DETECTED")

        self._send_notifications(message, ["security_team"], "incident_detected")

    async def notify_incident_contained(self, incident: Incident) -> None:
        """Send notifications for incident containment."""
        if not self.config.notify_on_containment:
            return

        message = self._format_incident_notification(incident, "CONTAINED")

        self._send_notifications(
            message, ["security_team", "management"], "incident_contained"
        )

    async def send_escalation_notification(
        self, incident: Incident, level: int
    ) -> None:
        """Send escalation notification."""
        message = self._format_incident_notification(
            incident, f"ESCALATION LEVEL {level}"
        )

        # Escalate to more stakeholders
        stakeholders = ["security_team", "management"]
        if level >= 2:
            stakeholders.append("executives")
        if level >= 3:
            stakeholders.extend(["legal", "compliance"])

        self._send_notifications(
            message, stakeholders, f"incident_escalation_level_{level}"
        )

    def _format_incident_notification(self, incident: Incident, event_type: str) -> str:
        """Format incident notification message."""
        return f"""
INCIDENT {event_type}: {incident.incident_id}

Title: {incident.title}
Severity: {incident.severity.value.upper()}
Category: {incident.category.value.replace("_", " ").title()}
Status: {incident.status.value.replace("_", " ").title()}

Description: {incident.description}

Detected: {datetime.fromtimestamp(incident.detected_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
Duration: {int(incident.duration)} seconds

Impact:
- Affected Users: {incident.affected_users}
- Affected Systems: {incident.affected_systems}
- Business Impact: {incident.business_impact or "Under assessment"}

Detection Source: {incident.detection_source}

Please review and take appropriate action.
        """.strip()

    def _send_notifications(
        self, message: str, stakeholders: list[str], notification_type: str
    ) -> None:
        """Send notifications via configured channels."""
        # This would integrate with actual notification systems
        # For now, just log

        for stakeholder in stakeholders:
            for channel in self.config.notification_channels:
                logger.info(
                    "notification_sent",
                    stakeholder=stakeholder,
                    channel=channel,
                    type=notification_type,
                    message_preview=message[:100] + "...",
                )


class IncidentResponseEngine:
    """
    Main incident response engine with automated detection and response.

    Features:
    - Automated incident detection and classification
    - Intelligent response action execution
    - Stakeholder notification and escalation
    - Response playbook management
    - Post-incident analysis and reporting
    - Continuous learning and improvement
    """

    def __init__(self, config: IncidentResponseConfig | None = None):
        self.config = config or IncidentResponseConfig()

        # Core components
        self.detector = IncidentDetector(self.config)
        self.responder = IncidentResponder(self.config)
        self.notifier = NotificationManager(self.config)

        # Incident management
        self.active_incidents: dict[str, Incident] = {}
        self.incident_history: list[Incident] = []
        self.response_playbooks: dict[str, ResponsePlaybook] = {}

        # Event buffering for detection
        self.event_buffer: deque = deque(maxlen=1000)
        self.detection_window = 600  # 10 minutes

        # Statistics
        self.incidents_detected = 0
        self.automated_responses = 0
        self.manual_interventions = 0
        self.average_response_time = 0.0

        # Background tasks
        self._detection_task: asyncio.Task | None = None
        self._escalation_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Initialize playbooks
        self._initialize_playbooks()

    def _initialize_playbooks(self) -> None:
        """Initialize incident response playbooks."""
        playbooks = [
            ResponsePlaybook(
                playbook_id="pb_unauthorized_access",
                name="Unauthorized Access Response",
                description="Response playbook for unauthorized access incidents",
                category=IncidentCategory.UNAUTHORIZED_ACCESS,
                severity_range=(IncidentSeverity.MEDIUM, IncidentSeverity.CRITICAL),
                triage_steps=[
                    "Verify incident details and impact",
                    "Determine affected systems and data",
                    "Assess attacker capabilities and intent",
                ],
                containment_steps=[
                    "Isolate affected systems from network",
                    "Revoke compromised credentials",
                    "Implement emergency access controls",
                ],
                eradication_steps=[
                    "Remove malicious code or backdoors",
                    "Patch exploited vulnerabilities",
                    "Clean compromised systems",
                ],
                recovery_steps=[
                    "Restore systems from clean backups",
                    "Validate system integrity",
                    "Monitor for re-compromise",
                ],
                automated_actions=[
                    ResponseAction(
                        "isolate_ip", "Isolate IP", "", "isolate", IncidentSeverity.HIGH
                    ),
                    ResponseAction(
                        "notify_security_team",
                        "Notify Team",
                        "",
                        "notify",
                        IncidentSeverity.MEDIUM,
                    ),
                ],
                notify_stakeholders=["security_team", "it_ops"],
            ),
            ResponsePlaybook(
                playbook_id="pb_data_breach",
                name="Data Breach Response",
                description="Response playbook for data breach incidents",
                category=IncidentCategory.DATA_BREACH,
                severity_range=(IncidentSeverity.HIGH, IncidentSeverity.CRITICAL),
                triage_steps=[
                    "Assess breach scope and data exposure",
                    "Notify legal and compliance teams",
                    "Determine notification requirements",
                ],
                containment_steps=[
                    "Stop data exfiltration",
                    "Isolate affected databases",
                    "Secure backup systems",
                ],
                eradication_steps=[
                    "Remove unauthorized access",
                    "Audit and patch systems",
                    "Enhance monitoring",
                ],
                recovery_steps=[
                    "Restore from clean backups",
                    "Validate data integrity",
                    "Implement additional controls",
                ],
                automated_actions=[
                    ResponseAction(
                        "backup_critical_data",
                        "Backup Data",
                        "",
                        "protect",
                        IncidentSeverity.HIGH,
                    ),
                    ResponseAction(
                        "notify_security_team",
                        "Notify Team",
                        "",
                        "notify",
                        IncidentSeverity.HIGH,
                    ),
                ],
                notify_stakeholders=["security_team", "legal", "executives"],
            ),
        ]

        for playbook in playbooks:
            self.response_playbooks[playbook.playbook_id] = playbook

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the incident response engine.

        Args:
            tg: Optional TaskGroup to run the background tasks in
        """
        if self._running:
            return

        self._running = True
        if tg:
            self._detection_task = tg.create_task(
                self._incident_detection_worker(), name="incident_detection_worker"
            )
            self._escalation_task = tg.create_task(
                self._escalation_worker(), name="escalation_worker"
            )
            self._cleanup_task = tg.create_task(
                self._cleanup_worker(), name="cleanup_worker"
            )
        else:
            self._detection_task = track_task(
                self._incident_detection_worker(), name="incident_detection_worker"
            )
            self._escalation_task = track_task(
                self._escalation_worker(), name="escalation_worker"
            )
            self._cleanup_task = track_task(self._cleanup_worker(), name="cleanup_worker")

        logger.info("Incident response engine started", method="task_group" if tg else "track_task")

    async def stop(self) -> None:
        """Stop the incident response engine."""
        if not self._running:
            return

        self._running = False

        for task in [self._detection_task, self._escalation_task, self._cleanup_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("Incident response engine stopped")

    async def process_security_event(self, event: dict[str, Any]) -> None:
        """Process a security event for incident detection."""
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = time.time()

        # Add to event buffer
        self.event_buffer.append(event)

        # Immediate check for critical events
        if event.get("severity") == "critical":
            await self._check_for_incident()

    async def create_incident_manually(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        category: IncidentCategory,
        detection_source: str = "manual_creation",
        **kwargs,
    ) -> str:
        """Create an incident manually."""
        incident_id = f"inc_manual_{int(time.time())}_{hash(title) % 10000}"

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            status=IncidentStatus.TRIAGED,  # Skip detection phase
            triaged_at=time.time(),
            detection_source=detection_source,
            **kwargs,
        )

        self.active_incidents[incident_id] = incident
        self.incidents_detected += 1

        # Start response process
        await self._start_incident_response(incident)

        logger.info("Manual incident created: %s", incident_id)
        return incident_id

    async def update_incident_status(
        self, incident_id: str, new_status: IncidentStatus, **kwargs
    ) -> bool:
        """Update incident status and execute appropriate actions."""
        if incident_id not in self.active_incidents:
            return False

        incident = self.active_incidents[incident_id]
        old_status = incident.status
        incident.status = new_status

        # Update timestamps
        if new_status == IncidentStatus.TRIAGED:
            incident.triaged_at = time.time()
        elif new_status == IncidentStatus.CONTAINMENT:
            incident.contained_at = time.time()
        elif new_status == IncidentStatus.ERADICATION:
            incident.eradicated_at = time.time()
        elif new_status == IncidentStatus.RECOVERY:
            incident.recovered_at = time.time()
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = time.time()
            # Move to history
            self.incident_history.append(incident)
            del self.active_incidents[incident_id]

        # Execute status-specific actions
        if new_status == IncidentStatus.CONTAINMENT:
            await self.notifier.notify_incident_contained(incident)
        elif new_status == IncidentStatus.CLOSED:
            self._generate_post_mortem(incident)

        logger.info(
            "incident_status_updated",
            incident_id=incident_id,
            old_status=old_status.value,
            new_status=new_status.value,
        )

        return True

    def get_incident(self, incident_id: str) -> Incident | None:
        """Get incident by ID."""
        return self.active_incidents.get(incident_id)

    def get_active_incidents(self) -> list[Incident]:
        """Get all active incidents."""
        return list(self.active_incidents.values())

    def get_incident_history(
        self, limit: int = 50, severity_filter: IncidentSeverity | None = None
    ) -> list[Incident]:
        """Get incident history with optional filtering."""
        incidents = self.incident_history[-limit:]

        if severity_filter:
            incidents = [i for i in incidents if i.severity == severity_filter]

        return incidents

    def get_response_metrics(self) -> dict[str, Any]:
        """Get incident response metrics."""
        total_incidents = len(self.incident_history) + len(self.active_incidents)
        if total_incidents == 0:
            return {"no_incidents": True}

        # Calculate metrics
        contained_incidents = [i for i in self.incident_history if i.contained_at]
        avg_containment_time = (
            sum(i.contained_at - i.detected_at for i in contained_incidents)
            / len(contained_incidents)
            if contained_incidents
            else 0
        )

        severity_distribution = {}
        for incident in self.incident_history + list(self.active_incidents.values()):
            severity_distribution[incident.severity.value] = (
                severity_distribution.get(incident.severity.value, 0) + 1
            )

        category_distribution = {}
        for incident in self.incident_history + list(self.active_incidents.values()):
            category_distribution[incident.category.value] = (
                category_distribution.get(incident.category.value, 0) + 1
            )

        return {
            "total_incidents": total_incidents,
            "active_incidents": len(self.active_incidents),
            "resolved_incidents": len(self.incident_history),
            "average_containment_time_hours": avg_containment_time / 3600,
            "severity_distribution": severity_distribution,
            "category_distribution": category_distribution,
            "automated_responses": self.automated_responses,
            "manual_interventions": self.manual_interventions,
            "response_effectiveness": sum(
                i.response_effectiveness for i in self.incident_history
            )
            / max(1, len(self.incident_history)),
        }

    async def _check_for_incident(self) -> None:
        """Check event buffer for incident patterns."""
        if not self.event_buffer:
            return

        # Get recent events within detection window
        cutoff_time = time.time() - self.detection_window
        recent_events = [
            e for e in self.event_buffer if e.get("timestamp", 0) > cutoff_time
        ]

        if len(recent_events) < 3:  # Need minimum events for pattern detection
            return

        # Check for incidents
        incident = self.detector.detect_incident(recent_events)
        if incident:
            self.active_incidents[incident.incident_id] = incident
            self.incidents_detected += 1

            # Start response process
            await self._start_incident_response(incident)

    async def _start_incident_response(self, incident: Incident) -> None:
        """Start the incident response process."""
        # Find applicable playbook
        applicable_playbook = None
        for playbook in self.response_playbooks.values():
            if playbook.applies_to(incident):
                applicable_playbook = playbook
                break

        if applicable_playbook:
            incident.tags.add(f"playbook_{applicable_playbook.playbook_id}")
            logger.info(
                f"Applied playbook {applicable_playbook.playbook_id} to incident {incident.incident_id}"
            )

        # Execute automated response
        if self.config.enable_automated_response:
            executed_actions = await self.responder.execute_response(incident)
            self.automated_responses += len(
                [a for a in executed_actions if a["status"] == "executed"]
            )

        # Send notifications
        await self.notifier.notify_incident_detected(incident)

        # Update status to triaged
        await self.update_incident_status(incident.incident_id, IncidentStatus.TRIAGED)

    def _generate_post_mortem(self, incident: Incident) -> None:
        """Generate post-mortem analysis for resolved incident."""
        if not self.config.enable_learning:
            return

        # This would generate detailed post-mortem reports
        # For now, just log key metrics

        analysis = {
            "incident_id": incident.incident_id,
            "duration_hours": incident.duration / 3600,
            "response_effectiveness": incident.response_effectiveness,
            "lessons_learned": [
                "Review automated response effectiveness",
                "Update detection rules if needed",
                "Improve stakeholder communication",
            ],
        }

        logger.info(
            "incident_post_mortem", incident_id=incident.incident_id, analysis=analysis
        )

    async def _incident_detection_worker(self) -> None:
        """Background worker for incident detection."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._check_for_incident()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Incident detection worker error: %s", e)

    async def _escalation_worker(self) -> None:
        """Background worker for incident escalation."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                current_time = time.time()
                for incident in list(self.active_incidents.values()):
                    if not incident.is_active:
                        continue

                    # Check escalation intervals
                    age_minutes = (current_time - incident.detected_at) / 60

                    for i, interval in enumerate(
                        self.config.escalation_intervals_minutes
                    ):
                        if age_minutes >= interval:
                            # Check if we already escalated at this level
                            escalation_tag = f"escalated_level_{i + 1}"
                            if escalation_tag not in incident.tags:
                                incident.tags.add(escalation_tag)
                                await self.notifier.send_escalation_notification(
                                    incident, i + 1
                                )
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Escalation worker error: %s", e)

    async def _cleanup_worker(self) -> None:
        """Background worker for incident cleanup."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Close old incidents
                current_time = time.time()
                cutoff_time = current_time - (
                    self.config.review_incidents_after_days * 24 * 3600
                )

                to_close = []
                for incident_id, incident in self.active_incidents.items():
                    if incident.detected_at < cutoff_time:
                        to_close.append(incident_id)

                for incident_id in to_close:
                    await self.update_incident_status(
                        incident_id, IncidentStatus.CLOSED
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup worker error: %s", e)


# Global incident response engine instance (lazy-initialized)
_incident_response_engine: IncidentResponseEngine | None = None


def _get_incident_response_engine_instance() -> IncidentResponseEngine:
    """Get or create the incident response engine (lazy init)."""
    global _incident_response_engine
    if _incident_response_engine is None:
        _incident_response_engine = IncidentResponseEngine()
    return _incident_response_engine


# Backward compatibility alias
incident_response_engine: IncidentResponseEngine | None = None  # type: ignore[assignment]


class IncidentResponse:
    """Basic incident response class for compatibility."""

    def __init__(self):
        self.engine = _get_incident_response_engine_instance()

    async def log_incident(self, incident_type: str, details: dict):
        """Log an incident."""
        return await self.engine.process_security_event(details)


async def get_incident_response_engine(tg: asyncio.TaskGroup | None = None) -> IncidentResponseEngine:
    """
    Get the global incident response engine instance.

    Args:
        tg: Optional TaskGroup to start the engine in
    """
    engine = _get_incident_response_engine_instance()
    if not engine._running:
        engine.start(tg=tg)
    return engine
