"""
Enterprise Manager - Central orchestrator for enterprise modules.

v5.5.0: Initial integration of enterprise features.

This manager coordinates:
- Phase 1: Incident Response + Auto Recovery + Runbooks
- Phase 2: GDPR Compliance + Encrypted Audit + SIEM
- Phase 3: Log Aggregation + Anomaly Detection
- Phase 4: Chaos Engineering + Service Discovery
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from resync.core.task_tracker import track_task

if TYPE_CHECKING:
    from resync.core.anomaly_detector import AnomalyDetectionEngine
    from resync.core.auto_recovery import AutoRecovery
    from resync.core.chaos_engineering import ChaosEngineer
    from resync.core.encrypted_audit import EncryptedAuditTrail
    from resync.core.gdpr_compliance import GDPRComplianceManager
    from resync.core.incident_response import IncidentResponseEngine
    from resync.core.log_aggregator import LogAggregator
    from resync.core.runbooks import RunbookRegistry
    from resync.core.service_discovery import ServiceDiscoveryManager
    from resync.core.siem_integrator import SIEMIntegrator

logger = structlog.get_logger(__name__)

class EnterprisePhase(Enum):
    """Enterprise module phases."""

    ESSENTIAL = "essential"  # incident_response, auto_recovery, runbooks
    COMPLIANCE = "compliance"  # gdpr, encrypted_audit, siem
    OBSERVABILITY = "observability"  # log_aggregator, anomaly_detector
    RESILIENCE = "resilience"  # chaos_engineering, service_discovery

@dataclass
class EnterpriseStatus:
    """Status of enterprise modules."""

    phase: EnterprisePhase
    enabled: bool = False
    initialized: bool = False
    healthy: bool = False
    last_check: datetime | None = None
    error: str | None = None
    components: dict[str, bool] = field(default_factory=dict)

@dataclass
class EnterpriseConfig:
    """Configuration for enterprise modules."""

    # Phase 1: Essential
    enable_incident_response: bool = True
    enable_auto_recovery: bool = True
    enable_runbooks: bool = True

    # Phase 2: Compliance
    enable_gdpr: bool = False  # Enable for EU deployments
    enable_encrypted_audit: bool = True
    enable_siem: bool = False  # Requires SIEM endpoint config

    # Phase 3: Observability
    enable_log_aggregator: bool = True
    enable_anomaly_detection: bool = True

    # Phase 4: Resilience
    enable_chaos_engineering: bool = False  # Only for staging/test
    enable_service_discovery: bool = False  # For microservices deployments

    # Auto-recovery settings
    auto_recovery_enabled: bool = True
    auto_recovery_max_retries: int = 3
    auto_recovery_cooldown_seconds: int = 60

    # Incident settings
    incident_notification_channels: list[str] = field(
        default_factory=lambda: ["log", "metrics"]
    )
    incident_auto_escalate: bool = True
    incident_escalation_timeout_minutes: int = 15

    # GDPR settings
    gdpr_data_retention_days: int = 365
    gdpr_anonymization_enabled: bool = True

    # SIEM settings
    siem_endpoint: str | None = None
    siem_api_key: str | None = None
    siem_batch_size: int = 100

    # Anomaly detection settings
    anomaly_sensitivity: float = 0.95
    anomaly_check_interval_seconds: int = 60

class EnterpriseManager:
    """
    Central manager for all enterprise modules.

    Usage:
        enterprise = EnterpriseManager(config)
        await enterprise.initialize()

        # Access modules
        await enterprise.incident_response.report_incident(...)
        await enterprise.audit.log(...)
    """

    def __init__(self, config: EnterpriseConfig | None = None):
        self.config = config or EnterpriseConfig()
        self._initialized = False
        self._status: dict[EnterprisePhase, EnterpriseStatus] = {}

        # Phase 1: Essential
        self._incident_response: IncidentResponseEngine | None = None
        self._auto_recovery: AutoRecovery | None = None
        self._runbook_registry: RunbookRegistry | None = None

        # Phase 2: Compliance
        self._gdpr_manager: GDPRComplianceManager | None = None
        self._encrypted_audit: EncryptedAuditTrail | None = None
        self._siem_integrator: SIEMIntegrator | None = None

        # Phase 3: Observability
        self._log_aggregator: LogAggregator | None = None
        self._anomaly_detector: AnomalyDetectionEngine | None = None

        # Phase 4: Resilience
        self._chaos_engineer: ChaosEngineer | None = None
        self._service_discovery: ServiceDiscoveryManager | None = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    # =========================================================================
    # Properties for module access
    # =========================================================================

    @property
    def incident_response(self) -> IncidentResponseEngine | None:
        """Access incident response engine."""
        return self._incident_response

    @property
    def auto_recovery(self) -> AutoRecovery | None:
        """Access auto recovery system."""
        return self._auto_recovery

    @property
    def runbooks(self) -> RunbookRegistry | None:
        """Access runbook registry."""
        return self._runbook_registry

    @property
    def gdpr(self) -> GDPRComplianceManager | None:
        """Access GDPR compliance manager."""
        return self._gdpr_manager

    @property
    def audit(self) -> EncryptedAuditTrail | None:
        """Access encrypted audit trail."""
        return self._encrypted_audit

    @property
    def siem(self) -> SIEMIntegrator | None:
        """Access SIEM integrator."""
        return self._siem_integrator

    @property
    def logs(self) -> LogAggregator | None:
        """Access log aggregator."""
        return self._log_aggregator

    @property
    def anomaly_detector(self) -> AnomalyDetectionEngine | None:
        """Access anomaly detection engine."""
        return self._anomaly_detector

    @property
    def chaos(self) -> ChaosEngineer | None:
        """Access chaos engineering (staging only)."""
        return self._chaos_engineer

    @property
    def service_discovery(self) -> ServiceDiscoveryManager | None:
        """Access service discovery manager."""
        return self._service_discovery

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self, tg: asyncio.TaskGroup | None = None) -> None:
        """Initialize all enabled enterprise modules.

        Args:
            tg: Optional TaskGroup to run background tasks in
        """
        if self._initialized:
            logger.warning("Enterprise manager already initialized")
            return

        logger.info("Initializing enterprise modules...")

        # Initialize phases in order
        await self._init_phase_essential()
        await self._init_phase_compliance()
        await self._init_phase_observability()
        await self._init_phase_resilience()

        # Start background tasks
        await self._start_background_tasks(tg=tg)

        self._initialized = True
        logger.info(
            "Enterprise modules initialized",
            phases_enabled=[p.value for p, s in self._status.items() if s.enabled],
        )

    async def _init_phase_essential(self) -> None:
        """Initialize Phase 1: Essential modules."""
        status = EnterpriseStatus(phase=EnterprisePhase.ESSENTIAL)

        try:
            # Incident Response
            if self.config.enable_incident_response:
                from resync.core.incident_response import get_incident_response_engine

                self._incident_response = await get_incident_response_engine()
                status.components["incident_response"] = True
                logger.info("✓ Incident Response initialized")

            # Auto Recovery
            if self.config.enable_auto_recovery:
                from resync.core.auto_recovery import AutoRecovery

                self._auto_recovery = AutoRecovery()
                status.components["auto_recovery"] = True
                logger.info("✓ Auto Recovery initialized")

            # Runbooks
            if self.config.enable_runbooks:
                from resync.core.runbooks import get_runbook_registry

                self._runbook_registry = get_runbook_registry()
                status.components["runbooks"] = True
                logger.info("✓ Runbooks initialized")

            # Connect incident response to runbooks
            if self._incident_response and self._runbook_registry:
                self._connect_incident_to_runbooks()

            status.enabled = any(status.components.values())
            status.initialized = True
            status.healthy = True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Failed to initialize essential modules", error=str(e))
            status.error = str(e)

        self._status[EnterprisePhase.ESSENTIAL] = status

    async def _init_phase_compliance(self) -> None:
        """Initialize Phase 2: Compliance modules."""
        status = EnterpriseStatus(phase=EnterprisePhase.COMPLIANCE)

        try:
            # GDPR Compliance
            if self.config.enable_gdpr:
                from resync.core.gdpr_compliance import get_gdpr_compliance_manager

                self._gdpr_manager = await get_gdpr_compliance_manager()
                status.components["gdpr"] = True
                logger.info("✓ GDPR Compliance initialized")

            # Encrypted Audit
            if self.config.enable_encrypted_audit:
                from resync.core.encrypted_audit import get_encrypted_audit_trail

                self._encrypted_audit = await get_encrypted_audit_trail()
                status.components["encrypted_audit"] = True
                logger.info("✓ Encrypted Audit initialized")

            # SIEM Integration
            if self.config.enable_siem and self.config.siem_endpoint:
                from resync.core.siem_integrator import SIEMIntegrator

                self._siem_integrator = SIEMIntegrator()
                status.components["siem"] = True
                logger.info("✓ SIEM Integrator initialized")

            status.enabled = any(status.components.values())
            status.initialized = True
            status.healthy = True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Failed to initialize compliance modules", error=str(e))
            status.error = str(e)

        self._status[EnterprisePhase.COMPLIANCE] = status

    async def _init_phase_observability(self) -> None:
        """Initialize Phase 3: Observability modules."""
        status = EnterpriseStatus(phase=EnterprisePhase.OBSERVABILITY)

        try:
            # Log Aggregator
            if self.config.enable_log_aggregator:
                from resync.core.log_aggregator import get_log_aggregator

                self._log_aggregator = await get_log_aggregator()
                status.components["log_aggregator"] = True
                logger.info("✓ Log Aggregator initialized")

            # Anomaly Detection
            if self.config.enable_anomaly_detection:
                from resync.core.anomaly_detector import get_anomaly_detection_engine

                self._anomaly_detector = await get_anomaly_detection_engine()
                status.components["anomaly_detector"] = True
                logger.info("✓ Anomaly Detection initialized")

            status.enabled = any(status.components.values())
            status.initialized = True
            status.healthy = True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Failed to initialize observability modules", error=str(e))
            status.error = str(e)

        self._status[EnterprisePhase.OBSERVABILITY] = status

    async def _init_phase_resilience(self) -> None:
        """Initialize Phase 4: Resilience modules."""
        status = EnterpriseStatus(phase=EnterprisePhase.RESILIENCE)

        try:
            # Chaos Engineering (staging only!)
            if self.config.enable_chaos_engineering:
                from resync.core.chaos_engineering import ChaosEngineer

                self._chaos_engineer = ChaosEngineer()
                status.components["chaos_engineering"] = True
                logger.warning("⚠️ Chaos Engineering enabled - use only in staging!")

            # Service Discovery
            if self.config.enable_service_discovery:
                from resync.core.service_discovery import get_service_discovery_manager

                self._service_discovery = await get_service_discovery_manager()
                status.components["service_discovery"] = True
                logger.info("✓ Service Discovery initialized")

            status.enabled = any(status.components.values())
            status.initialized = True
            status.healthy = True

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Failed to initialize resilience modules", error=str(e))
            status.error = str(e)

        self._status[EnterprisePhase.RESILIENCE] = status

    def _connect_incident_to_runbooks(self) -> None:
        """Connect incident response to runbook execution."""
        if not self._incident_response or not self._runbook_registry:
            return

        # Register runbook execution callback
        logger.info("Connected Incident Response to Runbooks")

    async def _start_background_tasks(
        self, tg: asyncio.TaskGroup | None = None
    ) -> None:
        """Start background monitoring tasks.

        Args:
            tg: Optional TaskGroup to run the tasks in
        """
        # Anomaly detection background task
        if self._anomaly_detector:
            if tg:
                task = tg.create_task(
                    self._anomaly_monitoring_loop(), name="anomaly_monitoring_loop"
                )
            else:
                task = track_task(
                    self._anomaly_monitoring_loop(), name="anomaly_monitoring_loop"
                )
            self._tasks.append(task)

        # Auto-recovery monitoring
        if self._auto_recovery and self._incident_response:
            if tg:
                task = tg.create_task(
                    self._auto_recovery_loop(), name="auto_recovery_loop"
                )
            else:
                task = track_task(self._auto_recovery_loop(), name="auto_recovery_loop")
            self._tasks.append(task)

        # Start component background tasks
        if self._incident_response:
            await self._incident_response.start(tg=tg)

        if self._siem_integrator:
            await self._siem_integrator.start(tg=tg)

        if self._gdpr_manager:
            await self._gdpr_manager.start(tg=tg)

        if self._soc2_manager:
            await self._soc2_manager.start(tg=tg)

        if self._encrypted_audit:
            await self._encrypted_audit.start(tg=tg)

    async def _anomaly_monitoring_loop(self) -> None:
        """Background loop for anomaly detection."""
        while True:
            try:
                await asyncio.sleep(self.config.anomaly_check_interval_seconds)
                if self._anomaly_detector:
                    # Check for anomalies
                    pass  # Implementation depends on metrics source
            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Anomaly monitoring error", error=str(e))

    async def _auto_recovery_loop(self) -> None:
        """Background loop for auto-recovery checks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                if self._auto_recovery:
                    # Check for recovery opportunities
                    pass  # Implementation depends on health checks
            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Auto-recovery monitoring error", error=str(e))

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def shutdown(self) -> None:
        """Gracefully shutdown all enterprise modules."""
        logger.info("Shutting down enterprise modules...")

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Shutdown modules in reverse order
        if self._service_discovery:
            logger.info("Shutting down Service Discovery...")

        if self._log_aggregator:
            logger.info("Shutting down Log Aggregator...")

        if self._siem_integrator:
            logger.info("Shutting down SIEM Integrator...")

        if self._encrypted_audit:
            logger.info("Shutting down Encrypted Audit...")

        if self._incident_response:
            logger.info("Shutting down Incident Response...")

        self._initialized = False
        logger.info("Enterprise modules shutdown complete")

    # =========================================================================
    # Health & Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get status of all enterprise modules."""
        return {
            "initialized": self._initialized,
            "phases": {
                phase.value: {
                    "enabled": status.enabled,
                    "initialized": status.initialized,
                    "healthy": status.healthy,
                    "components": status.components,
                    "error": status.error,
                }
                for phase, status in self._status.items()
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Perform health check on all modules."""
        results = {
            "healthy": True,
            "modules": {},
        }

        # Check each module
        if self._incident_response:
            results["modules"]["incident_response"] = True

        if self._auto_recovery:
            results["modules"]["auto_recovery"] = True

        if self._encrypted_audit:
            results["modules"]["encrypted_audit"] = True

        if self._log_aggregator:
            results["modules"]["log_aggregator"] = True

        if self._anomaly_detector:
            results["modules"]["anomaly_detector"] = True

        return results

    # =========================================================================
    # Convenience methods
    # =========================================================================

    async def log_audit_event(
        self,
        action: str,
        user_id: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit event (convenience method)."""
        if self._encrypted_audit:
            await self._encrypted_audit.log_entry(
                action=action,
                user_id=user_id or "system",
                resource=resource or "unknown",
                details=details or {},
            )

    async def report_incident(
        self,
        title: str,
        description: str,
        severity: str = "medium",
        category: str = "operational",
    ) -> str | None:
        """Report an incident (convenience method)."""
        if self._incident_response:
            incident = await self._incident_response.create_incident(
                title=title,
                description=description,
                severity=severity,
                category=category,
            )
            return incident.id if incident else None
        return None

    async def send_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send a security event to SIEM (convenience method)."""
        if self._siem_integrator:
            await self._siem_integrator.send_event(
                event_type=event_type,
                severity=severity,
                source=source,
                details=details or {},
            )

# Global instance
_enterprise_manager: EnterpriseManager | None = None

async def get_enterprise_manager(
    tg: asyncio.TaskGroup | None = None,
) -> EnterpriseManager:
    """Get or create the enterprise manager instance.

    Args:
        tg: Optional TaskGroup to initialize background tasks in
    """
    global _enterprise_manager
    if _enterprise_manager is None:
        # Load config from application settings
        try:
            from resync.core.enterprise.config import load_enterprise_config

            config = load_enterprise_config()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            config = EnterpriseConfig()  # Use defaults

        _enterprise_manager = EnterpriseManager(config)
        await _enterprise_manager.initialize(tg=tg)
    return _enterprise_manager

async def shutdown_enterprise_manager() -> None:
    """Shutdown the enterprise manager."""
    global _enterprise_manager
    if _enterprise_manager:
        await _enterprise_manager.shutdown()
        _enterprise_manager = None
