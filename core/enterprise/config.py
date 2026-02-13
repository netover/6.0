"""
Enterprise Configuration - Loads settings into EnterpriseConfig.

v5.5.0: Initial implementation.
"""

from resync.core.enterprise.manager import EnterpriseConfig
from resync.settings import get_settings


def load_enterprise_config() -> EnterpriseConfig:
    """Load enterprise configuration from application settings."""
    settings = get_settings()

    return EnterpriseConfig(
        # Phase 1: Essential
        enable_incident_response=settings.enterprise_enable_incident_response,
        enable_auto_recovery=settings.enterprise_enable_auto_recovery,
        enable_runbooks=settings.enterprise_enable_runbooks,
        # Phase 2: Compliance
        enable_gdpr=settings.enterprise_enable_gdpr,
        enable_encrypted_audit=settings.enterprise_enable_encrypted_audit,
        enable_siem=settings.enterprise_enable_siem,
        # Phase 3: Observability
        enable_log_aggregator=settings.enterprise_enable_log_aggregator,
        enable_anomaly_detection=settings.enterprise_enable_anomaly_detection,
        # Phase 4: Resilience
        enable_chaos_engineering=settings.enterprise_enable_chaos_engineering,
        enable_service_discovery=settings.enterprise_enable_service_discovery,
        # Auto-recovery settings
        auto_recovery_enabled=settings.enterprise_enable_auto_recovery,
        auto_recovery_max_retries=settings.enterprise_auto_recovery_max_retries,
        auto_recovery_cooldown_seconds=settings.enterprise_auto_recovery_cooldown,
        # Incident settings
        incident_auto_escalate=settings.enterprise_incident_auto_escalate,
        incident_escalation_timeout_minutes=settings.enterprise_incident_escalation_timeout,
        # GDPR settings
        gdpr_data_retention_days=settings.enterprise_gdpr_data_retention_days,
        gdpr_anonymization_enabled=settings.enterprise_gdpr_anonymization_enabled,
        # SIEM settings
        siem_endpoint=settings.enterprise_siem_endpoint,
        siem_api_key=(
            settings.enterprise_siem_api_key.get_secret_value()
            if settings.enterprise_siem_api_key
            else None
        ),
        # Anomaly detection settings
        anomaly_sensitivity=settings.enterprise_anomaly_sensitivity,
    )
