"""
Enterprise Module - Production-ready enterprise features.

v5.5.0: Initial integration.

This module provides:
- Incident Response with automatic runbook execution
- Auto-Recovery for self-healing
- GDPR Compliance for EU deployments
- Encrypted Audit trails
- SIEM Integration for security
- Log Aggregation and Anomaly Detection
- Chaos Engineering for resilience testing
- Service Discovery for microservices

Usage:
    from resync.core.enterprise import get_enterprise_manager

    enterprise = await get_enterprise_manager()

    # Report incidents
    await enterprise.report_incident("Database slow", "High latency detected")

    # Log audit events
    await enterprise.log_audit_event("user_login", user_id="123")

    # Send security events
    await enterprise.send_security_event("auth_failure", "high", "api")
"""

from .config import load_enterprise_config
from .manager import (
    EnterpriseConfig,
    EnterpriseManager,
    EnterprisePhase,
    EnterpriseStatus,
    get_enterprise_manager,
    shutdown_enterprise_manager,
)

__all__ = [
    "EnterpriseConfig",
    "EnterpriseManager",
    "EnterprisePhase",
    "EnterpriseStatus",
    "get_enterprise_manager",
    "load_enterprise_config",
    "shutdown_enterprise_manager",
]
