"""Shared types and interfaces for compliance modules."""

from enum import Enum
from typing import Any, Protocol


class SOC2TrustServiceCriteria(Enum):
    """SOC 2 Trust Service Criteria."""

    SECURITY = "security"  # Protects against unauthorized access
    AVAILABILITY = "availability"  # Systems available for operation
    PROCESSING_INTEGRITY = "processing_integrity"  # System processing complete, accurate, timely
    CONFIDENTIALITY = "confidentiality"  # Information designated confidential protected
    PRIVACY = "privacy"  # Personal information collected, used, retained appropriately


class SOC2ComplianceManagerProtocol(Protocol):
    """Protocol for SOC2 compliance manager."""

    def generate_report(self, report_type: str, **kwargs) -> dict[str, Any]:
        """Generate a compliance report."""
        ...

    def get_compliance_status(self) -> dict[str, Any]:
        """Get current compliance status."""
        ...

    def update_compliance_record(self, record_id: str, data: dict[str, Any]) -> bool:
        """Update a compliance record."""
        ...


class SOC2ComplianceManager:
    """
    Shared SOC 2 Type II compliance management class.

    This is a shared version used to break circular dependencies.
    """

    def __init__(self):
        # Basic initialization - full implementation in soc2_compliance.py
        self.config = None
        self.controls = {}
        self.control_tests = {}

    def generate_report(self, report_type: str, **kwargs) -> dict[str, Any]:
        """Generate a compliance report."""
        # This will be overridden by the full implementation
        return {"report_type": report_type, "status": "placeholder"}

    def get_compliance_status(self) -> dict[str, Any]:
        """Get current compliance status."""
        # This will be overridden by the full implementation
        return {"status": "placeholder"}

    def update_compliance_record(self, record_id: str, data: dict[str, Any]) -> bool:
        """Update a compliance record."""
        # This will be overridden by the full implementation
        return True
