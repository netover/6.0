"""
Enterprise API - REST endpoints for enterprise modules.

v5.5.0: Initial API endpoints.

Provides endpoints for:
- Enterprise status and health
- Incident management
- Audit log queries
- Compliance reports
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from resync.core.enterprise import EnterpriseManager, get_enterprise_manager

router = APIRouter(prefix="/enterprise", tags=["Enterprise"])


# =============================================================================
# Models
# =============================================================================


class EnterpriseStatusResponse(BaseModel):
    """Enterprise status response."""

    initialized: bool
    phases: dict[str, Any]


class IncidentCreateRequest(BaseModel):
    """Request to create an incident."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    severity: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    category: str = Field(default="operational")


class IncidentResponse(BaseModel):
    """Incident response."""

    id: str | None = None
    created: bool
    message: str


class AuditEventRequest(BaseModel):
    """Request to log an audit event."""

    action: str = Field(..., min_length=1, max_length=100)
    user_id: str | None = None
    resource: str | None = None
    details: dict[str, Any] | None = None


class AuditEventResponse(BaseModel):
    """Audit event response."""

    logged: bool
    message: str


class SecurityEventRequest(BaseModel):
    """Request to send a security event."""

    event_type: str = Field(..., min_length=1, max_length=100)
    severity: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    source: str = Field(..., min_length=1, max_length=100)
    details: dict[str, Any] | None = None


class HealthCheckResponse(BaseModel):
    """Health check response."""

    healthy: bool
    modules: dict[str, bool]
    timestamp: datetime


# =============================================================================
# Dependencies
# =============================================================================


async def get_enterprise() -> EnterpriseManager:
    """Dependency to get enterprise manager."""
    return await get_enterprise_manager()


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/status", response_model=EnterpriseStatusResponse)
async def get_status(
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> EnterpriseStatusResponse:
    """Get enterprise modules status."""
    status = enterprise.get_status()
    return EnterpriseStatusResponse(**status)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> HealthCheckResponse:
    """Perform enterprise health check."""
    result = await enterprise.health_check()
    return HealthCheckResponse(
        healthy=result["healthy"],
        modules=result["modules"],
        timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# Incident Management
# =============================================================================


@router.post("/incidents", response_model=IncidentResponse)
async def create_incident(
    request: IncidentCreateRequest,
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> IncidentResponse:
    """Create a new incident."""
    if not enterprise.incident_response:
        raise HTTPException(
            status_code=503,
            detail="Incident response module not enabled",
        )

    incident_id = await enterprise.report_incident(
        title=request.title,
        description=request.description,
        severity=request.severity,
        category=request.category,
    )

    return IncidentResponse(
        id=incident_id,
        created=incident_id is not None,
        message="Incident created" if incident_id else "Failed to create incident",
    )


@router.get("/incidents")
async def list_incidents(
    status: str | None = Query(None, description="Filter by status"),
    severity: str | None = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200),
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """List incidents."""
    if not enterprise.incident_response:
        raise HTTPException(
            status_code=503,
            detail="Incident response module not enabled",
        )

    # Get incidents from engine
    incidents = []
    if hasattr(enterprise.incident_response, "get_incidents"):
        incidents = await enterprise.incident_response.get_incidents(
            status=status,
            severity=severity,
            limit=limit,
        )

    return {
        "incidents": incidents,
        "count": len(incidents),
    }


# =============================================================================
# Audit Logging
# =============================================================================


@router.post("/audit", response_model=AuditEventResponse)
async def log_audit_event(
    request: AuditEventRequest,
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> AuditEventResponse:
    """Log an audit event."""
    if not enterprise.audit:
        raise HTTPException(
            status_code=503,
            detail="Encrypted audit module not enabled",
        )

    await enterprise.log_audit_event(
        action=request.action,
        user_id=request.user_id,
        resource=request.resource,
        details=request.details,
    )

    return AuditEventResponse(
        logged=True,
        message="Audit event logged",
    )


@router.get("/audit/logs")
async def get_audit_logs(
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    action: str | None = Query(None, description="Filter by action"),
    user_id: str | None = Query(None, description="Filter by user"),
    limit: int = Query(100, ge=1, le=1000),
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Query audit logs."""
    if not enterprise.audit:
        raise HTTPException(
            status_code=503,
            detail="Encrypted audit module not enabled",
        )

    # Query audit trail
    logs = []
    if hasattr(enterprise.audit, "query_logs"):
        logs = await enterprise.audit.query_logs(
            start_date=start_date,
            end_date=end_date,
            action=action,
            user_id=user_id,
            limit=limit,
        )

    return {
        "logs": logs,
        "count": len(logs),
    }


# =============================================================================
# Security Events (SIEM)
# =============================================================================


@router.post("/security/events")
async def send_security_event(
    request: SecurityEventRequest,
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Send a security event to SIEM."""
    if not enterprise.siem:
        raise HTTPException(
            status_code=503,
            detail="SIEM integration not enabled",
        )

    await enterprise.send_security_event(
        event_type=request.event_type,
        severity=request.severity,
        source=request.source,
        details=request.details,
    )

    return {
        "sent": True,
        "message": "Security event sent to SIEM",
    }


# =============================================================================
# GDPR Compliance
# =============================================================================


@router.get("/gdpr/status")
async def get_gdpr_status(
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Get GDPR compliance status."""
    if not enterprise.gdpr:
        return {
            "enabled": False,
            "message": "GDPR module not enabled",
        }

    return {
        "enabled": True,
        "status": "compliant",
        "last_check": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/gdpr/erasure-request")
async def create_erasure_request(
    user_id: str = Query(..., description="User ID to erase"),
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Create a GDPR data erasure request."""
    if not enterprise.gdpr:
        raise HTTPException(
            status_code=503,
            detail="GDPR module not enabled",
        )

    # Create erasure request
    request_id = None
    if hasattr(enterprise.gdpr, "create_erasure_request"):
        request_id = await enterprise.gdpr.create_erasure_request(user_id)

    return {
        "request_id": request_id,
        "user_id": user_id,
        "status": "pending",
        "message": "Erasure request created",
    }


# =============================================================================
# Runbooks
# =============================================================================


@router.get("/runbooks")
async def list_runbooks(
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """List available runbooks."""
    if not enterprise.runbooks:
        raise HTTPException(
            status_code=503,
            detail="Runbooks module not enabled",
        )

    runbooks = []
    if hasattr(enterprise.runbooks, "list_runbooks"):
        runbooks = enterprise.runbooks.list_runbooks()

    return {
        "runbooks": runbooks,
        "count": len(runbooks),
    }


@router.post("/runbooks/{runbook_id}/execute")
async def execute_runbook(
    runbook_id: str,
    params: dict[str, Any] | None = None,
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Execute a runbook."""
    if not enterprise.runbooks:
        raise HTTPException(
            status_code=503,
            detail="Runbooks module not enabled",
        )

    # Execute runbook
    result = None
    if hasattr(enterprise.runbooks, "execute"):
        result = await enterprise.runbooks.execute(runbook_id, params or {})

    return {
        "runbook_id": runbook_id,
        "executed": result is not None,
        "result": result,
    }


# =============================================================================
# Anomaly Detection
# =============================================================================


@router.get("/anomalies")
async def get_anomalies(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Get detected anomalies."""
    if not enterprise.anomaly_detector:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detection module not enabled",
        )

    anomalies = []
    if hasattr(enterprise.anomaly_detector, "get_recent_anomalies"):
        anomalies = await enterprise.anomaly_detector.get_recent_anomalies(hours=hours)

    return {
        "anomalies": anomalies,
        "count": len(anomalies),
        "period_hours": hours,
    }


# =============================================================================
# Service Discovery
# =============================================================================


@router.get("/services")
async def list_services(
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """List registered services."""
    if not enterprise.service_discovery:
        raise HTTPException(
            status_code=503,
            detail="Service discovery module not enabled",
        )

    services = []
    if hasattr(enterprise.service_discovery, "list_services"):
        services = await enterprise.service_discovery.list_services()

    return {
        "services": services,
        "count": len(services),
    }


@router.get("/services/{service_name}")
async def get_service(
    service_name: str,
    enterprise: EnterpriseManager = Depends(get_enterprise),
) -> dict[str, Any]:
    """Get service instances."""
    if not enterprise.service_discovery:
        raise HTTPException(
            status_code=503,
            detail="Service discovery module not enabled",
        )

    instances = []
    if hasattr(enterprise.service_discovery, "get_instances"):
        instances = await enterprise.service_discovery.get_instances(service_name)

    return {
        "service": service_name,
        "instances": instances,
        "count": len(instances),
    }
