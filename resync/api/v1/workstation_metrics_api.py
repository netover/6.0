"""
Workstation Metrics API Endpoint

Receives CPU, memory, and disk metrics from TWS FTAs/Workstations
via bash scripts executed via cron.

v6.0.2:
- Moved WorkstationMetricsHistory to core/database/models/metrics.py
- Fixed security logic (uses verify_metrics_api_key)
- Standardized documentation to English
"""

from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_db
from resync.core.database.models.metrics import WorkstationMetricsHistory
from resync.core.security import verify_metrics_api_key

logger = structlog.get_logger(__name__)

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics Collection"])


# ============================================================================
# Pydantic Models
# ============================================================================


class WorkstationMetrics(BaseModel):
    """Workstation resource metrics."""

    cpu_percent: float = Field(
        ..., ge=0.0, le=100.0, description="CPU usage percentage"
    )
    memory_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Memory usage percentage"
    )
    disk_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Disk usage percentage"
    )

    load_avg_1min: float | None = Field(
        None, ge=0.0, description="Load average (1 minute)"
    )
    cpu_count: int | None = Field(None, ge=1, description="Number of CPU cores")
    total_memory_gb: int | None = Field(None, ge=0, description="Total memory in GB")
    total_disk_gb: int | None = Field(None, ge=0, description="Total disk space in GB")


class WorkstationMetadata(BaseModel):
    """Workstation metadata."""

    os_type: str | None = Field(None, description="Operating system type")
    hostname: str | None = Field(None, description="Full hostname")
    collector_version: str | None = Field(None, description="Collector script version")


class MetricsPayload(BaseModel):
    """Complete metrics payload."""

    workstation: str = Field(
        ..., min_length=1, max_length=100, description="Workstation identifier"
    )
    timestamp: datetime = Field(
        ..., description="Timestamp when metrics were collected (UTC)"
    )
    metrics: WorkstationMetrics = Field(..., description="Resource metrics")
    metadata: WorkstationMetadata | None = Field(
        None, description="Additional metadata"
    )

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_recent(cls, v: datetime) -> datetime:
        """Validates that the timestamp is not too old (> 1 hour)."""
        now = datetime.now(timezone.utc)

        # Ensure v is timezone-aware
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)

        age = (now - v).total_seconds()

        # Accept up to 1 hour in the past
        if age > 3600:
            raise ValueError(f"Timestamp too old: {age} seconds")

        # Accept up to 5 minutes in the future (clock skew)
        if age < -300:
            raise ValueError(f"Timestamp too far in future: {age} seconds")

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workstation": "WS-PROD-01",
                "timestamp": "2024-12-25T10:30:00Z",
                "metrics": {
                    "cpu_percent": 45.2,
                    "memory_percent": 62.8,
                    "disk_percent": 78.5,
                    "load_avg_1min": 2.15,
                    "cpu_count": 8,
                    "total_memory_gb": 32,
                    "total_disk_gb": 500,
                },
                "metadata": {
                    "os_type": "linux-gnu",
                    "hostname": "ws-prod-01.company.com",
                    "collector_version": "1.0.0",
                },
            }
        }
    )


class MetricsResponse(BaseModel):
    """Endpoint response."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Descriptive message")
    workstation: str = Field(..., description="Identified workstation")
    timestamp: datetime = Field(..., description="Processed timestamp")
    metrics_saved: bool = Field(..., description="Whether metrics were saved")


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/workstation",
    response_model=MetricsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Receive workstation metrics",
    description="""
    Receives resource metrics from a TWS workstation FTA.
    
    Security: Requires X-API-Key header validated against settings.metrics_api_key_hash.
    """,
)
async def receive_workstation_metrics(
    payload: MetricsPayload,
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Receives and stores workstation metrics in the history table.
    """
    # 1. Security check
    if not await verify_metrics_api_key(x_api_key):
        logger.warning(
            "metrics_auth_failed",
            workstation=payload.workstation,
            api_key_prefix=x_api_key[:8] if x_api_key else "None",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key"
        )

    logger.debug(
        "metrics_received",
        workstation=payload.workstation,
        cpu=payload.metrics.cpu_percent,
        mem=payload.metrics.memory_percent,
    )

    try:
        # 2. Create database record
        record = WorkstationMetricsHistory(
            workstation=payload.workstation,
            timestamp=payload.timestamp,
            cpu_percent=payload.metrics.cpu_percent,
            memory_percent=payload.metrics.memory_percent,
            disk_percent=payload.metrics.disk_percent,
            load_avg_1min=payload.metrics.load_avg_1min,
            cpu_count=payload.metrics.cpu_count,
            total_memory_gb=payload.metrics.total_memory_gb,
            total_disk_gb=payload.metrics.total_disk_gb,
            os_type=payload.metadata.os_type if payload.metadata else None,
            hostname=payload.metadata.hostname if payload.metadata else None,
            collector_version=payload.metadata.collector_version
            if payload.metadata
            else None,
        )

        db.add(record)
        await db.commit()

        # 3. Analyze for critical conditions
        await _check_critical_metrics(payload)

        return MetricsResponse(
            status="success",
            message=f"Metrics recorded for {payload.workstation}",
            workstation=payload.workstation,
            timestamp=payload.timestamp,
            metrics_saved=True,
        )

    except Exception as e:
        logger.error(
            "metrics_storage_failed", error=str(e), workstation=payload.workstation
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store metrics",
        )


@router.get(
    "/workstation/{workstation_name}",
    summary="Retrieve metrics history",
    description="Returns historical resource metrics for a specific workstation.",
)
async def get_metrics_history(
    workstation_name: str,
    hours: int = 24,
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Queries historical metrics for a given workstation.
    """
    # Security check
    if not await verify_metrics_api_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    stmt = (
        select(WorkstationMetricsHistory)
        .where(
            WorkstationMetricsHistory.workstation == workstation_name,
            WorkstationMetricsHistory.timestamp >= cutoff,
        )
        .order_by(WorkstationMetricsHistory.timestamp.desc())
    )

    result = await db.execute(stmt)
    records = result.scalars().all()

    return {
        "workstation": workstation_name,
        "records_count": len(records),
        "history": [
            {
                "timestamp": r.timestamp,
                "cpu": r.cpu_percent,
                "mem": r.memory_percent,
                "disk": r.disk_percent,
                "load": r.load_avg_1min,
            }
            for r in records
        ],
    }


@router.get("/health", summary="Service health check")
async def health():
    """Simple health check for the metrics API."""
    return {"status": "healthy", "module": "metrics-v1"}


# ============================================================================
# Private Helpers
# ============================================================================


async def _check_critical_metrics(payload: MetricsPayload) -> None:
    """
    Analyzes metrics for threshold violations and logs alerts.
    """
    alerts = []

    # CPU > 95%
    if payload.metrics.cpu_percent > 95:
        alerts.append(f"CPU usage high: {payload.metrics.cpu_percent}%")

    # Memory > 95%
    if payload.metrics.memory_percent > 95:
        alerts.append(f"Memory usage high: {payload.metrics.memory_percent}%")

    # Disk > 90%
    if payload.metrics.disk_percent > 90:
        alerts.append(f"Disk usage high: {payload.metrics.disk_percent}%")

    if alerts:
        logger.warning(
            "critical_metrics_alert", workstation=payload.workstation, alerts=alerts
        )
        # TODO: Integration with notification bus/EventBus
