"""
Workstation Metrics API Endpoint

Recebe métricas de CPU, memory e disk das FTAs/Workstations TWS
via scripts bash executados via cron.

Author: Resync Team
Version: 1.0.0
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import get_db
from resync.core.security import verify_api_key

logger = structlog.get_logger(__name__)

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/v1/metrics",
    tags=["Metrics Collection"]
)


# ============================================================================
# MODELS
# ============================================================================

class WorkstationMetrics(BaseModel):
    """Métricas de recursos da workstation."""

    cpu_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage"
    )
    memory_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Memory usage percentage"
    )
    disk_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Disk usage percentage"
    )
    load_avg_1min: float | None = Field(
        None,
        ge=0.0,
        description="Load average (1 minute)"
    )
    cpu_count: int | None = Field(
        None,
        ge=1,
        description="Number of CPU cores"
    )
    total_memory_gb: int | None = Field(
        None,
        ge=0,
        description="Total memory in GB"
    )
    total_disk_gb: int | None = Field(
        None,
        ge=0,
        description="Total disk space in GB"
    )


class WorkstationMetadata(BaseModel):
    """Metadata da workstation."""

    os_type: str | None = Field(
        None,
        description="Operating system type"
    )
    hostname: str | None = Field(
        None,
        description="Full hostname"
    )
    collector_version: str | None = Field(
        None,
        description="Collector script version"
    )


class MetricsPayload(BaseModel):
    """Payload completo de métricas."""

    workstation: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Workstation identifier"
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when metrics were collected (UTC)"
    )
    metrics: WorkstationMetrics = Field(
        ...,
        description="Resource metrics"
    )
    metadata: WorkstationMetadata | None = Field(
        None,
        description="Additional metadata"
    )

    @field_validator('timestamp')
    @classmethod
    def timestamp_must_be_recent(cls, v):
        """Valida que timestamp não é muito antigo (> 1 hora)."""
        now = datetime.now(timezone.utc)
        age = (now - v).total_seconds()

        # Aceita até 1 hora no passado
        if age > 3600:
            raise ValueError(f"Timestamp too old: {age} seconds")

        # Aceita até 5 minutos no futuro (clock skew)
        if age < -300:
            raise ValueError(f"Timestamp too far in future: {age} seconds")

        return v

    model_config = ConfigDict(json_schema_extra={
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
                    "total_disk_gb": 500
                },
                "metadata": {
                    "os_type": "linux-gnu",
                    "hostname": "ws-prod-01.company.com",
                    "collector_version": "1.0.0"
                }
            }
        })


class MetricsResponse(BaseModel):
    """Resposta do endpoint."""

    status: str = Field(..., description="Status da operação")
    message: str = Field(..., description="Mensagem descritiva")
    workstation: str = Field(..., description="Workstation identificada")
    timestamp: datetime = Field(..., description="Timestamp processado")
    metrics_saved: bool = Field(..., description="Se métricas foram salvas")


# ============================================================================
# DATABASE MODEL
# ============================================================================

from sqlalchemy import Column, DateTime, Float, Index, Integer, String

from resync.core.database import Base


class WorkstationMetricsHistory(Base):
    """
    Histórico de métricas das workstations TWS.

    Armazena CPU, memory, disk coletados via scripts bash.
    """

    __tablename__ = "workstation_metrics_history"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identificação
    workstation = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Métricas principais (required)
    cpu_percent = Column(Float, nullable=False)
    memory_percent = Column(Float, nullable=False)
    disk_percent = Column(Float, nullable=False)

    # Métricas adicionais (optional)
    load_avg_1min = Column(Float, nullable=True)
    cpu_count = Column(Integer, nullable=True)
    total_memory_gb = Column(Integer, nullable=True)
    total_disk_gb = Column(Integer, nullable=True)

    # Metadata
    os_type = Column(String(50), nullable=True)
    hostname = Column(String(255), nullable=True)
    collector_version = Column(String(20), nullable=True)

    # Audit
    received_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    # Indexes compostos para queries comuns
    __table_args__ = (
        # Query por workstation + time range
        Index(
            'ix_workstation_timestamp',
            'workstation',
            'timestamp',
            postgresql_using='btree'
        ),
        # Query recentes
        Index(
            'ix_received_at',
            'received_at',
            postgresql_using='btree'
        ),
    )

    def __repr__(self):
        return (
            f"<WorkstationMetricsHistory("
            f"workstation={self.workstation}, "
            f"timestamp={self.timestamp}, "
            f"cpu={self.cpu_percent}%, "
            f"mem={self.memory_percent}%, "
            f"disk={self.disk_percent}%"
            f")>"
        )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/workstation",
    response_model=MetricsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Receive workstation metrics",
    description="""
    Recebe métricas de CPU, memory e disk de uma workstation TWS.

    Este endpoint é chamado pelos scripts bash nas FTAs via cron.

    Autenticação: X-API-Key header
    Rate limit: 1000 requests/hour por API key
    """,
)
async def receive_workstation_metrics(
    payload: MetricsPayload,
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Recebe e armazena métricas de workstation.

    Args:
        payload: Métricas coletadas
        x_api_key: API key para autenticação
        db: Database session

    Returns:
        MetricsResponse com status da operação

    Raises:
        HTTPException 401: API key inválida
        HTTPException 422: Payload inválido
        HTTPException 500: Erro ao salvar no banco
    """
    # 1. Validar API key
    if not await verify_api_key(x_api_key):
        logger.warning(
            "invalid_api_key",
            workstation=payload.workstation,
            api_key_prefix=x_api_key[:8] if x_api_key else None
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    logger.info(
        "metrics_received",
        workstation=payload.workstation,
        timestamp=payload.timestamp,
        cpu=payload.metrics.cpu_percent,
        memory=payload.metrics.memory_percent,
        disk=payload.metrics.disk_percent
    )

    try:
        # 2. Criar registro no banco
        metrics_record = WorkstationMetricsHistory(
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
            collector_version=payload.metadata.collector_version if payload.metadata else None,
        )

        db.add(metrics_record)
        await db.commit()
        await db.refresh(metrics_record)

        logger.info(
            "metrics_saved",
            workstation=payload.workstation,
            record_id=metrics_record.id
        )

        # 3. Trigger análise se métricas críticas
        await _check_critical_metrics(payload)

        return MetricsResponse(
            status="success",
            message=f"Metrics stored successfully for {payload.workstation}",
            workstation=payload.workstation,
            timestamp=payload.timestamp,
            metrics_saved=True
        )

    except Exception as e:
        logger.error(
            "metrics_save_failed",
            workstation=payload.workstation,
            error=str(e)
        )
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save metrics. Check server logs for details."
        ) from None


async def _check_critical_metrics(payload: MetricsPayload):
    """
    Verifica se métricas estão em níveis críticos e gera alertas.

    Args:
        payload: Métricas recebidas
    """
    alerts = []

    # CPU crítico (> 95%)
    if payload.metrics.cpu_percent > 95:
        alerts.append({
            "severity": "critical",
            "metric": "cpu",
            "value": payload.metrics.cpu_percent,
            "threshold": 95,
            "message": f"CPU critically high: {payload.metrics.cpu_percent}%"
        })

    # Memory crítico (> 95%)
    if payload.metrics.memory_percent > 95:
        alerts.append({
            "severity": "critical",
            "metric": "memory",
            "value": payload.metrics.memory_percent,
            "threshold": 95,
            "message": f"Memory critically high: {payload.metrics.memory_percent}%"
        })

    # Disk crítico (> 90%)
    if payload.metrics.disk_percent > 90:
        alerts.append({
            "severity": "critical",
            "metric": "disk",
            "value": payload.metrics.disk_percent,
            "threshold": 90,
            "message": f"Disk critically high: {payload.metrics.disk_percent}%"
        })

    # Load average crítico (> cpu_count * 2)
    if payload.metrics.load_avg_1min and payload.metrics.cpu_count:
        threshold = payload.metrics.cpu_count * 2
        if payload.metrics.load_avg_1min > threshold:
            alerts.append({
                "severity": "warning",
                "metric": "load_avg",
                "value": payload.metrics.load_avg_1min,
                "threshold": threshold,
                "message": f"Load average high: {payload.metrics.load_avg_1min} (threshold: {threshold})"
            })

    # Se há alertas, logar e (futuramente) notificar
    if alerts:
        logger.warning(
            "critical_metrics_detected",
            workstation=payload.workstation,
            alerts=alerts
        )

        # DEBT: Integrate with alerting system (PagerDuty/OpsGenie) when available
        # await alert_manager.send_alert(
        #     workstation=payload.workstation,
        #     alerts=alerts
        # )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get(
    "/health",
    summary="Health check",
    description="Verifica se o endpoint está operacional"
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "workstation-metrics-api",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc)
    }


# ============================================================================
# QUERY ENDPOINTS (para consultar métricas armazenadas)
# ============================================================================

@router.get(
    "/workstation/{workstation_name}",
    summary="Get workstation metrics history",
    description="Retorna histórico de métricas de uma workstation específica"
)
async def get_workstation_metrics(
    workstation_name: str,
    hours: int = 24,
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Consulta métricas históricas de uma workstation.

    Args:
        workstation_name: Nome da workstation
        hours: Número de horas para trás (default: 24)
        x_api_key: API key
        db: Database session

    Returns:
        Lista de métricas ordenadas por timestamp
    """
    # Validar API key
    if not await verify_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Query
    from datetime import timedelta

    from sqlalchemy import select

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    stmt = select(WorkstationMetricsHistory).where(
        WorkstationMetricsHistory.workstation == workstation_name,
        WorkstationMetricsHistory.timestamp >= cutoff_time
    ).order_by(WorkstationMetricsHistory.timestamp.desc())

    result = await db.execute(stmt)
    metrics = result.scalars().all()

    return {
        "workstation": workstation_name,
        "hours": hours,
        "count": len(metrics),
        "metrics": [
            {
                "timestamp": m.timestamp,
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "disk_percent": m.disk_percent,
                "load_avg_1min": m.load_avg_1min,
            }
            for m in metrics
        ]
    }
