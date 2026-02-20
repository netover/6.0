# mypy: ignore-errors
# pylint: disable=not-callable
"""Admin API para gerenciar notificações do Teams."""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl, ConfigDict
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.database.models.teams_notifications import (
    TeamsChannel,
    TeamsJobMapping,
    TeamsNotificationConfig,
    TeamsNotificationLog,
    TeamsPatternRule,
)
from resync.core.database.session import get_db
from resync.core.teams_notifier import TeamsNotificationManager

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/admin/teams-notifications",
    tags=["Teams Notifications Admin"],
    dependencies=[Depends(verify_admin_credentials)],
)


# ============================================================================
# MODELS
# ============================================================================


class ChannelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    webhook_url: HttpUrl
    color: str = Field(default="#0078D4", pattern="^#[0-9A-Fa-f]{6}$")
    icon: str = Field(default="📢", max_length=20)


class ChannelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    webhook_url: HttpUrl | None = None
    is_active: bool | None = None
    color: str | None = None
    icon: str | None = None


class ChannelResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    webhook_url_masked: str
    is_active: bool
    color: str
    icon: str
    notification_count: int
    last_notification_sent: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class JobMappingCreate(BaseModel):
    job_name: str = Field(..., min_length=1, max_length=255)
    channel_id: int
    priority: int = Field(default=0)


class JobMappingResponse(BaseModel):
    id: int
    job_name: str
    channel_id: int
    channel_name: str | None = None
    priority: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class PatternRuleCreate(BaseModel):
    pattern: str = Field(..., min_length=1, max_length=255)
    channel_id: int
    description: str | None = Field(None, max_length=500)
    priority: int = Field(default=0)
    pattern_type: str = Field(
        default="glob", pattern="^(glob|regex|prefix|suffix|contains)$"
    )


class PatternRuleResponse(BaseModel):
    id: int
    pattern: str
    channel_id: int
    channel_name: str | None = None
    description: str | None = None
    priority: int
    pattern_type: str
    is_active: bool
    match_count: int

    model_config = ConfigDict(from_attributes=True)


class ConfigUpdate(BaseModel):
    notify_on_status: list[str] | None = None
    quiet_hours_enabled: bool | None = None
    quiet_hours_start: str | None = Field(
        None, pattern="^([0-1][0-9]|2[0-3]):[0-5][0-9]$"
    )
    quiet_hours_end: str | None = Field(
        None, pattern="^([0-1][0-9]|2[0-3]):[0-5][0-9]$"
    )
    rate_limit_enabled: bool | None = None
    max_notifications_per_job: int | None = Field(None, ge=1, le=100)
    rate_limit_window_minutes: int | None = Field(None, ge=1, le=1440)
    default_channel_id: int | None = None
    include_mention_on_critical: bool | None = None
    mention_text: str | None = Field(None, max_length=100)


class StatsResponse(BaseModel):
    total_channels: int
    active_channels: int
    total_mappings: int
    total_rules: int
    notifications_sent_today: int
    notifications_failed_today: int
    top_notified_jobs: list[dict]


# ============================================================================
# CHANNELS ENDPOINTS
# ============================================================================


@router.get("/channels", response_model=list[ChannelResponse])
async def list_channels(active_only: bool = False, db: Session = Depends(get_db)):
    """Lista todos os canais configurados."""
    stmt = select(TeamsChannel)
    if active_only:
        stmt = stmt.where(TeamsChannel.is_active)
    stmt = stmt.order_by(TeamsChannel.name)

    channels = db.execute(stmt).scalars().all()

    return [
        ChannelResponse(
            **channel.__dict__,
            webhook_url_masked=_mask_webhook_url(str(channel.webhook_url)),
        )
        for channel in channels
    ]


@router.post(
    "/channels", response_model=ChannelResponse, status_code=status.HTTP_201_CREATED
)
async def create_channel(channel_data: ChannelCreate, db: Session = Depends(get_db)):
    """Cria novo canal."""
    # Verificar duplicados
    stmt = select(TeamsChannel).where(TeamsChannel.name == channel_data.name)
    if db.execute(stmt).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Canal '{channel_data.name}' já existe",
        )

    channel = TeamsChannel(
        name=channel_data.name,
        description=channel_data.description,
        webhook_url=str(channel_data.webhook_url),
        color=channel_data.color,
        icon=channel_data.icon,
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)

    logger.info("teams_channel_created", channel=channel.name)

    return ChannelResponse(
        **channel.__dict__,
        webhook_url_masked=_mask_webhook_url(str(channel.webhook_url)),
    )


@router.put("/channels/{channel_id}", response_model=ChannelResponse)
async def update_channel(
    channel_id: int, update_data: ChannelUpdate, db: Session = Depends(get_db)
):
    """Atualiza canal."""
    channel = db.get(TeamsChannel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Canal não encontrado")

    if update_data.name:
        channel.name = update_data.name
    if update_data.description is not None:
        channel.description = update_data.description
    if update_data.webhook_url:
        channel.webhook_url = str(update_data.webhook_url)
    if update_data.is_active is not None:
        channel.is_active = update_data.is_active
    if update_data.color:
        channel.color = update_data.color
    if update_data.icon:
        channel.icon = update_data.icon

    channel.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(channel)

    return ChannelResponse(
        **channel.__dict__,
        webhook_url_masked=_mask_webhook_url(str(channel.webhook_url)),
    )


@router.delete("/channels/{channel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_channel(channel_id: int, db: Session = Depends(get_db)):
    """Deleta canal."""
    channel = db.get(TeamsChannel, channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Canal não encontrado")

    db.delete(channel)
    db.commit()
    logger.info("teams_channel_deleted", channel=channel.name)


@router.post("/channels/{channel_id}/test")
async def test_channel(channel_id: int, db: Session = Depends(get_db)):
    """Envia notificação de teste para o canal."""
    manager = TeamsNotificationManager(db)
    success = await manager.test_channel(channel_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao enviar notificação de teste",
        )

    return {"success": True, "message": "Notificação de teste enviada com sucesso"}


# ============================================================================
# JOB MAPPINGS ENDPOINTS
# ============================================================================


@router.get("/mappings", response_model=list[JobMappingResponse])
async def list_job_mappings(db: Session = Depends(get_db)):
    """Lista todos os mapeamentos de jobs."""
    stmt = select(TeamsJobMapping).order_by(
        TeamsJobMapping.priority.desc(), TeamsJobMapping.job_name
    )
    mappings = db.execute(stmt).scalars().all()

    result = []
    for mapping in mappings:
        channel = db.get(TeamsChannel, mapping.channel_id)
        result.append(
            JobMappingResponse(
                **mapping.__dict__, channel_name=channel.name if channel else None
            )
        )

    return result


@router.post("/mappings", response_model=JobMappingResponse, status_code=201)
async def create_job_mapping(
    mapping_data: JobMappingCreate, db: Session = Depends(get_db)
):
    """Cria novo mapeamento de job."""
    mapping = TeamsJobMapping(
        job_name=mapping_data.job_name,
        channel_id=mapping_data.channel_id,
        priority=mapping_data.priority,
    )
    db.add(mapping)
    db.commit()
    db.refresh(mapping)

    channel = db.get(TeamsChannel, mapping.channel_id)
    return JobMappingResponse(
        **mapping.__dict__, channel_name=channel.name if channel else None
    )


@router.delete("/mappings/{mapping_id}", status_code=204)
async def delete_job_mapping(mapping_id: int, db: Session = Depends(get_db)):
    """Deleta mapeamento de job."""
    mapping = db.get(TeamsJobMapping, mapping_id)
    if not mapping:
        raise HTTPException(status_code=404)

    db.delete(mapping)
    db.commit()


# ============================================================================
# PATTERN RULES ENDPOINTS
# ============================================================================


@router.get("/rules", response_model=list[PatternRuleResponse])
async def list_pattern_rules(db: Session = Depends(get_db)):
    """Lista todas as regras de padrões."""
    stmt = select(TeamsPatternRule).order_by(TeamsPatternRule.priority.desc())
    rules = db.execute(stmt).scalars().all()

    result = []
    for rule in rules:
        channel = db.get(TeamsChannel, rule.channel_id)
        result.append(
            PatternRuleResponse(
                **rule.__dict__, channel_name=channel.name if channel else None
            )
        )

    return result


@router.post("/rules", response_model=PatternRuleResponse, status_code=201)
async def create_pattern_rule(
    rule_data: PatternRuleCreate, db: Session = Depends(get_db)
):
    """Cria nova regra de padrão."""
    rule = TeamsPatternRule(
        pattern=rule_data.pattern,
        channel_id=rule_data.channel_id,
        description=rule_data.description,
        priority=rule_data.priority,
        pattern_type=rule_data.pattern_type,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)

    channel = db.get(TeamsChannel, rule.channel_id)
    return PatternRuleResponse(
        **rule.__dict__, channel_name=channel.name if channel else None
    )


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_pattern_rule(rule_id: int, db: Session = Depends(get_db)):
    """Deleta regra de padrão."""
    rule = db.get(TeamsPatternRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404)

    db.delete(rule)
    db.commit()


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================


@router.get("/config")
async def get_config(db: Session = Depends(get_db)):
    """Obtém configuração global."""
    config = db.query(TeamsNotificationConfig).first()
    if not config:
        # Criar configuração padrão
        config = TeamsNotificationConfig(notify_on_status=["ABEND", "ERROR", "FAILED"])
        db.add(config)
        db.commit()
        db.refresh(config)

    return config


@router.put("/config")
async def update_config(config_data: ConfigUpdate, db: Session = Depends(get_db)):
    """Atualiza configuração global."""
    config = db.query(TeamsNotificationConfig).first()
    if not config:
        config = TeamsNotificationConfig()
        db.add(config)

    for field, value in config_data.dict(exclude_unset=True).items():
        setattr(config, field, value)

    config.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(config)

    return config


# ============================================================================
# STATS ENDPOINTS
# ============================================================================


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """Obtém estatísticas de uso."""
    total_channels = db.query(func.count(TeamsChannel.id)).scalar()
    active_channels = (
        db.query(func.count(TeamsChannel.id)).filter(TeamsChannel.is_active).scalar()
    )

    total_mappings = db.query(func.count(TeamsJobMapping.id)).scalar()
    total_rules = db.query(func.count(TeamsPatternRule.id)).scalar()

    today = datetime.now(timezone.utc).date()
    notifications_sent_today = (
        db.query(func.count(TeamsNotificationLog.id))
        .filter(
            func.date(TeamsNotificationLog.timestamp) == today,
            TeamsNotificationLog.notification_sent,
        )
        .scalar()
    )

    notifications_failed_today = (
        db.query(func.count(TeamsNotificationLog.id))
        .filter(
            func.date(TeamsNotificationLog.timestamp) == today,
            TeamsNotificationLog.notification_sent.is_(False),
        )
        .scalar()
    )

    # Top jobs notificados
    top_jobs = (
        db.query(
            TeamsNotificationLog.job_name,
            func.count(TeamsNotificationLog.id).label("count"),
        )
        .filter(func.date(TeamsNotificationLog.timestamp) == today)
        .group_by(TeamsNotificationLog.job_name)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )

    return StatsResponse(
        total_channels=total_channels,
        active_channels=active_channels,
        total_mappings=total_mappings,
        total_rules=total_rules,
        notifications_sent_today=notifications_sent_today,
        notifications_failed_today=notifications_failed_today,
        top_notified_jobs=[{"job": job, "count": count} for job, count in top_jobs],
    )


@router.get("/logs")
async def get_notification_logs(
    limit: int = 100, channel_id: int | None = None, db: Session = Depends(get_db)
):
    """Obtém logs de notificações."""
    stmt = (
        select(TeamsNotificationLog)
        .order_by(desc(TeamsNotificationLog.timestamp))
        .limit(limit)
    )

    if channel_id:
        stmt = stmt.where(TeamsNotificationLog.channel_id == channel_id)

    logs = db.execute(stmt).scalars().all()
    return [log.__dict__ for log in logs]


@router.post("/test")
async def send_test_notification(test_data: dict = None, db: Session = Depends(get_db)):
    """Envia uma notificação de teste para o canal padrão."""
    manager = TeamsNotificationManager(db)

    # Verifica se há um canal padrão configurado
    config = db.query(TeamsNotificationConfig).first()
    if not config or not config.default_channel_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No default channel configured. Please configure a default channel first.",
        )

    # Dados do teste
    job_name = (
        test_data.get("job_name", "TEST_JOB_ALERT") if test_data else "TEST_JOB_ALERT"
    )
    job_status = test_data.get("job_status", "ABEND") if test_data else "ABEND"
    error_message = (
        test_data.get("error_message", "This is a test alert from Resync Admin")
        if test_data
        else "This is a test alert from Resync Admin"
    )

    # Envia notificação
    success = await manager.send_job_notification(
        job_name=job_name,
        job_status=job_status,
        instance_name="TEST_INSTANCE",
        return_code=999,
        error_message=error_message,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if success:
        return {"status": "success", "message": "Test notification sent successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send test notification",
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _mask_webhook_url(url: str) -> str:
    """Mascara URL do webhook para segurança."""
    if len(url) > 50:
        return url[:25] + "..." + url[-10:]
    return url[:15] + "..."
