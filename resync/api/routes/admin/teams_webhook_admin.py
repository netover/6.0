# pylint: disable=not-callable
"""Teams Webhook Administration API."""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.database.models.teams import TeamsWebhookAudit, TeamsWebhookUser
from resync.core.database.session import get_db
from resync.core.teams_permissions import TeamsPermissionsManager

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/admin/teams-webhook",
    tags=["Teams Webhook Admin"],
    dependencies=[Depends(verify_admin_credentials)],
)


# Models
class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="viewer", pattern="^(viewer|operator|admin)$")
    can_execute_commands: bool = False
    aad_object_id: str | None = None


class UserUpdate(BaseModel):
    name: str | None = None
    role: str | None = Field(None, pattern="^(viewer|operator|admin)$")
    can_execute_commands: bool | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    can_execute_commands: bool
    is_active: bool
    created_at: datetime
    last_activity: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class AuditLogResponse(BaseModel):
    id: int
    user_email: str
    user_name: str
    channel_name: str | None = None
    message_text: str
    command_type: str
    was_authorized: bool
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class StatsResponse(BaseModel):
    total_users: int
    active_users: int
    users_with_execute_permission: int
    total_interactions: int
    interactions_today: int
    authorized_commands: int
    unauthorized_attempts: int


# Endpoints
@router.get("/users", response_model=list[UserResponse])
async def list_users(active_only: bool = True, db: Session = Depends(get_db)):
    """Lista todos os usuários cadastrados."""
    stmt = select(TeamsWebhookUser)
    if active_only:
        stmt = stmt.where(TeamsWebhookUser.is_active)
    stmt = stmt.order_by(TeamsWebhookUser.created_at.desc())

    result = db.execute(stmt)
    users = result.scalars().all()

    return [UserResponse.model_validate(user) for user in users]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Cria novo usuário."""
    # Verifica se já existe
    stmt = select(TeamsWebhookUser).where(TeamsWebhookUser.email == user_data.email)
    existing = db.execute(stmt).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email {user_data.email} already exists",
        )

    pm = TeamsPermissionsManager(db)
    user = await pm.create_user(
        email=user_data.email,
        name=user_data.name,
        role=user_data.role,
        can_execute=user_data.can_execute_commands,
        aad_object_id=user_data.aad_object_id,
    )

    return UserResponse.model_validate(user)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Obtém usuário por ID."""
    stmt = select(TeamsWebhookUser).where(TeamsWebhookUser.id == user_id)
    user = db.execute(stmt).scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found"
        )

    return UserResponse.model_validate(user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int, update_data: UserUpdate, db: Session = Depends(get_db)
):
    """Atualiza usuário."""
    stmt = select(TeamsWebhookUser).where(TeamsWebhookUser.id == user_id)
    user = db.execute(stmt).scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found"
        )

    if update_data.name is not None:
        user.name = update_data.name
    if update_data.role is not None:
        user.role = update_data.role
    if update_data.can_execute_commands is not None:
        user.can_execute_commands = update_data.can_execute_commands
    if update_data.is_active is not None:
        user.is_active = update_data.is_active

    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    logger.info("teams_webhook_user_updated", user_id=user_id, email=user.email)

    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Deleta usuário (soft delete - marca como inativo)."""
    stmt = select(TeamsWebhookUser).where(TeamsWebhookUser.id == user_id)
    user = db.execute(stmt).scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found"
        )

    user.is_active = False
    user.updated_at = datetime.now(timezone.utc)
    db.commit()

    logger.info("teams_webhook_user_deleted", user_id=user_id, email=user.email)


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def get_audit_logs(
    limit: int = 100, user_email: str | None = None, db: Session = Depends(get_db)
):
    """Obtém logs de auditoria."""
    stmt = select(TeamsWebhookAudit)

    if user_email:
        stmt = stmt.where(TeamsWebhookAudit.user_email == user_email)

    stmt = stmt.order_by(desc(TeamsWebhookAudit.timestamp)).limit(limit)

    result = db.execute(stmt)
    logs = result.scalars().all()

    return [AuditLogResponse.model_validate(log) for log in logs]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """Obtém estatísticas de uso."""
    from sqlalchemy import func

    # Total de usuários
    total_users = db.query(func.count(TeamsWebhookUser.id)).scalar()
    active_users = (
        db.query(func.count(TeamsWebhookUser.id))
        .filter(TeamsWebhookUser.is_active)
        .scalar()
    )
    users_with_execute = (
        db.query(func.count(TeamsWebhookUser.id))
        .filter(TeamsWebhookUser.can_execute_commands, TeamsWebhookUser.is_active)
        .scalar()
    )

    # Interações
    total_interactions = db.query(func.count(TeamsWebhookAudit.id)).scalar()

    today = datetime.now(timezone.utc).date()
    interactions_today = (
        db.query(func.count(TeamsWebhookAudit.id))
        .filter(func.date(TeamsWebhookAudit.timestamp) == today)
        .scalar()
    )

    authorized_commands = (
        db.query(func.count(TeamsWebhookAudit.id))
        .filter(
            TeamsWebhookAudit.command_type == "execute",
            TeamsWebhookAudit.was_authorized,
        )
        .scalar()
    )

    unauthorized_attempts = (
        db.query(func.count(TeamsWebhookAudit.id))
        .filter(
            TeamsWebhookAudit.command_type == "execute",
            not TeamsWebhookAudit.was_authorized,
        )
        .scalar()
    )

    return StatsResponse(
        total_users=total_users,
        active_users=active_users,
        users_with_execute_permission=users_with_execute,
        total_interactions=total_interactions,
        interactions_today=interactions_today,
        authorized_commands=authorized_commands,
        unauthorized_attempts=unauthorized_attempts,
    )
