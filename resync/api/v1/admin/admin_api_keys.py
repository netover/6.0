# mypy: ignore-errors
"""
Admin API - API Key Management

CRUD completo para gerenciamento de API Keys do Resync.

Features:
- Create API keys (com scopes e expiration)
- List all keys
- Get key details
- Delete/revoke keys
- Usage tracking
- Rate limiting

Author: Resync Team
Version: 1.0.0
"""

import secrets
from datetime import datetime, timezone, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database import Base, get_db
from resync.core.security import hash_api_key, verify_admin_token

logger = structlog.get_logger(__name__)


# ============================================================================
# DATABASE MODEL
# ============================================================================

class APIKey(Base):
    """
    API Keys para acesso ao Resync.

    Usado por:
    - FTAs enviando métricas
    - Scripts externos
    - Integrações
    """

    __tablename__ = "api_keys"

    # Primary key
    id = Column(String(36), primary_key=True)  # UUID

    # Key (hashed)
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    key_prefix = Column(String(20), nullable=False)  # "rsk_abc..." (primeiros 10 chars)

    # Metadata
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)

    # Scopes (JSON array)
    scopes = Column(JSON, nullable=False, default=list)
    # Examples: ["metrics:write", "metrics:read", "admin:read", "admin:write"]

    # Expiration
    expires_at = Column(DateTime, nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_revoked = Column(Boolean, nullable=False, default=False)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by = Column(String(100), nullable=True)
    revoked_reason = Column(String(500), nullable=True)

    # Usage tracking
    last_used_at = Column(DateTime, nullable=True, index=True)
    usage_count = Column(Integer, nullable=False, default=0)

    # Audit
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    created_by = Column(String(100), nullable=False)

    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, active={self.is_active})>"

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if key is valid (active, not revoked, not expired)."""
        return self.is_active and not self.is_revoked and not self.is_expired


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class APIKeyCreate(BaseModel):
    """Request model para criar API key."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Nome descritivo da API key"
    )
    description: str | None = Field(
        None,
        max_length=500,
        description="Descrição opcional"
    )
    scopes: list[str] = Field(
        default=["metrics:write"],
        description="Scopes/permissões da key"
    )
    expires_in_days: int | None = Field(
        None,
        ge=1,
        le=3650,  # Max 10 anos
        description="Dias até expiração (None = nunca expira)"
    )

    @field_validator('scopes')
    @classmethod
    def validate_scopes(cls, v):
        """Valida scopes."""
        valid_scopes = {
            "metrics:read",
            "metrics:write",
            "admin:read",
            "admin:write",
            "workflows:read",
            "workflows:write"
        }

        for scope in v:
            if scope not in valid_scopes:
                raise ValueError(
                    f"Invalid scope: {scope}. "
                    f"Valid scopes: {', '.join(valid_scopes)}"
                )

        return v

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "name": "FTA Production Metrics",
                "description": "API key for production FTAs to send metrics",
                "scopes": ["metrics:write"],
                "expires_in_days": 365
            }
        })


class APIKeyResponse(BaseModel):
    """Response model para API key."""

    id: str
    key_prefix: str
    name: str
    description: str | None = None
    scopes: list[str]
    expires_at: datetime | None = None
    is_active: bool
    is_revoked: bool
    is_expired: bool
    is_valid: bool
    last_used_at: datetime | None = None
    usage_count: int
    created_at: datetime
    created_by: str

    model_config = ConfigDict(from_attributes=True)


class APIKeyCreatedResponse(BaseModel):
    """Response quando key é criada (inclui key completa)."""

    id: str
    key: str  # Full key (rsk_abc123...) - ONLY shown once!
    key_prefix: str
    name: str
    description: str | None = None
    scopes: list[str]
    expires_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "key": "rsk_abc123xyz789...",
                "key_prefix": "rsk_abc123",
                "name": "FTA Production Metrics",
                "description": "API key for production FTAs",
                "scopes": ["metrics:write"],
                "expires_at": "2025-12-25T00:00:00Z",
                "created_at": "2024-12-25T10:00:00Z"
            }
        })


class APIKeyRevoke(BaseModel):
    """Request para revogar key."""

    reason: str | None = Field(
        None,
        max_length=500,
        description="Motivo da revogação"
    )


class APIKeyListResponse(BaseModel):
    """Response para list de keys."""

    total: int
    keys: list[APIKeyResponse]


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/v1/admin/api-keys",
    tags=["Admin - API Keys"]
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="""
    Cria uma nova API key.

    **ATENÇÃO:** A key completa só é retornada uma vez!
    Guarde-a em local seguro.

    Requer autenticação admin.
    """
)
async def create_api_key(
    payload: APIKeyCreate,
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Cria nova API key."""

    # Generate key
    key_id = secrets.token_urlsafe(16)
    key_secret = secrets.token_urlsafe(32)
    full_key = f"rsk_{key_secret}"

    # Hash key
    key_hash = hash_api_key(full_key)
    key_prefix = full_key[:10]

    # Calculate expiration
    expires_at = None
    if payload.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)

    # Create DB record
    api_key = APIKey(
        id=key_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=payload.name,
        description=payload.description,
        scopes=payload.scopes,
        expires_at=expires_at,
        created_by=admin_token["user"]  # From admin token
    )

    db.add(api_key)

    try:
        await db.commit()
        await db.refresh(api_key)

        logger.info(
            "api_key_created",
            key_id=key_id,
            name=payload.name,
            created_by=admin_token["user"]
        )

        return APIKeyCreatedResponse(
            id=api_key.id,
            key=full_key,  # ONLY time we return full key!
            key_prefix=api_key.key_prefix,
            name=api_key.name,
            description=api_key.description,
            scopes=api_key.scopes,
            expires_at=api_key.expires_at,
            created_at=api_key.created_at
        )

    except Exception as e:
        await db.rollback()
        logger.error("api_key_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key. Check server logs for details."
        ) from None


@router.get(
    "",
    response_model=APIKeyListResponse,
    summary="List API keys",
    description="Lista todas API keys (sem mostrar key completa)"
)
async def list_api_keys(
    include_revoked: bool = False,
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Lista todas API keys."""

    # Build query
    stmt = select(APIKey).order_by(APIKey.created_at.desc())

    if not include_revoked:
        stmt = stmt.where(APIKey.is_revoked.is_(False))

    result = await db.execute(stmt)
    keys = result.scalars().all()

    # Convert to response models
    key_responses = []
    for key in keys:
        key_responses.append(
            APIKeyResponse(
                id=key.id,
                key_prefix=key.key_prefix,
                name=key.name,
                description=key.description,
                scopes=key.scopes,
                expires_at=key.expires_at,
                is_active=key.is_active,
                is_revoked=key.is_revoked,
                is_expired=key.is_expired,
                is_valid=key.is_valid,
                last_used_at=key.last_used_at,
                usage_count=key.usage_count,
                created_at=key.created_at,
                created_by=key.created_by
            )
        )

    return APIKeyListResponse(
        total=len(key_responses),
        keys=key_responses
    )


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API key details",
    description="Retorna detalhes de uma API key específica"
)
async def get_api_key(
    key_id: str,
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Get API key details."""

    stmt = select(APIKey).where(APIKey.id == key_id)
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    return APIKeyResponse(
        id=api_key.id,
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        description=api_key.description,
        scopes=api_key.scopes,
        expires_at=api_key.expires_at,
        is_active=api_key.is_active,
        is_revoked=api_key.is_revoked,
        is_expired=api_key.is_expired,
        is_valid=api_key.is_valid,
        last_used_at=api_key.last_used_at,
        usage_count=api_key.usage_count,
        created_at=api_key.created_at,
        created_by=api_key.created_by
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoga (desativa) uma API key"
)
async def revoke_api_key(
    key_id: str,
    payload: APIKeyRevoke,
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Revoke API key."""

    # Find key
    stmt = select(APIKey).where(APIKey.id == key_id)
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    if api_key.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key already revoked"
        )

    # Revoke
    stmt = update(APIKey).where(APIKey.id == key_id).values(
        is_revoked=True,
        is_active=False,
        revoked_at=datetime.now(timezone.utc),
        revoked_by=admin_token["user"],
        revoked_reason=payload.reason
    )

    await db.execute(stmt)
    await db.commit()

    logger.info(
        "api_key_revoked",
        key_id=key_id,
        revoked_by=admin_token["user"],
        reason=payload.reason
    )

    return


@router.delete(
    "/{key_id}/permanent",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete API key permanently",
    description="Deleta permanentemente uma API key (não pode ser desfeito!)"
)
async def delete_api_key_permanent(
    key_id: str,
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Delete API key permanently."""

    # Verify exists
    stmt = select(APIKey).where(APIKey.id == key_id)
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    # Delete
    stmt = delete(APIKey).where(APIKey.id == key_id)
    await db.execute(stmt)
    await db.commit()

    logger.warning(
        "api_key_deleted_permanently",
        key_id=key_id,
        deleted_by=admin_token["user"]
    )

    return


# ============================================================================
# HEALTH & STATS
# ============================================================================

@router.get(
    "/stats/summary",
    summary="API keys statistics",
    description="Estatísticas de uso das API keys"
)
async def get_api_keys_stats(
    admin_token: str = Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db)
):
    """Get API keys usage statistics."""

    # Count total using func.count for efficiency
    stmt_total = select(func.count()).select_from(APIKey)
    total = (await db.execute(stmt_total)).scalar() or 0

    # Count active (is_active=True AND is_revoked=False)
    stmt_active = select(func.count()).select_from(APIKey).where(
        and_(
            APIKey.is_active.is_(True),
            APIKey.is_revoked.is_(False)
        )
    )
    active = (await db.execute(stmt_active)).scalar() or 0

    # Count revoked
    stmt_revoked = select(func.count()).select_from(APIKey).where(
        APIKey.is_revoked.is_(True)
    )
    revoked = (await db.execute(stmt_revoked)).scalar() or 0

    # Count expired
    stmt_expired = select(func.count()).select_from(APIKey).where(
        APIKey.expires_at < datetime.now(timezone.utc)
    )
    expired = (await db.execute(stmt_expired)).scalar() or 0

    # Most used
    stmt_most_used = select(APIKey).order_by(APIKey.usage_count.desc()).limit(5)
    most_used = (await db.execute(stmt_most_used)).scalars().all()

    # Recently created
    stmt_recent = select(APIKey).order_by(APIKey.created_at.desc()).limit(5)
    recent = (await db.execute(stmt_recent)).scalars().all()

    return {
        "counts": {
            "total": total,
            "active": active,
            "revoked": revoked,
            "expired": expired
        },
        "most_used": [
            {
                "id": k.id,
                "name": k.name,
                "usage_count": k.usage_count,
                "last_used_at": k.last_used_at
            }
            for k in most_used
        ],
        "recently_created": [
            {
                "id": k.id,
                "name": k.name,
                "created_at": k.created_at,
                "created_by": k.created_by
            }
            for k in recent
        ]
    }
