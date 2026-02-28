"""
Admin Backup Routes.

Provides endpoints for:
- Creating manual backups (database, config, full)
- Managing backup schedules
- Listing and downloading backups
- Backup statistics

Endpoints:
    POST   /api/v1/admin/backup/database     - Create database backup
    POST   /api/v1/admin/backup/config       - Create config backup
    POST   /api/v1/admin/backup/full         - Create full backup
    GET    /api/v1/admin/backup/list         - List all backups
    GET    /api/v1/admin/backup/{id}         - Get backup details
    GET    /api/v1/admin/backup/{id}/download - Download backup file
    DELETE /api/v1/admin/backup/{id}         - Delete backup

    GET    /api/v1/admin/backup/schedules          - List schedules
    POST   /api/v1/admin/backup/schedules          - Create schedule
    PUT    /api/v1/admin/backup/schedules/{id}     - Update schedule
    DELETE /api/v1/admin/backup/schedules/{id}     - Delete schedule

    GET    /api/v1/admin/backup/stats        - Backup statistics
    POST   /api/v1/admin/backup/cleanup      - Manual cleanup
"""

from typing import Any

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.backup import (
    BackupInfo,
    BackupSchedule,
    BackupStatus,
    BackupType,
    get_backup_service,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# v5.9.5: Added authentication - was exposed without auth
router = APIRouter(
    prefix="/admin/backup",
    tags=["Admin - Backup"],
    dependencies=[Depends(verify_admin_credentials)],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateBackupRequest(BaseModel):
    """Request to create a backup."""

    description: str = Field(default="", description="Optional backup description")
    compress: bool = Field(
        default=True, description="Compress the backup (database only)"
    )
    include_env: bool = Field(
        default=True, description="Include .env files (config only)"
    )
    collection_name: str | None = Field(
        default=None, description="Collection name for RAG backup"
    )

class BackupResponse(BaseModel):
    """Response with backup information."""

    id: str
    type: str
    status: str
    filename: str
    size_bytes: int
    size_human: str
    created_at: str
    completed_at: str | None = None
    duration_seconds: float
    checksum_sha256: str
    metadata: dict[str, Any]
    error: str | None = None

    @classmethod
    def from_backup_info(cls, backup: BackupInfo) -> "BackupResponse":
        return cls(
            id=backup.id,
            type=backup.type.value,
            status=backup.status.value,
            filename=backup.filename,
            size_bytes=backup.size_bytes,
            size_human=backup.size_human,
            created_at=backup.created_at.isoformat(),
            completed_at=backup.completed_at.isoformat()
            if backup.completed_at
            else None,
            duration_seconds=backup.duration_seconds,
            checksum_sha256=backup.checksum_sha256,
            metadata=backup.metadata,
            error=backup.error,
        )

class BackupListResponse(BaseModel):
    """Response with list of backups."""

    backups: list[BackupResponse]
    total: int

class CreateScheduleRequest(BaseModel):
    """Request to create a backup schedule."""

    name: str = Field(..., description="Schedule name")
    backup_type: str = Field(..., description="Type: database, config, or full")
    cron_expression: str = Field(
        ...,
        description="Cron expression (e.g., '0 2 * * *' for 2 AM daily)",
        examples=["0 2 * * *", "0 0 * * 0", "0 0 1 * *"],
    )
    retention_days: int = Field(
        default=30, ge=1, le=365, description="Days to retain backups"
    )

class UpdateScheduleRequest(BaseModel):
    """Request to update a backup schedule."""

    enabled: bool | None = None
    cron_expression: str | None = None
    retention_days: int | None = Field(default=None, ge=1, le=365)

class ScheduleResponse(BaseModel):
    """Response with schedule information."""

    id: str
    name: str
    backup_type: str
    cron_expression: str
    enabled: bool
    retention_days: int
    last_run: str | None = None
    next_run: str | None = None
    created_at: str

    @classmethod
    def from_schedule(cls, schedule: BackupSchedule) -> "ScheduleResponse":
        return cls(
            id=schedule.id,
            name=schedule.name,
            backup_type=schedule.backup_type.value,
            cron_expression=schedule.cron_expression,
            enabled=schedule.enabled,
            retention_days=schedule.retention_days,
            last_run=schedule.last_run.isoformat() if schedule.last_run else None,
            next_run=schedule.next_run.isoformat() if schedule.next_run else None,
            created_at=schedule.created_at.isoformat(),
        )

class ScheduleListResponse(BaseModel):
    """Response with list of schedules."""

    schedules: list[ScheduleResponse]

class BackupStatsResponse(BaseModel):
    """Response with backup statistics."""

    total_backups: int
    total_size_bytes: int
    total_size_human: str
    by_type: dict[str, dict[str, Any]]
    active_schedules: int
    backup_directory: str

class CleanupRequest(BaseModel):
    """Request to cleanup old backups."""

    retention_days: int = Field(default=30, ge=1, le=365)

class CleanupResponse(BaseModel):
    """Response with cleanup result."""

    deleted_count: int
    retention_days: int

# =============================================================================
# BACKUP ENDPOINTS
# =============================================================================

@router.post("/database", response_model=BackupResponse)
async def create_database_backup(
    request: CreateBackupRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create a PostgreSQL database backup.

    The backup is created using pg_dump and compressed with gzip.
    """
    service = get_backup_service()

    try:
        backup = await service.create_database_backup(
            description=request.description,
            compress=request.compress,
        )
        return BackupResponse.from_backup_info(backup)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("create_database_backup_failed", error=str(e))  # type: ignore[call-arg]
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.post("/config", response_model=BackupResponse)
async def create_config_backup(request: CreateBackupRequest):
    """
    Create a system configuration backup.

    Includes config files, prompts, environment variables, etc.
    """
    service = get_backup_service()

    try:
        backup = await asyncio.to_thread(
            service.create_config_backup,
            description=request.description,
            include_env=request.include_env,
        )
        return BackupResponse.from_backup_info(backup)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("create_config_backup_failed", error=str(e))  # type: ignore[call-arg]
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.post("/full", response_model=BackupResponse)
async def create_full_backup(request: CreateBackupRequest):
    """
    Create a full backup (database + config).

    Creates both backups and packages them into a single ZIP file.
    """
    service = get_backup_service()

    try:
        backup = await service.create_full_backup(description=request.description)
        return BackupResponse.from_backup_info(backup)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("create_full_backup_failed", error=str(e))  # type: ignore[call-arg]
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check server logs for details.",
        ) from e

@router.get("/list", response_model=BackupListResponse)
async def list_backups(
    backup_type: str | None = Query(
        None, description="Filter by type: database, config, full"
    ),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
):
    """
    List all backups with optional filters.
    """
    service = get_backup_service()

    # Parse filters
    type_filter = None
    if backup_type:
        try:
            type_filter = BackupType(backup_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid backup type: {backup_type}"
            ) from None

    status_filter = None
    if status:
        try:
            status_filter = BackupStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid status: {status}"
            ) from None

    backups = service.list_backups(
        backup_type=type_filter,
        status=status_filter,
        limit=limit,
    )

    return BackupListResponse(
        backups=[BackupResponse.from_backup_info(b) for b in backups],
        total=len(backups),
    )

@router.get("/{backup_id}", response_model=BackupResponse)
async def get_backup(backup_id: str):
    """
    Get details of a specific backup.
    """
    service = get_backup_service()
    backup = service.get_backup(backup_id)

    if not backup:
        raise HTTPException(status_code=404, detail="Backup not found")

    return BackupResponse.from_backup_info(backup)

@router.get("/{backup_id}/download")
async def download_backup(backup_id: str):
    """
    Download a backup file.
    """
    service = get_backup_service()

    backup = service.get_backup(backup_id)
    if not backup:
        raise HTTPException(status_code=404, detail="Backup not found")

    filepath = service.get_backup_filepath(backup_id)
    if not filepath:
        raise HTTPException(status_code=404, detail="Backup file not found")

    # Determine media type
    if backup.filename.endswith(".sql.gz"):
        media_type = "application/gzip"
    elif backup.filename.endswith(".zip"):
        media_type = "application/zip"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(filepath),
        filename=backup.filename,
        media_type=media_type,
    )

@router.delete("/{backup_id}")
async def delete_backup(backup_id: str):
    """
    Delete a backup.
    """
    service = get_backup_service()

    deleted = service.delete_backup(backup_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Backup not found")

    return {"message": "Backup deleted", "backup_id": backup_id}

# =============================================================================
# SCHEDULE ENDPOINTS
# =============================================================================

@router.get("/schedules", response_model=ScheduleListResponse)
async def list_schedules():
    """
    List all backup schedules.
    """
    service = get_backup_service()
    schedules = service.list_schedules()

    return ScheduleListResponse(
        schedules=[ScheduleResponse.from_schedule(s) for s in schedules]
    )

@router.post("/schedules", response_model=ScheduleResponse)
async def create_schedule(request: CreateScheduleRequest):
    """
    Create a new backup schedule.

    Cron expression format: minute hour day month weekday
    Examples:
    - "0 2 * * *" = 2:00 AM every day
    - "0 0 * * 0" = Midnight every Sunday
    - "0 0 1 * *" = Midnight on the 1st of every month
    """
    service = get_backup_service()

    try:
        backup_type = BackupType(request.backup_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid backup type: {request.backup_type}. "
                "Use: database, config, or full"
            ),
        ) from None

    schedule = service.create_schedule(
        name=request.name,
        backup_type=backup_type,
        cron_expression=request.cron_expression,
        retention_days=request.retention_days,
    )

    return ScheduleResponse.from_schedule(schedule)

@router.put("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: str, request: UpdateScheduleRequest):
    """
    Update a backup schedule.
    """
    service = get_backup_service()

    schedule = service.update_schedule(
        schedule_id=schedule_id,
        enabled=request.enabled,
        cron_expression=request.cron_expression,
        retention_days=request.retention_days,
    )

    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return ScheduleResponse.from_schedule(schedule)

@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """
    Delete a backup schedule.
    """
    service = get_backup_service()

    deleted = service.delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return {"message": "Schedule deleted", "schedule_id": schedule_id}

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/stats", response_model=BackupStatsResponse)
async def get_backup_stats():
    """
    Get backup statistics.
    """
    service = get_backup_service()
    stats = service.get_statistics()

    return BackupStatsResponse(**stats)

@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_old_backups(request: CleanupRequest):
    """
    Manually cleanup old backups.

    Deletes all backups older than the specified retention period.
    """
    service = get_backup_service()

    deleted_count = await service.cleanup_old_backups(request.retention_days)

    return CleanupResponse(
        deleted_count=deleted_count,
        retention_days=request.retention_days,
    )

@router.post("/scheduler/start")
async def start_scheduler():
    """
    Start the backup scheduler.
    """
    service = get_backup_service()
    await service.start_scheduler()

    return {"message": "Backup scheduler started"}

@router.post("/scheduler/stop")
async def stop_scheduler():
    """
    Stop the backup scheduler.
    """
    service = get_backup_service()
    await service.stop_scheduler()

    return {"message": "Backup scheduler stopped"}

@router.post(
    "/rag/index",
    response_model=BackupResponse,
    responses={
        500: {"description": "Failed to create RAG index backup"},
    },
)
async def create_rag_index_backup(request: CreateBackupRequest):
    """
    Create a backup of the RAG BM25 index.
    """
    from resync.knowledge.retrieval.hybrid_retriever import (
        INDEX_STORAGE_PATH,
        BM25Index,
    )

    service = get_backup_service()

    index = BM25Index.load(INDEX_STORAGE_PATH)
    backup = service.create_rag_index_backup(index, request.collection_name or "bm25")

    if backup:
        return BackupResponse.from_backup_info(backup)

    raise HTTPException(status_code=500, detail="Failed to create RAG index backup")

@router.get("/rag/list", response_model=BackupListResponse)
async def list_rag_backups():
    """
    List all RAG index backups.
    """
    service = get_backup_service()
    backups = service.list_rag_backups()

    return BackupListResponse(
        backups=[BackupResponse.from_backup_info(b) for b in backups],
        total=len(backups),
    )

@router.post(
    "/rag/{backup_id}/restore",
    responses={
        404: {"description": "Backup not found"},
    },
)
async def restore_rag_index_backup(backup_id: str):
    """
    Restore RAG index from backup.
    """
    service = get_backup_service()

    success = service.restore_rag_index_backup(backup_id)

    if success:
        return {"message": f"RAG index restored from backup {backup_id}"}

    raise HTTPException(status_code=404, detail=f"Backup {backup_id} not found")
