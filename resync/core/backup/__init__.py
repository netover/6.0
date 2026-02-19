"""
Backup Module for Resync.

Provides database and configuration backup capabilities:
- PostgreSQL database backup via pg_dump
- System configuration backup
- Scheduled backups with retention policy
- Backup listing and management

Usage:
    from resync.core.backup import get_backup_service

    service = get_backup_service()
    backup = await service.create_database_backup()
"""

from resync.core.backup.backup_service import (
    BackupInfo,
    BackupSchedule,
    BackupService,
    BackupStatus,
    BackupType,
    get_backup_service,
)

__all__ = [
    "BackupService",
    "BackupInfo",
    "BackupSchedule",
    "BackupType",
    "BackupStatus",
    "get_backup_service",
]
