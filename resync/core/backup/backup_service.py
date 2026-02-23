# pylint: skip-file
# mypy: ignore-errors
"""
Backup Service for PostgreSQL and System Configuration.

Provides:
- PostgreSQL database backup (pg_dump compressed)
- System configuration backup (YAML, ENV, configs)
- Scheduled backups with cron-like expressions
- Backup listing, download, and cleanup

Usage:
    from resync.core.backup import get_backup_service

    service = get_backup_service()

    # Manual database backup
    backup = await service.create_database_backup()

    # System config backup
    config_backup = await service.create_config_backup()

    # List all backups
    backups = await service.list_backups()
"""

from __future__ import annotations

import asyncio
from resync.core.task_tracker import track_task
import contextlib
import hashlib
import json
import os
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from resync.core.database.config import get_database_config
from resync.core.structured_logger import get_logger
import aiofiles

logger = get_logger(__name__)


class BackupType(str, Enum):
    """Type of backup."""

    DATABASE = "database"
    CONFIG = "config"
    FULL = "full"
    RAG_INDEX = "rag_index"


class BackupStatus(str, Enum):
    """Backup status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackupInfo:
    """Information about a backup."""

    id: str
    type: BackupType
    status: BackupStatus
    filename: str
    filepath: str
    size_bytes: int = 0
    size_human: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float = 0
    checksum_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration_seconds,
            "checksum_sha256": self.checksum_sha256,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class BackupSchedule:
    """Backup schedule configuration."""

    id: str
    name: str
    backup_type: BackupType
    cron_expression: str  # e.g., "0 2 * * *" for 2 AM daily
    enabled: bool = True
    retention_days: int = 30
    last_run: datetime | None = None
    next_run: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "backup_type": self.backup_type.value,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "retention_days": self.retention_days,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "created_at": self.created_at.isoformat(),
        }


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def _generate_backup_id() -> str:
    """Generate unique backup ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.sha256(os.urandom(8)).hexdigest()[:6]
    return f"{timestamp}_{random_suffix}"


def _calculate_sha256(filepath: str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class BackupService:
    """
    Service for managing database and configuration backups.

    Features:
    - PostgreSQL backup via pg_dump
    - Configuration files backup
    - Automatic compression (ZIP/GZIP)
    - Scheduled backups
    - Retention policy
    """

    _instance: BackupService | None = None

    def __new__(cls) -> BackupService:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Backup storage directory
        self._backup_dir = Path(os.getenv("BACKUP_DIR", "/var/backups/resync"))
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self._db_backup_dir = self._backup_dir / "database"
        self._config_backup_dir = self._backup_dir / "config"
        self._db_backup_dir.mkdir(exist_ok=True)
        self._config_backup_dir.mkdir(exist_ok=True)

        # Backup metadata file
        self._metadata_file = self._backup_dir / "backups.json"

        # Schedules
        self._schedules: dict[str, BackupSchedule] = {}
        self._scheduler_task: asyncio.Task | None = None

        # Load existing metadata
        self._backups: dict[str, BackupInfo] = {}
        self._load_metadata()

        # Configuration paths to backup
        self._config_paths = [
            "config/",
            "resync/prompts/",
            ".env",
            ".env.local",
            "alembic.ini",
            "requirements.txt",
            "setup.cfg",
            "ruff.toml",
        ]

        self._initialized = True
        logger.info("backup_service_initialized", backup_dir=str(self._backup_dir))

    def _load_metadata(self) -> None:
        """Load backup metadata from file."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)

                for backup_data in data.get("backups", []):
                    backup = BackupInfo(
                        id=backup_data["id"],
                        type=BackupType(backup_data["type"]),
                        status=BackupStatus(backup_data["status"]),
                        filename=backup_data["filename"],
                        filepath=backup_data.get("filepath", ""),
                        size_bytes=backup_data.get("size_bytes", 0),
                        size_human=backup_data.get("size_human", ""),
                        created_at=datetime.fromisoformat(backup_data["created_at"]),
                        completed_at=datetime.fromisoformat(backup_data["completed_at"])
                        if backup_data.get("completed_at")
                        else None,
                        duration_seconds=backup_data.get("duration_seconds", 0),
                        checksum_sha256=backup_data.get("checksum_sha256", ""),
                        metadata=backup_data.get("metadata", {}),
                        error=backup_data.get("error"),
                    )
                    self._backups[backup.id] = backup

                for schedule_data in data.get("schedules", []):
                    schedule = BackupSchedule(
                        id=schedule_data["id"],
                        name=schedule_data["name"],
                        backup_type=BackupType(schedule_data["backup_type"]),
                        cron_expression=schedule_data["cron_expression"],
                        enabled=schedule_data.get("enabled", True),
                        retention_days=schedule_data.get("retention_days", 30),
                        last_run=datetime.fromisoformat(schedule_data["last_run"])
                        if schedule_data.get("last_run")
                        else None,
                        next_run=datetime.fromisoformat(schedule_data["next_run"])
                        if schedule_data.get("next_run")
                        else None,
                    )
                    self._schedules[schedule.id] = schedule

                logger.debug(
                    "backup_metadata_loaded",
                    backups=len(self._backups),
                    schedules=len(self._schedules),
                )
            except Exception as e:
                logger.warning("backup_metadata_load_failed", error=str(e))

    def _save_metadata(self) -> None:
        """Save backup metadata to file."""
        try:
            data = {
                "backups": [b.to_dict() for b in self._backups.values()],
                "schedules": [s.to_dict() for s in self._schedules.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self._metadata_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("backup_metadata_saved")
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("backup_metadata_save_failed", error=str(e))

    async def create_database_backup(
        self,
        description: str = "",
        compress: bool = True,
    ) -> BackupInfo:
        """
        Create a PostgreSQL database backup.

        Uses pg_dump to create a SQL dump, then compresses with gzip.

        Args:
            description: Optional description for the backup
            compress: Whether to compress the backup (default: True)

        Returns:
            BackupInfo with backup details
        """
        backup_id = _generate_backup_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Get database configuration
        db_config = get_database_config()

        # Determine filename
        extension = ".sql.gz" if compress else ".sql"
        filename = f"resync_db_{timestamp}{extension}"
        filepath = self._db_backup_dir / filename

        backup = BackupInfo(
            id=backup_id,
            type=BackupType.DATABASE,
            status=BackupStatus.IN_PROGRESS,
            filename=filename,
            filepath=str(filepath),
            metadata={"description": description, "database": db_config.name},
        )
        self._backups[backup_id] = backup

        start_time = datetime.now(timezone.utc)

        try:
            # Build pg_dump command
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config.password

            pg_dump_cmd = [
                "pg_dump",
                "-h",
                db_config.host,
                "-p",
                str(db_config.port),
                "-U",
                db_config.user,
                "-d",
                db_config.name,
                "--format=plain",
                "--no-owner",
                "--no-privileges",
            ]

            logger.info(
                "database_backup_started", backup_id=backup_id, database=db_config.name
            )

            if compress:
                # Pipe to gzip
                async with aiofiles.open(filepath, "wb") as f:
                    pg_dump = await asyncio.create_subprocess_exec(
                        *pg_dump_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    gzip_proc = await asyncio.create_subprocess_exec(
                        "gzip",
                        "-9",
                        stdin=pg_dump.stdout,
                        stdout=f,
                        stderr=subprocess.PIPE,
                    )

                    _, pg_stderr = await pg_dump.communicate()
                    _, gzip_stderr = await gzip_proc.communicate()

                    if pg_dump.returncode != 0:
                        raise RuntimeError(f"pg_dump failed: {pg_stderr.decode()}")
                    if gzip_proc.returncode != 0:
                        raise RuntimeError(f"gzip failed: {gzip_stderr.decode()}")
            else:
                # Direct output
                async with aiofiles.open(filepath, "w") as f:
                    pg_dump = await asyncio.create_subprocess_exec(
                        *pg_dump_cmd,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    _, stderr = await pg_dump.communicate()

                    if pg_dump.returncode != 0:
                        raise RuntimeError(f"pg_dump failed: {stderr.decode()}")

            # Get file info
            stat = os.stat(filepath)
            backup.size_bytes = stat.st_size
            backup.size_human = _human_size(stat.st_size)
            backup.checksum_sha256 = _calculate_sha256(str(filepath))
            backup.status = BackupStatus.COMPLETED
            backup.completed_at = datetime.now(timezone.utc)
            backup.duration_seconds = (backup.completed_at - start_time).total_seconds()

            logger.info(
                "database_backup_completed",
                backup_id=backup_id,
                size=backup.size_human,
                duration=backup.duration_seconds,
            )

        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.error = str(e)
            backup.completed_at = datetime.now(timezone.utc)

            logger.error("database_backup_failed", backup_id=backup_id, error=str(e))

            # Clean up partial file
            if filepath.exists():
                filepath.unlink()

        self._save_metadata()
        return backup

    def create_config_backup(
        self,
        description: str = "",
        include_env: bool = True,
    ) -> BackupInfo:
        """
        Create a backup of system configuration files.

        Includes:
        - config/ directory
        - prompts/ directory
        - .env files (if include_env=True)
        - requirements.txt, setup.cfg, etc.

        Args:
            description: Optional description
            include_env: Whether to include .env files

        Returns:
            BackupInfo with backup details
        """
        backup_id = _generate_backup_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        filename = f"resync_config_{timestamp}.zip"
        filepath = self._config_backup_dir / filename

        backup = BackupInfo(
            id=backup_id,
            type=BackupType.CONFIG,
            status=BackupStatus.IN_PROGRESS,
            filename=filename,
            filepath=str(filepath),
            metadata={"description": description, "include_env": include_env},
        )
        self._backups[backup_id] = backup

        start_time = datetime.now(timezone.utc)

        try:
            # Get project root
            project_root = Path(__file__).parent.parent.parent.parent

            files_added = []

            with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
                for config_path in self._config_paths:
                    full_path = project_root / config_path

                    # Skip .env files if not included
                    if not include_env and config_path.startswith(".env"):
                        continue

                    if full_path.is_file():
                        arcname = config_path
                        zf.write(full_path, arcname)
                        files_added.append(arcname)

                    elif full_path.is_dir() and full_path.exists():
                        for file_path in full_path.rglob("*"):
                            if file_path.is_file() and "__pycache__" not in str(
                                file_path
                            ):
                                arcname = str(file_path.relative_to(project_root))
                                zf.write(file_path, arcname)
                                files_added.append(arcname)

                # Add manifest
                manifest = {
                    "backup_id": backup_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "files": files_added,
                    "description": description,
                }
                zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

            # Get file info
            stat = os.stat(filepath)
            backup.size_bytes = stat.st_size
            backup.size_human = _human_size(stat.st_size)
            backup.checksum_sha256 = _calculate_sha256(str(filepath))
            backup.status = BackupStatus.COMPLETED
            backup.completed_at = datetime.now(timezone.utc)
            backup.duration_seconds = (backup.completed_at - start_time).total_seconds()
            backup.metadata["files_count"] = len(files_added)

            logger.info(
                "config_backup_completed",
                backup_id=backup_id,
                files=len(files_added),
                size=backup.size_human,
            )

        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.error = str(e)
            backup.completed_at = datetime.now(timezone.utc)

            logger.error("config_backup_failed", backup_id=backup_id, error=str(e))

            if filepath.exists():
                filepath.unlink()

        self._save_metadata()
        return backup

    async def create_full_backup(self, description: str = "") -> BackupInfo:
        """
        Create a full backup (database + config).

        Creates both backups and packages them together.
        """
        backup_id = _generate_backup_id()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        filename = f"resync_full_{timestamp}.zip"
        filepath = self._backup_dir / filename

        backup = BackupInfo(
            id=backup_id,
            type=BackupType.FULL,
            status=BackupStatus.IN_PROGRESS,
            filename=filename,
            filepath=str(filepath),
            metadata={"description": description},
        )
        self._backups[backup_id] = backup

        start_time = datetime.now(timezone.utc)

        try:
            # Create individual backups
            db_backup = await self.create_database_backup(
                description=f"Part of full backup {backup_id}"
            )
            config_backup = self.create_config_backup(
                description=f"Part of full backup {backup_id}"
            )

            # Package together
            with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
                if db_backup.status == BackupStatus.COMPLETED:
                    zf.write(db_backup.filepath, f"database/{db_backup.filename}")

                if config_backup.status == BackupStatus.COMPLETED:
                    zf.write(config_backup.filepath, f"config/{config_backup.filename}")

                # Add manifest
                manifest = {
                    "backup_id": backup_id,
                    "type": "full",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "components": {
                        "database": db_backup.to_dict()
                        if db_backup.status == BackupStatus.COMPLETED
                        else None,
                        "config": config_backup.to_dict()
                        if config_backup.status == BackupStatus.COMPLETED
                        else None,
                    },
                }
                zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

            # Get file info
            stat = os.stat(filepath)
            backup.size_bytes = stat.st_size
            backup.size_human = _human_size(stat.st_size)
            backup.checksum_sha256 = _calculate_sha256(str(filepath))
            backup.status = BackupStatus.COMPLETED
            backup.completed_at = datetime.now(timezone.utc)
            backup.duration_seconds = (backup.completed_at - start_time).total_seconds()
            backup.metadata["db_backup_id"] = db_backup.id
            backup.metadata["config_backup_id"] = config_backup.id

            logger.info(
                "full_backup_completed",
                backup_id=backup_id,
                size=backup.size_human,
            )

        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.error = str(e)
            backup.completed_at = datetime.now(timezone.utc)

            logger.error("full_backup_failed", backup_id=backup_id, error=str(e))

        self._save_metadata()
        return backup

    def list_backups(
        self,
        backup_type: BackupType | None = None,
        status: BackupStatus | None = None,
        limit: int = 100,
    ) -> list[BackupInfo]:
        """
        List all backups with optional filters.

        Args:
            backup_type: Filter by type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of BackupInfo objects
        """
        backups = list(self._backups.values())

        if backup_type:
            backups = [b for b in backups if b.type == backup_type]

        if status:
            backups = [b for b in backups if b.status == status]

        # Sort by creation date (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)

        return backups[:limit]

    def get_backup(self, backup_id: str) -> BackupInfo | None:
        """Get a specific backup by ID."""
        return self._backups.get(backup_id)

    def get_backup_filepath(self, backup_id: str) -> Path | None:
        """Get the file path for a backup."""
        backup = self._backups.get(backup_id)
        if backup and backup.filepath:
            path = Path(backup.filepath)
            try:
                # v5.9.7: Prevent path traversal by ensuring the backup file lives
                # under the configured backup directory.
                resolved = path.resolve()
                backup_root = self._backup_dir.resolve()
                if not str(resolved).startswith(str(backup_root) + os.sep):
                    logger.warning(
                        "backup_filepath_outside_root",
                        backup_id=backup_id,
                        path=str(resolved),
                        root=str(backup_root),
                    )
                    return None

                # Optional: ensure filename matches metadata
                if backup.filename and resolved.name != backup.filename:
                    logger.warning(
                        "backup_filename_mismatch",
                        backup_id=backup_id,
                        resolved_name=resolved.name,
                        expected=backup.filename,
                    )
                    return None

                if resolved.exists():
                    return resolved
            except Exception as e:
                logger.warning(
                    "backup_filepath_validation_failed",
                    backup_id=backup_id,
                    error=str(e),
                )
        return None

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.

        Args:
            backup_id: ID of backup to delete

        Returns:
            True if deleted, False if not found
        """
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        # Delete file
        if backup.filepath:
            path = Path(backup.filepath)
            if path.exists():
                path.unlink()

        # Remove from metadata
        del self._backups[backup_id]
        self._save_metadata()

        logger.info("backup_deleted", backup_id=backup_id)
        return True

    async def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """
        Delete backups older than retention period.

        Args:
            retention_days: Number of days to keep backups

        Returns:
            Number of backups deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        deleted = 0

        for backup_id, backup in list(self._backups.items()):
            if backup.created_at < cutoff:
                self.delete_backup(backup_id)
                deleted += 1

        logger.info(
            "backup_cleanup_completed", deleted=deleted, retention_days=retention_days
        )
        return deleted

    # =========================================================================
    # SCHEDULING
    # =========================================================================

    def create_schedule(
        self,
        name: str,
        backup_type: BackupType,
        cron_expression: str,
        retention_days: int = 30,
    ) -> BackupSchedule:
        """
        Create a backup schedule.

        Args:
            name: Schedule name
            backup_type: Type of backup
            cron_expression: Cron expression (e.g., "0 2 * * *" for 2 AM daily)
            retention_days: How long to keep backups

        Returns:
            BackupSchedule object
        """
        schedule_id = _generate_backup_id()

        schedule = BackupSchedule(
            id=schedule_id,
            name=name,
            backup_type=backup_type,
            cron_expression=cron_expression,
            retention_days=retention_days,
            next_run=self._calculate_next_run(cron_expression),
        )

        self._schedules[schedule_id] = schedule
        self._save_metadata()

        logger.info(
            "backup_schedule_created",
            schedule_id=schedule_id,
            name=name,
            cron=cron_expression,
        )

        return schedule

    def list_schedules(self) -> list[BackupSchedule]:
        """List all backup schedules."""
        return list(self._schedules.values())

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a backup schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._save_metadata()
            logger.info("backup_schedule_deleted", schedule_id=schedule_id)
            return True
        return False

    def update_schedule(
        self,
        schedule_id: str,
        enabled: bool | None = None,
        cron_expression: str | None = None,
        retention_days: int | None = None,
    ) -> BackupSchedule | None:
        """Update a backup schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None

        if enabled is not None:
            schedule.enabled = enabled
        if cron_expression is not None:
            schedule.cron_expression = cron_expression
            schedule.next_run = self._calculate_next_run(cron_expression)
        if retention_days is not None:
            schedule.retention_days = retention_days

        self._save_metadata()
        return schedule

    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """
        Calculate next run time from cron expression.

        Simplified parser for common patterns:
        - "0 2 * * *" = 2:00 AM daily
        - "0 0 * * 0" = Midnight on Sunday
        - "0 0 1 * *" = Midnight on 1st of month
        """
        try:
            parts = cron_expression.split()
            if len(parts) != 5:
                raise ValueError("Invalid cron expression")

            minute, hour, day, month, weekday = parts

            now = datetime.now(timezone.utc)
            next_run = now.replace(second=0, microsecond=0)

            # Set hour and minute
            if minute != "*":
                next_run = next_run.replace(minute=int(minute))
            if hour != "*":
                next_run = next_run.replace(hour=int(hour))

            # If already passed today, move to tomorrow
            if next_run <= now:
                next_run += timedelta(days=1)

            return next_run

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            # Default to 2 AM tomorrow
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            return tomorrow.replace(hour=2, minute=0, second=0, microsecond=0)

    def start_scheduler(self, tg: asyncio.TaskGroup | None = None) -> None:
        """Start the backup scheduler.

        Args:
            tg: Optional TaskGroup to run the scheduler in
        """
        if self._scheduler_task:
            return

        if tg:
            self._scheduler_task = tg.create_task(
                self._run_scheduler(), name="run_scheduler"
            )
        else:
            self._scheduler_task = track_task(
                self._run_scheduler(), name="run_scheduler"
            )

        logger.info(
            "backup_scheduler_started", method="task_group" if tg else "track_task"
        )

    async def stop_scheduler(self) -> None:
        """Stop the backup scheduler."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
            logger.info("backup_scheduler_stopped")

    async def _run_scheduler(self) -> None:
        """Background task that checks and runs scheduled backups."""
        while True:
            try:
                now = datetime.now(timezone.utc)

                for schedule in self._schedules.values():
                    if not schedule.enabled:
                        continue

                    if schedule.next_run and now >= schedule.next_run:
                        # Run backup
                        logger.info(
                            "scheduled_backup_starting",
                            schedule_id=schedule.id,
                            name=schedule.name,
                        )

                        try:
                            if schedule.backup_type == BackupType.DATABASE:
                                await self.create_database_backup(
                                    description=f"Scheduled: {schedule.name}"
                                )
                            elif schedule.backup_type == BackupType.CONFIG:
                                self.create_config_backup(
                                    description=f"Scheduled: {schedule.name}"
                                )
                            else:
                                await self.create_full_backup(
                                    description=f"Scheduled: {schedule.name}"
                                )

                            # Cleanup old backups
                            await self.cleanup_old_backups(schedule.retention_days)

                        except Exception as e:
                            logger.error(
                                "scheduled_backup_failed",
                                schedule_id=schedule.id,
                                error=str(e),
                            )

                        # Update schedule
                        schedule.last_run = now
                        schedule.next_run = self._calculate_next_run(
                            schedule.cron_expression
                        )
                        self._save_metadata()

                # Check every minute
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduler_error", error=str(e))
                await asyncio.sleep(60)

    def get_statistics(self) -> dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(b.size_bytes for b in self._backups.values())

        by_type = {}
        for bt in BackupType:
            backups = [b for b in self._backups.values() if b.type == bt]
            by_type[bt.value] = {
                "count": len(backups),
                "total_size": sum(b.size_bytes for b in backups),
            }

        return {
            "total_backups": len(self._backups),
            "total_size_bytes": total_size,
            "total_size_human": _human_size(total_size),
            "by_type": by_type,
            "active_schedules": sum(1 for s in self._schedules.values() if s.enabled),
            "backup_directory": str(self._backup_dir),
        }

    def create_rag_index_backup(
        self,
        index,
        collection_name: str = "bm25",
    ) -> BackupInfo | None:
        """
        Create a backup of the RAG BM25 index.

        Args:
            index: BM25Index instance to backup
            collection_name: Name of the collection for identification

        Returns:
            BackupInfo object or None if failed
        """
        from resync.knowledge.retrieval.hybrid_retriever import INDEX_STORAGE_PATH

        backup_id = _generate_backup_id()
        timestamp = datetime.now(timezone.utc)

        try:
            source_path = INDEX_STORAGE_PATH

            if not os.path.exists(source_path):
                logger.warning("rag_index_backup_source_not_found", path=source_path)
                return None

            filename = (
                f"rag_{collection_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.bin.gz"
            )
            dest_path = self._backup_dir / filename

            import shutil

            shutil.copy2(source_path, dest_path)

            file_size = os.path.getsize(dest_path)
            file_hash = _calculate_sha256(str(dest_path))

            backup = BackupInfo(
                id=backup_id,
                type=BackupType.RAG_INDEX,
                status=BackupStatus.COMPLETED,
                filename=os.path.basename(dest_path),
                filepath=str(dest_path),
                size_bytes=file_size,
                checksum_sha256=file_hash,
                created_at=timestamp,
                metadata={
                    "collection": collection_name,
                    "source_path": source_path,
                },
            )

            self._backups[backup_id] = backup
            self._save_metadata()

            logger.info(
                "rag_index_backup_created",
                backup_id=backup_id,
                path=str(dest_path),
                size_bytes=file_size,
            )

            return backup

        except Exception as e:
            logger.error("rag_index_backup_failed", backup_id=backup_id, error=str(e))
            return None

    def restore_rag_index_backup(
        self,
        backup_id: str,
        target_path: str | None = None,
    ) -> bool:
        """
        Restore RAG index from backup.

        Args:
            backup_id: ID of the backup to restore
            target_path: Optional target path (defaults to INDEX_STORAGE_PATH)

        Returns:
            True if successful
        """
        from resync.knowledge.retrieval.hybrid_retriever import INDEX_STORAGE_PATH

        backup = self._backups.get(backup_id)

        if not backup or backup.type != BackupType.RAG_INDEX:
            logger.error("rag_restore_backup_not_found", backup_id=backup_id)
            return False

        try:
            target = target_path or INDEX_STORAGE_PATH

            import shutil

            shutil.copy2(backup.filepath, target)

            logger.info(
                "rag_index_restored",
                backup_id=backup_id,
                target=target,
            )

            return True

        except Exception as e:
            logger.error("rag_restore_failed", backup_id=backup_id, error=str(e))
            return False

    def list_rag_backups(self) -> list[BackupInfo]:
        """
        List all RAG index backups.

        Returns:
            List of RAG backup info objects
        """
        return [b for b in self._backups.values() if b.type == BackupType.RAG_INDEX]


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_backup_service: BackupService | None = None


def get_backup_service() -> BackupService:
    """Get or create the backup service singleton."""
    global _backup_service
    if _backup_service is None:
        _backup_service = BackupService()
    return _backup_service
