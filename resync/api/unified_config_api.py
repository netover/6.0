# pylint: disable=broad-except
"""
Unified Configuration API

REST API for managing ALL Resync configurations with hot reload.
Ensures zero-blocking execution for Python 3.14 + FastAPI.

Endpoints:
- GET  /api/admin/config/all
- GET  /api/admin/config/status
- POST /api/admin/config/reload
- GET  /api/admin/config/backups
- POST /api/admin/config/backups/{filename}/restore
- GET  /api/admin/config/{config_name}
- POST /api/admin/config/{config_name}/update

Security: All endpoints require JWT with role 'admin'.
          Uses resync.api.core.security.require_role (existing infrastructure).

Author: Resync Team
Version: 6.1.0

Changes from 5.9.9 (360° Audit):
- [P0-01] Auth via require_role(["admin"]) — existing project infrastructure
- [P0-02] config_name/section validated with strict pattern via FastApiPath/Field
- [P1-01] TOCTOU fixed: O_NOFOLLOW + atomic fd in _perform_restore
- [P1-02] .tmp_* cleanup on any restore failure
- [P1-03] Rate limiting via slowapi (already in requirements.txt)
- [P1-04] asyncio.Lock on /reload against concurrent calls
- [P1-05] data dict limited to 100 keys with key pattern validation
- [P2-01] 404 no longer echoes user-supplied config_name
- [P2-02] _BACKUP_DIR as module constant (single source of truth)
- [P2-03] ConfigUpdateRequest uses ConfigDict(extra="forbid")
- [P2-04] get_config_manager() wrapped in asyncio.to_thread
"""

import asyncio
import contextlib
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi import Path as FastApiPath
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

# [P0-01] Use existing project auth infrastructure — do NOT reinvent
from resync.api.core.security import require_role
from resync.core.unified_config import get_config_manager

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# [P2-02] Single source of truth — avoids silent divergence if file is moved
_BACKUP_DIR: Path = Path(__file__).parent.parent.parent / "config" / "backups"

# [P1-04] Serialize concurrent reload_all_configs calls
_reload_lock: asyncio.Lock = asyncio.Lock()

# [P1-03] slowapi already in requirements.txt>=0.1.9
limiter = Limiter(key_func=get_remote_address)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin/config", tags=["configuration"])

# Strict identifier pattern reused across path params and field validators
_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration attributes."""

    # [P2-03] extra="forbid" — unknown fields raise 422
    # [P0-02] section validated with strict pattern
    model_config = ConfigDict(extra="forbid")

    section: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Configuration section (alphanumeric, _ and - only)",
    )
    data: dict[str, Any] = Field(..., description="Key-value pairs to apply")
    create_backup: bool = Field(True)

    @field_validator("data")
    @classmethod
    def validate_data_shape(cls, v: dict[str, Any]) -> dict[str, Any]:
        """[P1-05] Limit key count and enforce key pattern."""
        if len(v) > 100:
            raise ValueError("data must contain at most 100 keys")
        for key in v:
            if not _SAFE_ID.match(str(key)):
                raise ValueError(
                    f"Invalid key {key!r}: only [a-zA-Z0-9_-] allowed"
                )
        return v


class ConfigReloadResponse(BaseModel):
    """Response structure for config reload events."""

    status: str
    configs_loaded: list[str]
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _get_manager() -> Any:
    """[P2-04] Guard against lazy-init blocking the event loop."""
    return await asyncio.to_thread(get_config_manager)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/all", response_model=dict[str, Any])
async def get_all_configs(
    # [P0-01] require_role from resync.api.core.security
    # includes: decode_access_token → JTI revocation → role check
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """Get all configuration files. Requires: JWT role 'admin'."""
    try:
        manager = await _get_manager()
        configs = manager.get_all_configs()
        return {
            "status": "success",
            "configs": configs,
            "hot_reload_active": manager.observer is not None,
        }
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("Failed to get configs", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error managing configurations.",
        ) from e


@router.post("/{config_name}/update", response_model=dict[str, Any])
async def update_config(
    request: ConfigUpdateRequest,
    config_name: str = FastApiPath(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",  # [P0-02]
        description="Configuration identifier",
    ),
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """Update configuration with zero-downtime hot reload."""
    try:
        manager = await _get_manager()

        if config_name not in manager.CONFIG_FILES:
            # [P2-01] Generic — do not confirm valid names
            raise HTTPException(status_code=404, detail="Configuration not found")

        await manager.update_config(
            config_name=config_name,
            section=request.section,
            data=request.data,
            create_backup=request.create_backup,
        )
        return {
            "status": "success",
            "message": "Config updated with hot reload (persists across restarts)",
            "config_name": config_name,
            "section": request.section,
            "updated_fields": list(request.data.keys()),
            "hot_reload": True,
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error(
            "Failed to update config",
            config_name=config_name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to update configuration. Check server logs.",
        ) from e


@router.get("/status", response_model=dict[str, Any])
async def get_config_status(
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """Get configuration subsystem health and runtime status."""
    try:
        manager = await _get_manager()
        configs_loaded = list(manager.configs.keys())
        config_files = {
            name: str(path) for name, path in manager.CONFIG_FILES.items()
        }
        return {
            "status": "operational",
            "hot_reload_active": manager.observer is not None,
            "configs_loaded": configs_loaded,
            "config_files": config_files,
            "total_configs": len(configs_loaded),
        }
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("Failed to get config status", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving configuration status.",
        ) from e


@router.post("/reload", response_model=ConfigReloadResponse)
@limiter.limit("5/minute")  # [P1-03]
async def reload_all_configs(
    request: Request,  # required by slowapi
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> ConfigReloadResponse:
    """Force hot-reload for all tracked configs. Rate: 5/min per IP."""
    async with _reload_lock:  # [P1-04]
        manager = await _get_manager()
        configs_loaded: list[str] = []
        errors: list[str] = []

        for name, path in manager.CONFIG_FILES.items():
            try:
                exists = await asyncio.to_thread(path.exists)
                if exists:
                    await manager.reload_config_file(path)
                    configs_loaded.append(name)
                else:
                    errors.append(f"Config file not found: {name}")
            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error(
                    "Failed to reload config file",
                    config_name=name,
                    error=str(e),
                    exc_info=True,
                )
                errors.append(f"Failed to reload {name}: {str(e)}")

        return ConfigReloadResponse(
            status="success" if not errors else "partial",
            configs_loaded=configs_loaded,
            errors=errors,
        )


@router.get("/backups", response_model=dict[str, Any])
async def list_config_backups(
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """List all configuration backups on disk (non-blocking)."""
    try:
        # [P2-02] _BACKUP_DIR is the single module-level constant
        dir_exists = await asyncio.to_thread(_BACKUP_DIR.exists)
        if not dir_exists:
            return {"status": "success", "backups": [], "total": 0}

        def _get_backups() -> list[dict[str, Any]]:
            results = []
            for backup_file in _BACKUP_DIR.glob("*.toml.bak"):
                stat_info = backup_file.stat()
                results.append({
                    "filename": backup_file.name,
                    "size_bytes": stat_info.st_size,
                    "modified": stat_info.st_mtime,
                    "path": str(backup_file),
                })
            results.sort(key=lambda x: x["modified"], reverse=True)
            return results

        backups = await asyncio.to_thread(_get_backups)
        return {
            "status": "success",
            "backups": backups,
            "total": len(backups),
            "backup_dir": str(_BACKUP_DIR),
        }
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("Failed to list backups", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error listing backups.",
        ) from e


@router.post("/backups/{filename}/restore", response_model=dict[str, Any])
@limiter.limit("3/minute")  # [P1-03]
async def restore_config_backup(
    request: Request,
    filename: str = FastApiPath(..., description="Target backup file name"),
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """Restore config from backup atomically. Rate: 3/min per IP."""
    try:
        # [P0-02] Strict upfront validation — fully mitigates path traversal
        match = re.match(r"^([a-zA-Z0-9]+)_[a-zA-Z0-9_]+\.toml\.bak$", filename)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid backup filename. "
                    "Expected: {config_name}_{timestamp}.toml.bak"
                ),
            )
        config_name = match.group(1)

        manager = await _get_manager()
        if config_name not in manager.CONFIG_FILES:
            raise HTTPException(
                status_code=400,
                detail="Unknown configuration identifier in backup filename",
            )

        backup_file = _BACKUP_DIR / filename  # [P2-02]
        config_file = manager.CONFIG_FILES[config_name]
        timestamp = int(datetime.now(timezone.utc).timestamp())
        current_backup = (
            _BACKUP_DIR / f"{config_name}_before_restore_{timestamp}.toml.bak"
        )
        temp_restore = _BACKUP_DIR / f".tmp_{config_file.name}_{timestamp}"

        def _perform_restore() -> None:
            """
            [P1-01] O_NOFOLLOW prevents symlink TOCTOU.
            [P1-02] contextlib.suppress cleans up .tmp_* on any failure.
            """
            try:
                fd = os.open(str(backup_file), os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Backup file not found: {backup_file}"
                ) from exc
            except OSError as exc:
                raise OSError(
                    f"Backup file is a symlink or inaccessible: {backup_file}"
                ) from exc

            try:
                with os.fdopen(fd, "rb") as src:
                    temp_restore.write_bytes(src.read())
                shutil.copy2(config_file, current_backup)
                temp_restore.replace(config_file)  # atomic on POSIX (rename(2))
            except Exception:
                # [P1-02] Always clean orphaned temp file
                with contextlib.suppress(OSError):
                    temp_restore.unlink(missing_ok=True)
                raise

        await asyncio.to_thread(_perform_restore)
        await manager.reload_config_file(config_file)

        return {
            "status": "success",
            "message": "Config restored from backup via zero-downtime hot reload",
            "config_name": config_name,
            "backup_file": filename,
            "current_backed_up_to": current_backup.name,
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error(
            "Failed to restore backup",
            backup_file=filename,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to restore configuration backup safely.",
        ) from e


@router.get("/{config_name}", response_model=dict[str, Any])
async def get_config(
    config_name: str = FastApiPath(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",  # [P0-02]
        description="Configuration identifier",
    ),
    section: str | None = None,
    _: dict[str, Any] = Depends(require_role(["admin"])),
) -> dict[str, Any]:
    """Get a specific loaded configuration, optionally filtered by section."""
    try:
        manager = await _get_manager()

        if config_name not in manager.CONFIG_FILES:
            raise HTTPException(
                status_code=404,
                detail="Configuration not found",  # [P2-01]
            )

        config = manager.get_config(config_name, section)
        return {
            "status": "success",
            "config_name": config_name,
            "section": section,
            "data": config,
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error(
            "Failed to retrieve config",
            config_name=config_name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving configuration.",
        ) from e
