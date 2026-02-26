# pylint: disable=broad-except
"""
Unified Configuration API

REST API for managing ALL Resync configurations with hot reload.
Ensures zero-blocking execution for Python 3.14 + FastAPI.

Endpoints:
- GET  /api/admin/config/all                          - Get all configs
- GET  /api/admin/config/status                       - Get config system status
- POST /api/admin/config/reload                       - Force reload all configs
- GET  /api/admin/config/backups                      - List backups
- POST /api/admin/config/backups/{filename}/restore   - Restore backup
- GET  /api/admin/config/{config_name}                - Get specific config
- POST /api/admin/config/{config_name}/update         - Update config (hot reload)

Security: All endpoints require a valid JWT with role 'admin'.
          Rate limiting is enforced on mutating and expensive operations.

Author: Resync Team
Version: 6.1.0

Changes from v6.0.0 (360° Audit — Feb 2026):
- [P0-01] All endpoints now require JWT admin authentication via require_admin()
- [P0-02] config_name and section validated with strict regex patterns
- [P1-01] TOCTOU in restore_config_backup fixed: O_NOFOLLOW + atomic open
- [P1-02] Orphaned .tmp_* files now cleaned up on any restore failure
- [P1-03] Rate limiting added to /reload (5/min) and /restore (3/min)
- [P1-04] reload_all_configs protected by asyncio.Lock against concurrent calls
- [P1-05] ConfigUpdateRequest.data limited to 100 keys with key pattern validation
- [P2-01] 404 detail no longer echoes user-supplied config_name (information leak)
- [P2-02] _BACKUP_DIR extracted as module-level constant (single source of truth)
- [P2-03] ConfigUpdateRequest uses ConfigDict(extra="forbid")
- [P2-04] get_config_manager() wrapped in asyncio.to_thread for lazy-init safety
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
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from resync.core.config import get_settings
from resync.core.unified_config import get_config_manager

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# [P2-02] Single source of truth — avoids silent divergence if file is moved
_BACKUP_DIR: Path = Path(__file__).parent.parent.parent / "config" / "backups"

# [P1-04] Global lock: prevents concurrent reload_all_configs race conditions
_reload_lock: asyncio.Lock = asyncio.Lock()

# [P1-03] Rate limiter keyed by remote IP address
limiter = Limiter(key_func=get_remote_address)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin/config", tags=["configuration"])

# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=True)


async def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict[str, Any]:
    """
    [P0-01] Validate JWT and enforce admin role on every request.

    Raises:
        HTTPException 401: Token is missing, malformed, or expired.
        HTTPException 403: Token is valid but role 'admin' is absent.
    """
    import jwt  # PyJWT — imported here to allow test-time monkey-patching

    settings = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc

    if "admin" not in payload.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation",
        )
    return payload


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# Strict pattern for all user-supplied identifiers (config names, sections, keys)
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration attributes."""

    # [P2-03] extra="forbid" — unknown fields raise 422, not silently ignored
    # [P0-02] section validated with strict alphanumeric pattern
    model_config = ConfigDict(extra="forbid")

    section: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Configuration section to update (alphanumeric, _ and - only)",
    )
    data: dict[str, Any] = Field(
        ...,
        description="Key-value pairs to set in the config section",
    )
    create_backup: bool = Field(
        True,
        description="Whether to create a backup before applying changes",
    )

    @field_validator("data")
    @classmethod
    def validate_data_shape(cls, v: dict[str, Any]) -> dict[str, Any]:
        """
        [P1-05] Enforce key count and key pattern on incoming data payload.
        Prevents DoS via oversized payloads and TOML key injection.
        """
        if len(v) > 100:
            raise ValueError("data must contain at most 100 keys per request")
        for key in v:
            if not _SAFE_IDENTIFIER_PATTERN.match(str(key)):
                raise ValueError(
                    f"Invalid key {key!r}: only [a-zA-Z0-9_-] allowed in config keys"
                )
        return v


class ConfigReloadResponse(BaseModel):
    """Response structure for config reload events."""

    status: str
    configs_loaded: list[str]
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: safe get_config_manager
# ---------------------------------------------------------------------------


async def _get_manager() -> Any:
    """
    [P2-04] Wrap get_config_manager() in asyncio.to_thread to guard against
    lazy-initialization blocking (file I/O, watcher setup, etc.).
    """
    return await asyncio.to_thread(get_config_manager)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/all", response_model=dict[str, Any])
async def get_all_configs(
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    Get all configuration files.

    Returns complete configuration tree for the entire system from memory.
    Requires: JWT with role 'admin'.
    """
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
        pattern=r"^[a-zA-Z0-9_-]+$",  # [P0-02] strict identifier validation
        description="Configuration identifier (alphanumeric, _ and - only)",
    ),
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    Update configuration dynamically with zero-downtime hot reload.

    Changes are applied to runtime immediately and persisted to disk.
    Requires: JWT with role 'admin'.
    """
    try:
        manager = await _get_manager()

        if config_name not in manager.CONFIG_FILES:
            # [P2-01] Do not echo config_name in the response — avoids enumeration
            raise HTTPException(
                status_code=404,
                detail="Configuration not found",
            )

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
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    Get configuration subsystem health and runtime status.

    Requires: JWT with role 'admin'.
    """
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
@limiter.limit("5/minute")  # [P1-03] Rate limit: reload is expensive I/O
async def reload_all_configs(
    request: Request,  # required by slowapi
    _: dict[str, Any] = Depends(require_admin),
) -> ConfigReloadResponse:
    """
    Force hot-reload cycle for all tracked configurations.

    Rate limited to 5 requests per minute per IP.
    Requires: JWT with role 'admin'.
    """
    # [P1-04] Serialize concurrent reload calls to prevent race conditions
    async with _reload_lock:
        manager = await _get_manager()
        configs_loaded: list[str] = []
        errors: list[str] = []

        for name, path in manager.CONFIG_FILES.items():
            try:
                # path.exists() is blocking I/O — offload to thread pool
                exists = await asyncio.to_thread(path.exists)
                if exists:
                    await manager.reload_config_file(path)
                    configs_loaded.append(name)
                else:
                    errors.append(f"Config file not found: {name}")
            except (
                OSError,
                ValueError,
                RuntimeError,
                TimeoutError,
                ConnectionError,
            ) as e:
                logger.error(
                    "Failed to reload config file",
                    config_name=name,
                    error=str(e),
                    exc_info=True,
                )
                errors.append(f"Failed to reload {name}: {str(e)}")

        status_str = "success" if not errors else "partial"
        return ConfigReloadResponse(
            status=status_str,
            configs_loaded=configs_loaded,
            errors=errors,
        )


@router.get("/backups", response_model=dict[str, Any])
async def list_config_backups(
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    List all configuration backups on disk (non-blocking).

    Requires: JWT with role 'admin'.
    """
    try:
        # [P2-02] Use module-level _BACKUP_DIR constant
        dir_exists = await asyncio.to_thread(_BACKUP_DIR.exists)
        if not dir_exists:
            return {"status": "success", "backups": [], "total": 0}

        def _get_backups() -> list[dict[str, Any]]:
            """Synchronous helper: glob + stat in a single thread."""
            results = []
            for backup_file in _BACKUP_DIR.glob("*.toml.bak"):
                stat_info = backup_file.stat()
                results.append(
                    {
                        "filename": backup_file.name,
                        "size_bytes": stat_info.st_size,
                        "modified": stat_info.st_mtime,
                        "path": str(backup_file),
                    }
                )
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
@limiter.limit("3/minute")  # [P1-03] Rate limit: restore creates backups on disk
async def restore_config_backup(
    request: Request,  # required by slowapi
    filename: str = FastApiPath(
        ...,
        description="Target backup file name",
    ),
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    Restore a specific configuration from an atomic backup.

    WARNING: Overwrites the live configuration file.
    Rate limited to 3 requests per minute per IP.
    Requires: JWT with role 'admin'.
    """
    try:
        # [P0-02] Strict filename validation upfront — fully mitigates path traversal
        # Accepted format: graphrag_20241225_150300.toml.bak
        match = re.match(r"^([a-zA-Z0-9]+)_[a-zA-Z0-9_]+\.toml\.bak$", filename)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid backup filename format. "
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

        # [P2-02] Use module-level _BACKUP_DIR constant
        backup_file = _BACKUP_DIR / filename
        config_file = manager.CONFIG_FILES[config_name]
        timestamp = int(datetime.now(timezone.utc).timestamp())
        current_backup = (
            _BACKUP_DIR / f"{config_name}_before_restore_{timestamp}.toml.bak"
        )
        temp_restore = _BACKUP_DIR / f".tmp_{config_file.name}_{timestamp}"

        def _perform_restore() -> None:
            """
            [P1-01] TOCTOU fix: open backup with O_NOFOLLOW + O_RDONLY to
            prevent symlink race. The file descriptor is held open across
            the entire operation, eliminating the check-then-use window.

            [P1-02] Cleanup fix: temp file is always removed on failure via
            contextlib.suppress inside the except block.
            """
            # O_NOFOLLOW refuses to open if backup_file is a symlink
            try:
                fd = os.open(str(backup_file), os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Backup file not found: {backup_file}"
                ) from exc
            except OSError as exc:
                # ELOOP on Linux when O_NOFOLLOW hits a symlink
                raise OSError(
                    f"Backup file is a symlink or inaccessible: {backup_file}"
                ) from exc

            try:
                with os.fdopen(fd, "rb") as src:
                    temp_restore.write_bytes(src.read())

                # Preserve current config before overwriting
                shutil.copy2(config_file, current_backup)

                # Atomic replace: on POSIX this is a single syscall (rename(2))
                temp_restore.replace(config_file)
            except Exception:
                # [P1-02] Always clean up orphaned temp file on any failure
                with contextlib.suppress(OSError):
                    temp_restore.unlink(missing_ok=True)
                raise

        await asyncio.to_thread(_perform_restore)

        # Trigger application-level hot reload after successful restore
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
        pattern=r"^[a-zA-Z0-9_-]+$",  # [P0-02] strict identifier validation
        description="Configuration identifier (alphanumeric, _ and - only)",
    ),
    section: str | None = None,
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """
    Get the nested dictionary of a specific loaded configuration.

    Optionally filter by section name.
    Requires: JWT with role 'admin'.
    """
    try:
        manager = await _get_manager()

        if config_name not in manager.CONFIG_FILES:
            # [P2-01] Generic message — do not confirm which names are valid
            raise HTTPException(
                status_code=404,
                detail="Configuration not found",
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
