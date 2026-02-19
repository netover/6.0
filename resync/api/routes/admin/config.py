"""
API routes for admin configuration management.

This module provides simple endpoints to persist and retrieve configuration
settings for administration UI.  Currently it supports only the
Microsoft Teams integration settings, but it can be extended to cover
additional configuration domains (e.g., TWS instances, system settings)
without changing the core logic.

Configurations are persisted to a JSON file under the ``config`` folder
within the project.  If the file does not exist, a default configuration
is returned.  Modifying environment variables or system settings at
runtime is outside the scope of this demonstration implementation.

Endpoints:

* ``GET /admin/teams-config`` – Return current Teams integration settings.
* ``POST /admin/teams-config`` – Update Teams integration settings.

These endpoints are mounted under the ``/api/v1/admin`` prefix via the
main application (see ``main.py``).
"""

import datetime
import json
import logging
from datetime import timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from resync.api.routes.admin.main import verify_admin_credentials
import aiofiles

logger = logging.getLogger(__name__)


# v5.9.5: Added authentication - was exposed without auth
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])

# Determine the path to the configuration file.  We place the file
# alongside other configuration resources (e.g., settings.toml) inside the
# top-level ``config`` directory.  ``__file__`` is
# ``.../resync/fastapi_app/api/v1/routes/admin_config.py``.
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "admin_config.json"

# Default configuration values used when the config file does not exist.
DEFAULT_CONFIG: dict[str, Any] = {
    "teams_config": {
        "enabled": False,
        "webhook_url": "",
        "channel_name": "",
        "bot_name": "Resync Bot",
        "avatar_url": "",
    },
    # Configuration for TWS instances.  ``primary_instance`` is
    # required; ``monitored_instances`` is a list of instance names.
    "tws_config": {
        "primary_instance": "TWS_NAZ",
        "monitored_instances": ["TWS_NAZ", "TWS_SAZ"],
    },
    # System-wide settings controlling environment and security features.
    "system_settings": {
        "environment": "Production",
        "debug_mode": False,
        "ssl_enabled": True,
        "csp_enabled": True,
        "cors_enabled": True,
    },
    # Notification configuration (job status filters and alert toggles).
    "notifications": {
        "job_status_filters": ["ABEND", "ERROR", "FAILED"],
        "notify_job_status": True,
        "notify_alerts": True,
        "notify_performance": False,
    },
}


# -----------------------------------------------------------------------------
# Health monitoring helpers
#
# The admin UI may request a high-level health report for key services such as
# the primary database, cache and external integrations.  For demonstration
# purposes we avoid deep dependency checks and instead return simple boolean
# flags.  Integrators can replace these stubs with real connectivity tests.
async def _get_health_report() -> dict[str, Any]:
    """Assemble a basic health report.

    Returns
    -------
    dict
        A dictionary with boolean flags indicating the status of key services.
    """
    report = {
        "database": True,
        "cache": True,
        "teams": False,
        "llm": True,
        "timestamp": datetime.datetime.now(timezone.utc).isoformat() + "Z",
    }
    # If Teams integration is enabled in config, mark as active
    cfg = await _load_config()
    teams_cfg = cfg.get("teams_config", DEFAULT_CONFIG["teams_config"])
    report["teams"] = bool(teams_cfg.get("enabled"))
    return report


async def _load_config() -> dict[str, Any]:
    """Read configuration from disk or return defaults.

    Returns
    -------
    dict
        The configuration dictionary containing all configurable sections.
    """
    import asyncio
    
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
        
    try:
        # Use asyncio.to_thread for blocking file read
        def _read():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        
        data = await asyncio.to_thread(_read)
        
        # Merge missing top-level keys from defaults without
        # overriding existing values.
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception as e:
        logger.error("config_load_failed", error=str(e), exc_info=True)  # type: ignore[call-arg]
        # If the file is corrupted, fall back to defaults but do not
        # overwrite the file immediately.  The next save will update it.
        return DEFAULT_CONFIG.copy()


async def _save_config(config: dict[str, Any]) -> None:
    """Persist the entire configuration to disk as JSON.

    Parameters
    ----------
    config : dict
        The configuration to persist.
    """
    import asyncio
    
    try:
        def _write():
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        await asyncio.to_thread(_write)
    except Exception as exc:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration. Check server logs for details.",
        ) from exc


@router.get("/teams-config", tags=["Admin"])
async def get_teams_config() -> dict[str, Any]:
    """Retrieve current Teams integration settings.

    Returns
    -------
    dict
        A dictionary containing Teams integration configuration such as
        ``enabled``, ``webhook_url``, ``channel_name``, ``bot_name`` and
        ``avatar_url``.
    """
    config = await _load_config()
    return config.get("teams_config", DEFAULT_CONFIG["teams_config"]).copy()


@router.post("/teams-config", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def update_teams_config(settings: dict[str, Any]) -> None:
    """Persist Teams integration settings provided by the admin UI.

    Parameters
    ----------
    settings : dict
        A dictionary containing Teams integration configuration fields.  Any
        unspecified fields will be left unchanged from the current config.

    Raises
    ------
    HTTPException
        If saving the configuration fails.
    """
    config = await _load_config()
    teams_config = config.get("teams_config", DEFAULT_CONFIG["teams_config"]).copy()
    # Update only known keys; ignore unknown keys for forward compatibility.
    for key in ["enabled", "webhook_url", "channel_name", "bot_name", "avatar_url"]:
        if key in settings:
            teams_config[key] = settings[key]
    config["teams_config"] = teams_config
    await _save_config(config)
    # Returning None with status 204 implies success with no content


# ----- TWS Configuration Endpoints -----


@router.get("/tws-config", tags=["Admin"])
async def get_tws_config() -> dict[str, Any]:
    """Return current TWS connection configuration."""
    config = await _load_config()
    return config.get("tws_config", DEFAULT_CONFIG["tws_config"]).copy()


@router.post("/tws-config", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def update_tws_config(settings: dict[str, Any]) -> None:
    """Update TWS connection configuration.

    Accepts ``primary_instance`` (str) and ``monitored_instances`` (list of str).
    """
    config = await _load_config()
    tws_config = config.get("tws_config", DEFAULT_CONFIG["tws_config"]).copy()
    if "primary_instance" in settings:
        tws_config["primary_instance"] = settings["primary_instance"]
    if "monitored_instances" in settings and isinstance(settings["monitored_instances"], list):
        tws_config["monitored_instances"] = settings["monitored_instances"]
    config["tws_config"] = tws_config
    await _save_config(config)


# ----- System Settings Endpoints -----


@router.get("/system-settings", tags=["Admin"])
async def get_system_settings() -> dict[str, Any]:
    """Return current system settings."""
    config = await _load_config()
    return config.get("system_settings", DEFAULT_CONFIG["system_settings"]).copy()


@router.post("/system-settings", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def update_system_settings(settings: dict[str, Any]) -> None:
    """Update system settings configuration.

    Accepts any subset of keys defined in the ``system_settings`` section.
    """
    config = await _load_config()
    sys_cfg = config.get("system_settings", DEFAULT_CONFIG["system_settings"]).copy()
    for key in ["environment", "debug_mode", "ssl_enabled", "csp_enabled", "cors_enabled"]:
        if key in settings:
            sys_cfg[key] = settings[key]
    config["system_settings"] = sys_cfg
    await _save_config(config)


# ----- Notifications Endpoints -----


@router.get("/notifications", tags=["Admin"])
async def get_notifications() -> dict[str, Any]:
    """Return current notification configuration."""
    config = await _load_config()
    return config.get("notifications", DEFAULT_CONFIG["notifications"]).copy()


@router.post("/notifications", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
async def update_notifications(settings: dict[str, Any]) -> None:
    """Update notification configuration.

    Accepts ``job_status_filters`` (list of str) and boolean toggles for
    ``notify_job_status``, ``notify_alerts``, and ``notify_performance``.
    """
    config = await _load_config()
    notifications = config.get("notifications", DEFAULT_CONFIG["notifications"]).copy()
    if "job_status_filters" in settings and isinstance(settings["job_status_filters"], list):
        notifications["job_status_filters"] = settings["job_status_filters"]
    for key in ["notify_job_status", "notify_alerts", "notify_performance"]:
        if key in settings:
            notifications[key] = bool(settings[key])
    config["notifications"] = notifications
    await _save_config(config)


# ----- Logs Endpoint -----


@router.get("/logs", tags=["Admin"])
async def get_logs(file: str = "app.log", lines: int = 100) -> dict[str, Any]:
    """Return the last ``lines`` lines of the specified log file.

    Parameters
    ----------
    file : str, optional
        Name of log file (must be located in the project's ``logs`` directory).
    lines : int, optional
        Number of lines to return from the end of the file.

    Returns
    -------
    dict
        A dictionary containing the filename and text content of the log.
    """
    logs_dir = Path(__file__).resolve().parent.parent.parent.parent / "logs"
    log_path = logs_dir / file
    if not log_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")
    try:
        # Read the last ``lines`` lines efficiently
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            # Use deque to store the last ``lines`` lines
            from collections import deque

            dq = deque(f, maxlen=lines)
            content = "".join(dq)
        return {"file": file, "content": content}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read log file. Check server logs for details.",
        ) from exc


# ----- Backup and Restore Endpoints -----


@router.post("/backup", tags=["Admin"])
async def create_backup() -> dict[str, str]:
    """Create a backup of audit data.

    With PostgreSQL, this exports audit entries to a JSON file
    for archival purposes.

    Returns:
        Dictionary with backup status and filename.
    """
    from resync.core.database.repositories import AuditEntryRepository

    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = AuditEntryRepository()
        entries = await repo.get_all(limit=10000)

        backup_data = [
            {
                "id": e.id,
                "action": e.action,
                "user_id": e.user_id,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in entries
        ]

        backup_file = backup_dir / f"audit_backup_{timestamp}.json"
        async with aiofiles.open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)

        return {"created": [backup_file.name], "count": len(backup_data)}
    except Exception as exc:
        logger.error("backup_failed", error=str(exc), exc_info=True)  # type: ignore[call-arg]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Backup failed. Check server logs for details."
        ) from exc


@router.post("/restore", tags=["Admin"])
async def restore_from_latest() -> dict[str, str]:
    """Restore audit data from the most recent backup.

    Note: With PostgreSQL, this is primarily for disaster recovery.
    Normal operations should use PostgreSQL's native backup tools (pg_dump).

    Returns:
        Dictionary with restore status.
    """
    backup_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "backup"
    if not backup_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No backup directory found"
        )

    # Find latest JSON backup
    backups = sorted(
        backup_dir.glob("audit_backup_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not backups:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No backup files found")

    return {
        "message": "PostgreSQL restore should be done via pg_restore",
        "latest_backup": backups[0].name,
        "recommendation": "Use: pg_restore -d resync backup.dump",
    }


@router.get("/audit", tags=["Admin"])
async def get_audit_logs(limit: int = 50) -> dict[str, Any]:
    """Return the latest entries from the audit log.

    Parameters
    ----------
    limit : int, optional
        Maximum number of audit records to return.

    Returns
    -------
    dict
        A dictionary containing a list of audit records.
    """
    from resync.core.database.repositories import AuditEntryRepository

    try:
        repo = AuditEntryRepository()
        entries = await repo.get_all(limit=limit, order_by="timestamp", desc=True)

        records = [
            {
                "id": e.id,
                "action": e.action,
                "user_id": e.user_id,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "metadata": e.metadata_,
            }
            for e in entries
        ]

        return {"records": records, "count": len(records)}
    except Exception as exc:
        logger.error("audit_query_failed", error=str(exc), exc_info=True)  # type: ignore[call-arg]
        return {"records": [], "error": str(exc)}


# ----- Health Monitoring Endpoint -----


@router.get("/health", tags=["Admin"])
async def get_health() -> dict[str, Any]:
    """Return a simple health report for key services.

    The report includes boolean flags for the database, cache, Teams integration
    (if enabled in the configuration) and the LLM service.  In a production
    deployment, these values should be determined via connectivity checks or
    service probes.

    Returns
    -------
    dict
        A dictionary containing health flags and a timestamp.
    """
    return await _get_health_report()
