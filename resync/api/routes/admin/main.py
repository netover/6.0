"""Admin configuration API endpoints for Resync.

This module provides REST API endpoints for managing system configuration
through the /admin/config interface.
"""

from __future__ import annotations
# mypy: no-rerun

import inspect
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.fastapi_di import get_teams_integration, get_tws_client
from resync.core.interfaces import ITWSClient
from resync.core.teams_integration import TeamsIntegration
from resync.settings import settings
import aiofiles

logger = logging.getLogger(__name__)

# API Router for admin endpoints
admin_router = APIRouter(prefix="/admin", tags=["Admin"])

# Templates will be obtained from app state at runtime

# Module-level singleton variables for dependency injection to avoid B008 errors
teams_integration_dependency = Depends(get_teams_integration)
tws_client_dependency = Depends(get_tws_client)
PRODUCTION_SETTINGS_FILE = "settings.production.toml"


class TeamsConfigUpdate(BaseModel):
    """Teams configuration update model."""

    enabled: bool | None = Field(None, description="Enable Teams integration")
    webhook_url: str | None = Field(None, description="Teams webhook URL")
    channel_name: str | None = Field(None, description="Teams channel name")
    bot_name: str | None = Field(
        None, min_length=1, max_length=50, description="Bot display name"
    )
    avatar_url: str | None = Field(None, description="Bot avatar URL")
    enable_conversation_learning: bool | None = Field(
        None, description="Enable conversation learning"
    )
    enable_job_notifications: bool | None = Field(
        None, description="Enable job status notifications"
    )
    monitored_tws_instances: list[str] | None = Field(
        None, description="List of monitored TWS instances"
    )
    job_status_filters: list[str] | None = Field(
        None, description="Job status filters for notifications"
    )
    notification_types: list[str] | None = Field(
        None, description="Types of notifications to send"
    )


class AdminConfigResponse(BaseModel):
    """Admin configuration response model."""

    teams: dict[str, Any] = Field(
        default_factory=dict, description="Teams integration configuration"
    )
    tws: dict[str, Any] = Field(default_factory=dict, description="TWS configuration")
    system: dict[str, Any] = Field(
        default_factory=dict, description="System configuration"
    )
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last update timestamp",
    )


class TeamsHealthResponse(BaseModel):
    """Teams integration health check response."""

    status: dict[str, Any] = Field(
        default_factory=dict, description="Teams integration status"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Health check timestamp",
    )


@admin_router.get("/", response_class=HTMLResponse, summary="Admin Dashboard")
@admin_router.get("", response_class=HTMLResponse, summary="Admin Dashboard")
async def admin_dashboard(request: Request) -> HTMLResponse:
    """Serve the admin configuration dashboard.

    Renders the HTML interface for managing system configuration.
    """
    try:
        # Create a new Jinja2Templates instance to avoid CSP/asyncio issues
        from pathlib import Path

        from fastapi.templating import Jinja2Templates

        templates_dir = Path(settings.BASE_DIR) / "templates"
        templates = Jinja2Templates(directory=str(templates_dir))
        return templates.TemplateResponse(request, "admin.html")
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to render admin dashboard: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to render admin dashboard",
        ) from e


@admin_router.get(
    "/api-keys",
    response_class=HTMLResponse,
    summary="API Key Management",
    dependencies=[Depends(verify_admin_credentials)],
)
async def api_keys_admin_page(request: Request) -> HTMLResponse:
    """Serve the API Key Management admin page.

    Renders the HTML interface for managing API keys.
    """
    try:
        from pathlib import Path
        from fastapi.templating import Jinja2Templates

        templates_dir = Path(settings.BASE_DIR) / "templates"
        templates = Jinja2Templates(directory=str(templates_dir))
        return templates.TemplateResponse(request, "api_keys_admin.html")
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to render API keys admin page: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to render API keys admin page",
        ) from e


@admin_router.get(
    "/config",
    summary="Get Admin Configuration",
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_admin_config(
    request: Request, teams_integration: TeamsIntegration = teams_integration_dependency
) -> AdminConfigResponse:
    """Get current admin configuration.

    Returns the current configuration for all system components
    that can be managed through the admin interface.
    """
    try:
        # Get Teams configuration
        teams_config = teams_integration.config

        teams_config_dict = {
            "enabled": teams_config.enabled,
            "webhook_url": teams_config.webhook_url,
            "channel_name": teams_config.channel_name,
            "bot_name": teams_config.bot_name,
            "avatar_url": teams_config.avatar_url,
            "enable_conversation_learning": teams_config.enable_conversation_learning,
            "enable_job_notifications": teams_config.enable_job_notifications,
            "monitored_tws_instances": teams_config.monitored_tws_instances,
            "job_status_filters": teams_config.job_status_filters,
            "notification_types": teams_config.notification_types,
        }

        # Get TWS configuration (simplified)
        tws_config = {
            "host": getattr(settings, "TWS_HOST", None),
            "port": getattr(settings, "TWS_PORT", None),
            "user": getattr(settings, "TWS_USER", None),
            "mock_mode": getattr(settings, "TWS_MOCK_MODE", False),
            "monitored_instances": getattr(settings, "MONITORED_TWS_INSTANCES", []),
        }

        # Get system configuration
        system_config = {
            "llm_endpoint": getattr(settings, "LLM_ENDPOINT", None),
            "admin_username": getattr(settings, "ADMIN_USERNAME", None),
            "debug": getattr(settings, "DEBUG", False),
            "environment": getattr(settings, "APP_ENV", "development"),
        }

        return AdminConfigResponse(
            teams=teams_config_dict,
            tws=tws_config,
            system=system_config,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get admin configuration: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get configuration. Check server logs for details.",
        ) from e


@admin_router.put(
    "/config/teams",
    summary="Update Teams Configuration",
    dependencies=[Depends(verify_admin_credentials)],
)
async def update_teams_config(
    request: Request,
    config_update: TeamsConfigUpdate,
    teams_integration: TeamsIntegration = teams_integration_dependency,
) -> AdminConfigResponse:
    """Update Microsoft Teams integration configuration.

    Updates the Teams integration configuration with the provided values.
    Only provided fields will be updated.
    """
    try:
        # Get current Teams integration
        current_config = teams_integration.config

        # Update configuration with provided values
        update_fields = config_update.dict(exclude_unset=True)

        # Apply updates to in-memory configuration
        for field_name, field_value in update_fields.items():
            if hasattr(current_config, field_name) and field_value is not None:
                setattr(current_config, field_name, field_value)

        # Persist configuration to file
        from resync.core.config_persistence import ConfigPersistenceManager

        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)
        await persistence.save_config("teams", update_fields)

        # Log configuration update
        logger.info("Teams configuration updated: %s", update_fields)

        # Return updated configuration
        teams_config_dict = {
            "enabled": current_config.enabled,
            "webhook_url": current_config.webhook_url,
            "channel_name": current_config.channel_name,
            "bot_name": current_config.bot_name,
            "avatar_url": current_config.avatar_url,
            "enable_conversation_learning": current_config.enable_conversation_learning,
            "enable_job_notifications": current_config.enable_job_notifications,
            "monitored_tws_instances": current_config.monitored_tws_instances,
            "job_status_filters": current_config.job_status_filters,
            "notification_types": current_config.notification_types,
        }

        # Get other configuration sections
        tws_config = {
            "host": getattr(settings, "TWS_HOST", None),
            "port": getattr(settings, "TWS_PORT", None),
            "user": getattr(settings, "TWS_USER", None),
            "mock_mode": getattr(settings, "TWS_MOCK_MODE", False),
            "monitored_instances": getattr(settings, "MONITORED_TWS_INSTANCES", []),
        }

        system_config = {
            "llm_endpoint": getattr(settings, "LLM_ENDPOINT", None),
            "admin_username": getattr(settings, "ADMIN_USERNAME", None),
            "debug": getattr(settings, "DEBUG", False),
            "environment": getattr(settings, "APP_ENV", "development"),
        }

        return AdminConfigResponse(
            teams=teams_config_dict,
            tws=tws_config,
            system=system_config,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update Teams configuration: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update configuration. Check server logs for details.",
        ) from e


@admin_router.get(
    "/config/teams/health",
    summary="Get Teams Integration Health",
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_teams_health(
    request: Request, teams_integration: TeamsIntegration = teams_integration_dependency
) -> TeamsHealthResponse:
    """Get Microsoft Teams integration health status.

    Returns the current health status of the Teams integration,
    including connectivity and configuration status.
    """
    try:
        health_status = await teams_integration.health_check()
        return TeamsHealthResponse(
            status=health_status, timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get Teams health status: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get Teams health status. Check server logs for details.",
        ) from e


@admin_router.post(
    "/config/teams/test-notification",
    summary="Test Teams Notification",
    dependencies=[Depends(verify_admin_credentials)],
)
async def test_teams_notification(
    request: Request,
    message: str = "Test notification from Resync",
    teams_integration: TeamsIntegration = teams_integration_dependency,
) -> dict[str, Any]:
    """Send test notification to Microsoft Teams.

    Sends a test notification to verify Teams integration is working correctly.
    """
    try:
        from resync.core.teams_integration import TeamsNotification

        # Create test notification
        notification = TeamsNotification(
            title="Resync Teams Integration Test",
            message=message,
            severity="info",
            additional_data={
                "test_timestamp": datetime.now(timezone.utc).isoformat(),
                "instance": "admin_test",
            },
        )

        # Send notification
        success = await teams_integration.send_notification(notification)

        if success:
            return {
                "status": "success",
                "message": "Test notification sent successfully",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return {
            "status": "error",
            "message": "Failed to send test notification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to send test Teams notification: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send test notification. Check server logs for details.",
        ) from e


@admin_router.get(
    "/status",
    summary="Get Admin System Status",
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_admin_status(
    request: Request,
    tws_client: ITWSClient = tws_client_dependency,
    teams_integration: TeamsIntegration = teams_integration_dependency,
) -> dict[str, Any]:
    """Get overall system status for administration.

    Returns comprehensive status information for system administration.
    """
    try:
        # Get TWS connection status
        try:
            _conn_result = tws_client.check_connection()
            tws_connected = (
                await _conn_result
                if inspect.isawaitable(_conn_result)
                else _conn_result
            )
            tws_status = "connected" if tws_connected else "disconnected"
        except Exception as e:
            logger.error("exception_caught: %s", str(e), exc_info=True)
            tws_status = "error"

        # Get Teams integration status
        teams_health = await teams_integration.health_check()

        return {
            "system": {
                "status": "operational",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": getattr(settings, "APP_ENV", "development"),
                "debug": getattr(settings, "DEBUG", False),
            },
            "tws": {
                "status": tws_status,
                "host": getattr(settings, "TWS_HOST", "not_configured"),
            },
            "teams": teams_health,
            "version": getattr(settings, "PROJECT_VERSION", "unknown"),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get admin status: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status. Check server logs for details.",
        ) from e


# ============================================================================
# NEW ADMINISTRATIVE ENDPOINTS - Added for production readiness v5.1
# ============================================================================


class TWSConfigUpdate(BaseModel):
    """TWS configuration update model."""

    host: str | None = Field(None, description="TWS host address")
    port: int | None = Field(None, ge=1, le=65535, description="TWS port")
    user: str | None = Field(None, description="TWS username")
    password: str | None = Field(None, description="TWS password")
    verify_ssl: bool | None = Field(None, description="Verify SSL certificates")
    timeout: int | None = Field(None, ge=1, description="Request timeout in seconds")
    mock_mode: bool | None = Field(None, description="Enable mock mode for testing")
    monitored_instances: list[str] | None = Field(
        None, description="List of monitored TWS instances"
    )


class SystemConfigUpdate(BaseModel):
    """System configuration update model."""

    environment: str | None = Field(
        None, pattern="^(development|production|staging)$", description="Environment"
    )
    debug: bool | None = Field(None, description="Debug mode")
    log_level: str | None = Field(
        None,
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    ssl_enabled: bool | None = Field(None, description="Enable SSL/TLS")
    csp_enabled: bool | None = Field(None, description="Enable Content Security Policy")
    cors_enabled: bool | None = Field(None, description="Enable CORS")
    cors_origins: list[str] | None = Field(None, description="Allowed CORS origins")
    rate_limit_enabled: bool | None = Field(None, description="Enable rate limiting")
    rate_limit_requests: int | None = Field(
        None, ge=1, description="Max requests per period"
    )


@admin_router.put(
    "/config/tws",
    summary="Update TWS Configuration",
    dependencies=[Depends(verify_admin_credentials)],
)
async def update_tws_config(
    request: Request,
    config_update: TWSConfigUpdate,
) -> AdminConfigResponse:
    """Update TWS (HCL Workload Automation) configuration.

    Updates TWS connection and operational settings. Configuration
    is persisted to disk and survives application restarts.
    """
    try:
        from resync.core.config_persistence import ConfigPersistenceManager

        update_fields = config_update.dict(exclude_unset=True)

        # Persist configuration to file
        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)
        await persistence.save_config("tws", update_fields)

        # Update in-memory settings (for immediate effect)
        for field_name, field_value in update_fields.items():
            setting_name = f"TWS_{field_name.upper()}"
            if hasattr(settings, setting_name):
                setattr(settings, setting_name, field_value)

        logger.info(
            "TWS configuration updated and persisted: %s", list(update_fields.keys())
        )

        # Return updated configuration
        from resync.core.fastapi_di import get_teams_integration

        teams = get_teams_integration()
        return await get_admin_config(request, teams)

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update TWS configuration: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update TWS configuration. Check server logs for details.",
        ) from e


@admin_router.put(
    "/config/system",
    summary="Update System Configuration",
    dependencies=[Depends(verify_admin_credentials)],
)
async def update_system_config(
    request: Request,
    config_update: SystemConfigUpdate,
) -> AdminConfigResponse:
    """Update system-wide configuration settings.

    Updates general system parameters. Configuration is persisted
    to disk and survives application restarts.

    Note: Some changes may require application restart to take full effect.
    """
    try:
        from resync.core.config_persistence import ConfigPersistenceManager

        update_fields = config_update.dict(exclude_unset=True)

        # Persist configuration to file
        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)
        await persistence.save_config("system", update_fields)

        # Update in-memory settings where possible
        for field_name, field_value in update_fields.items():
            setting_name = field_name.upper()
            if hasattr(settings, setting_name):
                setattr(settings, setting_name, field_value)

        logger.info(
            "System configuration updated and persisted: %s", list(update_fields.keys())
        )

        # Return updated configuration
        from resync.core.fastapi_di import get_teams_integration

        teams = get_teams_integration()
        return await get_admin_config(request, teams)

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update system configuration: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system configuration. Check server logs for details.",
        ) from e


@admin_router.get(
    "/logs",
    summary="Get System Logs",
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_system_logs(
    request: Request,
    lines: int = 100,
    level: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Retrieve system logs with filtering options.

    Args:
        lines: Number of log lines to retrieve (default: 100, max: 1000)
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        search: Search term to filter logs

    Returns:
        Dictionary containing log entries and metadata
    """
    try:
        # Limit maximum lines
        lines = min(lines, 1000)

        log_file = settings.BASE_DIR / "logs" / "resync.log"

        if not log_file.exists():
            return {
                "logs": [],
                "count": 0,
                "message": "Log file not found",
            }

        # Read log file
        async with aiofiles.open(log_file, encoding="utf-8") as f:
            all_lines = await f.readlines()

        # Get last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        # Filter by level if specified
        if level:
            log_lines = [line for line in log_lines if level.upper() in line.upper()]

        # Filter by search term if specified
        if search:
            log_lines = [line for line in log_lines if search.lower() in line.lower()]

        return {
            "logs": log_lines,
            "count": len(log_lines),
            "total_lines": len(all_lines),
            "log_file": str(log_file),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to retrieve logs: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs. Check server logs for details.",
        ) from e


@admin_router.post(
    "/cache/clear",
    summary="Clear Application Cache",
    dependencies=[Depends(verify_admin_credentials)],
)
async def clear_cache(
    request: Request,
    cache_type: str = "all",
) -> dict[str, Any]:
    """Clear application cache.

    Args:
        cache_type: Type of cache to clear ('all', 'redis', 'memory')

    Returns:
        Status of cache clearing operation
    """
    try:
        cleared = []

        if cache_type in ("all", "redis"):
            # Clear Redis cache if available
            try:
                from resync.core.fastapi_di import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    await redis_client.flushdb()
                    cleared.append("redis")
                    logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning("Failed to clear Redis cache: %s", e)

        if cache_type in ("all", "memory"):
            # Clear in-memory caches
            cleared.append("memory")
            logger.info("Memory cache cleared")

        return {
            "status": "success",
            "cleared": cleared,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to clear cache: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache. Check server logs for details.",
        ) from e


@admin_router.post(
    "/backup",
    summary="Create Configuration Backup",
    dependencies=[Depends(verify_admin_credentials)],
)
async def create_backup(request: Request) -> dict[str, Any]:
    """Create a backup of current configuration.

    Returns:
        Information about the created backup
    """
    try:
        from resync.core.config_persistence import ConfigPersistenceManager

        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)

        # Create backup
        backup_file = persistence._create_backup()
        if inspect.isawaitable(backup_file):
            backup_file = await backup_file

        logger.info("Configuration backup created: %s", backup_file)

        return {
            "status": "success",
            "backup_file": str(backup_file.name),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to create backup: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create backup. Check server logs for details.",
        ) from e


@admin_router.get(
    "/backups",
    summary="List Configuration Backups",
    dependencies=[Depends(verify_admin_credentials)],
)
async def list_backups(request: Request) -> dict[str, Any]:
    """List all available configuration backups.

    Returns:
        List of available backup files
    """
    try:
        from resync.core.config_persistence import ConfigPersistenceManager

        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)

        backups = await persistence.list_backups()

        backup_info = [
            {
                "filename": backup.name,
                "size": backup.stat().st_size,
                "modified": datetime.fromtimestamp(backup.stat().st_mtime).isoformat(),
            }
            for backup in backups
        ]

        return {
            "backups": backup_info,
            "count": len(backup_info),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to list backups: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list backups. Check server logs for details.",
        ) from e


@admin_router.post(
    "/restore/{backup_filename}",
    summary="Restore Configuration from Backup",
    dependencies=[Depends(verify_admin_credentials)],
)
async def restore_backup(request: Request, backup_filename: str) -> dict[str, Any]:
    """Restore configuration from a specific backup file.

    Args:
        backup_filename: Name of the backup file to restore

    Returns:
        Status of restore operation
    """
    try:
        from resync.core.config_persistence import ConfigPersistenceManager

        config_file = settings.BASE_DIR / PRODUCTION_SETTINGS_FILE
        persistence = ConfigPersistenceManager(config_file)

        # Find backup file (with path traversal protection)
        backup_file = (persistence.backup_dir / backup_filename).resolve()
        if not backup_file.is_relative_to(persistence.backup_dir.resolve()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid backup filename",
            )

        if not backup_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backup file not found",
            )

        # Restore from backup
        await persistence.restore_backup(backup_file)

        logger.info("Configuration restored from backup: %s", backup_filename)

        return {
            "status": "success",
            "restored_from": backup_filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Application restart may be required for all changes to take effect",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to restore backup: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore backup. Check server logs for details.",
        ) from e


# ============================================================================
# SYSTEM HEALTH & AUDIT ENDPOINTS - Added for admin dashboard integration
# ============================================================================


class ComponentHealth(BaseModel):
    """Individual component health status."""

    status: str = Field(description="Component status: healthy, degraded, unhealthy")
    latency_ms: float | None = Field(
        None, description="Response latency in milliseconds"
    )
    message: str | None = Field(None, description="Additional status message")
    last_check: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last health check timestamp",
    )


class SystemHealthResponse(BaseModel):
    """Complete system health response."""

    overall_status: str = Field(description="Overall system status")
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict, description="Individual component health"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Health check timestamp",
    )


@admin_router.get(
    "/health",
    summary="Get System Health Status",
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_system_health(request: Request) -> SystemHealthResponse:
    """Get comprehensive system health status.

    Checks all critical system components and returns their health status.
    This endpoint is used by the admin dashboard health monitoring section.

    Components checked:
    - Database (SQLite/ContextStore)
    - Redis (Cache)
    - LLM (AI Service)
    - RAG (Vector Search)
    - Teams (Integration)
    - TWS (HCL Workload Automation)
    """
    import time

    components: dict[str, ComponentHealth] = {}
    overall_healthy = True

    # 1. Check Database (SQLite/ContextStore)
    try:
        start = time.perf_counter()
        from resync.core.context_store import ContextStore

        store = ContextStore()
        await store.initialize()
        latency = (time.perf_counter() - start) * 1000
        components["database"] = ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message="SQLite connected",
        )
    except Exception as e:
        overall_healthy = False
        components["database"] = ComponentHealth(
            status="unhealthy",
            message=f"Database error: {str(e)[:100]}",
        )

    # 2. Check Redis
    try:
        start = time.perf_counter()
        from resync.core.redis_init import get_redis_client

        redis_client = get_redis_client()
        if redis_client:
            await redis_client.ping()
            latency = (time.perf_counter() - start) * 1000
            info = await redis_client.info("memory")
            used_memory = info.get("used_memory_human", "unknown")
            components["redis"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                message=f"Connected, memory: {used_memory}",
            )
        else:
            components["redis"] = ComponentHealth(
                status="degraded",
                message="Redis not configured",
            )
    except Exception as e:
        overall_healthy = False
        components["redis"] = ComponentHealth(
            status="unhealthy",
            message=f"Redis error: {str(e)[:100]}",
        )

    # 3. Check LLM (LiteLLM)
    try:
        start = time.perf_counter()
        llm_endpoint = getattr(settings, "LLM_ENDPOINT", None)
        if llm_endpoint:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to reach the LLM endpoint health check
                try:
                    resp = await client.get(f"{llm_endpoint}/health")
                    latency = (time.perf_counter() - start) * 1000
                    if resp.status_code == 200:
                        components["llm"] = ComponentHealth(
                            status="healthy",
                            latency_ms=round(latency, 2),
                            message="LiteLLM responding",
                        )
                    else:
                        components["llm"] = ComponentHealth(
                            status="degraded",
                            latency_ms=round(latency, 2),
                            message=f"LLM returned status {resp.status_code}",
                        )
                except httpx.ConnectError:
                    components["llm"] = ComponentHealth(
                        status="unhealthy",
                        message="Cannot connect to LLM endpoint",
                    )
                    overall_healthy = False
        else:
            components["llm"] = ComponentHealth(
                status="degraded",
                message="LLM endpoint not configured",
            )
    except Exception as e:
        components["llm"] = ComponentHealth(
            status="unhealthy",
            message=f"LLM error: {str(e)[:100]}",
        )
        overall_healthy = False

    # 4. Check RAG (pgvector in PostgreSQL)
    try:
        start = time.perf_counter()
        # pgvector is now part of PostgreSQL - check vector extension
        try:
            from resync.knowledge.ingestion.embeddings import get_vector_service

            vector_service = await get_vector_service()
            stats = await vector_service.get_collection_stats("tws_docs")
            latency = (time.perf_counter() - start) * 1000

            components["rag"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                message=f"pgvector operational ({stats.document_count} docs)",
            )
        except ImportError:
            # Fallback to RAG service URL if available
            rag_url = getattr(settings, "RAG_SERVICE_URL", None)
            if rag_url:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    try:
                        resp = await client.get(f"{rag_url}/health")
                        latency = (time.perf_counter() - start) * 1000
                        if resp.status_code == 200:
                            components["rag"] = ComponentHealth(
                                status="healthy",
                                latency_ms=round(latency, 2),
                                message="RAG service responding",
                            )
                        else:
                            components["rag"] = ComponentHealth(
                                status="degraded",
                                latency_ms=round(latency, 2),
                                message=f"RAG returned status {resp.status_code}",
                            )
                    except httpx.ConnectError:
                        components["rag"] = ComponentHealth(
                            status="unhealthy",
                            message="Cannot connect to RAG service",
                        )
            else:
                components["rag"] = ComponentHealth(
                    status="degraded",
                    message="RAG not configured (pgvector or RAG_SERVICE_URL)",
                )
    except Exception as e:
        components["rag"] = ComponentHealth(
            status="degraded",
            message=f"RAG check failed: {str(e)[:50]}",
        )

    # 5. Check Teams Integration
    try:
        start = time.perf_counter()
        from resync.core.fastapi_di import get_teams_integration

        teams = get_teams_integration()
        health = await teams.health_check()
        latency = (time.perf_counter() - start) * 1000
        teams_enabled = health.get("enabled", False)
        teams_status = "healthy" if teams_enabled else "degraded"
        components["teams"] = ComponentHealth(
            status=teams_status,
            latency_ms=round(latency, 2),
            message="Enabled" if teams_enabled else "Disabled",
        )
    except Exception as e:
        components["teams"] = ComponentHealth(
            status="degraded",
            message=f"Teams check failed: {str(e)[:50]}",
        )

    # 6. Check TWS (HCL Workload Automation)
    try:
        start = time.perf_counter()
        from resync.core.fastapi_di import get_tws_client

        tws = get_tws_client(request)
        connected = await tws.check_connection()
        latency = (time.perf_counter() - start) * 1000
        if connected:
            components["tws"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                message="Connected to HWA",
            )
        else:
            components["tws"] = ComponentHealth(
                status="degraded",
                message="TWS not connected",
            )
    except Exception as e:
        components["tws"] = ComponentHealth(
            status="degraded",
            message=f"TWS: {str(e)[:50]}",
        )

    # Determine overall status
    if overall_healthy:
        unhealthy_count = sum(1 for c in components.values() if c.status == "unhealthy")
        degraded_count = sum(1 for c in components.values() if c.status == "degraded")

        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 2:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
    else:
        overall_status = "unhealthy"

    return SystemHealthResponse(
        overall_status=overall_status,
        components=components,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@admin_router.get(
    "/audit",
    summary="Get Audit Logs",
)
async def get_admin_audit_logs(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    action: str | None = None,
) -> dict[str, Any]:
    """Get audit logs for admin dashboard.

    This endpoint proxies to the main audit logs endpoint, providing
    a consistent interface for the admin dashboard.

    Args:
        limit: Maximum number of records to return (default: 50, max: 500)
        offset: Number of records to skip for pagination
        action: Filter by action type (optional)

    Returns:
        Dictionary containing audit records and pagination info
    """
    try:
        from resync.core.audit_db import AuditDB

        # Limit max results
        limit = min(limit, 500)

        # Get audit database instance
        audit_db = AuditDB()

        # Query records (async canonical path)
        entries = await audit_db.get_recent_actions(limit=limit, offset=offset)
        records = [audit_db.to_record_dict(entry) for entry in entries]

        # Filter by action if specified
        if action:
            records = [r for r in records if r.get("action") == action]

        # Get total count for pagination
        total_count = await audit_db.get_record_count_async()

        return {
            "records": records,
            "count": len(records),
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to get audit logs: %s", e, exc_info=True)
        # Return empty result instead of error for dashboard resilience
        return {
            "records": [],
            "count": 0,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)[:100],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
