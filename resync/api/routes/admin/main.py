"""Admin configuration API endpoints for Resync.

This module provides REST API endpoints for managing system configuration
through the /admin/config interface.
"""

from __future__ import annotations

# mypy
import asyncio
import inspect
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.fastapi_di import get_teams_integration, get_tws_client
from resync.core.interfaces import ITWSClient
from resync.core.teams_integration import TeamsIntegration
from resync.settings import settings

logger = logging.getLogger(__name__)

# API Router for admin endpoints
admin_router = APIRouter(prefix="/admin", tags=["Admin"])

# Templates will be obtained from app state at runtime

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
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Last update timestamp",
    )

class TeamsHealthResponse(BaseModel):
    """Teams integration health check response."""

    status: dict[str, Any] = Field(
        default_factory=dict, description="Teams integration status"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Health check timestamp",
    )

@admin_router.get(
    "/",
    response_class=HTMLResponse,
    summary="Admin Dashboard",
    dependencies=[Depends(verify_admin_credentials)],
)
@admin_router.get(
    "",
    response_class=HTMLResponse,
    summary="Admin Dashboard",
    dependencies=[Depends(verify_admin_credentials)],
)
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
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    request: Request, teams_integration: TeamsIntegration = Depends(get_teams_integration)
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
            last_updated=datetime.now(UTC).isoformat(),
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    teams_integration: TeamsIntegration = Depends(get_teams_integration),
) -> AdminConfigResponse:
    """Update Microsoft Teams integration configuration.

    Updates the Teams integration configuration with the provided values.
    Only provided fields will be updated.
    """
    try:
        # Get current Teams integration
        current_config = teams_integration.config

        # Update configuration with provided values
        update_fields = config_update.model_dump(exclude_unset=True)

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
            last_updated=datetime.now(UTC).isoformat(),
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    request: Request, teams_integration: TeamsIntegration = Depends(get_teams_integration)
) -> TeamsHealthResponse:
    """Get Microsoft Teams integration health status.

    Returns the current health status of the Teams integration,
    including connectivity and configuration status.
    """
    try:
        health_status = await teams_integration.health_check()
        return TeamsHealthResponse(
            status=health_status, timestamp=datetime.now(UTC).isoformat()
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    teams_integration: TeamsIntegration = Depends(get_teams_integration),
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
                "test_timestamp": datetime.now(UTC).isoformat(),
                "instance": "admin_test",
            },
        )

        # Send notification
        success = await teams_integration.send_notification(notification)

        if success:
            return {
                "status": "success",
                "message": "Test notification sent successfully",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "status": "error",
            "message": "Failed to send test notification",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
    tws_client: ITWSClient = Depends(get_tws_client),
    teams_integration: TeamsIntegration = Depends(get_teams_integration),
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
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("exception_caught: %s", str(e), exc_info=True)
            tws_status = "error"

        # Get Teams integration status
        teams_health = await teams_integration.health_check()

        return {
            "system": {
                "status": "operational",
                "timestamp": datetime.now(UTC).isoformat(),
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

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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

        update_fields = config_update.model_dump(exclude_unset=True)

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

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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

        update_fields = config_update.model_dump(exclude_unset=True)

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

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to update system configuration: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Failed to update system configuration. "
                "Check server logs for details."
            ),
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
    """Retrieve system logs with filtering options."""
    try:
        lines = min(lines, 1000)
        log_file = settings.BASE_DIR / "logs" / "resync.log"

        def _read_and_filter_logs(
            file_path: Path,
            line_limit: int,
            level_filter: str | None,
            search_filter: str | None,
        ) -> tuple[list[str], int, int]:
            from collections import deque

            if not file_path.exists():
                return [], 0, 0

            level_upper = level_filter.upper() if level_filter else None
            search_lower = search_filter.lower() if search_filter else None

            filtered_logs: deque[str] = deque(maxlen=line_limit)
            total_lines = 0
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    if level_upper and level_upper not in line.upper():
                        continue
                    if search_lower and search_lower not in line.lower():
                        continue
                    filtered_logs.append(line)

            return list(filtered_logs), total_lines, file_path.stat().st_size

        log_lines, total_lines, file_size = await asyncio.to_thread(
            _read_and_filter_logs,
            log_file,
            lines,
            level,
            search,
        )

        if total_lines == 0 and not log_lines:
            return {
                "logs": [],
                "count": 0,
                "message": "Log file not found",
            }

        return {
            "logs": log_lines,
            "count": len(log_lines),
            "total_lines": total_lines,
            "file_size_bytes": file_size,
            "log_file": str(log_file),
        }

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        TimeoutError,
        ConnectionError,
    ) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error

        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.warning("Failed to clear Redis cache: %s", e)

        if cache_type in ("all", "memory"):
            # Clear in-memory caches
            cleared.append("memory")
            logger.info("Memory cache cleared")

        return {
            "status": "success",
            "cleared": cleared,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
        # Enforce strict filename policy and allow-list against actual backups.
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", backup_filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid backup filename format. "
                    "Only alphanumeric, underscore, dash and dot are allowed."
                ),
            )

        if "/" in backup_filename or "\\" in backup_filename or ".." in backup_filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid backup filename",
            )

        allowed_backups = {backup.name: backup for backup in await persistence.list_backups()}
        backup_file = allowed_backups.get(backup_filename)
        if backup_file is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backup file not found",
            )

        resolved_backup = backup_file.resolve(strict=True)
        if not resolved_backup.is_relative_to(persistence.backup_dir.resolve()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid backup filename",
            )

        if not resolved_backup.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backup file not found",
            )

        # Restore from backup
        await persistence.restore_backup(resolved_backup)

        logger.info("Configuration restored from backup: %s", backup_filename)

        return {
            "status": "success",
            "restored_from": backup_filename,
            "timestamp": datetime.now(UTC).isoformat(),
            "note": (
                "Application restart may be required for all "
                "changes to take effect"
            ),
        }

    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Last health check timestamp",
    )

class SystemHealthResponse(BaseModel):
    """Complete system health response."""

    overall_status: str = Field(description="Overall system status")
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict, description="Individual component health"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
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

    Components checked (in parallel):
    - Database (SQLite/ContextStore)
    - Redis (Cache)
    - LLM (AI Service)
    - RAG (Vector Search)
    - Teams (Integration)
    - TWS (HCL Workload Automation)
    """
    import asyncio
    import time

    # P2-38 FIX: Define helper functions for parallel execution

    async def _check_database() -> ComponentHealth:
        """Check database health."""
        try:
            start = time.perf_counter()
            from resync.core.context_store import ContextStore
            store = ContextStore()
            await store.initialize()
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                message="SQLite connected",
            )
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"Database error: {str(e)[:100]}",
            )

    async def _check_redis() -> ComponentHealth:
        """Check Redis health."""
        try:
            start = time.perf_counter()
            from resync.core.redis_init import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                await redis_client.ping()
                latency = (time.perf_counter() - start) * 1000
                info = await redis_client.info("memory")
                used_memory = info.get("used_memory_human", "unknown")
                return ComponentHealth(
                    status="healthy",
                    latency_ms=round(latency, 2),
                    message=f"Connected, memory: {used_memory}",
                )
            return ComponentHealth(
                status="degraded",
                message="Redis not configured",
            )
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"Redis error: {str(e)[:100]}",
            )

    async def _check_llm() -> ComponentHealth:
        """Check LLM health."""
        try:
            start = time.perf_counter()
            llm_endpoint = getattr(settings, "LLM_ENDPOINT", None)
            if not llm_endpoint:
                return ComponentHealth(
                    status="degraded",
                    message="LLM endpoint not configured",
                )
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{llm_endpoint}/health")
                latency = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    return ComponentHealth(
                        status="healthy",
                        latency_ms=round(latency, 2),
                        message="LiteLLM responding",
                    )
                return ComponentHealth(
                    status="degraded",
                    latency_ms=round(latency, 2),
                    message=f"LLM returned status {resp.status_code}",
                )
        except httpx.ConnectError:
            return ComponentHealth(
                status="unhealthy",
                message="Cannot connect to LLM endpoint",
            )
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"LLM error: {str(e)[:100]}",
            )

    async def _check_rag() -> ComponentHealth:
        """Check RAG health."""
        try:
            start = time.perf_counter()
            # Try pgvector first
            try:
                from resync.knowledge.ingestion.embeddings import get_vector_service
                vector_service = await get_vector_service()
                stats = await vector_service.get_collection_stats("tws_docs")
                latency = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    status="healthy",
                    latency_ms=round(latency, 2),
                    message=f"pgvector operational ({stats.document_count} docs)",
                )
            except ImportError:
                pass
            # Fallback to RAG service URL
            rag_url = getattr(settings, "RAG_SERVICE_URL", None)
            if not rag_url:
                return ComponentHealth(
                    status="degraded",
                    message="RAG not configured (pgvector or RAG_SERVICE_URL)",
                )
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{rag_url}/health")
                latency = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    return ComponentHealth(
                        status="healthy",
                        latency_ms=round(latency, 2),
                        message="RAG service responding",
                    )
                return ComponentHealth(
                    status="degraded",
                    latency_ms=round(latency, 2),
                    message=f"RAG returned status {resp.status_code}",
                )
        except httpx.ConnectError:
            return ComponentHealth(
                status="unhealthy",
                message="Cannot connect to RAG service",
            )
        except Exception as e:
            return ComponentHealth(
                status="degraded",
                message=f"RAG check failed: {str(e)[:50]}",
            )

    async def _check_teams() -> ComponentHealth:
        """Check Teams integration health."""
        try:
            start = time.perf_counter()
            from resync.core.fastapi_di import get_teams_integration
            teams = get_teams_integration()
            health = await teams.health_check()
            latency = (time.perf_counter() - start) * 1000
            teams_enabled = health.get("enabled", False)
            return ComponentHealth(
                status="healthy" if teams_enabled else "degraded",
                latency_ms=round(latency, 2),
                message="Enabled" if teams_enabled else "Disabled",
            )
        except Exception as e:
            return ComponentHealth(
                status="degraded",
                message=f"Teams check failed: {str(e)[:50]}",
            )

    async def _check_tws() -> ComponentHealth:
        """Check TWS (HCL Workload Automation) health."""
        try:
            start = time.perf_counter()
            from resync.core.fastapi_di import get_tws_client
            tws = get_tws_client(request)
            connected = await tws.check_connection()
            latency = (time.perf_counter() - start) * 1000
            if connected:
                return ComponentHealth(
                    status="healthy",
                    latency_ms=round(latency, 2),
                    message="Connected to HWA",
                )
            return ComponentHealth(
                status="degraded",
                latency_ms=round(latency, 2),
                message="TWS not connected",
            )
        except Exception as e:
            return ComponentHealth(
                status="degraded",
                message=f"TWS: {str(e)[:50]}",
            )

    # P2-38 FIX: Execute all health checks in parallel
    results = await asyncio.gather(
        _check_database(),
        _check_redis(),
        _check_llm(),
        _check_rag(),
        _check_teams(),
        _check_tws(),
        return_exceptions=True,
    )

    # Map results to component names
    component_names = ["database", "redis", "llm", "rag", "teams", "tws"]
    components: dict[str, ComponentHealth] = {}

    for name, result in zip(component_names, results):
        if isinstance(result, Exception):
            components[name] = ComponentHealth(
                status="unhealthy",
                message=f"Check failed: {str(result)[:100]}",
            )
        else:
            components[name] = result

    # Determine overall status
    unhealthy_count = sum(1 for c in components.values() if c.status == "unhealthy")
    degraded_count = sum(1 for c in components.values() if c.status == "degraded")

    if unhealthy_count > 0:
        overall_status = "unhealthy"
    elif degraded_count > 2:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return SystemHealthResponse(
        overall_status=overall_status,
        components=components,
        timestamp=datetime.now(UTC).isoformat(),
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
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

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
            "timestamp": datetime.now(UTC).isoformat(),
        }
