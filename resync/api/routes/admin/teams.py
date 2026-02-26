"""
Microsoft Teams Integration API Endpoints.

Provides REST endpoints for managing Teams integration:
- Configuration management
- Notification sending
- Health check
- Statistics
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.structured_logger import get_logger
from resync.core.teams_integration import (
    TeamsNotification,
    get_teams_integration,
    send_teams_alert,
)

logger = get_logger(__name__)

# v5.9.5: Added authentication
router = APIRouter(
    prefix="/admin/teams",
    tags=["Teams Integration"],
    dependencies=[Depends(verify_admin_credentials)],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TeamsConfigResponse(BaseModel):
    """Teams configuration response."""

    enabled: bool
    channel_name: str | None = None
    bot_name: str
    enable_conversation_learning: bool
    enable_job_notifications: bool
    monitored_tws_instances: list[str]
    job_status_filters: list[str]
    notification_types: list[str]
    rate_limit_enabled: bool
    webhook_configured: bool
    webhook_masked: str

class TeamsConfigUpdate(BaseModel):
    """Teams configuration update request."""

    enabled: bool | None = None
    channel_name: str | None = None
    bot_name: str | None = None
    enable_conversation_learning: bool | None = None
    enable_job_notifications: bool | None = None
    monitored_tws_instances: list[str] | None = None
    job_status_filters: list[str] | None = None
    notification_types: list[str] | None = None
    rate_limit_enabled: bool | None = None

class NotificationRequest(BaseModel):
    """Request to send a Teams notification."""

    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=5000)
    severity: str = Field(
        default="info", pattern="^(info|success|warning|error|critical)$"
    )
    job_id: str | None = None
    instance_name: str | None = None
    additional_data: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, str]] = Field(default_factory=list)

class NotificationResponse(BaseModel):
    """Response from notification send."""

    success: bool
    message: str
    notification_id: str | None = None
    timestamp: str

class TeamsHealthResponse(BaseModel):
    """Teams integration health check response."""

    enabled: bool
    configured: bool
    webhook_accessible: bool | None = None
    webhook_error: str | None = None
    conversation_learning: bool
    job_notifications: bool
    monitored_instances: int
    rate_limiting: bool
    last_check: str
    statistics: dict[str, Any]

class TeamsStatsResponse(BaseModel):
    """Teams notification statistics."""

    notifications_sent: int
    notifications_failed: int
    retries: int
    rate_limit_waits: int
    success_rate: float
    last_notification: str | None = None

class TestNotificationResponse(BaseModel):
    """Response from test notification."""

    success: bool
    message: str
    response_time_ms: float

# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/config", response_model=TeamsConfigResponse)
async def get_teams_config():
    """
    Get current Teams integration configuration.

    Returns the current configuration with webhook URL masked for security.
    """
    integration = get_teams_integration()
    config = integration.config

    return TeamsConfigResponse(
        enabled=config.enabled,
        channel_name=config.channel_name,
        bot_name=config.bot_name,
        enable_conversation_learning=config.enable_conversation_learning,
        enable_job_notifications=config.enable_job_notifications,
        monitored_tws_instances=config.monitored_tws_instances,
        job_status_filters=config.job_status_filters,
        notification_types=config.notification_types,
        rate_limit_enabled=config.rate_limit_enabled,
        webhook_configured=bool(config.webhook_url),
        webhook_masked=integration._mask_webhook_url(config.webhook_url),
    )

@router.put("/config", response_model=TeamsConfigResponse)
async def update_teams_config(update: TeamsConfigUpdate):
    """
    Update Teams integration configuration.

    Updates only the provided fields. Webhook URL cannot be updated via API
    for security reasons - use environment variables or config file.
    """
    integration = get_teams_integration()
    config = integration.config

    if update.enabled is not None:
        config.enabled = update.enabled
    if update.channel_name is not None:
        config.channel_name = update.channel_name
    if update.bot_name is not None:
        config.bot_name = update.bot_name
    if update.enable_conversation_learning is not None:
        config.enable_conversation_learning = update.enable_conversation_learning
    if update.enable_job_notifications is not None:
        config.enable_job_notifications = update.enable_job_notifications
    if update.monitored_tws_instances is not None:
        config.monitored_tws_instances = update.monitored_tws_instances
    if update.job_status_filters is not None:
        config.job_status_filters = update.job_status_filters
    if update.notification_types is not None:
        config.notification_types = update.notification_types
    if update.rate_limit_enabled is not None:
        config.rate_limit_enabled = update.rate_limit_enabled

    logger.info("teams_config_updated", updates=update.model_dump(exclude_unset=True))

    return TeamsConfigResponse(
        enabled=config.enabled,
        channel_name=config.channel_name,
        bot_name=config.bot_name,
        enable_conversation_learning=config.enable_conversation_learning,
        enable_job_notifications=config.enable_job_notifications,
        monitored_tws_instances=config.monitored_tws_instances,
        job_status_filters=config.job_status_filters,
        notification_types=config.notification_types,
        rate_limit_enabled=config.rate_limit_enabled,
        webhook_configured=bool(config.webhook_url),
        webhook_masked=integration._mask_webhook_url(config.webhook_url),
    )

@router.post("/enable")
async def enable_teams_integration():
    """Enable Teams integration."""
    integration = get_teams_integration()

    if not integration.config.webhook_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot enable Teams integration: webhook URL not configured",
        )

    integration.config.enabled = True
    logger.info("teams_integration_enabled")

    return {"status": "enabled", "message": "Teams integration enabled successfully"}

@router.post("/disable")
async def disable_teams_integration():
    """Disable Teams integration."""
    integration = get_teams_integration()
    integration.config.enabled = False
    logger.info("teams_integration_disabled")

    return {"status": "disabled", "message": "Teams integration disabled"}

@router.post("/notifications", response_model=NotificationResponse)
async def send_notification(request: NotificationRequest):
    """
    Send a notification to Microsoft Teams.

    Sends an Adaptive Card notification to the configured Teams channel.
    """
    integration = get_teams_integration()

    if not integration.config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Teams integration is not enabled",
        )

    if not integration.config.webhook_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Teams webhook URL not configured",
        )

    notification = TeamsNotification(
        title=request.title,
        message=request.message,
        severity=request.severity,
        job_id=request.job_id,
        instance_name=request.instance_name,
        additional_data=request.additional_data,
        actions=request.actions,
    )

    try:
        success = await integration.send_notification(notification)

        return NotificationResponse(
            success=success,
            message="Notification sent successfully"
            if success
            else "Notification failed",
            notification_id=f"teams_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        logger.error("notification_send_error", error=str(e))  # type: ignore[call-arg]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send notification. Check server logs for details.",
        ) from e

@router.post("/test", response_model=TestNotificationResponse)
async def send_test_notification():
    """
    Send a test notification to verify Teams integration.

    Sends a simple test message and returns timing information.
    """
    import time

    integration = get_teams_integration()

    if not integration.config.webhook_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Teams webhook URL not configured",
        )

    # Temporarily enable for test
    was_enabled = integration.config.enabled
    integration.config.enabled = True

    start_time = time.time()

    try:
        notification = TeamsNotification(
            title="🧪 Test Notification",
            message=(
                "This is a test notification from Resync Admin. "
                "If you can see this message, Teams integration is "
                "working correctly!"
            ),
            severity="info",
            additional_data={
                "Environment": "Resync Admin",
                "Test Time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        success = await integration.send_notification(notification)
        response_time = (time.time() - start_time) * 1000

        return TestNotificationResponse(
            success=success,
            message="Test notification sent successfully!"
            if success
            else "Test notification failed",
            response_time_ms=round(response_time, 2),
        )

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        response_time = (time.time() - start_time) * 1000
        logger.error("test_notification_error", error=str(e))  # type: ignore[call-arg]
        return TestNotificationResponse(
            success=False,
            message=f"Test failed: {str(e)}",
            response_time_ms=round(response_time, 2),
        )

    finally:
        # Restore original state
        integration.config.enabled = was_enabled

@router.get("/health", response_model=TeamsHealthResponse)
async def check_teams_health():
    """
    Perform health check on Teams integration.

    Checks webhook accessibility and returns current status.
    """
    integration = get_teams_integration()
    health = await integration.health_check()

    return TeamsHealthResponse(**health)

@router.get("/stats", response_model=TeamsStatsResponse)
async def get_teams_stats():
    """
    Get Teams notification statistics.

    Returns counts of sent/failed notifications and success rate.
    """
    integration = get_teams_integration()
    stats = integration.get_stats()

    return TeamsStatsResponse(
        notifications_sent=stats.get("notifications_sent", 0),
        notifications_failed=stats.get("notifications_failed", 0),
        retries=stats.get("retries", 0),
        rate_limit_waits=stats.get("rate_limit_waits", 0),
        success_rate=stats.get("success_rate", 100.0),
    )

@router.post("/stats/reset")
async def reset_teams_stats():
    """Reset Teams notification statistics."""
    integration = get_teams_integration()
    integration._stats = {
        "notifications_sent": 0,
        "notifications_failed": 0,
        "retries": 0,
        "rate_limit_waits": 0,
    }

    logger.info("teams_stats_reset")

    return {"status": "success", "message": "Statistics reset successfully"}

@router.post("/alert")
async def send_quick_alert(
    title: str,
    message: str,
    severity: str = "warning",
    job_id: str | None = None,
    instance_name: str | None = None,
):
    """
    Quick endpoint to send an alert.

    Convenience method for sending simple alerts without full notification object.
    """
    success = await send_teams_alert(
        title=title,
        message=message,
        severity=severity,
        job_id=job_id,
        instance_name=instance_name,
    )

    return {
        "success": success,
        "message": "Alert sent" if success else "Alert failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@router.get("/filters")
async def get_notification_filters():
    """Get available notification filters and their current settings."""
    integration = get_teams_integration()

    return {
        "job_status_filters": integration.config.job_status_filters,
        "notification_types": integration.config.notification_types,
        "monitored_instances": integration.config.monitored_tws_instances,
        "available_severities": ["info", "success", "warning", "error", "critical"],
        "available_notification_types": [
            "job_status",
            "alerts",
            "performance",
            "system",
        ],
    }

@router.put("/filters")
async def update_notification_filters(
    job_status_filters: list[str] | None = None,
    notification_types: list[str] | None = None,
    monitored_instances: list[str] | None = None,
):
    """Update notification filters."""
    integration = get_teams_integration()

    if job_status_filters is not None:
        integration.config.job_status_filters = job_status_filters
    if notification_types is not None:
        integration.config.notification_types = notification_types
    if monitored_instances is not None:
        integration.config.monitored_tws_instances = monitored_instances

    logger.info(
        "teams_filters_updated",
        job_status_filters=integration.config.job_status_filters,
        notification_types=integration.config.notification_types,
    )

    return {
        "status": "updated",
        "job_status_filters": integration.config.job_status_filters,
        "notification_types": integration.config.notification_types,
        "monitored_instances": integration.config.monitored_tws_instances,
    }
