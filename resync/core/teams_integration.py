# pylint: disable=all
"""Microsoft Teams integration for Resync.

This module provides integration with Microsoft Teams for notifications,
job status monitoring, and conversational AI capabilities.

Improvements (v5.3):
- Retry with exponential backoff using tenacity
- Severity-based colors in Adaptive Cards
- Rate limiting to respect Teams API limits
- Integration with Knowledge Graph for conversation learning
- Webhook URL masking in logs
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from resync.core.exceptions import NotificationError
from resync.settings import settings

from .shared_utils import create_job_status_notification

logger = structlog.get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Severity-based colors for Adaptive Cards
SEVERITY_COLORS = {
    "info": "default",
    "success": "good",
    "warning": "warning",
    "error": "attention",
    "critical": "attention",
}

# Severity emoji indicators
SEVERITY_EMOJI = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "critical": "🚨",
}

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 4
RATE_LIMIT_PERIOD = 1.0


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TeamsConfig:
    """Configuration for Microsoft Teams integration."""

    enabled: bool = False
    webhook_url: str | None = None
    channel_name: str | None = None
    bot_name: str = "Resync Bot"
    avatar_url: str | None = None
    enable_conversation_learning: bool = False
    enable_job_notifications: bool = False
    monitored_tws_instances: list[str] = field(default_factory=list)
    job_status_filters: list[str] = field(
        default_factory=lambda: ["ABEND", "ERROR", "FAILED"]
    )
    notification_types: list[str] = field(
        default_factory=lambda: ["job_status", "alerts", "performance"]
    )
    rate_limit_enabled: bool = True
    rate_limit_requests: int = RATE_LIMIT_REQUESTS
    rate_limit_period: float = RATE_LIMIT_PERIOD
    max_retries: int = 3
    retry_min_wait: float = 2.0
    retry_max_wait: float = 10.0


@dataclass
class TeamsNotification:
    """Structure for Teams notification data."""

    title: str
    message: str
    severity: str = "info"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    job_id: str | None = None
    job_status: str | None = None
    instance_name: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, str]] = field(default_factory=list)


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Simple token bucket rate limiter for Teams API."""

    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.tokens: float = float(max_requests)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * (self.max_requests / self.period),
            )
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.period / self.max_requests)
                logger.debug("rate_limiter_waiting", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# =============================================================================
# TEAMS INTEGRATION SERVICE
# =============================================================================


class TeamsIntegration:
    """Microsoft Teams integration service with enhanced features."""

    def __init__(self, config: TeamsConfig | None = None):
        self.config = config or self._load_config_from_settings()
        self.session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

        self.rate_limiter = (
            RateLimiter(
                max_requests=self.config.rate_limit_requests,
                period=self.config.rate_limit_period,
            )
            if self.config.rate_limit_enabled
            else None
        )

        self._stats = {
            "notifications_sent": 0,
            "notifications_failed": 0,
            "retries": 0,
            "rate_limit_waits": 0,
        }

    def _load_config_from_settings(self) -> TeamsConfig:
        """Load Teams configuration from application settings."""
        config = TeamsConfig()

        teams_settings = getattr(settings, "TEAMS_INTEGRATION", {})
        if teams_settings:
            config.enabled = teams_settings.get("enabled", False)
            config.webhook_url = teams_settings.get("webhook_url")
            config.channel_name = teams_settings.get("channel_name")
            config.bot_name = teams_settings.get("bot_name", "Resync Bot")
            config.avatar_url = teams_settings.get("avatar_url")
            config.enable_conversation_learning = teams_settings.get(
                "enable_conversation_learning", False
            )
            config.enable_job_notifications = teams_settings.get(
                "enable_job_notifications", False
            )
            config.monitored_tws_instances = teams_settings.get(
                "monitored_tws_instances", []
            )
            config.job_status_filters = teams_settings.get(
                "job_status_filters", ["ABEND", "ERROR", "FAILED"]
            )
            config.notification_types = teams_settings.get(
                "notification_types", ["job_status", "alerts", "performance"]
            )
            config.rate_limit_enabled = teams_settings.get("rate_limit_enabled", True)
            config.max_retries = teams_settings.get("max_retries", 3)

        return config

    def _mask_webhook_url(self, url: str | None) -> str:
        """Mask webhook URL for safe logging."""
        if not url:
            return "not_configured"
        if len(url) < 20:
            return "***masked***"
        return f"{url[:30]}...{url[-10:]}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp client session."""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                connector = aiohttp.TCPConnector(
                    limit=100, limit_per_host=30, ttl_dns_cache=300
                )
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"},
                )
            return self.session

    async def _close_session(self) -> None:
        """Close aiohttp client session."""
        async with self._session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            "teams_notification_retry",
            attempt=retry_state.attempt_number,
        ),
    )
    async def _send_webhook_request(
        self, session: aiohttp.ClientSession, message: dict[str, Any]
    ) -> aiohttp.ClientResponse:
        """Send webhook request with retry logic."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
            self._stats["rate_limit_waits"] += 1

        async with session.post(self.config.webhook_url, json=message) as response:
            await response.read()
            return response

    async def send_notification(self, notification: TeamsNotification) -> bool:
        """Send notification to Microsoft Teams with retry and rate limiting."""
        if not self.config.enabled:
            logger.debug("teams_notification_skipped", reason="integration_disabled")
            return False

        if not self.config.webhook_url:
            logger.warning(
                "teams_notification_failed", reason="webhook_url_not_configured"
            )
            raise NotificationError("Teams webhook URL not configured")

        try:
            teams_message = self._format_teams_message(notification)
            session = await self._get_session()
            response = await self._send_webhook_request(session, teams_message)

            if response.status in [200, 201, 202, 204]:
                self._stats["notifications_sent"] += 1
                logger.info(
                    "teams_notification_sent",
                    title=notification.title,
                    severity=notification.severity,
                    webhook=self._mask_webhook_url(self.config.webhook_url),
                )
                return True
            self._stats["notifications_failed"] += 1
            error_text = await response.text()
            logger.error(
                "teams_notification_failed",
                status=response.status,
                error=error_text[:200],
                webhook=self._mask_webhook_url(self.config.webhook_url),
            )
            return False

        except aiohttp.ClientError as e:
            self._stats["notifications_failed"] += 1
            logger.error(
                "teams_notification_network_error",
                error=str(e),
                webhook=self._mask_webhook_url(self.config.webhook_url),
            )
            raise NotificationError(
                f"Network error sending Teams notification: {e}"
            ) from e
        except Exception as e:
            self._stats["notifications_failed"] += 1
            logger.error(
                "teams_notification_unexpected_error",
                error=str(e),
                webhook=self._mask_webhook_url(self.config.webhook_url),
            )
            raise NotificationError(
                f"Unexpected error sending Teams notification: {e}"
            ) from e

    def _format_teams_message(self, notification: TeamsNotification) -> dict[str, Any]:
        """Format notification as Microsoft Teams Adaptive Card with severity colors."""
        severity_color = SEVERITY_COLORS.get(notification.severity, "default")
        severity_emoji = SEVERITY_EMOJI.get(notification.severity, "")

        title_text = (
            f"{severity_emoji} {notification.title}"
            if severity_emoji
            else notification.title
        )

        card_body = [
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "text": title_text,
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": notification.message,
                "wrap": True,
            },
        ]

        if severity_color != "default":
            card_body[0]["color"] = severity_color

        facts = []

        if notification.instance_name:
            facts.append({"title": "Instance", "value": notification.instance_name})

        if notification.job_id:
            facts.append({"title": "Job ID", "value": notification.job_id})

        if notification.job_status:
            facts.append({"title": "Status", "value": notification.job_status})

        facts.append(
            {
                "title": "Timestamp",
                "value": notification.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        facts.append({"title": "Severity", "value": notification.severity.upper()})

        if facts:
            card_body.append({"type": "FactSet", "facts": facts})

        if notification.additional_data:
            additional_facts = [
                {"title": k, "value": str(v)[:100]}
                for k, v in notification.additional_data.items()
            ]
            if additional_facts:
                card_body.append(
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Additional Details",
                                "weight": "Bolder",
                            },
                            {"type": "FactSet", "facts": additional_facts},
                        ],
                    }
                )

        card_actions = []
        for action in notification.actions:
            if action.get("type") == "openUrl" and action.get("url"):
                card_actions.append(
                    {
                        "type": "Action.OpenUrl",
                        "title": action.get("title", "Open"),
                        "url": action["url"],
                    }
                )

        message_card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "msteams": {"width": "Full"},
                        "body": card_body,
                    },
                }
            ],
        }

        if card_actions:
            attachments = message_card.get("attachments")
            if isinstance(attachments, list) and attachments:
                first = attachments[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        content["actions"] = card_actions

        return message_card

    async def monitor_job_status(
        self, job_data: dict[str, Any], instance_name: str
    ) -> None:
        """Monitor job status and notify configured job state changes."""
        if not self.config.enabled or not self.config.enable_job_notifications:
            return

        if (
            self.config.monitored_tws_instances
            and instance_name not in self.config.monitored_tws_instances
        ):
            return

        job_status = job_data.get("status", "").upper()
        if job_status in [status.upper() for status in self.config.job_status_filters]:
            severity = (
                "error" if job_status in ["ABEND", "ERROR", "FAILED"] else "warning"
            )

            notification = create_job_status_notification(
                job_data, instance_name, self.config.job_status_filters
            )

            if notification is None:
                return

            notification.severity = severity

            try:
                await self.send_notification(notification)
            except NotificationError as e:
                logger.error("job_status_notification_failed", error=str(e))
            except Exception as e:
                logger.error("job_status_notification_unexpected_error", error=str(e))

    async def learn_from_conversation(
        self, message: str, context: dict[str, Any]
    ) -> None:
        """Learn from Teams conversation and store in Knowledge Graph."""
        if not self.config.enabled or not self.config.enable_conversation_learning:
            return

        logger.info(
            "learning_from_teams_conversation",
            message_preview=message[:100] if len(message) > 100 else message,
            sender=context.get("sender", "unknown"),
        )

        try:
            from resync.knowledge.retrieval.graph import get_knowledge_graph

            kg = await get_knowledge_graph()
            if kg:
                await kg.add_conversation_knowledge(
                    source="teams",
                    message=message,
                    context=context,
                    timestamp=datetime.now(timezone.utc),
                )
                logger.debug("conversation_stored_in_knowledge_graph")
        except ImportError:
            logger.debug("knowledge_graph_not_available")
        except Exception as e:
            logger.warning("knowledge_graph_storage_failed", error=str(e))

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        job_id: str | None = None,
        instance_name: str | None = None,
        actions: list[dict[str, str]] | None = None,
    ) -> bool:
        """Convenience method to send an alert notification."""
        notification = TeamsNotification(
            title=title,
            message=message,
            severity=severity,
            job_id=job_id,
            instance_name=instance_name,
            actions=actions or [],
        )
        return await self.send_notification(notification)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of Teams integration."""
        status = {
            "enabled": self.config.enabled,
            "configured": bool(self.config.webhook_url),
            "webhook_url_masked": self._mask_webhook_url(self.config.webhook_url),
            "conversation_learning": self.config.enable_conversation_learning,
            "job_notifications": self.config.enable_job_notifications,
            "monitored_instances": len(self.config.monitored_tws_instances),
            "rate_limiting": self.config.rate_limit_enabled,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "statistics": self._stats.copy(),
        }

        if self.config.enabled and self.config.webhook_url:
            try:
                session = await self._get_session()
                async with session.head(
                    self.config.webhook_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                    allow_redirects=True,
                ) as response:
                    status["webhook_accessible"] = response.status in [200, 405, 400]
            except Exception as e:
                status["webhook_accessible"] = False
                status["webhook_error"] = str(e)[:100]
                logger.warning(
                    "teams_webhook_health_check_failed",
                    error=str(e),
                    webhook=self._mask_webhook_url(self.config.webhook_url),
                )

        return status

    def get_stats(self) -> dict[str, Any]:
        """Get notification statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["notifications_sent"]
                / max(
                    1,
                    self._stats["notifications_sent"]
                    + self._stats["notifications_failed"],
                )
            )
            * 100,
        }

    async def shutdown(self) -> None:
        """Shutdown Teams integration service."""
        await self._close_session()
        logger.info("teams_integration_service_shutdown", stats=self._stats)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_teams_integration: TeamsIntegration | None = None


def get_teams_integration() -> TeamsIntegration:
    """Get global Teams integration instance."""
    global _teams_integration
    if _teams_integration is None:
        _teams_integration = TeamsIntegration()
    return _teams_integration


async def shutdown_teams_integration() -> None:
    """Shutdown global Teams integration instance."""
    global _teams_integration
    if _teams_integration is not None:
        await _teams_integration.shutdown()
        _teams_integration = None


async def send_teams_alert(
    title: str,
    message: str,
    severity: str = "warning",
    **kwargs,
) -> bool:
    """Send a Teams alert notification."""
    integration = get_teams_integration()
    return await integration.send_alert(title, message, severity, **kwargs)
