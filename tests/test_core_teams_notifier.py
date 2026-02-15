"""
Comprehensive tests for core/teams_notifier.py

Tests cover:
- Channel selection logic (mapping, pattern rules, default)
- Pattern matching (glob, regex, prefix, suffix, contains)
- Rate limiting
- Quiet hours
- Notification filtering
- Teams webhook integration
- Card creation
- Error handling
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp

from core.teams_notifier import TeamsNotificationManager


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.execute = MagicMock()
    db.commit = MagicMock()
    db.get = MagicMock()
    db.add = MagicMock()
    db.query = MagicMock()
    return db


@pytest.fixture
def mock_channel():
    """Create a mock Teams channel."""
    channel = MagicMock()
    channel.id = 1
    channel.name = "Test Channel"
    channel.icon = "📢"
    channel.webhook_url = "https://outlook.office.com/webhook/test"
    channel.is_active = True
    channel.notification_count = 0
    channel.last_notification_sent = None
    return channel


@pytest.fixture
def mock_config():
    """Create a mock notification config."""
    config = MagicMock()
    config.enabled = True
    config.notify_on_status = ["ABEND", "ERROR", "FAILED"]
    config.quiet_hours_enabled = False
    config.quiet_hours_start = None
    config.quiet_hours_end = None
    config.rate_limit_enabled = False
    config.max_notifications_per_job = 5
    config.rate_limit_window_minutes = 60
    config.default_channel_id = 1
    config.include_mention_on_critical = False
    config.mention_text = "@team"
    return config


class TestChannelSelection:
    """Tests for get_channel_for_job method."""

    def test_specific_mapping_takes_priority(self, mock_db, mock_channel):
        """Test that specific job mapping takes priority over pattern rules."""
        # Setup mapping
        mapping = MagicMock()
        mapping.channel_id = 1

        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=mapping)
        mock_db.execute.return_value = result
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)
        channel = manager.get_channel_for_job("CRITICAL_JOB_001")

        assert channel == mock_channel
        assert channel.name == "Test Channel"

    def test_pattern_rule_when_no_mapping(self, mock_db, mock_channel):
        """Test that pattern rules are used when no specific mapping exists."""
        # No mapping
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)

        # Pattern rule
        rule = MagicMock()
        rule.pattern = "BATCH_*"
        rule.pattern_type = "glob"
        rule.channel_id = 1
        rule.match_count = 0

        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[rule])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)
        channel = manager.get_channel_for_job("BATCH_DAILY_LOAD")

        assert channel == mock_channel
        assert rule.match_count == 1

    def test_default_channel_fallback(self, mock_db, mock_channel, mock_config):
        """Test that default channel is used when no mapping or pattern matches."""
        # No mapping
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)

        # No pattern rules
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)
        channel = manager.get_channel_for_job("UNKNOWN_JOB")

        assert channel == mock_channel

    def test_no_channel_found(self, mock_db):
        """Test that None is returned when no channel can be found."""
        # No mapping
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)

        # No pattern rules
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = None
        mock_db.query.return_value.first.return_value = None

        manager = TeamsNotificationManager(mock_db)
        channel = manager.get_channel_for_job("ORPHAN_JOB")

        assert channel is None


class TestPatternMatching:
    """Tests for _matches_pattern method."""

    def test_glob_pattern_matching(self, mock_db):
        """Test glob pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("BATCH_DAILY_001", "BATCH_*", "glob")
        assert manager._matches_pattern("PROD_JOB_123", "PROD_*", "glob")
        assert not manager._matches_pattern("DEV_JOB_001", "PROD_*", "glob")

    def test_regex_pattern_matching(self, mock_db):
        """Test regex pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("JOB_001", r"JOB_\d{3}", "regex")
        assert manager._matches_pattern("BATCH_2024", r"BATCH_\d{4}", "regex")
        assert not manager._matches_pattern("JOB_ABC", r"JOB_\d{3}", "regex")

    def test_prefix_pattern_matching(self, mock_db):
        """Test prefix pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("CRITICAL_JOB_001", "CRITICAL_", "prefix")
        assert not manager._matches_pattern("JOB_CRITICAL_001", "CRITICAL_", "prefix")

    def test_suffix_pattern_matching(self, mock_db):
        """Test suffix pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("DAILY_BATCH_ETL", "_ETL", "suffix")
        assert not manager._matches_pattern("ETL_DAILY_BATCH", "_ETL", "suffix")

    def test_contains_pattern_matching(self, mock_db):
        """Test contains pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("PROD_CRITICAL_ETL", "CRITICAL", "contains")
        assert not manager._matches_pattern("PROD_STANDARD_ETL", "CRITICAL", "contains")

    def test_unknown_pattern_type(self, mock_db):
        """Test that unknown pattern types return False."""
        manager = TeamsNotificationManager(mock_db)

        assert not manager._matches_pattern("ANY_JOB", "pattern", "unknown_type")

    def test_pattern_error_handling(self, mock_db):
        """Test that pattern matching errors are handled gracefully."""
        manager = TeamsNotificationManager(mock_db)

        # Invalid regex should return False
        assert not manager._matches_pattern("JOB", "[invalid(regex", "regex")


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_allows_initial_notifications(self, mock_db):
        """Test that rate limiting allows initial notifications."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

    def test_rate_limit_blocks_excess_notifications(self, mock_db):
        """Test that rate limiting blocks notifications after limit is reached."""
        manager = TeamsNotificationManager(mock_db)

        # Send max_count notifications
        for _ in range(5):
            assert manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

        # Next one should be blocked
        assert not manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

    def test_rate_limit_window_expiry(self, mock_db):
        """Test that rate limit window expires old notifications."""
        manager = TeamsNotificationManager(mock_db)

        # Add old notifications
        old_time = datetime.now(timezone.utc) - timedelta(minutes=61)
        manager._rate_limit_cache["JOB_001"] = [old_time] * 5

        # Should allow new notification (old ones expired)
        assert manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

    def test_rate_limit_per_job_isolation(self, mock_db):
        """Test that rate limits are per-job."""
        manager = TeamsNotificationManager(mock_db)

        # Fill limit for JOB_001
        for _ in range(5):
            manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

        # JOB_002 should still be allowed
        assert manager._check_rate_limit("JOB_002", max_count=5, window_minutes=60)


class TestQuietHours:
    """Tests for quiet hours functionality."""

    def test_quiet_hours_within_same_day(self, mock_db):
        """Test quiet hours within the same day (e.g., 22:00-06:00)."""
        manager = TeamsNotificationManager(mock_db)

        with patch('core.teams_notifier.datetime') as mock_datetime:
            # Current time: 23:00
            mock_datetime.now.return_value.time.return_value = datetime.strptime("23:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert manager._is_quiet_hours("22:00", "06:00")

    def test_quiet_hours_outside_range(self, mock_db):
        """Test that times outside quiet hours are not blocked."""
        manager = TeamsNotificationManager(mock_db)

        with patch('core.teams_notifier.datetime') as mock_datetime:
            # Current time: 12:00
            mock_datetime.now.return_value.time.return_value = datetime.strptime("12:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert not manager._is_quiet_hours("22:00", "06:00")

    def test_quiet_hours_crossing_midnight(self, mock_db):
        """Test quiet hours that cross midnight."""
        manager = TeamsNotificationManager(mock_db)

        with patch('core.teams_notifier.datetime') as mock_datetime:
            # Current time: 02:00
            mock_datetime.now.return_value.time.return_value = datetime.strptime("02:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            # Quiet hours: 22:00-06:00 (crosses midnight)
            assert manager._is_quiet_hours("22:00", "06:00")

    def test_quiet_hours_disabled(self, mock_db):
        """Test that quiet hours with None values are disabled."""
        manager = TeamsNotificationManager(mock_db)

        assert not manager._is_quiet_hours(None, None)
        assert not manager._is_quiet_hours("22:00", None)
        assert not manager._is_quiet_hours(None, "06:00")


class TestShouldNotify:
    """Tests for should_notify method."""

    def test_should_notify_with_matching_status(self, mock_db, mock_config):
        """Test that notification is allowed for matching status."""
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)
        assert manager.should_notify("JOB_001", "ABEND")

    def test_should_not_notify_with_non_matching_status(self, mock_db, mock_config):
        """Test that notification is blocked for non-matching status."""
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)
        assert not manager.should_notify("JOB_001", "SUCCESS")

    def test_should_not_notify_during_quiet_hours(self, mock_db, mock_config):
        """Test that notifications are blocked during quiet hours."""
        mock_config.quiet_hours_enabled = True
        mock_config.quiet_hours_start = "22:00"
        mock_config.quiet_hours_end = "06:00"
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)

        with patch('core.teams_notifier.datetime') as mock_datetime:
            # Current time: 23:00 (in quiet hours)
            mock_datetime.now.return_value.time.return_value = datetime.strptime("23:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert not manager.should_notify("JOB_001", "ABEND")

    def test_should_not_notify_when_rate_limited(self, mock_db, mock_config):
        """Test that notifications are blocked when rate limited."""
        mock_config.rate_limit_enabled = True
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)

        # Fill rate limit
        for _ in range(5):
            manager._check_rate_limit("JOB_001", 5, 60)

        assert not manager.should_notify("JOB_001", "ABEND")


class TestNotificationSending:
    """Tests for send_job_notification method."""

    @pytest.mark.asyncio
    async def test_send_notification_success(self, mock_db, mock_channel, mock_config):
        """Test successful notification sending."""
        mock_db.query.return_value.first.return_value = mock_config

        # Setup channel lookup
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        # Mock Teams webhook
        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (True, 200, None)

            result = await manager.send_job_notification(
                job_name="JOB_001",
                job_status="ABEND",
                instance_name="PROD_TWS",
                return_code=12,
                error_message="Out of memory"
            )

            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_skipped_by_filter(self, mock_db, mock_config):
        """Test that notification is skipped when filtered."""
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)

        result = await manager.send_job_notification(
            job_name="JOB_001",
            job_status="SUCCESS",  # Not in notify_on_status
            instance_name="PROD_TWS"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_notification_no_channel(self, mock_db, mock_config):
        """Test that notification fails when no channel is found."""
        mock_db.query.return_value.first.return_value = mock_config

        # No channel found
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = None

        manager = TeamsNotificationManager(mock_db)

        result = await manager.send_job_notification(
            job_name="JOB_001",
            job_status="ABEND",
            instance_name="PROD_TWS"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_notification_webhook_failure(self, mock_db, mock_channel, mock_config):
        """Test handling of webhook failure."""
        mock_db.query.return_value.first.return_value = mock_config

        # Setup channel
        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        # Mock webhook failure
        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (False, 400, "Bad Request")

            result = await manager.send_job_notification(
                job_name="JOB_001",
                job_status="ABEND",
                instance_name="PROD_TWS"
            )

            assert result is False


class TestCardCreation:
    """Tests for _create_job_notification_card method."""

    def test_card_structure_abend(self, mock_db, mock_channel):
        """Test card structure for ABEND status."""
        manager = TeamsNotificationManager(mock_db)

        card = manager._create_job_notification_card(
            job_name="JOB_001",
            job_status="ABEND",
            instance_name="PROD_TWS",
            return_code=12,
            error_message="Memory allocation failed",
            timestamp="2024-01-15 10:30:00",
            channel=mock_channel
        )

        assert card["type"] == "message"
        assert len(card["attachments"]) == 1
        assert card["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"

        content = card["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert "JOB ABEND" in content["body"][0]["text"]
        assert content["body"][0]["color"] == "attention"

    def test_card_includes_error_message(self, mock_db, mock_channel):
        """Test that error message is included in card."""
        manager = TeamsNotificationManager(mock_db)

        error_msg = "Database connection timeout"
        card = manager._create_job_notification_card(
            job_name="JOB_001",
            job_status="ERROR",
            instance_name="PROD_TWS",
            return_code=1,
            error_message=error_msg,
            timestamp=None,
            channel=mock_channel
        )

        body = card["attachments"][0]["content"]["body"]
        error_block = next((b for b in body if b.get("type") == "TextBlock" and "Erro:" in b.get("text", "")), None)
        assert error_block is not None
        assert error_msg in error_block["text"]

    def test_card_includes_mention_for_critical(self, mock_db, mock_channel, mock_config):
        """Test that mention is included for critical statuses."""
        mock_config.include_mention_on_critical = True
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)

        card = manager._create_job_notification_card(
            job_name="JOB_001",
            job_status="ABEND",
            instance_name="PROD_TWS",
            return_code=12,
            error_message=None,
            timestamp=None,
            channel=mock_channel
        )

        body = card["attachments"][0]["content"]["body"]
        mention_block = next((b for b in body if b.get("text") == "@team"), None)
        assert mention_block is not None


class TestWebhookIntegration:
    """Tests for _send_to_teams method."""

    @pytest.mark.asyncio
    async def test_webhook_success(self, mock_db):
        """Test successful webhook call."""
        manager = TeamsNotificationManager(mock_db)

        mock_response = AsyncMock()
        mock_response.status = 200

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            success, status, error = await manager._send_to_teams(
                "https://test.webhook.url",
                {"test": "card"}
            )

            assert success is True
            assert status == 200
            assert error is None

    @pytest.mark.asyncio
    async def test_webhook_failure(self, mock_db):
        """Test webhook failure handling."""
        manager = TeamsNotificationManager(mock_db)

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            success, status, error = await manager._send_to_teams(
                "https://test.webhook.url",
                {"test": "card"}
            )

            assert success is False
            assert status == 400
            assert "Bad Request" in error

    @pytest.mark.asyncio
    async def test_webhook_timeout(self, mock_db):
        """Test webhook timeout handling."""
        manager = TeamsNotificationManager(mock_db)

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = aiohttp.ClientTimeout()

            success, status, error = await manager._send_to_teams(
                "https://test.webhook.url",
                {"test": "card"}
            )

            assert success is False
            assert status is None
            assert error == "Timeout"


class TestTestChannel:
    """Tests for test_channel method."""

    @pytest.mark.asyncio
    async def test_channel_test_success(self, mock_db, mock_channel):
        """Test successful channel test."""
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (True, 200, None)

            result = await manager.test_channel(1)

            assert result is True

    @pytest.mark.asyncio
    async def test_channel_test_not_found(self, mock_db):
        """Test channel test with non-existent channel."""
        mock_db.get.return_value = None

        manager = TeamsNotificationManager(mock_db)

        result = await manager.test_channel(999)

        assert result is False


class TestIsEnabled:
    """Tests for is_enabled method."""

    def test_is_enabled_true(self, mock_db, mock_config):
        """Test that is_enabled returns True when configured."""
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)
        assert manager.is_enabled() is True

    def test_is_enabled_false(self, mock_db, mock_config):
        """Test that is_enabled returns False when disabled."""
        mock_config.enabled = False
        mock_db.query.return_value.first.return_value = mock_config

        manager = TeamsNotificationManager(mock_db)
        assert manager.is_enabled() is False

    def test_is_enabled_no_config(self, mock_db):
        """Test that is_enabled returns False when no config exists."""
        mock_db.query.return_value.first.return_value = None

        manager = TeamsNotificationManager(mock_db)
        assert manager.is_enabled() is False


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_pattern_matching_with_special_characters(self, mock_db):
        """Test pattern matching with special characters in job names."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("JOB_[PROD]_001", "JOB_*_001", "glob")
        assert manager._matches_pattern("JOB.DAILY.ETL", "*.ETL", "glob")

    def test_long_error_message_truncation(self, mock_db, mock_channel):
        """Test that long error messages are truncated in cards."""
        manager = TeamsNotificationManager(mock_db)

        long_error = "X" * 1000
        card = manager._create_job_notification_card(
            job_name="JOB_001",
            job_status="ERROR",
            instance_name="PROD_TWS",
            return_code=1,
            error_message=long_error,
            timestamp=None,
            channel=mock_channel
        )

        body = card["attachments"][0]["content"]["body"]
        error_block = next((b for b in body if b.get("type") == "TextBlock" and "Erro:" in b.get("text", "")), None)
        assert error_block is not None
        # Error should be truncated to 500 chars
        assert len(error_block["text"]) <= 510  # "**Erro:** " + 500 chars

    @pytest.mark.asyncio
    async def test_programming_error_reraise(self, mock_db):
        """Test that programming errors (TypeError, KeyError, etc.) are re-raised."""
        manager = TeamsNotificationManager(mock_db)

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = TypeError("Programming error")

            with pytest.raises(TypeError):
                await manager._send_to_teams("https://test.url", {"test": "card"})