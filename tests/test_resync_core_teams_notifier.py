"""
Comprehensive tests for resync/core/teams_notifier.py

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

from resync.core.teams_notifier import TeamsNotificationManager


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

    def test_inactive_channels_ignored(self, mock_db, mock_channel):
        """Test that inactive channels are skipped."""
        mock_channel.is_active = False

        mapping = MagicMock()
        mapping.channel_id = 1

        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=mapping)
        mock_db.execute.return_value = result
        mock_db.get.return_value = mock_channel
        mock_db.query.return_value.first.return_value = None

        manager = TeamsNotificationManager(mock_db)
        channel = manager.get_channel_for_job("JOB_001")

        assert channel is None


class TestPatternMatching:
    """Tests for _matches_pattern method."""

    def test_glob_pattern_matching(self, mock_db):
        """Test glob pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        assert manager._matches_pattern("BATCH_DAILY_001", "BATCH_*", "glob")
        assert manager._matches_pattern("PROD_JOB_123", "PROD_*", "glob")
        assert not manager._matches_pattern("DEV_JOB_001", "PROD_*", "glob")
        assert manager._matches_pattern("ETL_LOAD_001", "*_LOAD_*", "glob")

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

    def test_programming_error_reraise_in_pattern_match(self, mock_db):
        """Test that programming errors are re-raised in pattern matching."""
        manager = TeamsNotificationManager(mock_db)

        # This should raise TypeError (programming error)
        with pytest.raises(TypeError):
            manager._matches_pattern(None, "pattern", "glob")


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

    def test_rate_limit_zero_count(self, mock_db):
        """Test rate limiting with zero max count."""
        manager = TeamsNotificationManager(mock_db)

        # With max_count=0, should always block
        assert not manager._check_rate_limit("JOB_001", max_count=0, window_minutes=60)


class TestQuietHours:
    """Tests for quiet hours functionality."""

    def test_quiet_hours_within_same_day(self, mock_db):
        """Test quiet hours within the same day."""
        manager = TeamsNotificationManager(mock_db)

        with patch('resync.core.teams_notifier.datetime') as mock_datetime:
            # Current time: 23:00
            mock_datetime.now.return_value.time.return_value = datetime.strptime("23:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert manager._is_quiet_hours("22:00", "06:00")

    def test_quiet_hours_outside_range(self, mock_db):
        """Test that times outside quiet hours are not blocked."""
        manager = TeamsNotificationManager(mock_db)

        with patch('resync.core.teams_notifier.datetime') as mock_datetime:
            # Current time: 12:00
            mock_datetime.now.return_value.time.return_value = datetime.strptime("12:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert not manager._is_quiet_hours("22:00", "06:00")

    def test_quiet_hours_at_boundaries(self, mock_db):
        """Test quiet hours at exact boundaries."""
        manager = TeamsNotificationManager(mock_db)

        with patch('resync.core.teams_notifier.datetime') as mock_datetime:
            # Current time: exactly 22:00 (start of quiet hours)
            mock_datetime.now.return_value.time.return_value = datetime.strptime("22:00", "%H:%M").time()
            mock_datetime.strptime = datetime.strptime

            assert manager._is_quiet_hours("22:00", "06:00")


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

        with patch('resync.core.teams_notifier.datetime') as mock_datetime:
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

    def test_should_notify_when_no_config(self, mock_db):
        """Test notification behavior when no config exists."""
        mock_db.query.return_value.first.return_value = None

        manager = TeamsNotificationManager(mock_db)
        # Should return True when no config (permissive default)
        assert manager.should_notify("JOB_001", "ABEND")


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
            assert mock_channel.notification_count == 1
            assert mock_channel.last_notification_sent is not None

    @pytest.mark.asyncio
    async def test_send_notification_with_all_fields(self, mock_db, mock_channel, mock_config):
        """Test notification with all optional fields populated."""
        mock_db.query.return_value.first.return_value = mock_config

        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (True, 200, None)

            result = await manager.send_job_notification(
                job_name="JOB_001",
                job_status="FAILED",
                instance_name="PROD_TWS",
                return_code=8,
                error_message="Database timeout",
                timestamp="2024-01-15 10:30:00"
            )

            assert result is True


class TestCardCreation:
    """Tests for _create_job_notification_card method."""

    def test_card_structure_for_each_status(self, mock_db, mock_channel):
        """Test card structure for different statuses."""
        manager = TeamsNotificationManager(mock_db)

        statuses = ["ABEND", "ERROR", "FAILED", "WARNING"]

        for status in statuses:
            card = manager._create_job_notification_card(
                job_name="JOB_001",
                job_status=status,
                instance_name="PROD_TWS",
                return_code=1,
                error_message=None,
                timestamp=None,
                channel=mock_channel
            )

            assert card["type"] == "message"
            content = card["attachments"][0]["content"]
            assert content["type"] == "AdaptiveCard"
            assert status in content["body"][0]["text"]

    def test_card_includes_all_facts(self, mock_db, mock_channel):
        """Test that card includes all expected facts."""
        manager = TeamsNotificationManager(mock_db)

        card = manager._create_job_notification_card(
            job_name="TEST_JOB",
            job_status="ABEND",
            instance_name="PROD_TWS",
            return_code=12,
            error_message=None,
            timestamp="2024-01-15 10:30:00",
            channel=mock_channel
        )

        facts = card["attachments"][0]["content"]["body"][1]["facts"]
        fact_titles = [f["title"] for f in facts]

        assert "Job:" in fact_titles
        assert "Canal:" in fact_titles
        assert "Status:" in fact_titles
        assert "Instância:" in fact_titles
        assert "Return Code:" in fact_titles
        assert "Timestamp:" in fact_titles


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

    @pytest.mark.asyncio
    async def test_webhook_generic_exception(self, mock_db):
        """Test handling of generic exceptions."""
        manager = TeamsNotificationManager(mock_db)

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = Exception("Network error")

            success, status, error = await manager._send_to_teams(
                "https://test.webhook.url",
                {"test": "card"}
            )

            assert success is False
            assert status is None
            assert "Network error" in error


class TestTestChannel:
    """Tests for test_channel method."""

    @pytest.mark.asyncio
    async def test_channel_test_creates_proper_card(self, mock_db, mock_channel):
        """Test that channel test creates a proper test card."""
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (True, 200, None)

            result = await manager.test_channel(1)

            assert result is True
            # Verify test card was sent
            call_args = mock_send.call_args
            card = call_args[0][1]
            assert "Teste de Notificação" in card["attachments"][0]["content"]["body"][0]["text"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_job_name_pattern_matching(self, mock_db):
        """Test pattern matching with empty job name."""
        manager = TeamsNotificationManager(mock_db)

        assert not manager._matches_pattern("", "PATTERN", "prefix")
        assert manager._matches_pattern("", "", "prefix")

    def test_rate_limit_cache_cleanup(self, mock_db):
        """Test that rate limit cache is properly cleaned."""
        manager = TeamsNotificationManager(mock_db)

        # Add mix of old and new timestamps
        now = datetime.now(timezone.utc)
        old = now - timedelta(minutes=61)
        manager._rate_limit_cache["JOB_001"] = [old, old, now, now]

        # Check rate limit (should clean old entries)
        manager._check_rate_limit("JOB_001", max_count=5, window_minutes=60)

        # Should have 3 entries now (2 old removed, 1 new added)
        assert len(manager._rate_limit_cache["JOB_001"]) == 3

    @pytest.mark.asyncio
    async def test_notification_log_created_on_failure(self, mock_db, mock_channel, mock_config):
        """Test that notification log is created even on webhook failure."""
        mock_db.query.return_value.first.return_value = mock_config

        result_mapping = MagicMock()
        result_mapping.scalar_one_or_none = MagicMock(return_value=None)
        result_rules = MagicMock()
        result_rules.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

        mock_db.execute.side_effect = [result_mapping, result_rules]
        mock_db.get.return_value = mock_channel

        manager = TeamsNotificationManager(mock_db)

        with patch.object(manager, '_send_to_teams', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (False, 400, "Bad Request")

            await manager.send_job_notification(
                job_name="JOB_001",
                job_status="ABEND",
                instance_name="PROD_TWS"
            )

            # Verify log entry was added
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called()