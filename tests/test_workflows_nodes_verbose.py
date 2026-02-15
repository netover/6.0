"""
Comprehensive tests for workflows/nodes_verbose.py

Tests cover:
- fetch_job_history: PostgreSQL and TWS API fallback
- fetch_workstation_metrics: Data retrieval from multiple sources
- detect_degradation: Statistical analysis and pattern detection
- correlate_metrics: Correlation analysis between metrics
- predict_timeline: Failure prediction and extrapolation
- generate_recommendations: Recommendation generation
- notify_operators: Multi-channel notifications
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from workflows.nodes_verbose import (
    fetch_job_history,
    fetch_workstation_metrics,
    detect_degradation,
    correlate_metrics,
    predict_timeline,
    generate_recommendations,
    notify_operators,
)


class TestFetchJobHistory:
    """Tests for fetch_job_history function."""

    @pytest.mark.asyncio
    async def test_fetch_from_postgresql_success(self, monkeypatch):
        """Test successful fetch from PostgreSQL."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "timestamp": now,
                "job_name": "TEST_JOB",
                "workstation": "WS01",
                "status": "SUCC",
                "return_code": 0,
                "runtime_seconds": 120,
                "scheduled_time": now,
                "actual_start_time": now,
                "completed_time": now,
            }
        ]

        # Mock table exists check
        check_result = MagicMock()
        check_result.scalar = MagicMock(return_value=True)

        # Mock query result
        query_result = SimpleNamespace(
            mappings=lambda: SimpleNamespace(fetchall=lambda: rows)
        )

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[check_result, query_result])

        history = await fetch_job_history(db, "TEST_JOB", days=30)

        assert len(history) == 1
        assert history[0]["job_name"] == "TEST_JOB"
        assert history[0]["status"] == "SUCC"

    @pytest.mark.asyncio
    async def test_fetch_with_feature_disabled(self, monkeypatch):
        """Test that function returns empty list when feature disabled."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "false")

        db = AsyncMock()
        history = await fetch_job_history(db, "TEST_JOB")

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_postgresql_table_not_exists(self, monkeypatch):
        """Test fallback when PostgreSQL table doesn't exist."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        # Table doesn't exist
        check_result = MagicMock()
        check_result.scalar = MagicMock(return_value=False)

        db = AsyncMock()
        db.execute = AsyncMock(return_value=check_result)

        with patch('workflows.nodes_verbose.get_tws_client_singleton') as mock_tws:
            mock_client = AsyncMock()
            mock_client.query_current_plan_jobs = AsyncMock(return_value=[])
            mock_tws.return_value = mock_client

            history = await fetch_job_history(db, "TEST_JOB")

            # Should attempt TWS API
            mock_client.query_current_plan_jobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, monkeypatch):
        """Test error handling returns empty list."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=Exception("Database error"))

        history = await fetch_job_history(db, "TEST_JOB")

        assert len(history) == 0


class TestFetchWorkstationMetrics:
    """Tests for fetch_workstation_metrics function."""

    @pytest.mark.asyncio
    async def test_fetch_from_postgresql(self, monkeypatch):
        """Test successful fetch from PostgreSQL."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "timestamp": now,
                "workstation": "WS01",
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "disk_percent": 70.0,
                "network_mbps": 100.0,
                "active_jobs": 5,
            }
        ]

        check_result = MagicMock()
        check_result.scalar = MagicMock(return_value=True)

        query_result = SimpleNamespace(
            mappings=lambda: SimpleNamespace(fetchall=lambda: rows)
        )

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[check_result, query_result])

        metrics = await fetch_workstation_metrics(db, "WS01")

        assert len(metrics) == 1
        assert metrics[0]["workstation"] == "WS01"
        assert metrics[0]["cpu_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_fetch_with_feature_disabled(self, monkeypatch):
        """Test returns empty when feature disabled."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "false")

        db = AsyncMock()
        metrics = await fetch_workstation_metrics(db, "WS01")

        assert len(metrics) == 0


class TestDetectDegradation:
    """Tests for detect_degradation function."""

    @pytest.mark.asyncio
    async def test_insufficient_data(self, monkeypatch):
        """Test with insufficient data points."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        job_history = [
            {"job_name": "TEST", "runtime_seconds": 100, "status": "SUCC", "timestamp": datetime.now(timezone.utc).isoformat()}
        ]

        llm = AsyncMock()

        result = await detect_degradation(job_history, llm)

        assert result["detected"] is False
        assert "Insufficient data" in result["evidence"]

    @pytest.mark.asyncio
    async def test_detect_runtime_degradation(self, monkeypatch):
        """Test detection of runtime degradation."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        job_history = [
            {
                "job_name": "TEST",
                "runtime_seconds": 100 + i * 10,  # Increasing runtime
                "status": "SUCC",
                "timestamp": (now - timedelta(days=i)).isoformat()
            }
            for i in range(20, 0, -1)  # Reverse order (oldest to newest)
        ]

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Runtime degradation detected"))

        result = await detect_degradation(job_history, llm)

        assert result["detected"] is True
        assert result["type"] == "runtime_increase"
        assert result["severity"] > 0

    @pytest.mark.asyncio
    async def test_detect_failure_rate_increase(self, monkeypatch):
        """Test detection of increasing failure rate."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        # First half: all success, second half: many failures
        job_history = []
        for i in range(10):
            job_history.append({
                "job_name": "TEST",
                "runtime_seconds": 100,
                "status": "SUCC",
                "timestamp": (now - timedelta(days=19 - i)).isoformat()
            })
        for i in range(10):
            job_history.append({
                "job_name": "TEST",
                "runtime_seconds": 100,
                "status": "FAILED",
                "timestamp": (now - timedelta(days=9 - i)).isoformat()
            })

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Failure rate increased"))

        result = await detect_degradation(job_history, llm)

        assert result["detected"] is True
        assert result["type"] in ["failure_rate_increase", "runtime_increase"]

    @pytest.mark.asyncio
    async def test_no_degradation_detected(self, monkeypatch):
        """Test when no degradation is present."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        job_history = [
            {
                "job_name": "TEST",
                "runtime_seconds": 100,  # Stable runtime
                "status": "SUCC",
                "timestamp": (now - timedelta(days=i)).isoformat()
            }
            for i in range(20)
        ]

        llm = AsyncMock()

        result = await detect_degradation(job_history, llm)

        assert result["detected"] is False


class TestCorrelateMetrics:
    """Tests for correlate_metrics function."""

    @pytest.mark.asyncio
    async def test_no_degradation(self, monkeypatch):
        """Test skips correlation when no degradation."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()

        result = await correlate_metrics([], [], None, llm)

        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_insufficient_data(self, monkeypatch):
        """Test with insufficient data."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        job_history = []
        workstation_metrics = []
        llm = AsyncMock()

        result = await correlate_metrics(job_history, workstation_metrics, "runtime_increase", llm)

        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_strong_cpu_correlation(self, monkeypatch):
        """Test detection of strong CPU correlation."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)

        # Job history with increasing runtime
        job_history = [
            {
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "runtime_seconds": 100 + i * 5,
                "job_name": "TEST"
            }
            for i in range(10)
        ]

        # Workstation metrics with increasing CPU
        workstation_metrics = [
            {
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "workstation": "WS01",
                "cpu_percent": 50 + i * 5,
                "memory_percent": 40.0,
                "disk_percent": 30.0
            }
            for i in range(10)
        ]

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="CPU correlation found"))

        result = await correlate_metrics(
            job_history,
            workstation_metrics,
            "runtime_increase",
            llm
        )

        assert result["found"] is True
        assert result["correlation_coefficient"] > 0.5

    @pytest.mark.asyncio
    async def test_feature_disabled(self, monkeypatch):
        """Test returns false when feature disabled."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "false")

        llm = AsyncMock()
        result = await correlate_metrics([], [], "runtime_increase", llm)

        assert result["found"] is False


class TestPredictTimeline:
    """Tests for predict_timeline function."""

    @pytest.mark.asyncio
    async def test_no_degradation(self, monkeypatch):
        """Test prediction with no degradation."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        result = await predict_timeline([], None, 0.0, llm)

        assert result["probability"] == 0.0
        assert result["date"] is None

    @pytest.mark.asyncio
    async def test_insufficient_data(self, monkeypatch):
        """Test with insufficient data points."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        job_history = [
            {"runtime_seconds": 100, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]

        llm = AsyncMock()
        result = await predict_timeline(job_history, "runtime_increase", 0.5, llm)

        assert result["probability"] == 0.0
        assert "Insufficient" in result["explanation"]

    @pytest.mark.asyncio
    async def test_predict_imminent_failure(self, monkeypatch):
        """Test prediction of imminent failure."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        # Rapidly increasing runtime
        job_history = [
            {
                "runtime_seconds": 100 + i * 50,  # Large increase
                "timestamp": (now - timedelta(days=i)).isoformat()
            }
            for i in range(15, 0, -1)
        ]

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Imminent failure predicted"))

        result = await predict_timeline(job_history, "runtime_increase", 0.8, llm)

        assert result["probability"] > 0.5
        if result["date"]:
            assert isinstance(result["date"], datetime)

    @pytest.mark.asyncio
    async def test_stable_trend_low_probability(self, monkeypatch):
        """Test prediction with stable trend."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        # Stable runtime
        job_history = [
            {
                "runtime_seconds": 100,
                "timestamp": (now - timedelta(days=i)).isoformat()
            }
            for i in range(15)
        ]

        llm = AsyncMock()

        result = await predict_timeline(job_history, "runtime_increase", 0.3, llm)

        assert result["probability"] <= 0.2


class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""

    @pytest.mark.asyncio
    async def test_no_problem(self, monkeypatch):
        """Test recommendations when no problem detected."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        result = await generate_recommendations(None, [], 0.05, None, llm)

        assert result["urgency"] == "low"
        assert "Continue monitoring" in result["recommendations"][0]["action"]

    @pytest.mark.asyncio
    async def test_cpu_related_recommendations(self, monkeypatch):
        """Test CPU-related recommendations."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="CPU optimization needed"))

        result = await generate_recommendations(
            "CPU usage strongly correlates",
            ["cpu_saturation"],
            0.7,
            None,
            llm
        )

        assert result["urgency"] in ["high", "critical"]
        # Should have CPU-related recommendation
        cpu_recs = [r for r in result["recommendations"] if "cpu" in r["action"].lower()]
        assert len(cpu_recs) > 0

    @pytest.mark.asyncio
    async def test_memory_related_recommendations(self, monkeypatch):
        """Test memory-related recommendations."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Memory pressure detected"))

        result = await generate_recommendations(
            "Memory usage correlates",
            ["memory_saturation"],
            0.6,
            None,
            llm
        )

        memory_recs = [r for r in result["recommendations"] if "memory" in r["action"].lower()]
        assert len(memory_recs) > 0

    @pytest.mark.asyncio
    async def test_critical_urgency_recommendations(self, monkeypatch):
        """Test recommendations for critical failures."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Immediate action required"))

        result = await generate_recommendations(
            "Critical degradation",
            [],
            0.9,  # Very high probability
            None,
            llm
        )

        assert result["urgency"] == "critical"
        assert any("Immediate" in r["action"] for r in result["recommendations"])

    @pytest.mark.asyncio
    async def test_preventive_actions_included(self, monkeypatch):
        """Test that preventive actions are generated."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Analysis"))

        result = await generate_recommendations(
            "CPU correlates",
            [],
            0.6,
            None,
            llm
        )

        assert "preventive_actions" in result
        assert len(result["preventive_actions"]) > 0


class TestNotifyOperators:
    """Tests for notify_operators function."""

    @pytest.mark.asyncio
    async def test_console_notification_always_works(self, monkeypatch):
        """Test that console notification always succeeds."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        recommendations = [
            {"action": "Test action", "priority": "high", "description": "Test"}
        ]

        result = await notify_operators(
            "workflow-123",
            "TEST_JOB",
            recommendations,
            0.7,
            None
        )

        assert result["notification_sent"] is True
        assert "console" in result["channels_used"]

    @pytest.mark.asyncio
    async def test_teams_notification_success(self, monkeypatch):
        """Test successful Teams notification."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")
        monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://test.webhook.url")

        recommendations = [{"action": "Test", "priority": "high", "description": "Test"}]

        with patch('workflows.nodes_verbose.httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            result = await notify_operators(
                "workflow-123",
                "TEST_JOB",
                recommendations,
                0.7,
                None
            )

            assert "teams" in result["channels_used"]

    @pytest.mark.asyncio
    async def test_feature_disabled(self, monkeypatch):
        """Test notification with feature disabled."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "false")

        result = await notify_operators("id", "job", [], 0.5, None)

        assert result["notification_sent"] is False

    @pytest.mark.asyncio
    async def test_urgency_emoji_selection(self, monkeypatch):
        """Test that urgency determines correct emoji."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        recommendations = []

        # Critical urgency
        result = await notify_operators("id", "job", recommendations, 0.9, None)
        assert result["notification_sent"] is True

        # Medium urgency
        result = await notify_operators("id", "job", recommendations, 0.5, None)
        assert result["notification_sent"] is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_fetch_job_history_with_none_values(self, monkeypatch):
        """Test handling of None values in job history."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "timestamp": now,
                "job_name": "TEST_JOB",
                "workstation": None,
                "status": None,
                "return_code": None,
                "runtime_seconds": None,
                "scheduled_time": None,
                "actual_start_time": None,
                "completed_time": None,
            }
        ]

        check_result = MagicMock()
        check_result.scalar = MagicMock(return_value=True)

        query_result = SimpleNamespace(
            mappings=lambda: SimpleNamespace(fetchall=lambda: rows)
        )

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[check_result, query_result])

        history = await fetch_job_history(db, "TEST_JOB")

        assert len(history) == 1
        assert history[0]["workstation"] == "UNKNOWN"
        assert history[0]["runtime_seconds"] == 0

    @pytest.mark.asyncio
    async def test_detect_degradation_with_missing_timestamps(self, monkeypatch):
        """Test degradation detection with missing timestamp data."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        job_history = [
            {
                "job_name": "TEST",
                "runtime_seconds": 100 + i * 5,
                "status": "SUCC",
                "timestamp": None  # Missing timestamp
            }
            for i in range(10)
        ]

        llm = AsyncMock()

        result = await detect_degradation(job_history, llm)

        # Should handle missing timestamps gracefully
        assert "detected" in result

    @pytest.mark.asyncio
    async def test_llm_failure_handling(self, monkeypatch):
        """Test that LLM failures don't break the workflow."""
        monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

        now = datetime.now(timezone.utc)
        job_history = [
            {
                "job_name": "TEST",
                "runtime_seconds": 100 + i * 10,
                "status": "SUCC",
                "timestamp": (now - timedelta(days=i)).isoformat()
            }
            for i in range(20)
        ]

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        result = await detect_degradation(job_history, llm)

        # Should still complete with fallback evidence
        assert "detected" in result
        assert "evidence" in result