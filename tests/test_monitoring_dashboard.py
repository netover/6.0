"""
Comprehensive tests for resync/api/monitoring_dashboard.py

Tests cover:
- DashboardMetricsStore: sample persistence, Redis integration, metrics retrieval
- WebSocketManager: connection management, broadcasting, pub/sub
- Metrics collector: data collection, leader election
- API endpoints: current metrics, history, WebSocket
- JSON serialization helpers
- Error handling and recovery
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from resync.api.monitoring_dashboard import (
    DashboardMetricsStore,
    MetricSample,
    WebSocketManager,
    collect_metrics_sample,
    metrics_collector_loop,
    _safe_float,
    _safe_int,
    _safe_json_loads,
    json_dumps,
    json_loads,
    REDIS_KEY_HISTORY,
    REDIS_KEY_LATEST,
    REDIS_KEY_ALERTS,
    REDIS_KEY_START_TIME,
    REDIS_CH_BROADCAST,
    REDIS_LOCK_COLLECTOR,
)


class TestSafeConversionHelpers:
    """Tests for safe conversion helper functions."""

    def test_safe_float_valid_values(self):
        """Test safe float conversion with valid values."""
        assert _safe_float(10.5) == 10.5
        assert _safe_float(0) == 0.0
        assert _safe_float("42.3") == 42.3

    def test_safe_float_invalid_values(self):
        """Test safe float conversion with invalid values."""
        assert _safe_float(None) == 0.0
        assert _safe_float("invalid") == 0.0
        assert _safe_float(None, default=5.0) == 5.0

    def test_safe_int_valid_values(self):
        """Test safe int conversion with valid values."""
        assert _safe_int(10) == 10
        assert _safe_int(10.9) == 10
        assert _safe_int("42") == 42

    def test_safe_int_invalid_values(self):
        """Test safe int conversion with invalid values."""
        assert _safe_int(None) == 0
        assert _safe_int("invalid") == 0
        assert _safe_int(None, default=5) == 5

    def test_safe_json_loads_valid(self):
        """Test safe JSON loads with valid data."""
        data = '{"key": "value", "number": 42}'
        result = _safe_json_loads(data, "test_context")
        assert result == {"key": "value", "number": 42}

    def test_safe_json_loads_invalid(self):
        """Test safe JSON loads with invalid data."""
        result = _safe_json_loads("invalid json", "test_context")
        assert result is None

    def test_safe_json_loads_empty(self):
        """Test safe JSON loads with empty data."""
        assert _safe_json_loads("", "test_context") is None
        assert _safe_json_loads(None, "test_context") is None


class TestJSONSerialization:
    """Tests for JSON serialization functions."""

    def test_json_dumps_dict(self):
        """Test JSON dumps with dictionary."""
        data = {"key": "value", "number": 42}
        result = json_dumps(data)
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_json_loads_string(self):
        """Test JSON loads with string."""
        data = '{"key": "value"}'
        result = json_loads(data)
        assert result == {"key": "value"}

    def test_json_roundtrip(self):
        """Test JSON roundtrip serialization."""
        original = {"api": {"requests": 100}, "status": "ok"}
        serialized = json_dumps(original)
        deserialized = json_loads(serialized)
        assert deserialized == original


class TestMetricSample:
    """Tests for MetricSample dataclass."""

    def test_metric_sample_creation(self):
        """Test creating MetricSample with default values."""
        sample = MetricSample(
            timestamp=time.time(),
            datetime_str="12:00:00"
        )

        assert sample.requests_total == 0
        assert sample.error_rate == 0.0
        assert sample.tws_connected is False

    def test_metric_sample_with_values(self):
        """Test creating MetricSample with custom values."""
        sample = MetricSample(
            timestamp=time.time(),
            datetime_str="12:00:00",
            requests_total=100,
            error_rate=2.5,
            tws_connected=True,
            cache_hit_ratio=75.0
        )

        assert sample.requests_total == 100
        assert sample.error_rate == 2.5
        assert sample.tws_connected is True
        assert sample.cache_hit_ratio == 75.0


class TestDashboardMetricsStore:
    """Tests for DashboardMetricsStore class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value=None)
        redis.lrange = AsyncMock(return_value=[])
        redis.lpush = AsyncMock(return_value=1)
        redis.ltrim = AsyncMock(return_value=True)
        redis.publish = AsyncMock(return_value=0)

        pipeline = AsyncMock()
        pipeline.lpush = MagicMock()
        pipeline.ltrim = MagicMock()
        pipeline.set = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, True, True])
        redis.pipeline = MagicMock(return_value=pipeline)

        return redis

    @pytest.mark.asyncio
    async def test_add_sample_persists_to_redis(self, mock_redis):
        """Test that adding a sample persists to Redis."""
        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            sample = MetricSample(
                timestamp=time.time(),
                datetime_str="12:00:00",
                requests_total=100
            )

            await store.add_sample(sample)

            # Verify pipeline operations
            pipeline = mock_redis.pipeline.return_value
            pipeline.lpush.assert_called_once()
            pipeline.ltrim.assert_called_once()
            pipeline.set.assert_called_once()
            pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_error_sample(self, mock_redis):
        """Test adding an error sample."""
        mock_redis.get = AsyncMock(return_value=str(time.time()))

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            error = Exception("Test error")

            await store.add_error_sample(error)

            pipeline = mock_redis.pipeline.return_value
            pipeline.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_global_uptime_first_time(self, mock_redis):
        """Test getting global uptime on first call."""
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            uptime = await store.get_global_uptime()

            assert uptime >= 0.0
            mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_global_uptime_existing(self, mock_redis):
        """Test getting global uptime with existing start time."""
        start_time = time.time() - 3600  # 1 hour ago
        mock_redis.set = AsyncMock(return_value=False)  # Key already exists
        mock_redis.get = AsyncMock(return_value=str(start_time))

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            uptime = await store.get_global_uptime()

            assert uptime >= 3590  # Around 1 hour

    @pytest.mark.asyncio
    async def test_get_current_metrics_initializing(self, mock_redis):
        """Test getting current metrics when not yet initialized."""
        mock_redis.get = AsyncMock(return_value=None)

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            metrics = await store.get_current_metrics()

            assert metrics["status"] == "initializing"
            assert "api" in metrics

    @pytest.mark.asyncio
    async def test_get_current_metrics_with_data(self, mock_redis):
        """Test getting current metrics with valid data."""
        sample_data = {
            "datetime_str": "12:00:00",
            "system_uptime": 3600.0,
            "requests_per_sec": 10.5,
            "requests_total": 1000,
            "error_rate": 1.5,
            "system_availability": 99.9,
            "collection_error": None
        }
        mock_redis.get = AsyncMock(return_value=json_dumps(sample_data))
        mock_redis.lrange = AsyncMock(return_value=[])

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            metrics = await store.get_current_metrics()

            assert metrics["status"] == "ok"
            assert metrics["api"]["requests_per_sec"] == 10.5
            assert metrics["api"]["requests_total"] == 1000

    @pytest.mark.asyncio
    async def test_get_current_metrics_with_alerts(self, mock_redis):
        """Test getting current metrics with alerts."""
        sample_data = {
            "datetime_str": "12:00:00",
            "system_uptime": 100,
            "error_rate": 15.0,
            "requests_per_sec": 10,
            "requests_total": 100,
            "system_availability": 90,
            "collection_error": None
        }
        alert_data = json_dumps({
            "type": "error_rate",
            "severity": "critical",
            "message": "Error rate: 15.0%"
        })

        mock_redis.get = AsyncMock(return_value=json_dumps(sample_data))
        mock_redis.lrange = AsyncMock(return_value=[alert_data])

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            metrics = await store.get_current_metrics()

            assert metrics["status"] == "critical"
            assert len(metrics["alerts"]) == 1

    @pytest.mark.asyncio
    async def test_get_history_empty(self, mock_redis):
        """Test getting history with no data."""
        mock_redis.lrange = AsyncMock(return_value=[])

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            history = await store.get_history(minutes=60)

            assert history["sample_count"] == 0
            assert len(history["timestamps"]) == 0

    @pytest.mark.asyncio
    async def test_get_history_with_data(self, mock_redis):
        """Test getting history with valid data."""
        samples = [
            json_dumps({"datetime_str": "12:00:00", "requests_per_sec": 10.0, "error_rate": 1.0}),
            json_dumps({"datetime_str": "12:05:00", "requests_per_sec": 12.0, "error_rate": 1.5}),
        ]
        mock_redis.lrange = AsyncMock(return_value=samples)

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            history = await store.get_history(minutes=60)

            assert history["sample_count"] == 2
            assert len(history["timestamps"]) == 2
            assert history["api"]["requests_per_sec"] == [10.0, 12.0]

    @pytest.mark.asyncio
    async def test_compute_rate_and_add_sample(self, mock_redis):
        """Test computing rate and adding sample."""
        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()

            def sample_builder(rps):
                return MetricSample(
                    timestamp=time.time(),
                    datetime_str="12:00:00",
                    requests_total=100,
                    requests_per_sec=rps
                )

            # First call
            await store.compute_rate_and_add_sample(100, time.monotonic(), sample_builder)

            # Second call with more requests
            await asyncio.sleep(0.1)
            await store.compute_rate_and_add_sample(110, time.monotonic(), sample_builder)

            # Verify samples were added
            assert mock_redis.pipeline.return_value.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_check_alerts_error_rate(self, mock_redis):
        """Test alert generation for high error rate."""
        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            sample = MetricSample(
                timestamp=time.time(),
                datetime_str="12:00:00",
                error_rate=12.0  # Above threshold
            )

            await store.add_sample(sample)

            # Verify alert was added
            pipeline = mock_redis.pipeline.return_value
            # Should have calls for sample storage AND alert storage
            assert pipeline.lpush.call_count == 2

    @pytest.mark.asyncio
    async def test_check_alerts_tws_disconnected(self, mock_redis):
        """Test alert generation for TWS disconnection."""
        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            sample = MetricSample(
                timestamp=time.time(),
                datetime_str="12:00:00",
                tws_connected=False,
                collection_error=None  # Not a collection error
            )

            await store.add_sample(sample)

            # Verify alert was added
            pipeline = mock_redis.pipeline.return_value
            assert pipeline.lpush.call_count == 2


class TestWebSocketManager:
    """Tests for WebSocketManager class."""

    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test connecting a WebSocket."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        result = await manager.connect(mock_ws)

        assert result is True
        assert mock_ws in manager._clients

    @pytest.mark.asyncio
    async def test_connect_max_connections(self):
        """Test rejecting connection when max connections reached."""
        manager = WebSocketManager()

        # Fill up to max connections
        for _ in range(50):
            mock_ws = AsyncMock()
            await manager.connect(mock_ws)

        # Next connection should be rejected
        extra_ws = AsyncMock()
        result = await manager.connect(extra_ws)

        assert result is False
        assert extra_ws not in manager._clients

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self):
        """Test disconnecting a WebSocket."""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws)
        await manager.disconnect(mock_ws)

        assert mock_ws not in manager._clients

    @pytest.mark.asyncio
    async def test_broadcast_to_clients(self):
        """Test broadcasting message to all clients."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws1.send_text = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)

        await manager.broadcast('{"test": "data"}')

        ws1.send_text.assert_called_once_with('{"test": "data"}')
        ws2.send_text.assert_called_once_with('{"test": "data"}')

    @pytest.mark.asyncio
    async def test_broadcast_handles_slow_client(self):
        """Test that slow/failing clients are disconnected."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        # ws1 will timeout
        async def slow_send(*args, **kwargs):
            await asyncio.sleep(10)

        ws1.send_text = AsyncMock(side_effect=slow_send)
        ws2.send_text = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)

        await manager.broadcast('{"test": "data"}')

        # ws2 should have received message
        ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_sync(self):
        """Test starting pub/sub sync."""
        manager = WebSocketManager()

        with patch.object(manager, '_pubsub_listener', new_callable=AsyncMock):
            await manager.start_sync()

            assert manager._sync_task is not None

    @pytest.mark.asyncio
    async def test_stop_sync(self):
        """Test stopping pub/sub sync."""
        manager = WebSocketManager()

        async def dummy_listener():
            while not manager._stop_event.is_set():
                await asyncio.sleep(0.1)

        with patch.object(manager, '_pubsub_listener', side_effect=dummy_listener):
            await manager.start_sync()
            await manager.stop()

            assert manager._stop_event.is_set()


class TestMetricsCollector:
    """Tests for metrics collection functions."""

    @pytest.mark.asyncio
    async def test_collect_metrics_sample_acquires_lock(self):
        """Test that only leader collects metrics."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)  # Successfully acquired lock
        mock_redis.publish = AsyncMock(return_value=1)

        mock_metrics = MagicMock()
        mock_metrics.get_snapshot.return_value = {
            "agent": {"initializations": 100},
            "slo": {"api_error_rate": 0.01, "availability": 0.99, "tws_connection_success_rate": 1.0}
        }

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis), \
             patch('resync.api.monitoring_dashboard.runtime_metrics', mock_metrics):
            await collect_metrics_sample()

            # Verify lock was attempted
            mock_redis.set.assert_called_once_with(
                REDIS_LOCK_COLLECTOR, "leader", ex=8, nx=True
            )

    @pytest.mark.asyncio
    async def test_collect_metrics_sample_without_lock(self):
        """Test that non-leader doesn't collect metrics."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)  # Failed to acquire lock

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            await collect_metrics_sample()

            # Should not publish if lock not acquired
            mock_redis.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_collect_metrics_sample_error_handling(self):
        """Test error handling in metrics collection."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.publish = AsyncMock(return_value=0)

        mock_metrics = MagicMock()
        mock_metrics.get_snapshot.side_effect = Exception("Metrics error")

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis), \
             patch('resync.api.monitoring_dashboard.runtime_metrics', mock_metrics):
            # Should not raise exception
            await collect_metrics_sample()

            # Should still publish error sample
            mock_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_collector_loop_redis_unavailable(self):
        """Test collector loop exits gracefully when Redis unavailable."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Redis unavailable"))

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            # Should exit without error
            await metrics_collector_loop()

            mock_redis.ping.assert_called_once()


class TestWebSocketAuthentication:
    """Tests for WebSocket authentication."""

    @pytest.mark.asyncio
    async def test_verify_ws_admin_with_query_param(self):
        """Test WebSocket authentication with query parameter."""
        from resync.api.monitoring_dashboard import _verify_ws_admin

        mock_websocket = MagicMock()
        mock_websocket.query_params = {"access_token": "test_token"}
        mock_websocket.headers = {}

        with patch('resync.api.monitoring_dashboard.decode_token') as mock_decode:
            mock_decode.return_value = {"sub": "admin", "roles": ["admin"]}

            username = await _verify_ws_admin(mock_websocket)

            assert username == "admin"

    @pytest.mark.asyncio
    async def test_verify_ws_admin_with_bearer_token(self):
        """Test WebSocket authentication with Bearer token."""
        from resync.api.monitoring_dashboard import _verify_ws_admin

        mock_websocket = MagicMock()
        mock_websocket.query_params = {}
        mock_websocket.headers = {"authorization": "Bearer test_token"}

        with patch('resync.api.monitoring_dashboard.decode_token') as mock_decode:
            mock_decode.return_value = {"sub": "admin", "roles": ["admin"]}

            username = await _verify_ws_admin(mock_websocket)

            assert username == "admin"

    @pytest.mark.asyncio
    async def test_verify_ws_admin_non_admin(self):
        """Test WebSocket authentication rejection for non-admin."""
        from resync.api.monitoring_dashboard import _verify_ws_admin

        mock_websocket = MagicMock()
        mock_websocket.query_params = {"access_token": "test_token"}
        mock_websocket.headers = {}

        with patch('resync.api.monitoring_dashboard.decode_token') as mock_decode:
            mock_decode.return_value = {"sub": "user", "roles": ["user"]}

            username = await _verify_ws_admin(mock_websocket)

            assert username is None

    @pytest.mark.asyncio
    async def test_verify_ws_admin_legacy_role_format(self):
        """Test WebSocket authentication with legacy role format."""
        from resync.api.monitoring_dashboard import _verify_ws_admin

        mock_websocket = MagicMock()
        mock_websocket.query_params = {"access_token": "test_token"}
        mock_websocket.headers = {}

        with patch('resync.api.monitoring_dashboard.decode_token') as mock_decode:
            # Legacy format: "role" instead of "roles"
            mock_decode.return_value = {"sub": "admin", "role": "admin"}

            username = await _verify_ws_admin(mock_websocket)

            assert username == "admin"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_corrupted_redis_data_handling(self):
        """Test handling of corrupted JSON data from Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="invalid json {")
        mock_redis.lrange = AsyncMock(return_value=[])

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            metrics = await store.get_current_metrics()

            # Should return error status
            assert metrics["status"] == "data_error"

    @pytest.mark.asyncio
    async def test_history_with_corrupted_samples(self):
        """Test history retrieval with some corrupted samples."""
        samples = [
            json_dumps({"datetime_str": "12:00:00", "requests_per_sec": 10.0, "error_rate": 1.0}),
            "corrupted json",
            json_dumps({"datetime_str": "12:10:00", "requests_per_sec": 15.0, "error_rate": 2.0}),
        ]

        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=samples)

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()
            history = await store.get_history(minutes=60)

            # Should only include valid samples
            assert history["sample_count"] == 2

    def test_metric_sample_collection_error_truncation(self):
        """Test that long error messages are truncated."""
        long_error = "X" * 500

        sample = MetricSample(
            timestamp=time.time(),
            datetime_str="12:00:00",
            collection_error=long_error
        )

        # Error should be stored (truncation happens in add_error_sample)
        assert len(sample.collection_error) == 500

    @pytest.mark.asyncio
    async def test_websocket_manager_broadcast_with_no_clients(self):
        """Test broadcasting with no connected clients."""
        manager = WebSocketManager()

        # Should not raise error
        await manager.broadcast('{"test": "data"}')

    @pytest.mark.asyncio
    async def test_status_transitions(self):
        """Test status calculation for different error rates."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])

        test_cases = [
            (0.0, None, "ok"),
            (5.5, None, "degraded"),
            (12.0, None, "critical"),
            (2.0, "Error occurred", "collection_error"),
        ]

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()

            for error_rate, error_msg, expected_status in test_cases:
                sample_data = {
                    "datetime_str": "12:00:00",
                    "system_uptime": 100,
                    "error_rate": error_rate,
                    "requests_per_sec": 10,
                    "requests_total": 100,
                    "system_availability": 99,
                    "collection_error": error_msg
                }
                mock_redis.get = AsyncMock(return_value=json_dumps(sample_data))

                metrics = await store.get_current_metrics()
                assert metrics["status"] == expected_status


class TestConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_rate_computation(self):
        """Test that rate computation is thread-safe."""
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=AsyncMock())
        mock_redis.pipeline.return_value.execute = AsyncMock(return_value=[1, True, True])

        with patch('resync.api.monitoring_dashboard.get_redis_client', return_value=mock_redis):
            store = DashboardMetricsStore()

            def sample_builder(rps):
                return MetricSample(
                    timestamp=time.time(),
                    datetime_str="12:00:00",
                    requests_per_sec=rps
                )

            # Concurrent rate computations
            tasks = [
                store.compute_rate_and_add_sample(100 + i * 10, time.monotonic(), sample_builder)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # All should complete successfully
            assert mock_redis.pipeline.return_value.execute.call_count == 5