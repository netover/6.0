import pytest
import asyncio
import logging
import json
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import time

# ✅ Importar do caminho correto
from resync.api.monitoring_dashboard import (
    DashboardMetricsStore,
    MetricSample,
    ws_manager,
    collect_metrics_sample,
    REDIS_CH_BROADCAST,
)

@pytest.mark.asyncio
async def test_dashboard_redis_integration():
    # 1. Mock Redis Client
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value='{"status": "ok", "api": {"requests_per_sec": 0}}')
    mock_redis.lrange = AsyncMock(return_value=[])
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    # Pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[])
    mock_redis.pipeline.return_value = mock_pipeline

    # PubSub
    mock_pubsub = MagicMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.listen = AsyncMock()
    mock_redis.pubsub.return_value = mock_pubsub

    # 2. Patching dependencies
    with patch("resync.api.monitoring_dashboard.get_redis_client", return_value=mock_redis), \
         patch("resync.api.monitoring_dashboard.decode_token") as mock_decode, \
         patch("resync.core.metrics.runtime_metrics") as mock_metrics:

        # Setup mocks
        mock_decode.return_value = {"sub": "admin", "roles": ["admin"]}
        mock_metrics.get_snapshot.return_value = {
            "agent": {"initializations": 100, "active_count": 5, "creation_failures": 2},
            "slo": {"api_error_rate": 0.02, "availability": 0.99, "tws_connection_success_rate": 1.0}
        }

        store = DashboardMetricsStore()
        # Definir tws_connected=True para evitar alertas extras que chamam lpush/execute novamente
        sample = MetricSample(timestamp=time.time(), datetime_str="12:00:00", requests_total=100, tws_connected=True)

        # TESTE A: Persistência no Redis
        await store.add_sample(sample)
        mock_pipeline.lpush.assert_called_once_with(ANY, ANY)
        lpush_args, _ = mock_pipeline.lpush.call_args
        assert lpush_args[0]

        # Verificar payload via json.loads para ser robusto a variações de serializador
        payload = json.loads(lpush_args[1])
        assert payload["requests_total"] == 100
        mock_pipeline.execute.assert_called_once()

        # Reset mocks para TESTE B
        mock_redis.reset_mock()
        mock_pipeline.reset_mock()

        # TESTE B: Broadcast via Pub/Sub
        await collect_metrics_sample()
        mock_redis.publish.assert_called_once_with(REDIS_CH_BROADCAST, ANY)

        # TESTE C: WebSocket Relay
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        await ws_manager.connect(mock_ws)
        await ws_manager.broadcast('{"status": "sync_test"}')
        mock_ws.send_text.assert_called_once_with('{"status": "sync_test"}')
        await ws_manager.disconnect(mock_ws)
