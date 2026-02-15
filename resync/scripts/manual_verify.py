import sys
from unittest.mock import MagicMock, AsyncMock

# Mock all dependencies
for mod in ['fastapi', 'fastapi.responses', 'resync.core.redis_init', 'resync.api.security', 'resync.core.metrics', 'structlog']:
    sys.modules[mod] = MagicMock()

# Re-mock specifically what's needed for imports
import fastapi
fastapi.APIRouter = MagicMock
fastapi.WebSocket = MagicMock
fastapi.WebSocketDisconnect = Exception
fastapi.status = MagicMock()

import fastapi.responses
fastapi.responses.JSONResponse = MagicMock

# Import the code to test
from resync.api.monitoring_dashboard import (
    DashboardMetricsStore,
    MetricSample,
    ws_manager,
    collect_metrics_sample,
    REDIS_CH_BROADCAST,
)
from workflows.nodes_verbose import (
    fetch_job_execution_history,
    fetch_workstation_metrics_history,
)

import asyncio
import json
from unittest.mock import patch, ANY
import time
from datetime import datetime, timezone
from types import SimpleNamespace

async def test_dashboard_redis_integration():
    print("Running test_dashboard_redis_integration...")
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
         patch("resync.api.monitoring_dashboard.runtime_metrics") as mock_metrics:

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
    print("test_dashboard_redis_integration passed!")

async def test_workflows_history():
    print("Running test_workflows_history...")
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": now,
            "job_name": "JOB_A",
            "workstation": None,
            "status": None,
            "return_code": None,
            "runtime_seconds": None,
            "scheduled_time": now,
            "actual_start_time": now,
            "completed_time": now,
        }
    ]

    result = SimpleNamespace(mappings=lambda: SimpleNamespace(fetchall=lambda: rows))
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    # Mock environment variable for the function
    import os
    os.environ["WORKFLOW_MODE"] = "verbose" # Or whatever is needed

    history = await fetch_job_execution_history(db=db)

    assert len(history) == 1
    assert history[0]["timestamp"] == now.isoformat()
    assert history[0]["workstation"] == "UNKNOWN"

    # Test workstation metrics
    rows_metrics = [
        {
            "timestamp": now,
            "workstation": "WS01",
            "cpu_percent": 10.5,
            "memory_percent": 45.0,
            "disk_percent": 70.2,
            "load_avg_1min": 0.8,
            "cpu_count": 8,
            "total_memory_gb": 32,
            "total_disk_gb": 512,
        }
    ]
    result_metrics = SimpleNamespace(mappings=lambda: SimpleNamespace(fetchall=lambda: rows_metrics))
    db.execute.return_value = result_metrics

    history_metrics = await fetch_workstation_metrics_history(db=db)
    assert history_metrics[0]["total_memory_gb"] == 32.0
    print("test_workflows_history passed!")

if __name__ == "__main__":
    asyncio.run(test_dashboard_redis_integration())
    asyncio.run(test_workflows_history())
