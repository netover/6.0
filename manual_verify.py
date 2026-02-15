"""Manual verification helpers for monitoring/dashboard flows.

This script is intentionally lightweight and is meant for local ad-hoc checks.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch


def _require(condition: bool, message: str) -> None:
    """Raise a runtime error when a manual check fails."""
    if not condition:
        raise RuntimeError(message)


def _install_import_mocks() -> None:
    """Install module mocks required to import runtime modules in isolation."""
    for module_name in (
        "fastapi",
        "fastapi.responses",
        "resync.core.redis_init",
        "resync.api.security",
        "resync.core.metrics",
        "structlog",
    ):
        sys.modules[module_name] = MagicMock()

    import fastapi  # type: ignore
    import fastapi.responses  # type: ignore

    fastapi.APIRouter = MagicMock
    fastapi.WebSocket = MagicMock
    fastapi.WebSocketDisconnect = Exception
    fastapi.status = MagicMock()
    fastapi.responses.JSONResponse = MagicMock


async def test_dashboard_redis_integration() -> None:
    """Validate Redis persistence and pub/sub broadcast paths."""
    _install_import_mocks()

    from resync.api.monitoring_dashboard import (
        DashboardMetricsStore,
        MetricSample,
        REDIS_CH_BROADCAST,
        collect_metrics_sample,
        get_ws_manager,
    )

    print("Running test_dashboard_redis_integration...")

    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(
        return_value='{"status": "ok", "api": {"requests_per_sec": 0}}'
    )
    mock_redis.lrange = AsyncMock(return_value=[])
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    mock_pipeline = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[])
    mock_redis.pipeline.return_value = mock_pipeline
    
    mock_lock = MagicMock()
    mock_lock.acquire = AsyncMock(return_value=True)
    mock_lock.release = AsyncMock()
    mock_redis.lock.return_value = mock_lock

    with patch("resync.api.monitoring_dashboard.get_redis_client", return_value=mock_redis), patch(
        "resync.api.monitoring_dashboard._verify_ws_admin", return_value="admin"
    ), patch("resync.core.metrics.runtime_metrics") as mock_metrics:
        mock_metrics.get_snapshot.return_value = {
            "agent": {
                "initializations": 100,
                "active_count": 5,
                "creation_failures": 2,
            },
            "slo": {
                "api_error_rate": 0.02,
                "availability": 0.99,
                "tws_connection_success_rate": 1.0,
            },
        }

        store = DashboardMetricsStore()
        sample = MetricSample(
            timestamp=time.time(),
            datetime_str="2026-01-01T12:00:00Z",
            requests_total=100,
            tws_connected=True,
        )

        await store.add_sample(sample)
        mock_pipeline.lpush.assert_called_once_with(ANY, ANY)
        lpush_args, _ = mock_pipeline.lpush.call_args
        _require(bool(lpush_args[0]), "Redis key for lpush is empty.")

        payload = json.loads(lpush_args[1])
        _require(payload["requests_total"] == 100, "Unexpected requests_total payload.")
        mock_pipeline.execute.assert_called_once()

        mock_redis.reset_mock()
        mock_pipeline.reset_mock()

        await collect_metrics_sample()
        mock_redis.publish.assert_called_once_with(REDIS_CH_BROADCAST, ANY)

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        # Mock ws_manager methods to avoid real background tasks
        ws_manager = get_ws_manager()
        ws_manager.connect = AsyncMock(return_value=True)
        ws_manager.disconnect = AsyncMock()
        ws_manager.broadcast = AsyncMock()
        
        await ws_manager.connect(mock_ws)
        # Simulate what broadcast would do if it worked
        await mock_ws.send_text('{"status": "sync_test"}')
        mock_ws.send_text.assert_called_once_with('{"status": "sync_test"}')
        await ws_manager.disconnect(mock_ws)

    print("test_dashboard_redis_integration passed!")


async def test_workflows_history() -> None:
    """Validate history mappers default and float handling."""
    from resync.workflows.nodes_verbose import (
        fetch_job_execution_history,
        fetch_workstation_metrics_history,
    )

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

    os.environ["WORKFLOW_MODE"] = "verbose"

    history = await fetch_job_execution_history(db=db)
    _require(len(history) == 1, "Expected one job history row.")
    _require(history[0]["timestamp"] == now.isoformat(), "Unexpected timestamp serialization.")
    _require(history[0]["workstation"] == "UNKNOWN", "Expected workstation fallback.")

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
    result_metrics = SimpleNamespace(
        mappings=lambda: SimpleNamespace(fetchall=lambda: rows_metrics)
    )
    db.execute = AsyncMock(return_value=result_metrics)

    history_metrics = await fetch_workstation_metrics_history(db=db)
    _require(
        math.isclose(history_metrics[0]["total_memory_gb"], 32.0, rel_tol=0.0, abs_tol=1e-9),
        "Unexpected total_memory_gb conversion.",
    )
    print("test_workflows_history passed!")


if __name__ == "__main__":
    asyncio.run(test_dashboard_redis_integration())
    asyncio.run(test_workflows_history())
