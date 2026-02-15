from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from workflows.nodes_verbose import (
    fetch_job_execution_history,
    fetch_workstation_metrics_history,
)


@pytest.mark.asyncio
async def test_fetch_job_execution_history_uses_named_mappings_and_fallbacks(monkeypatch):
    monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

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

    history = await fetch_job_execution_history(db=db)

    assert len(history) == 1
    assert history[0]["timestamp"] == now.isoformat()
    assert history[0]["job_name"] == "JOB_A"
    assert history[0]["workstation"] == "UNKNOWN"
    assert history[0]["status"] == "UNKNOWN"
    assert history[0]["return_code"] == 0
    assert history[0]["runtime_seconds"] == 0
    assert history[0]["scheduled_time"] == now.isoformat()
    assert history[0]["actual_start_time"] == now.isoformat()
    assert history[0]["completed_time"] == now.isoformat()


@pytest.mark.asyncio
async def test_fetch_workstation_metrics_history_uses_named_mappings(monkeypatch):
    monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

    now = datetime(2026, 1, 2, 13, 0, tzinfo=timezone.utc)
    rows = [
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

    result = SimpleNamespace(mappings=lambda: SimpleNamespace(fetchall=lambda: rows))
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    history = await fetch_workstation_metrics_history(db=db)

    assert len(history) == 1
    assert history[0] == {
        "timestamp": now.isoformat(),
        "workstation": "WS01",
        "cpu_percent": 10.5,
        "memory_percent": 45.0,
        "disk_percent": 70.2,
        "load_avg_1min": 0.8,
        "cpu_count": 8,
        "total_memory_gb": 32.0,
        "total_disk_gb": 512.0,
    }


@pytest.mark.asyncio
async def test_fetch_workstation_metrics_history_handles_none_fields(monkeypatch):
    monkeypatch.setenv("ENABLE_PREDICTIVE_WORKFLOWS", "true")

    now = datetime(2026, 1, 3, 14, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": now,
            "workstation": "WS02",
            "cpu_percent": None,
            "memory_percent": None,
            "disk_percent": None,
            "load_avg_1min": None,
            "cpu_count": None,
            "total_memory_gb": None,
            "total_disk_gb": None,
        }
    ]

    result = SimpleNamespace(mappings=lambda: SimpleNamespace(fetchall=lambda: rows))
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    history = await fetch_workstation_metrics_history(db=db)

    assert len(history) == 1
    assert history[0] == {
        "timestamp": now.isoformat(),
        "workstation": "WS02",
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "disk_percent": 0.0,
        "load_avg_1min": 0.0,
        "cpu_count": 0,
        "total_memory_gb": 0.0,
        "total_disk_gb": 0.0,
    }
