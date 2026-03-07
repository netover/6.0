from __future__ import annotations

import pytest

import resync.core.metrics as metrics_module
from resync.api.routes.monitoring import metrics as metrics_routes


class _FailingMetricsStore:
    async def initialize(self) -> None:
        raise RuntimeError("database offline: internal detail")


@pytest.mark.asyncio
async def test_metrics_health_masks_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metrics_module, "get_metrics_store", lambda: _FailingMetricsStore()
    )

    result = await metrics_routes.metrics_health()

    assert result["status"] == "unhealthy"
    assert result["error"] == "metrics_health_check_failed"
