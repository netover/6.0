from __future__ import annotations

import pytest

from resync.api.routes.admin import semantic_cache as admin_semantic_cache


class _DummyCache:
    def __init__(self) -> None:
        self.threshold = 0.25

    def update_threshold(self, new_threshold: float) -> None:
        self.threshold = new_threshold


@pytest.mark.asyncio
async def test_update_threshold_route_supports_sync_cache_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _DummyCache()

    async def _get_cache() -> _DummyCache:
        return cache

    monkeypatch.setattr(admin_semantic_cache, "get_semantic_cache", _get_cache)

    response = await admin_semantic_cache.update_threshold(
        admin_semantic_cache.ThresholdUpdateRequest(threshold=0.4)
    )

    assert response.old_threshold == 0.25
    assert response.new_threshold == 0.4
    assert cache.threshold == 0.4
