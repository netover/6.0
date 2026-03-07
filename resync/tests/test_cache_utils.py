from __future__ import annotations

from datetime import timezone

import pytest

from resync.core import cache_utils
from resync.core.cache_utils import EnhancedCacheManager


@pytest.mark.asyncio
async def test_warm_cache_sets_last_warmup_in_utc() -> None:
    class _ValkeyStub:
        async def set(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return True

    cache = EnhancedCacheManager(valkey_client=_ValkeyStub())

    async def _fetch() -> dict[str, str]:
        return {"status": "ok"}

    await cache.warm_cache({"test:key": _fetch})

    assert cache.stats.last_warmup is not None
    assert cache.stats.last_warmup.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_warm_cache_serializes_datetime_values() -> None:
    captured: dict[str, str] = {}

    class _ValkeyStub:
        async def set(self, key, value, **kwargs):  # type: ignore[no-untyped-def]
            captured[key] = value
            return True

    cache = EnhancedCacheManager(valkey_client=_ValkeyStub())

    async def _fetch() -> dict[str, object]:
        from datetime import datetime, timezone as _timezone

        return {"generated_at": datetime(2026, 1, 1, tzinfo=_timezone.utc)}

    await cache.warm_cache({"test:dt": _fetch})

    assert "2026-01-01" in captured["test:dt"]


def test_cache_manager_lazy_valkey_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ValkeyStub:
        pass

    called = {"count": 0}

    def _fake_get_valkey_client():  # type: ignore[no-untyped-def]
        called["count"] += 1
        return _ValkeyStub()

    monkeypatch.setattr("resync.core.valkey_init.get_valkey_client", _fake_get_valkey_client)

    manager = cache_utils.EnhancedCacheManager()

    assert called["count"] == 0
    assert isinstance(manager.valkey, _ValkeyStub)
    assert called["count"] == 1
