from __future__ import annotations

from types import SimpleNamespace

import pytest

from resync.core.cache import semantic_cache as sc


class _DummyVectorizer:
    def embed(self, text: str) -> list[float]:
        base = float(len(text))
        return [base, base + 1.0, base + 2.0]


@pytest.mark.asyncio
async def test_semantic_cache_memory_mode_user_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sc, "REDISVL_AVAILABLE", False)
    monkeypatch.setattr(sc, "ResyncVectorizer", _DummyVectorizer)
    monkeypatch.setattr(
        sc,
        "get_redis_config",
        lambda: SimpleNamespace(
            semantic_cache_threshold=0.2,
            semantic_cache_ttl=60,
            semantic_cache_max_entries=100,
        ),
    )

    cache = sc.SemanticCache()
    assert await cache.initialize() is True

    await cache.set("status tws", "ok-user-a", user_id="user-a")
    await cache.set("status tws", "ok-user-b", user_id="user-b")

    result_a = await cache.get("status tws", user_id="user-a")
    result_b = await cache.get("status tws", user_id="user-b")

    assert result_a.hit is True
    assert result_b.hit is True
    assert result_a.response == "ok-user-a"
    assert result_b.response == "ok-user-b"


@pytest.mark.asyncio
async def test_semantic_cache_memory_mode_invalidate_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sc, "REDISVL_AVAILABLE", False)
    monkeypatch.setattr(sc, "ResyncVectorizer", _DummyVectorizer)
    monkeypatch.setattr(
        sc,
        "get_redis_config",
        lambda: SimpleNamespace(
            semantic_cache_threshold=0.2,
            semantic_cache_ttl=60,
            semantic_cache_max_entries=100,
        ),
    )

    cache = sc.SemanticCache()
    await cache.initialize()

    await cache.set("cpu usage", "42%")
    await cache.set("memory usage", "71%")
    await cache.set("disk usage", "33%")

    removed = await cache.invalidate_pattern("usage")
    miss_after_invalidate = await cache.get("cpu usage")

    assert removed == 3
    assert miss_after_invalidate.hit is False
