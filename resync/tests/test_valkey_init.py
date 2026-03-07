from __future__ import annotations

import asyncio

import pytest

from resync.core import valkey_init


def test_get_valkey_client_without_lazy_init_raises_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESYNC_VALKEY_LAZY_INIT", raising=False)
    monkeypatch.setattr(valkey_init, "_VALKEY_CLIENT", None)
    monkeypatch.setattr(valkey_init, "valkey", object())

    with pytest.raises(RuntimeError, match="Valkey client not initialized"):
        valkey_init.get_valkey_client()


@pytest.mark.asyncio
async def test_valkey_initializer_close_clears_global_state() -> None:
    class _Pool:
        async def disconnect(self) -> None:
            return None

    class _Client:
        def __init__(self) -> None:
            self.connection_pool = _Pool()

        async def close(self) -> None:
            return None

    initializer = valkey_init.ValkeyInitializer()
    client = _Client()
    initializer._client = client  # type: ignore[assignment]
    initializer._initialized = True
    valkey_init._VALKEY_CLIENT = client  # type: ignore[assignment]
    valkey_init._IDEMPOTENCY_MANAGER = object()  # type: ignore[assignment]

    await initializer.close()

    assert initializer._client is None
    assert initializer._health_task is None
    assert valkey_init._VALKEY_CLIENT is None
    assert valkey_init._IDEMPOTENCY_MANAGER is None


@pytest.mark.asyncio
async def test_close_valkey_initializer_resets_global_singleton() -> None:
    initializer = valkey_init.ValkeyInitializer()
    valkey_init._valkey_initializer = initializer

    await valkey_init.close_valkey_initializer()

    assert valkey_init._valkey_initializer is None


@pytest.mark.asyncio
async def test_initialize_skip_health_task_does_not_spawn_new_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pool:
        max_connections = 5

        async def disconnect(self) -> None:
            return None

    class _Client:
        def __init__(self) -> None:
            self.connection_pool = _Pool()

        async def set(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return True

        async def ping(self) -> bool:
            return True

        async def get(self, _key: str) -> str:
            return "ok"

        async def delete(self, _key: str) -> int:
            return 1

        async def eval(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return "1"

        async def close(self) -> None:
            return None

    initializer = valkey_init.ValkeyInitializer()
    monkeypatch.setattr(valkey_init, "valkey", object())
    monkeypatch.setattr(initializer, "_create_client_with_pool", lambda _url=None: _Client())
    monkeypatch.setattr(initializer, "_initialize_idempotency", lambda _client: None)

    tracked: list[asyncio.Task[object]] = []

    def _fake_create_tracked_task(coro, name=None, cancel_on_shutdown=True):  # type: ignore[no-untyped-def]
        task = asyncio.create_task(coro, name=name)
        tracked.append(task)
        return task

    monkeypatch.setattr(valkey_init, "create_tracked_task", _fake_create_tracked_task)

    client = await initializer.initialize(_skip_health_task=True)

    assert client is not None
    assert initializer._health_task is None
    assert tracked == []
