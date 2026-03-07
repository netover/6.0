from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.params import Depends

from resync.api.routes.monitoring import ai_monitoring
from resync.api.routes.core.auth import verify_admin_credentials


def test_ai_monitoring_router_requires_admin_dependency() -> None:
    dependency_calls = [
        dependency.dependency
        for dependency in ai_monitoring.router.dependencies
        if isinstance(dependency, Depends)
    ]

    assert verify_admin_credentials in dependency_calls


@pytest.mark.asyncio
async def test_get_ai_config_uses_async_file_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _load_config() -> dict[str, object]:
        calls.append("load")
        return {"monitoring": {"enabled": True}}

    async def _to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("to_thread")
        return func(*args, **kwargs)

    monkeypatch.setattr(ai_monitoring, "_load_config", _load_config)
    monkeypatch.setattr(ai_monitoring.asyncio, "to_thread", _to_thread)

    result = await ai_monitoring.get_ai_config()

    assert result == {"monitoring": {"enabled": True}}
    assert calls == ["to_thread", "load"]
