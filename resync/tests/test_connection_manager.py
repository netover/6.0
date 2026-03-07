from __future__ import annotations

import pytest

from resync.core.connection_manager import ConnectionManager


class _WebSocketStub:
    def __init__(self, *, fail_text: bool = False, fail_json: bool = False) -> None:
        self.fail_text = fail_text
        self.fail_json = fail_json
        self.closed = False

    async def send_text(self, _message: str) -> None:
        if self.fail_text:
            raise RuntimeError("socket closed")

    async def send_json(self, _data: dict[str, object]) -> None:
        if self.fail_json:
            raise RuntimeError("socket closed")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_broadcast_removes_failed_connections() -> None:
    manager = ConnectionManager()
    healthy = _WebSocketStub()
    failed = _WebSocketStub(fail_text=True)
    manager.active_connections = {"ok": healthy, "bad": failed}

    await manager.broadcast("hello")

    assert "ok" in manager.active_connections
    assert "bad" not in manager.active_connections
    assert failed.closed is True


@pytest.mark.asyncio
async def test_broadcast_json_removes_failed_connections() -> None:
    manager = ConnectionManager()
    healthy = _WebSocketStub()
    failed = _WebSocketStub(fail_json=True)
    manager.active_connections = {"ok": healthy, "bad": failed}

    await manager.broadcast_json({"ok": True})

    assert "ok" in manager.active_connections
    assert "bad" not in manager.active_connections
    assert failed.closed is True
