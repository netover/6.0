from __future__ import annotations

from types import SimpleNamespace

import pytest

from resync.api.routes.core import auth


@pytest.mark.asyncio
async def test_verify_admin_credentials_uses_verify_token_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"count": 0}

    async def _verify_token_async(token: str):  # type: ignore[no-untyped-def]
        called["count"] += 1
        assert token == "cookie-token"
        return {"sub": "admin"}

    monkeypatch.setattr(auth.jwt_security, "verify_token_async", _verify_token_async)
    monkeypatch.setattr(auth, "settings", SimpleNamespace(admin_username="admin"))

    request = SimpleNamespace(
        cookies={"access_token": "cookie-token"},
        client=SimpleNamespace(host="127.0.0.1"),
    )

    username = await auth.verify_admin_credentials(request, credentials=None)

    assert username == "admin"
    assert called["count"] == 1


def test_safe_log_username_sanitizes_control_chars() -> None:
    value = auth._safe_log_username("ad\r\nmin")

    assert "\r" not in value
    assert "\n" not in value
    assert value.endswith("***")
