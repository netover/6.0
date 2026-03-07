from __future__ import annotations

import importlib

import pytest

from resync.core import token_revocation


@pytest.mark.asyncio
async def test_revoke_jti_ignores_blank_jti(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ValkeyStub:
        called = False

        async def set(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.called = True

    stub = _ValkeyStub()
    monkeypatch.setattr(token_revocation, "_enabled", lambda: True)
    monkeypatch.setattr(
        "resync.core.valkey_init.get_valkey_client",
        lambda: stub,
        raising=False,
    )

    await token_revocation.revoke_jti("   ")

    assert stub.called is False


def test_token_revocation_prefix_is_namespaced_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "staging")
    monkeypatch.delenv("TOKEN_REVOCATION_PREFIX", raising=False)

    module = importlib.reload(token_revocation)

    assert module._PREFIX == "revoked:staging:jti:"
