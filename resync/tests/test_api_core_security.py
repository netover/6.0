from __future__ import annotations

from types import SimpleNamespace

import pytest

from resync.api.core import security


def test_decode_access_token_raises_non_jwt_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(
            environment=SimpleNamespace(value="test"),
            secret_key=SimpleNamespace(get_secret_value=lambda: "secret"),
            jwt_leeway_seconds=0,
            jwt_algorithm="HS256",
        ),
    )

    def _decode_token(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(security, "decode_token", _decode_token)

    with pytest.raises(RuntimeError, match="boom"):
        security.decode_access_token("token")
