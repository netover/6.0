from __future__ import annotations

import pytest

from resync.core import jwt_utils


def test_decode_access_token_passes_algorithms_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: dict[str, object] = {}

    def _decode(token, secret_key, algorithms=None, **kwargs):  # type: ignore[no-untyped-def]
        calls["token"] = token
        calls["secret_key"] = secret_key
        calls["algorithms"] = algorithms
        return {"sub": "user-1"}

    monkeypatch.setattr(jwt_utils, "jwt", type("JwtStub", (), {"decode": staticmethod(_decode)})())
    monkeypatch.setattr(jwt_utils, "JWT_LIBRARY", "pyjwt")

    payload = jwt_utils.decode_access_token("token", "secret", "HS512")

    assert payload == {"sub": "user-1"}
    assert calls["algorithms"] == ["HS512"]


def test_create_token_requires_positive_expiration() -> None:
    with pytest.raises(ValueError, match="expires_in must be a positive int"):
        jwt_utils.create_token({"sub": "user-1"}, "secret", expires_in=None)
