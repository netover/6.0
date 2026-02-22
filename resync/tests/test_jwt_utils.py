from __future__ import annotations

from resync.core import jwt_utils


def test_create_and_decode_token_roundtrip() -> None:
    payload = {"sub": "user-123", "role": "admin"}
    token = jwt_utils.create_token(payload, secret_key="test-secret", expires_in=60)

    decoded = jwt_utils.decode_token(token, secret_key="test-secret")

    assert decoded["sub"] == "user-123"
    assert decoded["role"] == "admin"
    assert "iat" in decoded
    assert "exp" in decoded


def test_verify_token_invalid_secret_returns_error_tuple() -> None:
    token = jwt_utils.create_token({"sub": "user-123"}, secret_key="right-secret")

    ok, data = jwt_utils.verify_token(token, secret_key="wrong-secret")

    assert ok is False
    assert isinstance(data, str)
    assert "invalid token" in data.lower() or "signature" in data.lower()
