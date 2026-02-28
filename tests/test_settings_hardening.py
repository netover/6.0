import os
import warnings

import pytest
from pydantic import SecretStr

from resync import settings as settings_module
from resync.settings_types import Environment


def _reset_env(prefix: str = "APP_") -> None:
    for k in list(os.environ.keys()):
        if k.startswith(prefix) or k in {"DATABASE_URL", "REDIS_URL", "SECRET_KEY", "ADMIN_PASSWORD"}:
            os.environ.pop(k, None)


def test_debug_legacy_alias_tracks_debug_flag() -> None:
    _reset_env()
    settings_module.clear_settings_cache()

    s = settings_module.Settings(APP_ENVIRONMENT=Environment.DEVELOPMENT, debug=False)
    assert s.DEBUG is False

    s2 = settings_module.Settings(APP_ENVIRONMENT=Environment.DEVELOPMENT, debug=True)
    assert s2.DEBUG is True


def test_redis_url_legacy_warns_and_returns_plaintext() -> None:
    _reset_env()
    settings_module.clear_settings_cache()

    s = settings_module.Settings(redis_url=SecretStr("redis://localhost:6379/0"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = s.REDIS_URL
        assert val.startswith("redis://")
        assert any(isinstance(x.message, DeprecationWarning) for x in w)


def test_teams_config_proxy_does_not_unwrap_secretstr() -> None:
    _reset_env()
    settings_module.clear_settings_cache()

    os.environ["APP_TEAMS_OUTGOING_WEBHOOK_ENABLED"] = "true"
    os.environ["APP_TEAMS_OUTGOING_WEBHOOK_SECURITY_TOKEN"] = "super-secret"

    # Ensure singleton picks up env vars
    settings_module.get_settings()
    config = settings_module.TEAMS_OUTGOING_WEBHOOK

    token_val = config["security_token"]
    assert isinstance(token_val, SecretStr)
    assert config.get_secret("security_token") == "super-secret"


def test_database_url_default_has_no_credentials() -> None:
    _reset_env()
    settings_module.clear_settings_cache()

    s = settings_module.Settings()
    raw = s.database_url.get_secret_value()
    assert "resync:resync@" not in raw


def test_production_rejects_localhost_database_url_and_requires_metrics_hash() -> None:
    _reset_env()
    settings_module.clear_settings_cache()

    strong_pwd = SecretStr("StrongPassw0rd!")
    strong_key = SecretStr("x" * 32)

    # localhost should be rejected in production
    with pytest.raises(ValueError):
        settings_module.Settings(
            APP_ENVIRONMENT=Environment.PRODUCTION,
            admin_password=strong_pwd,
            secret_key=strong_key,
            metrics_api_key_hash=SecretStr("abcd"),
            database_url=SecretStr("postgresql+asyncpg://localhost:5432/resync"),
        )

    # metrics hash required
    with pytest.raises(ValueError):
        settings_module.Settings(
            APP_ENVIRONMENT=Environment.PRODUCTION,
            admin_password=strong_pwd,
            secret_key=strong_key,
            database_url=SecretStr("postgresql+asyncpg://db.example.com:5432/resync"),
            metrics_api_key_hash=SecretStr(""),
        )
