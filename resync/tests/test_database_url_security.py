import pytest

from resync.core.database.config import get_database_config, get_effective_database_url


@pytest.mark.parametrize(
    "env_value",
    ["prod", "production", "PROD", "Production"],
)
def test_database_url_insecure_password_blocked_in_production(monkeypatch, env_value) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", env_value)
    # Ensure any variations of postgres/postgresql are handled
    monkeypatch.setenv("APP_DATABASE_URL", "postgresql://resync:password@db:5432/resync")

    with pytest.raises(ValueError, match="insecure database password"):
        get_database_config()


def test_database_url_insecure_password_allowed_outside_production(monkeypatch) -> None:
    monkeypatch.delenv("APP_ENVIRONMENT", raising=False)
    monkeypatch.setenv("APP_ENVIRONMENT", "development")
    monkeypatch.setenv(
        "APP_DATABASE_URL", "postgresql://resync:password@localhost:5432/resync"
    )

    # Should not raise
    cfg = get_database_config()
    assert cfg.user == "resync"
    assert cfg.password == "password"  # noqa: S105 - test fixture value


def test_database_url_safe_password_allowed_in_production(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "APP_DATABASE_URL", "postgresql://resync:secure_long_password_123!@db:5432/resync"
    )

    # Should not raise
    cfg = get_database_config()
    assert cfg.password == "secure_long_password_123!"  # noqa: S105 - test fixture value


def test_database_url_no_password_allowed_in_production(monkeypatch) -> None:
    # Some environments use IAM or peer auth
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.setenv("APP_DATABASE_URL", "postgresql://resync@db:5432/resync")

    # Should not raise
    cfg = get_database_config()
    assert cfg.password == ""


def test_app_database_url_is_single_source_of_truth(monkeypatch) -> None:
    monkeypatch.setenv(
        "APP_DATABASE_URL", "postgresql://preferred:secure@primary:5432/appdb"
    )

    assert get_effective_database_url() == "postgresql://preferred:secure@primary:5432/appdb"
    cfg = get_database_config()
    assert cfg.user == "preferred"
    assert cfg.host == "primary"
    assert cfg.name == "appdb"


def test_database_url_alias_is_ignored(monkeypatch) -> None:
    monkeypatch.delenv("APP_DATABASE_URL", raising=False)
    assert get_effective_database_url() is None
    cfg = get_database_config()
    assert cfg.user == "resync"
    assert cfg.host == "localhost"
    assert cfg.name == "resync"
