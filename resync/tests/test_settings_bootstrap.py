from __future__ import annotations

from pathlib import Path

from resync.settings import Settings
import resync.settings as settings_module


def test_settings_parse_proxy_env_aliases(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TRUSTED_HOSTS", "api.example.com,*.example.com")
    monkeypatch.setenv("PROXY_HEADERS", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8,192.168.0.0/16")

    settings = Settings()

    assert settings.trusted_hosts == ["api.example.com", "*.example.com"]
    assert settings.proxy_headers_enabled is True
    assert settings.proxy_trusted_hosts == ["10.0.0.0/8", "192.168.0.0/16"]


def test_settings_knowledge_paths_are_anchored_to_package() -> None:
    settings = Settings()
    base_dir = Path(settings_module.__file__).resolve().parent

    assert settings.knowledge_base_dirs == [base_dir / "RAG"]
    assert settings.protected_directories == [base_dir / "RAG" / "BASE"]


def test_settings_startup_field_aliases(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("STARTUP_LLM_HEALTH_TIMEOUT", "7.5")
    monkeypatch.setenv("SHUTDOWN_TASK_CANCEL_TIMEOUT", "9")

    settings = Settings()

    assert settings.graphrag_enabled is True
    assert settings.startup_llm_health_timeout == 7.5
    assert settings.shutdown_task_cancel_timeout == 9.0
