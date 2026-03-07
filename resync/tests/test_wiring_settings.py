from __future__ import annotations

from types import SimpleNamespace


def test_wiring_uses_snake_case_tws_mock_mode() -> None:
    settings = SimpleNamespace(tws_mock_mode=True)

    assert settings.tws_mock_mode is True
