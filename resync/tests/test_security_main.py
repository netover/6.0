from __future__ import annotations

from resync.core.security_main import InputSanitizer, validate_input


def test_validate_input_rejects_newline_characters() -> None:
    result = validate_input("user\r\nAdmin: true")

    assert result.is_valid is False


def test_sanitize_dict_preserves_value_with_fallback_key() -> None:
    sanitized = InputSanitizer.sanitize_dict({"###": "value"})

    assert sanitized == {"_key_0": "value"}
