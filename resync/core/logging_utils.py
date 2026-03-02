"""Logging utilities for the Resync application."""

import logging
import re
from re import error as ReError
from typing import Any

logger = logging.getLogger(__name__)

_SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "password",
        "token",
        "secret",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "pin",
        "cvv",
        "ssn",
        "credit_card",
        "card_number",
        "tws_password",
        "llm_api_key",
        "jwt",
        "session",
        "cookie",
    }
)

_SENSITIVE_VALUE_PATTERNS: tuple[str, ...] = (
    r'(?:password|pwd|token|secret|key|api_key|apikey|auth|authorization)\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
    r"(?:authorization)[:\s]*bearer\s+([^\s]+)",
    r"(?:basic)\s+([a-zA-Z0-9+/=]+)",
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    r"\b\d{3}-?\d{2}-?\d{4}\b",
    r"ey[A-Za-z0-9-_]+\.ey[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
)

_RESERVED_LOGRECORD_ATTRS: frozenset[str] = frozenset(logging.makeLogRecord({}).__dict__.keys())

class SecretRedactor(logging.Filter):
    """
    A logging filter that redacts sensitive information from log records.

    This filter will redact fields containing sensitive data like passwords,
    API keys, tokens, etc. from log messages and structured log data.
    """

    def __init__(self, name: str = "") -> None:
        """
        Initialize the SecretRedactor filter.

        Args:
            name: Optional name for the filter
        """
        super().__init__(name)
        self.sensitive_patterns = _SENSITIVE_FIELD_NAMES
        self.sensitive_value_patterns = _SENSITIVE_VALUE_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the log record, redacting sensitive information.

        Args:
            record: The log record to filter

        Returns:
            Always True to ensure the record is not filtered out
        """
        # Redact from the message
        if isinstance(record.msg, str):
            record.msg = self._redact_sensitive_data(record.msg)
        elif isinstance(record.msg, dict):
            record.msg = self._redact_dict(record.msg)

        # If the record has an args attribute, redact from there too
        if record.args:
            record.args = self._redact_args(record.args)

        # If the record has additional structured data, redact from there
        if hasattr(record, "__dict__"):
            for key, value in vars(record).items():
                if key.startswith("_") or key in _RESERVED_LOGRECORD_ATTRS:
                    continue
                if isinstance(value, str):
                    setattr(record, key, self._redact_sensitive_data(value))
                elif isinstance(value, dict):
                    setattr(record, key, self._redact_dict(value))

        return True

    def _redact_args(self, args: Any) -> Any:
        """
        Redact sensitive data from log record args.

        Args:
            args: The arguments to redact

        Returns:
            The redacted arguments
        """
        if isinstance(args, (list, tuple)):
            redacted_args: list[Any] = []
            for arg in args:
                if isinstance(arg, str):
                    redacted_args.append(self._redact_sensitive_data(arg))
                elif isinstance(arg, dict):
                    redacted_args.append(self._redact_dict(arg))
                else:
                    redacted_args.append(arg)
            return redacted_args if isinstance(args, list) else tuple(redacted_args)
        if isinstance(args, dict):
            return self._redact_dict(args)
        return args

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively redact sensitive data from a dictionary.

        Args:
            data: The dictionary to redact

        Returns:
            The redacted dictionary
        """
        if not isinstance(data, dict):
            return data

        redacted: dict[str, Any] = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            # Check if key matches sensitive patterns
            if any(sensitive in key_lower for sensitive in self.sensitive_patterns):
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self._redact_dict(item)
                    if isinstance(item, dict)
                    else self._redact_sensitive_data(str(item))
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                redacted[key] = self._redact_sensitive_data(value)
            else:
                redacted[key] = value
        return redacted

    def _redact_sensitive_data(self, data: str) -> str:
        """
        Redact sensitive data from a string.

        Args:
            data: The string to redact

        Returns:
            The redacted string
        """
        if not isinstance(data, str):
            return str(data)

        redacted = data
        for pattern in self.sensitive_value_patterns:
            try:

                def replace_match(match: re.Match[str]) -> str:
                    full_match = match.group(0)
                    if match.groups():
                        redacted_full_match = full_match
                        spans = [
                            match.span(i)
                            for i in range(1, len(match.groups()) + 1)
                            if match.group(i) is not None
                        ]
                        for start, end in sorted(spans, reverse=True):
                            rel_start = start - match.start(0)
                            rel_end = end - match.start(0)
                            redacted_full_match = (
                                redacted_full_match[:rel_start]
                                + "***REDACTED***"
                                + redacted_full_match[rel_end:]
                            )
                        return redacted_full_match
                    return "***REDACTED***"

                redacted = re.sub(pattern, replace_match, redacted, flags=re.IGNORECASE)
            except (ReError, TypeError) as exc:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.debug(
                    "suppressed_exception: %s", exc, exc_info=True
                )  # was: pass

        return redacted
