from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"(?i)(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})")
_PHONE_RE = re.compile(r"(?x)(?P<phone>\+?\d[\d\s().-]{7,}\d)")
_TOKEN_RE = re.compile(r"(?i)\b(?P<tok>(?:sk-|rk-|pk-)[A-Z0-9_-]{12,}|[A-F0-9]{32,}|[A-Z0-9+/]{32,}={0,2})\b")

def redact_pii(text: str) -> str:
    """Redact common PII/secrets patterns from free-form text."""
    if not text:
        return text
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    return text
