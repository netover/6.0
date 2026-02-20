import hashlib
import uuid


def normalize_trace_id(value: str) -> str:
    """
    Normalize a correlation ID into a valid W3C trace ID (32-char hex).

    Strategies:
    1. If already 32-char hex, return as is.
    2. If UUID-like, strip hyphens.
    3. Otherwise, return sha256(value)[:32].
    """
    if not value:
        return uuid.uuid4().hex

    # Strip hyphens if it looks like a UUID
    cleaned = value.replace("-", "")

    # If it's a 32-char hex string, use it
    if len(cleaned) == 32 and all(c in "0123456789abcdefABCDEF" for c in cleaned):
        return cleaned.lower()

    # Otherwise, hash it to get a stable 32-char hex
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def hash_user_id(user_id: str) -> str:
    """
    Hash user_id to avoid PII leakage in traces.
    Uses SHA-256.
    """
    if not user_id:
        return "anonymous"
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()
