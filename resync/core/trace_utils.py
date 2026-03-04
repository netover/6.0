from __future__ import annotations

import hashlib
import re


def normalize_trace_id(value: str) -> str:
    """Convert an arbitrary string into a W3C Trace ID (32 hex chars).

    This is used for correlating logs/requests across systems without requiring
    any external tracing provider.
    """
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    # W3C trace-id must be 16 bytes (32 hex) and not all zeros.
    trace_id = h[:32]
    if re.fullmatch(r"0{32}", trace_id):
        trace_id = h[32:64]
    return trace_id


def hash_user_id(user_id: str) -> str:
    """One-way hash for user identifiers for logging/telemetry."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()
