"""Metadata normalization helpers for ingestion.

The advanced ingestion pipeline supports metadata fields such as ``platform`` and
``environment``. In earlier iterations, the ingestion service referenced a
``normalize_metadata_value`` function that was not present in the repository,
causing runtime ``ImportError``.

This module keeps the behavior conservative:
 - Accepts a small set of known values.
 - Falls back to "all" (meaning unfiltered) when input is missing/unknown.
"""

from __future__ import annotations

_ALLOWED: dict[str, set[str]] = {
    "platform": {"all", "web", "desktop", "mobile", "ios", "android"},
    "environment": {"all", "prod", "production", "staging", "dev", "development", "test"},
}


def normalize_metadata_value(field: str, value: str | None) -> str:
    """Normalize a metadata value.

    Args:
        field: The metadata field name (e.g. "platform", "environment").
        value: The raw value.

    Returns:
        Normalized value. Unknown values are normalized to "all".
    """
    allowed = _ALLOWED.get(field, {"all"})
    if value is None:
        return "all"

    v = str(value).strip().lower()
    if not v:
        return "all"

    # Normalize common synonyms
    if field == "environment":
        if v == "production":
            v = "prod"
        elif v == "development":
            v = "dev"

    return v if v in allowed else "all"
