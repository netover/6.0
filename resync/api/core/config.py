"""FastAPI core configuration compatibility layer.

Several modules under :mod:`resync.api.core` historically imported a local
``settings`` object from ``resync.api.core.config``.

The canonical settings now live in :mod:`resync.settings`.
This file re-exports the application settings to avoid import errors and
to keep backward compatibility.
"""

from __future__ import annotations

from resync.settings import get_settings

# Re-export a singleton settings instance for modules that expect it.
settings = get_settings()
