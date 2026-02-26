"""
resync.core.security.auth
=========================
Thin re-export shim so that routes can import ``get_current_admin``
from the canonical security package path.

The real implementation lives in ``resync.api.routes.core.auth``.
This module exists solely to avoid breaking imports in routes that
reference ``resync.core.security.auth``.
"""
from __future__ import annotations

# Re-export the canonical admin-verification dependency
from resync.api.routes.core.auth import verify_admin_credentials as get_current_admin

__all__ = ["get_current_admin"]
