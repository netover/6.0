"""
Compatibility bridge for authentication helpers.

This module re‑exports selected authentication utilities from their
implementation location so that imports like ``resync.api.auth`` continue to
function even if internal modules move.  Only the symbols documented here
should be considered part of the public API.
"""

from resync.api.routes.core.auth import verify_admin_credentials

__all__ = ["verify_admin_credentials"]