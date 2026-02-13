"""
Resync Admin API v1

API endpoints for administrative functions including API key management.
"""

from .admin_api_keys import router as admin_api_keys_router

__all__ = ["admin_api_keys_router"]
