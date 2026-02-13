"""
Auth module - migrated from fastapi_app/auth/

v5.8.0: Unified auth service layer.
"""

# Re-export from legacy auth module for backward compatibility
from resync.api.auth_legacy import (
    SecureAuthenticator,
    authenticate_admin,
    authenticator,
    create_access_token,
    verify_admin_credentials,
)

from .models import User, UserCreate, UserRole, UserUpdate
from .repository import UserRepository
from .service import AuthService

__all__ = [
    "User",
    "UserCreate",
    "UserRole",
    "UserUpdate",
    "UserRepository",
    "AuthService",
    # Legacy auth exports
    "SecureAuthenticator",
    "authenticator",
    "verify_admin_credentials",
    "create_access_token",
    "authenticate_admin",
]
