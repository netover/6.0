"""
Authentication Database Models

User, Role, and Audit models for authentication system.
Migrated from fastapi_app/db/models.py for consolidation.

Part of Resync v5.4.7 - Database Consolidation
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from resync.core.database.engine import Base


class UserRole(str, Enum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"
    OPERATOR = "operator"
    VIEWER = "viewer"


class User(Base):
    """
    User model for authentication.

    Optimized for PostgreSQL with proper indexes and JSON support.
    """

    __tablename__ = "users"

    # Primary key - UUID string for better distribution
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        index=True,
    )

    # Authentication
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Profile
    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        index=True,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    # Role and permissions
    role: Mapped[str] = mapped_column(
        String(20),
        default=UserRole.USER.value,
        index=True,
    )

    permissions_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    last_login: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    # Security
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    @property
    def permissions(self) -> list[str]:
        """Get permissions as list."""
        if self.permissions_json:
            import json

            return json.loads(self.permissions_json)
        return []

    @permissions.setter
    def permissions(self, value: list[str]):
        """Set permissions from list."""
        import json

        self.permissions_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "role": self.role,
            "permissions": self.permissions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def get_permissions(self) -> list[str]:
        """Get user permissions based on role."""
        role_permissions = {
            UserRole.ADMIN.value: ["read", "write", "delete", "admin", "manage_users"],
            UserRole.OPERATOR.value: ["read", "write", "delete"],
            UserRole.USER.value: ["read", "write"],
            UserRole.VIEWER.value: ["read"],
        }

        base_perms = role_permissions.get(self.role, [])
        custom_perms = self.permissions or []

        return list(set(base_perms + custom_perms))

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class AuditLog(Base):
    """
    Audit log for tracking user actions.

    Stores all important actions for compliance and debugging.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Who
    user_id: Mapped[str | None] = mapped_column(
        String(36),
        index=True,
        nullable=True,
    )
    username: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # What
    action: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )
    resource_type: Mapped[str | None] = mapped_column(
        String(100),
        index=True,
        nullable=True,
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Details (JSON as text for compatibility)
    details_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # When
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    # Context
    ip_address: Mapped[str | None] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    @property
    def details(self) -> dict:
        """Get details as dict."""
        if self.details_json:
            import json

            return json.loads(self.details_json)
        return {}

    @details.setter
    def details(self, value: dict):
        """Set details from dict."""
        import json

        self.details_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "ip_address": self.ip_address,
        }


__all__ = ["User", "UserRole", "AuditLog"]
