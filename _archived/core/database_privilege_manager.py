"""
Database Privilege Management System

This module implements principle of least privilege and role-based access control:
- User role management
- Database privilege separation
- Access control enforcement
- Privilege audit logging

Provides comprehensive database security through proper access controls.
"""

import logging
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """Database user roles with principle of least privilege."""

    READ_ONLY = "read_only"
    AUDITOR = "auditor"
    ANALYST = "analyst"
    ADMIN = "admin"
    SYSTEM = "system"


class DatabasePermission(str, Enum):
    """Database operation permissions."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE_TABLE = "CREATE_TABLE"
    ALTER_TABLE = "ALTER_TABLE"
    DROP_TABLE = "DROP_TABLE"
    EXECUTE = "EXECUTE"
    INDEX = "INDEX"


@dataclass
class RolePermissions:
    """Defines permissions for each user role."""

    role: UserRole
    permissions: set[DatabasePermission]
    description: str


class DatabasePrivilegeManager:
    """
    Manages database user privileges and access control.

    Implements principle of least privilege with role-based access control.
    """

    # Role-based permission definitions
    ROLE_PERMISSIONS = {
        UserRole.READ_ONLY: RolePermissions(
            role=UserRole.READ_ONLY,
            permissions={DatabasePermission.SELECT},
            description="Read-only access to audit logs and queue data",
        ),
        UserRole.AUDITOR: RolePermissions(
            role=UserRole.AUDITOR,
            permissions={DatabasePermission.SELECT},
            description="Audit and review access to compliance data",
        ),
        UserRole.ANALYST: RolePermissions(
            role=UserRole.ANALYST,
            permissions={
                DatabasePermission.SELECT,
                DatabasePermission.INSERT,
                DatabasePermission.UPDATE,
            },
            description="Analysis and reporting access to business data",
        ),
        UserRole.ADMIN: RolePermissions(
            role=UserRole.ADMIN,
            permissions={
                DatabasePermission.SELECT,
                DatabasePermission.INSERT,
                DatabasePermission.UPDATE,
                DatabasePermission.DELETE,
                DatabasePermission.CREATE_TABLE,
                DatabasePermission.ALTER_TABLE,
            },
            description="Full administrative access to database operations",
        ),
        UserRole.SYSTEM: RolePermissions(
            role=UserRole.SYSTEM,
            permissions=set(DatabasePermission),  # All permissions
            description="System-level access for maintenance and backups",
        ),
    }

    def __init__(self):
        """Initialize privilege manager."""
        self._user_roles: dict[str, UserRole] = {}
        self._session_tokens: dict[str, str] = {}
        self._active_sessions: dict[str, dict[str, Any]] = {}

    def register_user(
        self, user_id: str, role: UserRole, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Register a user with a specific role.

        Args:
            user_id: Unique user identifier
            role: User role to assign
            metadata: Additional user metadata

        Returns:
            Session token for the user
        """
        if not user_id or not role:
            raise ValueError("User ID and role are required")

        if role not in UserRole:
            raise ValueError(f"Invalid role: {role}")

        # Store user role
        self._user_roles[user_id] = role

        # Generate secure session token
        session_token = secrets.token_urlsafe(32)
        self._session_tokens[session_token] = user_id

        # Store session information
        self._active_sessions[session_token] = {
            "user_id": user_id,
            "role": role,
            "permissions": self.ROLE_PERMISSIONS[role].permissions,
            "created_at": logger.makeRecord(
                name=__name__,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            ).created,
            "metadata": metadata or {},
            "last_activity": None,
        }

        logger.info(
            "user_registered_with_role",
            user_id=user_id,
            role=role,
            session_token="***MASKED***"[:8] + "...",  # Log only prefix
        )

        return session_token

    def validate_session(self, session_token: str) -> dict[str, Any] | None:
        """
        Validate and retrieve session information.

        Args:
            session_token: Session token to validate

        Returns:
            Session information if valid, None otherwise
        """
        if not session_token or session_token not in self._session_tokens:
            logger.warning("invalid_session_token", token="***MASKED***"[:8] + "...")
            return None

        # Get session information
        session_info = self._active_sessions[session_token]
        if not session_info:
            logger.warning("session_not_found", token="***MASKED***"[:8] + "...")
            return None

        # Update last activity
        session_info["last_activity"] = logger.makeRecord(
            name=__name__, level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None
        ).created

        return session_info

    def has_permission(self, session_token: str, permission: DatabasePermission) -> bool:
        """
        Check if user session has specific permission.

        Args:
            session_token: User session token
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        session_info = self.validate_session(session_token)
        if not session_info:
            return False

        return permission in session_info["permissions"]

    def check_access(
        self, session_token: str, operation: DatabasePermission, table_name: str | None = None
    ) -> bool:
        """
        Check if user can perform specific database operation.

        Args:
            session_token: User session token
            operation: Database operation to perform
            table_name: Table being accessed (for additional restrictions)

        Returns:
            True if access is allowed, False otherwise
        """
        session_info = self.validate_session(session_token)
        if not session_info:
            logger.warning(
                "access_denied_invalid_session",
                operation=operation,
                table=table_name,
                token="***MASKED***"[:8] + "...",
            )
            return False

        # Check basic permission
        if operation not in session_info["permissions"]:
            logger.warning(
                "access_denied_insufficient_permissions",
                user_id=session_info["user_id"],
                role=session_info["role"],
                operation=operation,
                table=table_name,
            )
            return False

        # Additional table-specific restrictions could be implemented here
        # For example, admin role might have restrictions on certain tables

        logger.debug(
            "access_granted",
            user_id=session_info["user_id"],
            role=session_info["role"],
            operation=operation,
            table=table_name,
        )

        return True

    def invalidate_session(self, session_token: str) -> bool:
        """
        Invalidate a user session.

        Args:
            session_token: Session token to invalidate

        Returns:
            True if session was invalidated, False if not found
        """
        if session_token not in self._session_tokens:
            return False

        user_id = self._session_tokens[session_token]

        # Remove from active sessions
        del self._session_tokens[session_token]
        if session_token in self._active_sessions:
            del self._active_sessions[session_token]

        logger.info("session_invalidated", user_id=user_id, token="***MASKED***"[:8] + "...")

        return True

    def get_user_role(self, session_token: str) -> UserRole | None:
        """
        Get user role from session token.

        Args:
            session_token: Session token

        Returns:
            User role if valid session, None otherwise
        """
        session_info = self.validate_session(session_token)
        return session_info["role"] if session_info else None

    def get_all_active_sessions(self) -> dict[str, dict[str, Any]]:
        """
        Get all active sessions for monitoring.

        Returns:
            Dictionary of all active sessions
        """
        return self._active_sessions.copy()

    def get_user_permissions(self, session_token: str) -> set[DatabasePermission] | None:
        """
        Get user permissions from session token.

        Args:
            session_token: Session token

        Returns:
            Set of permissions if valid session, None otherwise
        """
        session_info = self.validate_session(session_token)
        return session_info["permissions"] if session_info else None

    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired sessions.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of sessions cleaned up
        """

        current_time = logger.makeRecord(
            name=__name__, level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None
        ).created

        expired_tokens = []

        for session_token, session_info in self._active_sessions.items():
            session_age = (current_time - session_info["created_at"]) / 3600  # Convert to hours

            if session_age > max_age_hours:
                expired_tokens.append(session_token)
                session_info["user_id"]

                # Remove from active sessions and tokens
                if session_token in self._active_sessions:
                    del self._active_sessions[session_token]
                if session_token in self._session_tokens:
                    del self._session_tokens[session_token]

        if expired_tokens:
            logger.info(
                "expired_sessions_cleaned", count=len(expired_tokens), max_age_hours=max_age_hours
            )

        return len(expired_tokens)


class DatabaseAccessController:
    """
    Controls database access based on user permissions.

    Integrates with database operations to enforce access controls.
    """

    def __init__(self, privilege_manager: DatabasePrivilegeManager):
        """
        Initialize access controller.

        Args:
            privilege_manager: Instance of DatabasePrivilegeManager
        """
        self.privilege_manager = privilege_manager
        self._access_attempts = 0
        self._access_denials = 0

    def enforce_access_control(
        self, session_token: str, operation: DatabasePermission, table_name: str | None = None
    ) -> None:
        """
        Enforce access control before database operation.

        Args:
            session_token: User session token
            operation: Database operation to perform
            table_name: Table being accessed

        Raises:
            PermissionError: If access is denied
        """
        self._access_attempts += 1

        if not self.privilege_manager.check_access(session_token, operation, table_name):
            self._access_denials += 1

            session_info = self.privilege_manager.validate_session(session_token)
            user_id = session_info["user_id"] if session_info else "unknown"
            user_role = session_info["role"] if session_info else "unknown"

            logger.warning(
                "database_access_denied",
                user_id=user_id,
                role=user_role,
                operation=operation,
                table=table_name,
                session_token="***MASKED***"[:8] + "...",
            )

            raise PermissionError(f"Access denied: {operation} on {table_name or 'database'}")

        logger.debug("database_access_granted", operation=operation, table=table_name)

    def log_operation_result(
        self,
        session_token: str,
        operation: DatabasePermission,
        table_name: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """
        Log the result of a database operation.

        Args:
            session_token: User session token
            operation: Database operation performed
            table_name: Table accessed
            success: Whether operation succeeded
            error: Error message if operation failed
        """
        session_info = self.privilege_manager.validate_session(session_token)
        if not session_info:
            return

        user_id = session_info["user_id"]
        user_role = session_info["role"]

        if success:
            logger.info(
                "database_operation_successful",
                user_id=user_id,
                role=user_role,
                operation=operation,
                table=table_name,
            )
        else:
            logger.error(
                "database_operation_failed",
                user_id=user_id,
                role=user_role,
                operation=operation,
                table=table_name,
                error=error,
            )

    def get_access_statistics(self) -> dict[str, Any]:
        """
        Get access control statistics.

        Returns:
            Dictionary of access statistics
        """
        denial_rate = (
            (self._access_denials / self._access_attempts * 100) if self._access_attempts > 0 else 0
        )

        return {
            "total_access_attempts": self._access_attempts,
            "access_denials": self._access_denials,
            "denial_rate_percent": round(denial_rate, 2),
            "active_sessions": len(self.privilege_manager.get_all_active_sessions()),
        }


class PermissionError(Exception):
    """Raised when database access is denied due to insufficient permissions."""


# Global privilege manager instance
_privilege_manager: DatabasePrivilegeManager | None = None


def get_privilege_manager() -> DatabasePrivilegeManager:
    """
    Get the global privilege manager instance.

    Returns:
        DatabasePrivilegeManager instance
    """
    global _privilege_manager
    if _privilege_manager is None:
        _privilege_manager = DatabasePrivilegeManager()
        logger.info("database_privilege_manager_initialized")

    return _privilege_manager


def create_database_access_controller() -> DatabaseAccessController:
    """
    Create a database access controller instance.

    Returns:
        DatabaseAccessController instance
    """
    return DatabaseAccessController(get_privilege_manager())


# Convenience functions for common operations
def require_permission(session_token: str, permission: DatabasePermission) -> bool:
    """
    Check if user has required permission.

    Args:
        session_token: User session token
        permission: Required permission

    Returns:
        True if user has permission
    """
    return get_privilege_manager().has_permission(session_token, permission)


def check_database_access(
    session_token: str, operation: DatabasePermission, table_name: str | None = None
) -> None:
    """
    Check and enforce database access control.

    Args:
        session_token: User session token
        operation: Database operation
        table_name: Table being accessed

    Raises:
        PermissionError: If access is denied
    """
    controller = create_database_access_controller()
    controller.enforce_access_control(session_token, operation, table_name)


def log_database_operation_with_context(
    session_token: str,
    operation: DatabasePermission,
    table_name: str,
    success: bool,
    error: str | None = None,
) -> None:
    """
    Log database operation with user context.

    Args:
        session_token: User session token
        operation: Database operation performed
        table_name: Table accessed
        success: Whether operation succeeded
        error: Error message if failed
    """
    controller = create_database_access_controller()
    controller.log_operation_result(session_token, operation, table_name, success, error)


def register_user_session(
    user_id: str, role: UserRole, metadata: dict[str, Any] | None = None
) -> str:
    """
    Register a new user session.

    Args:
        user_id: User identifier
        role: User role
        metadata: Additional metadata

    Returns:
        Session token
    """
    return get_privilege_manager().register_user(user_id, role, metadata)


def invalidate_user_session(session_token: str) -> bool:
    """
    Invalidate a user session.

    Args:
        session_token: Session token to invalidate

    Returns:
        True if session was invalidated
    """
    return get_privilege_manager().invalidate_session(session_token)
