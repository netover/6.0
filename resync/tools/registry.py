# pylint: disable=all
# mypy: no-rerun
"""
Tool Registry and Catalog.

v5.8.0: Centralized tool registration, permissions, and execution tracking.

This module provides:
- ToolCatalog: Singleton registry for all tools
- @tool decorator: Marks functions as tools with guardrails
- Permission checking and role-based access
- Execution tracing and approval workflow

Author: Resync Team
Version: 5.8.0
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from collections.abc import Callable
from typing import cast
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# PERMISSIONS AND ROLES
# =============================================================================


class ToolPermission(str, Enum):
    """Tool permission levels."""

    READ_ONLY = "read_only"  # Can read data, no state changes
    WRITE = "write"  # Can modify state
    EXECUTE = "execute"  # Can execute actions on external systems
    ADMIN = "admin"  # Administrative operations


class UserRole(str, Enum):
    """User roles for permission checking."""

    VIEWER = "viewer"  # Read-only access
    OPERATOR = "operator"  # Can execute approved actions
    ADMIN = "admin"  # Full access
    SYSTEM = "system"  # Internal system calls


# Role-based tool permissions allowlist
ROLE_PERMISSIONS: dict[UserRole, set[ToolPermission]] = {
    UserRole.VIEWER: {ToolPermission.READ_ONLY},
    UserRole.OPERATOR: {ToolPermission.READ_ONLY, ToolPermission.WRITE},
    UserRole.ADMIN: {
        ToolPermission.READ_ONLY,
        ToolPermission.WRITE,
        ToolPermission.EXECUTE,
        ToolPermission.ADMIN,
    },
    UserRole.SYSTEM: {
        ToolPermission.READ_ONLY,
        ToolPermission.WRITE,
        ToolPermission.EXECUTE,
        ToolPermission.ADMIN,
    },
}


# =============================================================================
# TOOL RUN STATUS (Observable for reactive UI)
# =============================================================================


class ToolRunStatus(str, Enum):
    """Observable status for tool execution."""

    QUEUED = "queued"
    BLOCKED_ON_USER = "blocked_on_user"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk classification for approval workflow."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolRun:
    """Observable tool run state for reactive UI."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    status: ToolRunStatus = ToolRunStatus.QUEUED
    progress: float = 0.0
    progress_message: str = ""
    error: str | None = None
    result: Any = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    approval_id: str | None = None
    can_cancel: bool = True
    can_undo: bool = False


@dataclass
class ToolExecutionTrace:
    """Execution trace for audit and debugging."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    input_params: dict = field(default_factory=dict)
    result: Any = None
    success: bool = False
    error: str | None = None
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str | None = None
    user_role: UserRole | None = None
    session_id: str | None = None
    permission_required: ToolPermission = ToolPermission.READ_ONLY
    approval_required: bool = False
    approval_id: str | None = None
    undo_data: Any | None = None
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class ToolDefinition:
    """Definition of a tool in the catalog."""

    name: str
    description: str
    function: Callable[..., Any]
    permission: ToolPermission = ToolPermission.READ_ONLY
    requires_approval: bool = False
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    rate_limit: int | None = None
    timeout_seconds: int = 30
    tags: list[str] = field(default_factory=list)
    supports_undo: bool = False


# =============================================================================
# TOOL CATALOG (Singleton Registry)
# =============================================================================


class ToolCatalog:
    _tools: dict[str, ToolDefinition]
    _execution_history: list[ToolExecutionTrace]
    _pending_approvals: dict[str, ToolExecutionTrace]
    _active_runs: dict[str, ToolRun]
    _undo_registry: dict[str, Any]
    _state_lock: threading.Lock
    """
    Central catalog for all available tools.

    Provides:
    - Tool registration
    - Permission checking
    - Usage tracking
    - Schema validation
    - Parallel/serial execution strategies
    """

    _instance: ClassVar[ToolCatalog | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> ToolCatalog:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tools = {}
                    cls._instance._execution_history = []
                    cls._instance._pending_approvals = {}
                    cls._instance._active_runs = {}
                    cls._instance._undo_registry = {}
                    cls._instance._state_lock = threading.Lock()
        return cls._instance

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool in the catalog."""
        with self._state_lock:
            self._tools[tool_def.name] = tool_def
        logger.debug(
            "tool_registered",
            tool_name=tool_def.name,
            permission=tool_def.permission.value,
        )

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        user_role: UserRole | None = None,
        tags: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """List available tools, optionally filtered by role and tags."""
        tools = list(self._tools.values())

        if user_role:
            allowed_permissions = ROLE_PERMISSIONS.get(user_role, set())
            tools = [t for t in tools if t.permission in allowed_permissions]

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        return tools

    def get_read_only_tools(self) -> list[ToolDefinition]:
        """Get only read-only tools (safe for parallel execution)."""
        return [
            t for t in self._tools.values() if t.permission == ToolPermission.READ_ONLY
        ]

    def can_execute(self, tool_name: str, user_role: UserRole) -> tuple[bool, str]:
        """Check if a user role can execute a tool."""
        tool = self.get(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"

        allowed_permissions = ROLE_PERMISSIONS.get(user_role, set())
        if tool.permission not in allowed_permissions:
            return (
                False,
                f"Role '{user_role.value}' cannot execute '{tool_name}' "
                f"(requires {tool.permission.value})",
            )

        return True, "OK"

    def is_read_only(self, tool_name: str) -> bool:
        """Check if a tool is read-only."""
        tool = self.get(tool_name)
        return tool is not None and tool.permission == ToolPermission.READ_ONLY

    # =========================================================================
    # Active Runs Management
    # =========================================================================

    def create_run(self, tool_name: str) -> ToolRun:
        """Create a new tool run for tracking."""
        run = ToolRun(tool_name=tool_name, status=ToolRunStatus.QUEUED)
        with self._state_lock:
            self._active_runs[run.run_id] = run
        return run

    def update_run_status(
        self,
        run_id: str,
        status: ToolRunStatus,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> ToolRun | None:
        """Update a run's status."""
        with self._state_lock:
            run = self._active_runs.get(run_id)
            if not run:
                return None

            run.status = status
            if progress is not None:
                run.progress = progress
            if message is not None:
                run.progress_message = message
            if error is not None:
                run.error = error

            if status == ToolRunStatus.IN_PROGRESS and run.started_at is None:
                run.started_at = datetime.now(timezone.utc)
            if status in {
                ToolRunStatus.DONE,
                ToolRunStatus.ERROR,
                ToolRunStatus.CANCELLED,
            }:
                run.completed_at = datetime.now(timezone.utc)

            return run

    def get_run(self, run_id: str) -> ToolRun | None:
        """Get a run by ID."""
        return self._active_runs.get(run_id)

    def get_active_runs(self) -> list[ToolRun]:
        """Get all active (non-terminal) runs."""
        return [
            r
            for r in self._active_runs.values()
            if r.status
            not in {ToolRunStatus.DONE, ToolRunStatus.ERROR, ToolRunStatus.CANCELLED}
        ]

    # =========================================================================
    # Execution History
    # =========================================================================

    def record_execution(self, trace: ToolExecutionTrace) -> None:
        """Record a tool execution for audit."""
        with self._state_lock:
            self._execution_history.append(trace)
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]

    def get_execution_history(
        self, tool_name: str | None = None, limit: int = 100
    ) -> list[ToolExecutionTrace]:
        """Get execution history, optionally filtered by tool."""
        history = self._execution_history
        if tool_name:
            history = [t for t in history if t.tool_name == tool_name]
        return history[-limit:]

    # =========================================================================
    # Approval Workflow
    # =========================================================================

    def request_approval(self, trace: ToolExecutionTrace) -> str:
        """Request HITL approval for a tool execution."""
        approval_id = str(uuid.uuid4())
        trace.approval_id = approval_id
        with self._state_lock:
            self._pending_approvals[approval_id] = trace
        logger.info(
            "approval_requested",
            tool_name=trace.tool_name,
            approval_id=approval_id,
        )
        return approval_id

    def approve(self, approval_id: str) -> ToolExecutionTrace | None:
        """Approve a pending tool execution."""
        with self._state_lock:
            trace = self._pending_approvals.pop(approval_id, None)
        if trace:
            logger.info("tool_approved", approval_id=approval_id)
        return trace

    def reject(self, approval_id: str, reason: str = "") -> ToolExecutionTrace | None:
        """Reject a pending tool execution."""
        with self._state_lock:
            trace = self._pending_approvals.pop(approval_id, None)
        if trace:
            trace.error = f"Rejected: {reason}"
            logger.info("tool_rejected", approval_id=approval_id, reason=reason)
        return trace


# Global catalog instance
_catalog = ToolCatalog()


def get_tool_catalog() -> ToolCatalog:
    """Get the global tool catalog instance."""
    return _catalog


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ApprovalRequiredError(Exception):
    """Raised when a tool requires HITL approval."""

    def __init__(self, message: str, approval_id: str, trace: ToolExecutionTrace):
        super().__init__(message)
        self.approval_id = approval_id
        self.trace = trace


# =============================================================================
# TOOL DECORATOR
# =============================================================================


def tool(
    permission: ToolPermission = ToolPermission.READ_ONLY,
    requires_approval: bool = False,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    rate_limit: int | None = None,
    timeout_seconds: int = 30,
    tags: list[str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a tool with guardrails.

    Args:
        permission: Required permission level
        requires_approval: Whether HITL approval is needed
        input_schema: Pydantic model for input validation
        output_schema: Pydantic model for output validation
        rate_limit: Max calls per minute
        timeout_seconds: Execution timeout
        tags: Tags for categorization
    """

    def decorator(func: F) -> F:
        tool_func = cast(Any, func)
        tool_func._is_tool = True
        tool_func._tool_name = func.__name__
        tool_func._tool_description = func.__doc__ or ""
        tool_func._tool_permission = permission
        tool_func._tool_requires_approval = requires_approval

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _execute_tool_with_guardrails(
                func,
                args,
                kwargs,
                permission,
                requires_approval,
                input_schema,
                output_schema,
                timeout_seconds,
            )

        # Register in catalog
        tool_def = ToolDefinition(
            name=func.__name__,
            description=func.__doc__ or "",
            function=func,
            permission=permission,
            requires_approval=requires_approval,
            input_schema=input_schema,
            output_schema=output_schema,
            rate_limit=rate_limit,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )
        _catalog.register(tool_def)

        return cast(F, wrapper)

    return decorator


def _execute_tool_with_guardrails(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    permission: ToolPermission,
    requires_approval: bool,
    input_schema: type[BaseModel] | None,
    output_schema: type[BaseModel] | None,
    timeout_seconds: int,
) -> Any:
    """Execute tool with guardrails."""
    trace = ToolExecutionTrace(
        tool_name=func.__name__,
        input_params=dict(kwargs),
        permission_required=permission,
    )

    start_time = time.time()

    try:
        # Extract context
        trace.user_id = kwargs.pop("_user_id", None)
        user_role_str = kwargs.pop("_user_role", None)
        trace.session_id = kwargs.pop("_session_id", None)

        if user_role_str:
            try:
                trace.user_role = UserRole(user_role_str)
            except ValueError:
                trace.user_role = UserRole.VIEWER

        # Check permission
        if trace.user_role:
            can_exec, reason = _catalog.can_execute(func.__name__, trace.user_role)
            if not can_exec:
                trace.success = False
                trace.error = reason
                _catalog.record_execution(trace)
                raise PermissionError(reason)

        # Validate input
        if input_schema:
            try:
                validated = input_schema(**kwargs)
                kwargs = validated.model_dump()
            except ValidationError as e:
                trace.success = False
                trace.error = f"Input validation failed: {e}"
                _catalog.record_execution(trace)
                raise ValueError(trace.error) from e

        # Check approval
        if requires_approval:
            trace.approval_required = True
            approval_id = _catalog.request_approval(trace)
            raise ApprovalRequiredError(
                f"Approval required for {func.__name__}",
                approval_id=approval_id,
                trace=trace,
            )

        # Execute
        result = func(*args, **kwargs)

        # Validate output
        if output_schema and isinstance(result, dict):
            try:
                result = output_schema(**result).model_dump()
            except ValidationError as e:
                logger.warning("Output validation warning: %s", e)

        trace.success = True
        trace.result = result

        return result

    except ApprovalRequiredError:
        raise
    except Exception as e:
        trace.success = False
        trace.error = str(e)
        raise
    finally:
        trace.duration_ms = int((time.time() - start_time) * 1000)
        _catalog.record_execution(trace)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ToolPermission",
    "UserRole",
    "ToolRunStatus",
    "RiskLevel",
    "ROLE_PERMISSIONS",
    # Dataclasses
    "ToolRun",
    "ToolExecutionTrace",
    "ToolDefinition",
    # Catalog
    "ToolCatalog",
    "get_tool_catalog",
    # Decorator
    "tool",
    # Exceptions
    "ApprovalRequiredError",
]
