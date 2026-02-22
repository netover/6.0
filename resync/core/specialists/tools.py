# pylint: skip-file
# mypy: ignore-errors
"""
TWS Specialist Tools with Guardrails.

Custom tools for each specialist agent to interact with TWS data,
logs, graphs, and documentation.

v5.4.2 Enhancements (PR-8 to PR-12):
- PR-8: Parallel tool execution (read-only tools run concurrently)
- PR-9: Observable ToolRunStatus for reactive UI
- PR-10: Sub-agent pattern with read-only restrictions
- PR-11: Undo/rollback support for stateful operations
- PR-12: Risk-based classification for approvals

v5.4.1 Enhancements (PR-1):
- Input/output schema validation
- Role-based permissions (allowlist)
- Tracing/logging per call
- Read-only vs write classification
- Tool catalog registry

Author: Resync Team
Version: 5.4.2
"""



import asyncio
import concurrent.futures
import functools
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, Field, ValidationError
from resync.core.utils.async_bridge import run_sync

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# TOOL PERMISSIONS AND CLASSIFICATION
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
# PR-9: OBSERVABLE TOOL RUN STATUS
# =============================================================================


class ToolRunStatus(str, Enum):
    """
    Observable status for tool execution.

    Enables reactive UI updates during tool execution.
    Based on patterns from Claude Code/Amp.
    """

    QUEUED = "queued"  # Tool is in queue, not started
    BLOCKED_ON_USER = "blocked_on_user"  # Waiting for HITL approval
    IN_PROGRESS = "in_progress"  # Currently executing
    DONE = "done"  # Completed successfully
    ERROR = "error"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user or timeout


@dataclass
class ToolRun:
    """
    Observable tool run state for reactive UI.

    Provides real-time progress tracking, cancellation support,
    and permission handling for collaborative safety.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    status: ToolRunStatus = ToolRunStatus.QUEUED

    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0 (100%)
    progress_message: str | None = None

    # Results
    result: Any = None
    error: str | None = None

    # HITL
    permissions_needed: list[str] | None = None
    blocked_reason: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Cancellation
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "error": self.error,
            "permissions_needed": self.permissions_needed,
            "blocked_reason": self.blocked_reason,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


# =============================================================================
# PR-12: RISK LEVEL CLASSIFICATION
# =============================================================================


class RiskLevel(str, Enum):
    """
    Risk level for tool operations.

    Used for UI styling and approval requirements.
    """

    LOW = "low"  # Green - safe, auto-approve possible
    MEDIUM = "medium"  # Yellow - requires attention
    HIGH = "high"  # Red - requires explicit approval
    CRITICAL = "critical"  # Red pulsing - multiple approvals required


# Keywords that indicate production/critical systems
CRITICAL_KEYWORDS = {"PROD", "PRD", "PRODUCTION", "CRITICAL", "MASTER", "MAIN"}
HIGH_RISK_KEYWORDS = {"STAGE", "STG", "UAT", "PREPROD", "BATCH"}


def calculate_risk_level(
    tool_name: str,
    params: dict[str, Any],
    permission: ToolPermission,
) -> RiskLevel:
    """
    Calculate risk level for a tool execution.

    Args:
        tool_name: Name of the tool
        params: Parameters being passed
        permission: Tool's permission level

    Returns:
        Calculated risk level
    """
    # Check params for critical keywords
    params_str = str(params).upper()

    if any(kw in params_str for kw in CRITICAL_KEYWORDS):
        if permission in {ToolPermission.WRITE, ToolPermission.EXECUTE}:
            return RiskLevel.CRITICAL
        return RiskLevel.HIGH

    if any(kw in params_str for kw in HIGH_RISK_KEYWORDS):
        if permission in {ToolPermission.WRITE, ToolPermission.EXECUTE}:
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    # Base risk on permission level
    if permission == ToolPermission.ADMIN:
        return RiskLevel.HIGH
    if permission == ToolPermission.EXECUTE:
        return RiskLevel.HIGH
    if permission == ToolPermission.WRITE:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


# =============================================================================
# PR-11: ENHANCED TOOL RESULT WITH UNDO SUPPORT
# =============================================================================


@dataclass
class ToolResult:
    """
    Enhanced tool result with undo/rollback support.

    Enables reverting changes if something goes wrong.
    """

    success: bool
    result: Any
    message: str

    # Undo/rollback support
    undo_fn: Callable[[], Awaitable[None]] | None = None
    files_changed: list[str] = field(default_factory=list)

    # TWS-specific
    jobs_affected: list[str] = field(default_factory=list)
    workstations_affected: list[str] = field(default_factory=list)
    original_state: dict[str, Any] | None = None

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW

    # Tracing
    trace_id: str | None = None
    duration_ms: int = 0

    async def undo(self) -> bool:
        """
        Attempt to undo the operation.

        Returns:
            True if undo was successful, False otherwise
        """
        if self.undo_fn is None:
            logger.warning("undo_not_available", trace_id=self.trace_id)
            return False

        try:
            await self.undo_fn()
            logger.info(
                "undo_successful",
                trace_id=self.trace_id,
                jobs_affected=self.jobs_affected,
            )
            return True
        except Exception as e:
            logger.error("undo_failed", trace_id=self.trace_id, error=str(e))
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "message": self.message,
            "files_changed": self.files_changed,
            "jobs_affected": self.jobs_affected,
            "workstations_affected": self.workstations_affected,
            "has_undo": self.undo_fn is not None,
            "risk_level": self.risk_level.value,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# TOOL EXECUTION TRACE (Enhanced)
# =============================================================================


@dataclass
class ToolExecutionTrace:
    """Trace of a tool execution for audit and debugging."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Input
    input_params: dict[str, Any] = field(default_factory=dict)

    # Execution
    user_id: str | None = None
    user_role: UserRole | None = None
    session_id: str | None = None

    # Output
    success: bool = False
    result: Any = None
    error: str | None = None
    duration_ms: int = 0

    # Security
    permission_required: ToolPermission = ToolPermission.READ_ONLY
    approval_required: bool = False
    approved_by: str | None = None

    # PR-12: Risk level
    risk_level: RiskLevel = RiskLevel.LOW

    # PR-11: Undo support
    can_undo: bool = False
    was_undone: bool = False

    # PR-9: Observable status
    run_status: ToolRunStatus = ToolRunStatus.DONE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "trace_id": self.trace_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat(),
            "input_params": self.input_params,
            "user_id": self.user_id,
            "user_role": self.user_role.value if self.user_role else None,
            "session_id": self.session_id,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "permission_required": self.permission_required.value,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "risk_level": self.risk_level.value,
            "can_undo": self.can_undo,
            "was_undone": self.was_undone,
            "run_status": self.run_status.value,
        }


# =============================================================================
# TOOL REGISTRY AND CATALOG (Enhanced)
# =============================================================================


@dataclass
class ToolDefinition:
    """Definition of a tool in the catalog."""

    name: str
    description: str
    function: Callable
    permission: ToolPermission = ToolPermission.READ_ONLY
    requires_approval: bool = False
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    rate_limit: int | None = None  # calls per minute
    timeout_seconds: int = 30
    tags: list[str] = field(default_factory=list)

    # PR-11: Supports undo
    supports_undo: bool = False


class ToolCatalog:
    """
    Central catalog for all available tools.

    v5.4.2 Enhancements:
    - Parallel execution support (PR-8)
    - Active runs tracking (PR-9)
    - Undo registry (PR-11)
    - Risk calculation (PR-12)

    Provides:
    - Tool registration
    - Permission checking
    - Usage tracking
    - Schema validation
    - Parallel/serial execution strategies
    """

    _instance: ToolCatalog | None = None

    def __new__(cls) -> ToolCatalog:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, ToolDefinition] = {}
            cls._instance._execution_history: list[ToolExecutionTrace] = []
            cls._instance._pending_approvals: dict[str, ToolExecutionTrace] = {}
            # PR-9: Active runs tracking
            cls._instance._active_runs: dict[str, ToolRun] = {}
            # PR-11: Undo registry (trace_id -> ToolResult with undo)
            cls._instance._undo_registry: dict[str, ToolResult] = {}
        return cls._instance

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool in the catalog."""
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
        """Get only read-only tools (for sub-agents)."""
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
                f"Role '{user_role.value}' cannot execute '{tool_name}' (requires {tool.permission.value})",
            )

        return True, "OK"

    def is_read_only(self, tool_name: str) -> bool:
        """Check if a tool is read-only (safe for parallel execution)."""
        tool = self.get(tool_name)
        return tool is not None and tool.permission == ToolPermission.READ_ONLY

    # =========================================================================
    # PR-9: Active Runs Management
    # =========================================================================

    def create_run(self, tool_name: str) -> ToolRun:
        """Create a new tool run for tracking."""
        run = ToolRun(tool_name=tool_name, status=ToolRunStatus.QUEUED)
        self._active_runs[run.run_id] = run
        logger.debug("run_created", run_id=run.run_id, tool_name=tool_name)
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
        if status in {ToolRunStatus.DONE, ToolRunStatus.ERROR, ToolRunStatus.CANCELLED}:
            run.completed_at = datetime.now(timezone.utc)

        logger.debug("run_updated", run_id=run_id, status=status.value)
        return run

    def get_active_runs(self) -> list[ToolRun]:
        """Get all active (non-completed) runs."""
        return [
            r
            for r in self._active_runs.values()
            if r.status
            in {
                ToolRunStatus.QUEUED,
                ToolRunStatus.IN_PROGRESS,
                ToolRunStatus.BLOCKED_ON_USER,
            }
        ]

    def cancel_run(self, run_id: str) -> bool:
        """Request cancellation of a run."""
        run = self._active_runs.get(run_id)
        if run and run.status in {ToolRunStatus.QUEUED, ToolRunStatus.IN_PROGRESS}:
            run.cancel_requested = True
            run.status = ToolRunStatus.CANCELLED
            run.completed_at = datetime.now(timezone.utc)
            logger.info("run_cancelled", run_id=run_id)
            return True
        return False

    def cleanup_completed_runs(self, max_age_seconds: int = 300) -> int:
        """Clean up completed runs older than max_age_seconds."""
        now = datetime.now(timezone.utc)
        to_remove = []

        for run_id, run in self._active_runs.items():
            if run.completed_at:
                age = (now - run.completed_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(run_id)

        for run_id in to_remove:
            del self._active_runs[run_id]

        return len(to_remove)

    # =========================================================================
    # PR-11: Undo Registry
    # =========================================================================

    def register_undoable(self, trace_id: str, result: ToolResult) -> None:
        """Register a result that can be undone."""
        if result.undo_fn is not None:
            self._undo_registry[trace_id] = result
            # Keep only last 100 undoable operations
            if len(self._undo_registry) > 100:
                oldest = list(self._undo_registry.keys())[0]
                del self._undo_registry[oldest]
            logger.debug("undoable_registered", trace_id=trace_id)

    async def undo_operation(self, trace_id: str) -> bool:
        """Attempt to undo an operation by trace_id."""
        result = self._undo_registry.get(trace_id)
        if not result:
            logger.warning("undo_not_found", trace_id=trace_id)
            return False

        success = await result.undo()
        if success:
            # Mark trace as undone
            for trace in self._execution_history:
                if trace.trace_id == trace_id:
                    trace.was_undone = True
                    break
            del self._undo_registry[trace_id]

        return success

    def get_undoable_operations(self) -> list[str]:
        """Get list of trace_ids that can be undone."""
        return list(self._undo_registry.keys())

    # =========================================================================
    # PR-12: Risk Assessment
    # =========================================================================

    def assess_risk(self, tool_name: str, params: dict[str, Any]) -> RiskLevel:
        """Assess risk level for a tool execution."""
        tool = self.get(tool_name)
        if not tool:
            return RiskLevel.HIGH  # Unknown tool = high risk

        return calculate_risk_level(tool_name, params, tool.permission)

    # =========================================================================
    # Existing Methods (Enhanced)
    # =========================================================================

    def record_execution(self, trace: ToolExecutionTrace) -> None:
        """Record a tool execution trace."""
        self._execution_history.append(trace)
        # Keep last 1000 traces in memory
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

    def request_approval(self, trace: ToolExecutionTrace) -> str:
        """Request approval for a tool execution."""
        trace.approval_required = True
        trace.run_status = ToolRunStatus.BLOCKED_ON_USER
        self._pending_approvals[trace.trace_id] = trace
        logger.info(
            "approval_requested",
            trace_id=trace.trace_id,
            tool_name=trace.tool_name,
            user_id=trace.user_id,
            risk_level=trace.risk_level.value,
        )
        return trace.trace_id

    def approve_execution(
        self, trace_id: str, approver_id: str
    ) -> ToolExecutionTrace | None:
        """Approve a pending execution."""
        trace = self._pending_approvals.pop(trace_id, None)
        if trace:
            trace.approved_by = approver_id
            trace.run_status = ToolRunStatus.IN_PROGRESS
            logger.info("execution_approved", trace_id=trace_id, approver=approver_id)
        return trace

    def reject_execution(self, trace_id: str, reason: str) -> ToolExecutionTrace | None:
        """Reject a pending execution."""
        trace = self._pending_approvals.pop(trace_id, None)
        if trace:
            trace.error = f"Rejected: {reason}"
            trace.success = False
            trace.run_status = ToolRunStatus.CANCELLED
            logger.info("execution_rejected", trace_id=trace_id, reason=reason)
        return trace

    def get_pending_approvals(self) -> list[ToolExecutionTrace]:
        """Get list of pending approval requests."""
        return list(self._pending_approvals.values())

    def get_execution_history(self, limit: int = 100) -> list[ToolExecutionTrace]:
        """Get recent execution history."""
        return self._execution_history[-limit:]


# Global catalog instance
_catalog = ToolCatalog()


def get_tool_catalog() -> ToolCatalog:
    """Get the global tool catalog."""
    return _catalog


# =============================================================================
# APPROVAL REQUIRED EXCEPTION
# =============================================================================


class ApprovalRequiredError(Exception):
    """Raised when a tool requires HITL approval."""

    def __init__(self, message: str, approval_id: str, trace: ToolExecutionTrace):
        super().__init__(message)
        self.approval_id = approval_id
        self.trace = trace


# =============================================================================
# TOOL DECORATOR (Agno-compatible with guardrails)
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
    Decorator to mark a function as a tool for Agno agents with guardrails.

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
        # Mark as tool
        func._is_tool = True
        func._tool_name = func.__name__
        func._tool_description = func.__doc__ or ""
        func._tool_permission = permission
        func._tool_requires_approval = requires_approval

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _execute_tool_with_guardrails_sync(
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

        return wrapper

    return decorator


def _execute_tool_with_guardrails_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    permission: ToolPermission,
    requires_approval: bool,
    input_schema: type[BaseModel] | None,
    output_schema: type[BaseModel] | None,
    timeout_seconds: int,
) -> Any:
    """Execute tool with guardrails (sync version)."""
    trace = ToolExecutionTrace(
        tool_name=func.__name__,
        input_params=dict(kwargs),
        permission_required=permission,
    )

    start_time = time.time()

    try:
        # Extract context from kwargs (but don't remove them to avoid breaking function signature)
        trace.user_id = kwargs.pop("_user_id", None)
        user_role_str = kwargs.pop("_user_role", None)
        trace.session_id = kwargs.pop("_session_id", None)

        if user_role_str:
            try:
                trace.user_role = UserRole(user_role_str)
            except ValueError:
                trace.user_role = UserRole.VIEWER

        # Check permission if role provided
        if trace.user_role:
            can_exec, reason = _catalog.can_execute(func.__name__, trace.user_role)
            if not can_exec:
                trace.success = False
                trace.error = reason
                _catalog.record_execution(trace)
                raise PermissionError(reason)

        # Validate input if schema provided
        if input_schema:
            try:
                validated = input_schema(**kwargs)
                kwargs = validated.model_dump()
            except ValidationError as e:
                trace.success = False
                trace.error = f"Input validation failed: {e}"
                _catalog.record_execution(trace)
                raise ValueError(trace.error) from e

        # Check if approval is required
        if requires_approval:
            trace.approval_required = True
            approval_id = _catalog.request_approval(trace)
            raise ApprovalRequiredError(
                f"Approval required for {func.__name__}",
                approval_id=approval_id,
                trace=trace,
            )

        # Execute function
        result = func(*args, **kwargs)

        # Validate output if schema provided
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

        logger.debug(
            "tool_executed",
            tool_name=trace.tool_name,
            success=trace.success,
            duration_ms=trace.duration_ms,
            user_id=trace.user_id,
            trace_id=trace.trace_id,
        )


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================


class JobLogInput(BaseModel):
    """Input schema for job log retrieval."""

    job_name: str = Field(..., min_length=1, max_length=100, description="Job name")
    run_date: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date (YYYY-MM-DD)"
    )
    max_lines: int = Field(100, ge=1, le=10000, description="Maximum log lines")


class JobLogOutput(BaseModel):
    """Output schema for job log retrieval."""

    job_name: str
    run_date: str
    status: str
    return_code: int | None = None
    abend_code: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: int | None = None
    workstation: str | None = None
    log_excerpt: list[str] = Field(default_factory=list)
    error_details: str | None = None


class JobHistoryInput(BaseModel):
    """Input schema for job history retrieval."""

    job_name: str = Field(..., min_length=1, max_length=100)
    days: int = Field(7, ge=1, le=365)


class WorkstationStatusInput(BaseModel):
    """Input schema for workstation status."""

    workstation_name: str | None = Field(None, max_length=100)


class ExecuteCommandInput(BaseModel):
    """Input schema for TWS command execution."""

    command: str = Field(..., min_length=1, max_length=500)
    target: str = Field(..., description="Job or workstation name")
    parameters: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# RAG AS TOOL (PR-1: Tool for agent to search knowledge base)
# =============================================================================


class RAGSearchInput(BaseModel):
    """Input schema for RAG search."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


class RAGSearchOutput(BaseModel):
    """Output schema for RAG search."""

    results: list[dict[str, Any]]
    query: str
    total_found: int
    search_time_ms: int


class RAGTool:
    """
    RAG as a tool for agents to search the knowledge base.

    Integrates with:
    - resync/RAG/microservice/core/retriever.py
    - resync/RAG/microservice/core/hybrid_retriever.py
    - resync/RAG/microservice/core/rag_reranker.py
    """

    def __init__(self):
        self._retriever = None
        self._hybrid_retriever = None
        self._reranker = None

    def _get_retriever(self):
        """Lazy initialization of retriever."""
        if not self._retriever:
            try:
                from resync.knowledge.ingestion.embedding_service import (
                    EmbeddingService,
                )
                from resync.knowledge.retrieval.retriever import RagRetriever
                from resync.knowledge.store.pgvector_store import PgVectorStore

                embedder = EmbeddingService()
                store = PgVectorStore()
                self._retriever = RagRetriever(embedder, store)
            except Exception as e:
                logger.warning("RAG retriever init failed: %s", e)
        return self._retriever

    def _get_hybrid_retriever(self):
        """Lazy initialization of hybrid retriever."""
        if not self._hybrid_retriever:
            try:
                from resync.knowledge.ingestion.embedding_service import (
                    EmbeddingService,
                )
                from resync.knowledge.retrieval.hybrid_retriever import HybridRetriever
                from resync.knowledge.store.pgvector_store import PgVectorStore

                embedder = EmbeddingService()
                store = PgVectorStore()
                self._hybrid_retriever = HybridRetriever(embedder, store)
            except Exception as e:
                logger.warning("Hybrid retriever init failed: %s", e)
        return self._hybrid_retriever

    def _run_async_retrieval(self, retriever: Any, query: str, top_k: int) -> list:
        """Execute async retrieval in sync context safely."""

        async def _retrieve() -> list:
            return await retriever.retrieve(query, top_k=top_k)

        return run_sync(_retrieve())

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["rag", "search", "knowledge"],
    )
    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> dict[str, Any]:
        """
        Search the TWS knowledge base using RAG.

        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use hybrid retrieval (BM25 + vector)

        Returns:
            Search results with relevance scores

        v5.7.1: Fixed - now actually calls retriever instead of returning stub
        """
        import asyncio
        import concurrent.futures

        start_time = time.time()

        try:
            retriever = (
                self._get_hybrid_retriever() if use_hybrid else self._get_retriever()
            )

            if not retriever:
                return {
                    "results": [],
                    "query": query,
                    "total_found": 0,
                    "search_time_ms": 0,
                    "error": "Retriever not available",
                }

            # v5.7.1 FIX: Execute async retrieval properly
            results = []
            try:
                # Check if we're already in an event loop
                try:
                    asyncio.get_running_loop()
                    # We're in async context - use thread pool to avoid nested loop
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            self._run_async_retrieval, retriever, query, top_k
                        )
                        results = future.result(timeout=30)
                except RuntimeError:
                    # No running loop - run through sync/async bridge helper
                    results = self._run_async_retrieval(retriever, query, top_k)

            except concurrent.futures.TimeoutError:
                logger.warning("RAG retrieval timeout for query: %s", query[:50])
                return {
                    "results": [],
                    "query": query,
                    "total_found": 0,
                    "search_time_ms": int((time.time() - start_time) * 1000),
                    "error": "Retrieval timeout",
                }

            search_time = int((time.time() - start_time) * 1000)

            # Normalize results format
            normalized_results = []
            for r in results if results else []:
                if isinstance(r, dict):
                    normalized_results.append(
                        {
                            "content": r.get("content", r.get("text", "")),
                            "score": r.get("score", 0.0),
                            "metadata": r.get("metadata", {}),
                        }
                    )
                elif isinstance(r, (tuple, list)) and len(r) >= 2:
                    normalized_results.append(
                        {
                            "content": str(r[0]),
                            "score": float(r[1]) if len(r) > 1 else 0.0,
                            "metadata": r[2] if len(r) > 2 else {},
                        }
                    )
                else:
                    normalized_results.append(
                        {
                            "content": str(r),
                            "score": 0.0,
                            "metadata": {},
                        }
                    )

            return {
                "results": normalized_results,
                "query": query,
                "total_found": len(normalized_results),
                "search_time_ms": search_time,
            }

        except Exception as e:
            logger.error("RAG search failed: %s", e, exc_info=True)
            return {
                "results": [],
                "query": query,
                "total_found": 0,
                "search_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e),
            }


# =============================================================================
# SEARCH HISTORY TOOL (PR-6: Operational memory)
# =============================================================================


class SearchHistoryInput(BaseModel):
    """Input schema for history search."""

    query: str = Field(..., min_length=1, max_length=1000)
    incident_type: str | None = None
    resolution_status: str | None = Field(None, pattern="^(resolved|pending|failed)$")
    days_back: int = Field(30, ge=1, le=365)
    limit: int = Field(10, ge=1, le=100)


class IncidentRecord(BaseModel):
    """Schema for incident record."""

    incident_id: str
    timestamp: datetime
    problem_description: str
    symptoms: list[str]
    root_cause: str | None = None
    resolution: str | None = None
    resolution_status: str
    affected_jobs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    feedback_score: float | None = None


class SearchHistoryTool:
    """
    Tool to search historical incidents and resolutions.

    Builds operational memory from:
    - resync/core/audit_db.py
    - resync/core/audit_log.py
    - resync/RAG/microservice/core/feedback_store.py
    """

    def __init__(self):
        self._db = None
        self._feedback_store = None

    def _get_db(self):
        """Get audit database connection."""
        if not self._db:
            try:
                from resync.core.audit_db import get_audit_db

                self._db = get_audit_db()
            except Exception as e:
                logger.warning("Audit DB init failed: %s", e)
        return self._db

    def _get_feedback_store(self):
        """Get feedback store."""
        if not self._feedback_store:
            try:
                from resync.knowledge.store.feedback_store import FeedbackStore

                self._feedback_store = FeedbackStore()
            except Exception as e:
                logger.warning("Feedback store init failed: %s", e)
        return self._feedback_store

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["history", "incidents", "memory"],
    )
    def search_history(
        self,
        query: str,
        incident_type: str | None = None,
        resolution_status: str | None = None,
        days_back: int = 30,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search historical incidents for similar problems and their resolutions.

        Args:
            query: Description of current problem
            incident_type: Filter by type (e.g., "job_failure", "dependency")
            resolution_status: Filter by status ("resolved", "pending", "failed")
            days_back: How far back to search
            limit: Maximum results

        Returns:
            Similar incidents with their resolutions and success rates
        """
        try:
            db = self._get_db()
            feedback = self._get_feedback_store()

            # Search for similar incidents
            incidents = self._search_incidents(
                db, query, incident_type, resolution_status, days_back, limit
            )

            # Enrich with feedback scores
            if feedback:
                for incident in incidents:
                    try:
                        score = feedback.get_resolution_score(
                            incident.get("incident_id", "")
                        )
                        if score:
                            incident["feedback_score"] = score
                    except Exception as exc:
                        logger.debug(
                            "suppressed_exception", error=str(exc), exc_info=True
                        )  # was: pass

            # Calculate success rate for similar resolutions
            success_rate = self._calculate_success_rate(incidents)

            return {
                "incidents": incidents,
                "total_found": len(incidents),
                "success_rate": success_rate,
                "query": query,
                "recommendations": self._generate_recommendations(incidents),
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("History search failed: %s", e)
            return {
                "incidents": [],
                "total_found": 0,
                "success_rate": 0.0,
                "query": query,
                "error": str(e),
            }

    def _run_async_search(
        self,
        db: Any,
        query: str,
        incident_type: str | None,
        status: str | None,
        limit: int,
    ) -> list:
        """Execute async search in sync context safely."""

        async def _search() -> list:
            return await db.search_incidents(
                query=query, incident_type=incident_type, status=status, limit=limit
            )

        return run_sync(_search())

    def _search_incidents(
        self,
        db: Any,
        query: str,
        incident_type: str | None,
        status: str | None,
        days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Search incidents in database.

        Uses the audit database as a local operational memory.
        """
        if not db:
            return []

        # Execute async search in sync context safely
        try:
            try:
                asyncio.get_running_loop()
                # We are in an async context (e.g. running under FastAPI/uvicorn)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._run_async_search, db, query, incident_type, status, limit
                    )
                    entries = future.result(timeout=30)
            except RuntimeError:
                # No running loop - run through sync/async bridge helper
                entries = self._run_async_search(
                    db, query, incident_type, status, limit
                )

            # Filter by date and map to incident records
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            results = []
            for entry in entries:
                if entry.timestamp >= cutoff:
                    results.append(self._map_audit_to_incident(entry))

            return results

        except Exception as e:
            logger.error("Database incident search failed: %s", e)
            return []

    def _map_audit_to_incident(self, entry: Any) -> dict[str, Any]:
        """Map AuditEntry to incident dictionary format."""
        details = entry.metadata_ or {}

        # Extract resolution info
        resolution = details.get("resolution", "")
        if not resolution and entry.action == "incident_resolved":
            resolution = details.get("solution", "Resolved in audit log")

        return {
            "incident_id": f"AUDIT_{entry.id}",
            "timestamp": entry.timestamp,
            "incident_type": entry.entity_type or entry.action,
            "problem_description": details.get("description", entry.action),
            "symptoms": details.get("symptoms", []),
            "root_cause": details.get("root_cause", ""),
            "resolution": resolution,
            "resolution_status": "resolved"
            if entry.action == "incident_resolved"
            else "pending",
            "affected_jobs": details.get(
                "affected_jobs", [entry.entity_id] if entry.entity_id else []
            ),
        }

    def _calculate_success_rate(self, incidents: list[dict]) -> float:
        """Calculate success rate of resolved incidents."""
        resolved = [i for i in incidents if i.get("resolution_status") == "resolved"]
        if not incidents:
            return 0.0
        return len(resolved) / len(incidents)

    def _generate_recommendations(self, incidents: list[dict]) -> list[str]:
        """Generate recommendations from historical incidents."""
        recommendations = []

        for incident in incidents[:3]:  # Top 3 similar
            if incident.get("resolution"):
                recommendations.append(
                    f"Similar incident '{incident.get('incident_id')}' was resolved by: "
                    f"{incident.get('resolution')}"
                )

        return recommendations


# =============================================================================
# JOB ANALYST TOOLS (with enhanced guardrails)
# =============================================================================


class JobLogTool:
    """
    Tool for analyzing TWS job logs and execution history.

    Capabilities:
    - Retrieve job execution logs
    - Parse return codes and ABEND codes
    - Identify error patterns
    - Get execution statistics
    """

    ABEND_CODES: dict[str, str] = {
        "S0C1": "Operation exception - invalid instruction",
        "S0C4": "Protection exception - invalid memory access",
        "S0C7": "Data exception - invalid decimal/numeric data",
        "S013": "I/O error - file not found or access denied",
        "S106": "Module not found",
        "S222": "Job cancelled by operator",
        "S322": "Job timed out - CPU or wait time exceeded",
        "S522": "Job wait time exceeded",
        "S806": "Module load failure",
        "S878": "Virtual storage exhausted",
        "S913": "Security violation - RACF/ACF2 denied",
        "U0016": "User abend - application error",
        "U1000": "User abend - custom application code",
        "U4038": "CICS transaction abend",
    }

    RETURN_CODES: dict[int, str] = {
        0: "Successful completion",
        4: "Warning - minor issues",
        8: "Error - processing problems",
        12: "Severe error - partial failure",
        16: "Critical error - job failed",
        20: "Fatal error - immediate termination",
    }

    def __init__(self, tws_client: Any | None = None):
        """Initialize with optional TWS client."""
        self.tws_client = tws_client

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["job", "log", "analysis"],
    )
    def get_job_log(
        self,
        job_name: str,
        run_date: str | None = None,
        max_lines: int = 100,
    ) -> dict[str, Any]:
        """
        Retrieve execution log for a specific job.

        Args:
            job_name: Name of the job (e.g., BATCH001, DAILY_BACKUP)
            run_date: Date to retrieve (YYYY-MM-DD), defaults to today
            max_lines: Maximum log lines to return

        Returns:
            Job log details including status, return code, and log content
        """
        logger.info("get_job_log", job_name=job_name, run_date=run_date)

        # Mock implementation - replace with actual TWS API call
        return {
            "job_name": job_name,
            "run_date": run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "status": "ABEND",
            "return_code": 16,
            "abend_code": "S0C7",
            "start_time": "08:30:00",
            "end_time": "08:35:22",
            "duration_seconds": 322,
            "workstation": "TWS_AGENT1",
            "log_excerpt": [
                "08:30:00 Job started on TWS_AGENT1",
                "08:32:15 Step STEP010 completed RC=0",
                "08:34:50 Step STEP020 processing file INPUT.DATA",
                "08:35:22 ABEND S0C7 in STEP030 - Data exception",
            ][:max_lines],
            "error_details": "Data exception at offset 0x1A2F in module PROC001",
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["analysis", "return_code"],
    )
    def analyze_return_code(self, return_code: int) -> dict[str, Any]:
        """
        Analyze a return code and provide interpretation.

        Args:
            return_code: The job return code (0-999)

        Returns:
            Analysis with severity, description, and recommendations
        """
        if return_code == 0:
            severity = "SUCCESS"
        elif return_code <= 4:
            severity = "WARNING"
        elif return_code <= 8:
            severity = "ERROR"
        elif return_code <= 12:
            severity = "SEVERE"
        else:
            severity = "CRITICAL"

        description = self.RETURN_CODES.get(
            return_code, f"Custom return code {return_code}"
        )

        return {
            "return_code": return_code,
            "severity": severity,
            "description": description,
            "action_required": severity in ("ERROR", "SEVERE", "CRITICAL"),
            "recommendations": self._get_rc_recommendations(return_code, severity),
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["analysis", "abend"],
    )
    def analyze_abend_code(self, abend_code: str) -> dict[str, Any]:
        """
        Analyze an ABEND code and provide interpretation.

        Args:
            abend_code: The ABEND code (e.g., S0C7, U0016)

        Returns:
            Analysis with description, common causes, and solutions
        """
        abend_upper = abend_code.upper()
        description = self.ABEND_CODES.get(
            abend_upper, f"Unknown ABEND code {abend_upper}"
        )

        return {
            "abend_code": abend_upper,
            "description": description,
            "category": "System" if abend_upper.startswith("S") else "User",
            "common_causes": self._get_abend_causes(abend_upper),
            "recommended_actions": self._get_abend_solutions(abend_upper),
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["job", "history", "statistics"],
    )
    def get_job_history(
        self,
        job_name: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Get execution history for a job over specified days.

        Args:
            job_name: Name of the job
            days: Number of days of history

        Returns:
            Execution history with statistics and trends
        """
        return {
            "job_name": job_name,
            "period_days": days,
            "total_executions": 21,
            "success_rate": 0.857,
            "avg_duration_seconds": 345,
            "max_duration_seconds": 890,
            "min_duration_seconds": 210,
            "failure_count": 3,
            "common_failure_codes": ["S0C7", "RC=16"],
            "trend": "degrading",
            "last_success": "2025-12-08T14:30:00Z",
            "last_failure": "2025-12-09T08:35:22Z",
        }

    def _get_rc_recommendations(self, rc: int, severity: str) -> list[str]:
        """Get recommendations based on return code."""
        if severity == "SUCCESS":
            return ["No action required"]
        if severity == "WARNING":
            return [
                "Review job output for warnings",
                "Verify data quality if applicable",
            ]
        if severity == "ERROR":
            return [
                "Check job log for error details",
                "Verify input files exist and are accessible",
                "Check for resource constraints",
            ]
        return [
            "Immediate investigation required",
            "Check system logs for related issues",
            "Verify job dependencies completed successfully",
            "Consider rerunning after investigation",
        ]

    def _get_abend_causes(self, abend: str) -> list[str]:
        """Get common causes for an ABEND code."""
        causes_map = {
            "S0C7": [
                "Invalid numeric data in input file",
                "Uninitialized working storage fields",
                "Packed decimal field contains invalid data",
            ],
            "S0C4": [
                "Array subscript out of bounds",
                "Invalid pointer reference",
                "Memory corruption",
            ],
            "S322": [
                "Job exceeded CPU time limit",
                "Infinite loop in program",
                "Excessive I/O operations",
            ],
            "S913": [
                "Missing RACF/ACF2 permissions",
                "Dataset protected",
                "Invalid user credentials",
            ],
        }
        return causes_map.get(abend, ["Unknown - check system logs"])

    def _get_abend_solutions(self, abend: str) -> list[str]:
        """Get recommended solutions for an ABEND code."""
        solutions_map = {
            "S0C7": [
                "Verify input file data format",
                "Check for invalid characters in numeric fields",
                "Initialize all working storage variables",
                "Add data validation before processing",
            ],
            "S0C4": [
                "Review array bounds in COBOL/PL1 code",
                "Check pointer initialization",
                "Run with debugging options enabled",
            ],
            "S322": [
                "Increase TIME parameter on JOB or EXEC",
                "Optimize program logic",
                "Add checkpoints for long-running processes",
            ],
            "S913": [
                "Request appropriate RACF permissions",
                "Verify dataset access requirements",
                "Contact security administrator",
            ],
        }
        return solutions_map.get(abend, ["Contact system support for assistance"])


# =============================================================================
# TWS COMMAND EXECUTION TOOL (HITL Required - PR-5)
# =============================================================================


class TWSCommandTool:
    """
    Tool for executing TWS commands (requires HITL approval).

    All write/execute operations require human approval.
    """

    def __init__(self, tws_client: Any | None = None):
        self.tws_client = tws_client

    @tool(
        permission=ToolPermission.EXECUTE,
        requires_approval=True,  # HITL required!
        tags=["tws", "command", "execute", "hitl"],
    )
    def execute_tws_command(
        self,
        command: str,
        target: str,
        parameters: dict | None = None,
    ) -> dict[str, Any]:
        """
        Execute a TWS command on a job or workstation.

        ⚠️ This action requires human approval before execution.

        Args:
            command: Command to execute (rerun, cancel, hold, release)
            target: Target job or workstation name
            parameters: Additional command parameters

        Returns:
            Command execution result
        """
        logger.info(
            "execute_tws_command",
            command=command,
            target=target,
            parameters=parameters,
        )

        # This will only execute after HITL approval
        # Mock implementation
        return {
            "command": command,
            "target": target,
            "status": "executed",
            "result": f"Command '{command}' executed on {target}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @tool(
        permission=ToolPermission.WRITE,
        requires_approval=True,  # HITL required!
        tags=["tws", "config", "write", "hitl"],
    )
    def modify_job_schedule(
        self,
        job_name: str,
        schedule_changes: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Modify job scheduling parameters.

        ⚠️ This action requires human approval before execution.

        Args:
            job_name: Job to modify
            schedule_changes: New schedule parameters

        Returns:
            Modification result
        """
        logger.info(
            "modify_job_schedule",
            job_name=job_name,
            changes=schedule_changes,
        )

        return {
            "job_name": job_name,
            "status": "modified",
            "changes_applied": schedule_changes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# DEPENDENCY GRAPH TOOL
# =============================================================================


class DependencyGraphTool:
    """Tool for analyzing TWS job dependencies."""

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["dependency", "graph", "analysis"],
    )
    def get_predecessors(
        self,
        job_name: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get predecessor jobs (upstream dependencies)."""
        logger.info("get_predecessors", job_name=job_name, depth=depth)

        return {
            "job_name": job_name,
            "depth": depth,
            "predecessors": [
                {
                    "job_name": "DATA_EXTRACT",
                    "level": 1,
                    "dependency_type": "SUCCESS",
                    "status": "COMPLETED",
                },
                {
                    "job_name": "FTP_TRANSFER",
                    "level": 2,
                    "dependency_type": "SUCCESS",
                    "status": "COMPLETED",
                },
            ],
            "total_predecessors": 2,
            "all_satisfied": True,
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["dependency", "graph", "analysis"],
    )
    def get_successors(
        self,
        job_name: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get successor jobs (downstream dependents)."""
        logger.info("get_successors", job_name=job_name, depth=depth)

        return {
            "job_name": job_name,
            "depth": depth,
            "successors": [
                {
                    "job_name": "REPORT_GEN",
                    "level": 1,
                    "dependency_type": "SUCCESS",
                    "status": "WAITING",
                },
                {
                    "job_name": "DATA_LOAD",
                    "level": 1,
                    "dependency_type": "SUCCESS",
                    "status": "WAITING",
                },
            ],
            "total_successors": 2,
            "impacted_jobs": ["REPORT_GEN", "DATA_LOAD"],
            "critical_successors": ["REPORT_GEN"],
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["dependency", "impact", "analysis"],
    )
    def analyze_impact(
        self,
        job_name: str,
        failure_scenario: bool = True,
    ) -> dict[str, Any]:
        """Analyze impact if a job fails or is delayed."""
        successors = self.get_successors(job_name)

        return {
            "job_name": job_name,
            "scenario": "failure" if failure_scenario else "delay",
            "direct_impact": len(successors.get("successors", [])),
            "total_impact": successors.get("total_successors", 0),
            "impacted_jobs": successors.get("impacted_jobs", []),
            "critical_jobs_affected": successors.get("critical_successors", []),
            "estimated_delay_minutes": 45,
            "risk_level": "HIGH" if successors.get("critical_successors") else "MEDIUM",
            "recommendations": [
                "Prioritize resolution of this job",
                "Notify downstream job owners",
                "Consider running backup procedures",
            ],
        }


# =============================================================================
# WORKSTATION AND RESOURCE TOOLS
# =============================================================================


class WorkstationTool:
    """Tool for analyzing workstation status and capacity."""

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["workstation", "status", "monitoring"],
    )
    def get_workstation_status(
        self,
        workstation_name: str | None = None,
    ) -> dict[str, Any]:
        """Get current workstation status."""
        if workstation_name:
            return {
                "workstation": workstation_name,
                "status": "ONLINE",
                "agent_status": "ACTIVE",
                "jobs_running": 3,
                "jobs_queued": 5,
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 62.8,
                "disk_usage_percent": 78.5,
                "last_heartbeat": "2025-12-09T10:45:00Z",
            }
        return {
            "workstations": [
                {"name": "TWS_MASTER", "status": "ONLINE", "jobs_running": 2},
                {"name": "TWS_AGENT1", "status": "ONLINE", "jobs_running": 5},
                {"name": "TWS_AGENT2", "status": "OFFLINE", "jobs_running": 0},
            ],
            "total_online": 2,
            "total_offline": 1,
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["resource", "availability"],
    )
    def check_resource_availability(
        self,
        resource_name: str,
    ) -> dict[str, Any]:
        """Check if a specific resource is available."""
        return {
            "resource_name": resource_name,
            "available": True,
            "current_owner": None,
            "max_concurrent": 1,
            "current_usage": 0,
            "queue_depth": 0,
            "waiting_jobs": [],
        }


class CalendarTool:
    """Tool for TWS calendar and scheduling analysis."""

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["calendar", "scheduling"],
    )
    def get_calendar_schedule(
        self,
        calendar_name: str,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Get calendar schedule information."""
        check_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return {
            "calendar": calendar_name,
            "date": check_date,
            "is_workday": True,
            "is_holiday": False,
            "special_processing": None,
            "next_run_date": check_date,
            "business_days_remaining": 15,
        }

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["scheduling", "window"],
    )
    def check_scheduling_window(
        self,
        job_name: str,
    ) -> dict[str, Any]:
        """Check the scheduling window for a job."""
        return {
            "job_name": job_name,
            "earliest_start": "06:00:00",
            "latest_start": "22:00:00",
            "deadline": "23:59:59",
            "within_window": True,
            "time_to_deadline_minutes": 480,
            "priority": 5,
        }


# =============================================================================
# METRICS AND OBSERVABILITY TOOL (PR-7)
# =============================================================================


class MetricsTool:
    """
    Tool for retrieving system metrics and observability data.

    Connects with:
    - resync/core/anomaly_detector.py
    - resync/core/alerting.py
    - resync/core/monitoring_config.py
    """

    @tool(
        permission=ToolPermission.READ_ONLY,
        tags=["metrics", "monitoring", "observability"],
    )
    def get_system_metrics(
        self,
        metric_type: str = "all",
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get system metrics and health indicators.

        Args:
            metric_type: Type of metrics (all, performance, errors, capacity)
            time_range_hours: Time range for metrics

        Returns:
            System metrics and health status
        """
        try:
            return {
                "metric_type": metric_type,
                "time_range_hours": time_range_hours,
                "metrics": {
                    "job_success_rate": 0.92,
                    "avg_job_duration_ms": 45000,
                    "active_workstations": 8,
                    "queue_depth": 12,
                    "error_rate": 0.08,
                },
                "anomalies": [],
                "health_status": "healthy",
            }
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("Metrics retrieval failed: %s", e)
            return {
                "metric_type": metric_type,
                "error": str(e),
                "health_status": "unknown",
            }


# =============================================================================
# ERROR CODE TOOL (Legacy compatibility)
# =============================================================================


class ErrorCodeTool:
    """Legacy compatibility - delegates to JobLogTool."""

    def __init__(self):
        self._job_log_tool = JobLogTool()

    def analyze_error(self, error_code: str) -> dict[str, Any]:
        """Analyze an error code."""
        if error_code.startswith(("S", "U")):
            return self._job_log_tool.analyze_abend_code(error_code)
        try:
            rc = int(error_code.replace("RC=", "").strip())
            return self._job_log_tool.analyze_return_code(rc)
        except ValueError:
            return {
                "error_code": error_code,
                "description": "Unknown error format",
                "recommendations": ["Check job log for details"],
            }


# =============================================================================
# EXPORT ALL TOOLS
# =============================================================================

__all__ = [
    # Permissions and types
    "ToolPermission",
    "UserRole",
    "ROLE_PERMISSIONS",
    "ToolExecutionTrace",
    "ToolDefinition",
    "ToolCatalog",
    "get_tool_catalog",
    "ApprovalRequiredError",
    # Decorator
    "tool",
    # Input/Output schemas
    "JobLogInput",
    "JobLogOutput",
    "JobHistoryInput",
    "WorkstationStatusInput",
    "ExecuteCommandInput",
    "RAGSearchInput",
    "RAGSearchOutput",
    "SearchHistoryInput",
    "IncidentRecord",
    # Tools
    "JobLogTool",
    "ErrorCodeTool",
    "DependencyGraphTool",
    "WorkstationTool",
    "CalendarTool",
    "RAGTool",
    "SearchHistoryTool",
    "TWSCommandTool",
    "MetricsTool",
]
