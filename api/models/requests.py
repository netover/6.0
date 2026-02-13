"""
Request models for FastAPI endpoints.

v5.4.1 Enhancements (PR-4):
- Goal model for agentic execution
- ToolResult for tool execution tracking
- ExecutionTrace for observability
- ApprovalRequest for HITL workflows

Author: Resync Team
Version: 5.4.1
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# EXISTING MODELS (maintained for compatibility)
# =============================================================================


class AuditReviewRequest(BaseModel):
    """Request model for audit review operations."""

    memory_id: str
    action: str


class ChatMessageRequest(BaseModel):
    """Request model for chat message operations."""

    message: str
    agent_id: str | None = None  # Deprecated: routing is automatic
    tws_instance_id: str | None = None  # TWS instance for multi-server queries
    session_id: str | None = None  # Session ID for conversation memory
    metadata: dict[str, Any] | None = None  # Additional metadata


class FileUploadRequest(BaseModel):
    """Request model for file upload operations."""

    filename: str
    content_type: str
    size: int


class SystemStatusFilter(BaseModel):
    """System status filter."""

    workstation_filter: str | None = None
    job_status_filter: str | None = None


class LoginRequest(BaseModel):
    """Request model for login operations."""

    username: str
    password: str


class RAGUploadRequest(BaseModel):
    """Request model for RAG upload operations."""

    filename: str
    content: str


class AuditFlagsQuery(BaseModel):
    """Query model for audit flags operations."""

    status_filter: str | None = None
    query: str | None = None
    limit: int = 50
    offset: int = 0


class ChatHistoryQuery(BaseModel):
    """Query model for chat history operations."""

    agent_id: str | None = None
    limit: int = 50


class RAGFileQuery(BaseModel):
    """Query model for RAG file operations."""

    file_id: str


class FileUploadValidation(BaseModel):
    """Validation model for file uploads"""

    filename: str
    content_type: str
    size: int

    def validate_file(self) -> None:
        """Validate file properties"""
        from pathlib import Path

        # Check file extension
        file_ext = Path(self.filename).suffix.lower()
        allowed_extensions = {".txt", ".pdf", ".docx", ".md", ".json"}
        if file_ext not in allowed_extensions:
            raise ValueError(
                f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024
        if self.size > max_size:
            raise ValueError(f"File too large. Maximum size: {max_size / (1024 * 1024)}MB")


# =============================================================================
# NEW MODELS FOR AGENTIC EXECUTION (PR-4)
# =============================================================================


class Goal(BaseModel):
    """
    High-level goal for agentic execution.

    The agent will break this down into steps and execute them.
    """

    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Description of what needs to be accomplished",
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the goal"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Constraints or requirements for execution"
    )
    priority: str = Field(
        default="normal", pattern="^(low|normal|high|critical)$", description="Priority level"
    )
    timeout_seconds: int = Field(default=120, ge=10, le=600, description="Maximum execution time")
    require_approval: bool = Field(
        default=False, description="Whether to require HITL approval before execution"
    )


class ToolCallRequest(BaseModel):
    """
    Request to execute a specific tool.
    """

    tool_name: str = Field(
        ..., min_length=1, max_length=100, description="Name of the tool to execute"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    user_role: str = Field(default="operator", description="User role for permission checking")


class AgentExecuteRequest(BaseModel):
    """
    Request for agent execution.

    Can include either a simple message or a structured goal.
    """

    # Simple message-based execution
    message: str | None = Field(None, description="Simple message for the agent")

    # Goal-based execution
    goal: Goal | None = Field(None, description="Structured goal for agentic execution")

    # Context
    tws_instance_id: str | None = Field(None, description="TWS instance for multi-server queries")
    session_id: str | None = Field(None, description="Session ID for conversation memory")

    # Execution options
    routing_mode: str | None = Field(
        None, pattern="^(rag_only|agentic|diagnostic)$", description="Force a specific routing mode"
    )
    include_trace: bool = Field(default=True, description="Include execution trace in response")

    def get_input(self) -> str:
        """Get the input message or goal description."""
        if self.message:
            return self.message
        if self.goal:
            return self.goal.description
        return ""


class ApprovalRequest(BaseModel):
    """
    Request for HITL approval of a pending action.
    """

    trace_id: str = Field(..., description="ID of the execution trace requiring approval")
    action: str = Field(..., pattern="^(approve|reject)$", description="Approval action")
    approver_id: str | None = Field(None, description="ID of the approver")
    reason: str | None = Field(None, max_length=500, description="Reason for approval/rejection")


class ApprovalListQuery(BaseModel):
    """
    Query for listing pending approvals.
    """

    status: str = Field(
        default="pending",
        pattern="^(pending|approved|rejected|all)$",
        description="Filter by approval status",
    )
    tool_name: str | None = Field(None, description="Filter by tool name")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


# =============================================================================
# DIAGNOSTIC EXECUTION MODELS
# =============================================================================


class DiagnosticRequest(BaseModel):
    """
    Request for diagnostic problem resolution.
    """

    problem_description: str = Field(
        ..., min_length=10, max_length=5000, description="Description of the problem to diagnose"
    )
    tws_instance_id: str | None = Field(None, description="TWS instance ID")
    affected_jobs: list[str] = Field(default_factory=list, description="List of affected job names")
    error_codes: list[str] = Field(default_factory=list, description="Error codes (ABEND, RC)")
    time_range_hours: int = Field(
        default=24, ge=1, le=168, description="Time range to search for related issues"
    )
    include_history: bool = Field(
        default=True, description="Include historical incidents in analysis"
    )
    max_iterations: int = Field(default=5, ge=1, le=10, description="Maximum diagnostic iterations")


# =============================================================================
# FEEDBACK AND LEARNING MODELS
# =============================================================================


class FeedbackRequest(BaseModel):
    """
    Request to provide feedback on a response.
    """

    trace_id: str = Field(..., description="ID of the execution trace to provide feedback for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    feedback_type: str = Field(
        default="general",
        pattern="^(general|accuracy|helpfulness|speed)$",
        description="Type of feedback",
    )
    comment: str | None = Field(None, max_length=1000, description="Optional comment")
    resolution_worked: bool | None = Field(
        None, description="Whether the suggested resolution worked"
    )


class IncidentResolutionRequest(BaseModel):
    """
    Request to record an incident resolution.
    """

    incident_id: str | None = Field(None, description="Existing incident ID (if updating)")
    problem_description: str = Field(
        ..., min_length=10, max_length=5000, description="Description of the problem"
    )
    symptoms: list[str] = Field(default_factory=list, description="List of symptoms observed")
    root_cause: str = Field(..., min_length=5, max_length=2000, description="Identified root cause")
    resolution: str = Field(..., min_length=5, max_length=5000, description="Resolution applied")
    affected_jobs: list[str] = Field(
        default_factory=list, description="Jobs affected by this incident"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Evidence collected (log lines, screenshots, etc.)"
    )
    resolution_status: str = Field(
        default="resolved",
        pattern="^(resolved|pending|failed)$",
        description="Status of the resolution",
    )


# =============================================================================
# METRICS AND OBSERVABILITY MODELS
# =============================================================================


class MetricsQuery(BaseModel):
    """
    Query for retrieving metrics.
    """

    metric_type: str = Field(
        default="all",
        pattern="^(all|performance|errors|capacity|tools)$",
        description="Type of metrics to retrieve",
    )
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range for metrics")
    group_by: str | None = Field(
        None, pattern="^(hour|day|tool|intent)$", description="Group results by"
    )
    include_anomalies: bool = Field(default=True, description="Include anomaly detection results")


class TraceQuery(BaseModel):
    """
    Query for retrieving execution traces.
    """

    trace_id: str | None = Field(None, description="Specific trace ID to retrieve")
    session_id: str | None = Field(None, description="Filter by session ID")
    user_id: str | None = Field(None, description="Filter by user ID")
    tool_name: str | None = Field(None, description="Filter by tool name")
    success: bool | None = Field(None, description="Filter by success status")
    start_time: datetime | None = Field(None, description="Filter traces after this time")
    end_time: datetime | None = Field(None, description="Filter traces before this time")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
