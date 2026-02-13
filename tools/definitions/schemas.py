"""
Tool Input/Output Schemas.

v5.8.0: Centralized Pydantic models for tool validation.

All tool input and output schemas are defined here for:
- Type safety
- Automatic validation
- OpenAPI documentation generation
- Consistent error messages

Author: Resync Team
Version: 5.8.0
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# JOB TOOLS SCHEMAS
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


class JobHistoryOutput(BaseModel):
    """Output schema for job history."""

    job_name: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    total_runs: int = 0
    success_rate: float = 0.0


# =============================================================================
# WORKSTATION TOOLS SCHEMAS
# =============================================================================


class WorkstationStatusInput(BaseModel):
    """Input schema for workstation status."""

    workstation_name: str | None = Field(None, max_length=100)


class WorkstationStatusOutput(BaseModel):
    """Output schema for workstation status."""

    workstation_name: str
    status: str
    linked_workstations: list[str] = Field(default_factory=list)
    jobs_running: int = 0
    jobs_waiting: int = 0
    last_updated: str | None = None


# =============================================================================
# COMMAND EXECUTION SCHEMAS
# =============================================================================


class ExecuteCommandInput(BaseModel):
    """Input schema for TWS command execution."""

    command: str = Field(..., min_length=1, max_length=500)
    target: str = Field(..., description="Job or workstation name")
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExecuteCommandOutput(BaseModel):
    """Output schema for command execution."""

    command: str
    target: str
    success: bool
    message: str
    output: str | None = None
    error: str | None = None


# =============================================================================
# RAG/SEARCH SCHEMAS
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


# =============================================================================
# DEPENDENCY GRAPH SCHEMAS
# =============================================================================


class DependencyGraphInput(BaseModel):
    """Input schema for dependency graph."""

    job_name: str = Field(..., min_length=1, max_length=100)
    depth: int = Field(2, ge=1, le=5, description="Max depth to traverse")
    direction: str = Field("both", pattern="^(upstream|downstream|both)$")


class DependencyGraphOutput(BaseModel):
    """Output schema for dependency graph."""

    job_name: str
    upstream: list[dict[str, Any]] = Field(default_factory=list)
    downstream: list[dict[str, Any]] = Field(default_factory=list)
    total_dependencies: int = 0


# =============================================================================
# SEARCH HISTORY SCHEMAS
# =============================================================================


class SearchHistoryInput(BaseModel):
    """Input schema for search history."""

    query: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field("job", pattern="^(job|workstation|schedule|incident)$")
    time_range: str = Field("7d", pattern=r"^\d+[hdwm]$")
    limit: int = Field(50, ge=1, le=200)


class IncidentRecord(BaseModel):
    """Model for incident records."""

    incident_id: str
    job_name: str
    timestamp: str
    severity: str
    status: str
    description: str
    resolution: str | None = None
    assigned_to: str | None = None


# =============================================================================
# METRICS SCHEMAS
# =============================================================================


class MetricsInput(BaseModel):
    """Input schema for metrics retrieval."""

    metric_type: str = Field(..., description="Type of metric to retrieve")
    time_range: str = Field("1h", pattern=r"^\d+[hdwm]$")
    aggregation: str = Field("avg", pattern="^(avg|sum|min|max|count)$")


class MetricsOutput(BaseModel):
    """Output schema for metrics."""

    metric_type: str
    values: list[dict[str, Any]] = Field(default_factory=list)
    aggregated_value: float | None = None
    time_range: str


# =============================================================================
# CALENDAR SCHEMAS
# =============================================================================


class CalendarInput(BaseModel):
    """Input schema for calendar operations."""

    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    calendar_name: str | None = None


class CalendarOutput(BaseModel):
    """Output schema for calendar operations."""

    date: str
    is_workday: bool
    calendar_name: str | None = None
    scheduled_jobs: list[str] = Field(default_factory=list)


# =============================================================================
# ERROR CODE SCHEMAS
# =============================================================================


class ErrorCodeInput(BaseModel):
    """Input schema for error code lookup."""

    error_code: str = Field(..., min_length=1, max_length=50)


class ErrorCodeOutput(BaseModel):
    """Output schema for error code lookup."""

    error_code: str
    description: str
    category: str
    severity: str
    suggested_actions: list[str] = Field(default_factory=list)
    documentation_url: str | None = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Job schemas
    "JobLogInput",
    "JobLogOutput",
    "JobHistoryInput",
    "JobHistoryOutput",
    # Workstation schemas
    "WorkstationStatusInput",
    "WorkstationStatusOutput",
    # Command schemas
    "ExecuteCommandInput",
    "ExecuteCommandOutput",
    # RAG schemas
    "RAGSearchInput",
    "RAGSearchOutput",
    # Dependency schemas
    "DependencyGraphInput",
    "DependencyGraphOutput",
    # Search schemas
    "SearchHistoryInput",
    "IncidentRecord",
    # Metrics schemas
    "MetricsInput",
    "MetricsOutput",
    # Calendar schemas
    "CalendarInput",
    "CalendarOutput",
    # Error code schemas
    "ErrorCodeInput",
    "ErrorCodeOutput",
]
