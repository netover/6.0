"""
TWS Tool Output Validators.

v5.3.17 - Pydantic validators for TWS/HWA tool responses.

Validates outputs from TWS tools before passing to LLM formatting:
- Job status responses
- Dependency information
- Resource usage data
- Error codes and messages

Benefits:
- Catches parsing errors early
- Ensures data consistency
- Provides meaningful error messages
- Enables proper error recovery

Architecture:
    Tool Output → Validator → Valid Data → LLM Response
                     ↓
               Validation Error → Error Handler → User Feedback
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# =============================================================================
# ENUMS
# =============================================================================


class JobStatus(str, Enum):
    """Valid TWS job statuses."""

    SUCC = "SUCC"  # Success
    ABEND = "ABEND"  # Abnormal end (error)
    EXEC = "EXEC"  # Executing
    HOLD = "HOLD"  # On hold
    READY = "READY"  # Ready to run
    WAIT = "WAIT"  # Waiting for dependencies
    SCHED = "SCHED"  # Scheduled
    PEND = "PEND"  # Pending
    CANC = "CANC"  # Cancelled
    UNKNOWN = "UNKNOWN"  # Unknown status


class DependencyType(str, Enum):
    """Types of job dependencies."""

    FOLLOWS = "follows"
    OPENS = "opens"
    NEEDS = "needs"
    PROMPTS = "prompts"


class ResourceType(str, Enum):
    """Types of resources."""

    FILE = "file"
    DATABASE = "database"
    WORKSTATION = "workstation"
    SPECIAL = "special"


# =============================================================================
# BASE VALIDATORS
# =============================================================================


class TWSBaseModel(BaseModel):
    """Base model for TWS validators with common configuration."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


# =============================================================================
# JOB STATUS VALIDATORS
# =============================================================================


class JobStatusResponse(TWSBaseModel):
    """Validates job status response from tws_status tool."""

    job_name: str = Field(..., min_length=1, max_length=255, description="Job name")
    status: JobStatus = Field(..., description="Current job status")
    rc: int | None = Field(None, ge=-9999, le=9999, description="Return code")

    workstation: str | None = Field(None, max_length=100)
    scheduled_time: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    duration_seconds: int | None = Field(None, ge=0)

    rerun_count: int = Field(default=0, ge=0, le=999)
    error_message: str | None = Field(None, max_length=1000)

    @field_validator("job_name")
    @classmethod
    def validate_job_name(cls, v: str) -> str:
        """Validate job name format."""
        v = v.strip().upper()
        if not v:
            raise ValueError("Job name cannot be empty")
        # TWS job names are typically uppercase alphanumeric with underscores
        return v

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> JobStatus:
        """Normalize status to enum."""
        if isinstance(v, JobStatus):
            return v
        if isinstance(v, str):
            v = v.strip().upper()
            try:
                return JobStatus(v)
            except ValueError:
                # Map common variations
                status_map = {
                    "SUCCESS": JobStatus.SUCC,
                    "OK": JobStatus.SUCC,
                    "FAILED": JobStatus.ABEND,
                    "ERROR": JobStatus.ABEND,
                    "RUNNING": JobStatus.EXEC,
                    "EXECUTING": JobStatus.EXEC,
                    "WAITING": JobStatus.WAIT,
                    "SCHEDULED": JobStatus.SCHED,
                    "CANCELLED": JobStatus.CANC,
                    "CANCELED": JobStatus.CANC,
                }
                return status_map.get(v, JobStatus.UNKNOWN)
        return JobStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_timing(self) -> "JobStatusResponse":
        """Validate timing consistency."""
        if self.actual_start and self.actual_end and self.actual_end < self.actual_start:
            raise ValueError("End time cannot be before start time")
        return self

    @property
    def is_error(self) -> bool:
        """Check if job is in error state."""
        return self.status == JobStatus.ABEND or (self.rc is not None and self.rc != 0)

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in (JobStatus.EXEC, JobStatus.READY)

    @property
    def is_complete(self) -> bool:
        """Check if job has completed (success or error)."""
        return self.status in (JobStatus.SUCC, JobStatus.ABEND, JobStatus.CANC)


class BulkJobStatusResponse(TWSBaseModel):
    """Validates bulk job status response."""

    jobs: list[JobStatusResponse] = Field(..., min_length=0)
    query_time: datetime | None = None
    total_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> "BulkJobStatusResponse":
        """Ensure count matches jobs list."""
        if self.total_count == 0:
            self.total_count = len(self.jobs)
        return self


# =============================================================================
# DEPENDENCY VALIDATORS
# =============================================================================


class DependencyInfo(TWSBaseModel):
    """Single dependency relationship."""

    from_job: str = Field(..., min_length=1)
    to_job: str = Field(..., min_length=1)
    dependency_type: DependencyType = Field(default=DependencyType.FOLLOWS)
    condition: str | None = Field(None, max_length=500)

    @field_validator("from_job", "to_job")
    @classmethod
    def normalize_job_name(cls, v: str) -> str:
        return v.strip().upper()


class DependencyChainResponse(TWSBaseModel):
    """Validates dependency chain response."""

    job_name: str = Field(..., min_length=1)
    dependencies: list[DependencyInfo] = Field(default_factory=list)
    depth: int = Field(default=0, ge=0, le=100)

    @property
    def predecessors(self) -> list[str]:
        """Get list of predecessor job names."""
        return [d.from_job for d in self.dependencies]

    @property
    def successors(self) -> list[str]:
        """Get list of successor job names."""
        return [d.to_job for d in self.dependencies if d.from_job == self.job_name]


class ImpactAnalysisResponse(TWSBaseModel):
    """Validates impact analysis response."""

    job_name: str = Field(..., min_length=1)
    affected_jobs: list[str] = Field(default_factory=list)
    affected_count: int = Field(default=0, ge=0)
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")

    estimated_recovery_time_minutes: int | None = Field(None, ge=0)
    affected_schedules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def calculate_severity(self) -> "ImpactAnalysisResponse":
        """Auto-calculate severity based on affected count."""
        if self.affected_count == 0:
            return self

        if self.affected_count > 50:
            self.severity = "critical"
        elif self.affected_count > 20:
            self.severity = "high"
        elif self.affected_count > 5:
            self.severity = "medium"
        else:
            self.severity = "low"

        return self


# =============================================================================
# RESOURCE VALIDATORS
# =============================================================================


class ResourceInfo(TWSBaseModel):
    """Resource information."""

    name: str = Field(..., min_length=1, max_length=255)
    resource_type: ResourceType = Field(default=ResourceType.SPECIAL)
    quantity: int = Field(default=1, ge=0)
    exclusive: bool = Field(default=False)


class ResourceConflictResponse(TWSBaseModel):
    """Validates resource conflict check response."""

    job_a: str = Field(..., min_length=1)
    job_b: str = Field(..., min_length=1)
    conflicts: list[ResourceInfo] = Field(default_factory=list)
    can_run_together: bool = Field(default=True)

    @model_validator(mode="after")
    def determine_conflict(self) -> "ResourceConflictResponse":
        """Determine if jobs can run together based on conflicts."""
        if any(c.exclusive for c in self.conflicts):
            self.can_run_together = False
        return self


class WorkstationStatus(TWSBaseModel):
    """Workstation status information."""

    name: str = Field(..., min_length=1)
    status: Literal["active", "inactive", "linked", "unlinked", "offline"] = "active"
    running_jobs: int = Field(default=0, ge=0)
    max_jobs: int = Field(default=10, ge=1)
    cpu_usage_percent: float | None = Field(None, ge=0, le=100)

    @property
    def is_available(self) -> bool:
        """Check if workstation can accept new jobs."""
        return self.status in ("active", "linked") and self.running_jobs < self.max_jobs


# =============================================================================
# ERROR VALIDATORS
# =============================================================================


class TWSError(TWSBaseModel):
    """TWS error information."""

    error_code: str = Field(..., min_length=1, max_length=50)
    message: str = Field(..., min_length=1, max_length=2000)
    severity: Literal["info", "warning", "error", "critical"] = "error"

    job_name: str | None = None
    workstation: str | None = None
    timestamp: datetime | None = None

    suggested_action: str | None = Field(None, max_length=1000)
    documentation_link: str | None = None


class ErrorLookupResponse(TWSBaseModel):
    """Validates error code lookup response."""

    error_code: str = Field(..., min_length=1)
    description: str = Field(default="Unknown error")
    category: str = Field(default="general")

    possible_causes: list[str] = Field(default_factory=list)
    resolution_steps: list[str] = Field(default_factory=list)
    related_errors: list[str] = Field(default_factory=list)

    @field_validator("error_code")
    @classmethod
    def normalize_error_code(cls, v: str) -> str:
        """Normalize error code format."""
        return v.strip().upper()


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_job_status(data: dict[str, Any]) -> JobStatusResponse | None:
    """
    Validate job status data, returning None on failure.

    Args:
        data: Raw data from TWS tool

    Returns:
        Validated JobStatusResponse or None
    """
    try:
        return JobStatusResponse.model_validate(data)
    except Exception as e:
        from resync.core.structured_logger import get_logger

        logger = get_logger(__name__)
        logger.warning(f"job_status_validation_failed: {e}", data=data)
        return None


def validate_dependency_chain(data: dict[str, Any]) -> DependencyChainResponse | None:
    """Validate dependency chain data."""
    try:
        return DependencyChainResponse.model_validate(data)
    except Exception as e:
        from resync.core.structured_logger import get_logger

        logger = get_logger(__name__)
        logger.warning(f"dependency_chain_validation_failed: {e}", data=data)
        return None


def validate_impact_analysis(data: dict[str, Any]) -> ImpactAnalysisResponse | None:
    """Validate impact analysis data."""
    try:
        return ImpactAnalysisResponse.model_validate(data)
    except Exception as e:
        from resync.core.structured_logger import get_logger

        logger = get_logger(__name__)
        logger.warning(f"impact_analysis_validation_failed: {e}", data=data)
        return None


def validate_tws_response(
    data: dict[str, Any],
    response_type: str,
) -> TWSBaseModel | None:
    """
    Generic validator dispatcher.

    Args:
        data: Raw response data
        response_type: Type of response (job_status, dependency, impact, etc.)

    Returns:
        Validated model or None
    """
    validators = {
        "job_status": validate_job_status,
        "dependency": validate_dependency_chain,
        "impact": validate_impact_analysis,
    }

    validator = validators.get(response_type)
    if validator:
        return validator(data)

    return None


__all__ = [
    # Enums
    "JobStatus",
    "DependencyType",
    "ResourceType",
    # Models
    "TWSBaseModel",
    "JobStatusResponse",
    "BulkJobStatusResponse",
    "DependencyInfo",
    "DependencyChainResponse",
    "ImpactAnalysisResponse",
    "ResourceInfo",
    "ResourceConflictResponse",
    "WorkstationStatus",
    "TWSError",
    "ErrorLookupResponse",
    # Utilities
    "validate_job_status",
    "validate_dependency_chain",
    "validate_impact_analysis",
    "validate_tws_response",
]
