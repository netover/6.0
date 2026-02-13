from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkstationStatus(BaseModel):
    """Represents the status of a single TWS workstation."""

    name: str = Field(..., description="The name of the workstation.")
    status: str = Field(
        ...,
        description="The current status of the workstation (e.g., 'LINKED', 'DOWN').",
    )
    type: str = Field(..., description="The type of the workstation (e.g., 'FTA', 'MASTER').")


class JobStatus(BaseModel):
    """Represents the status of a single TWS job."""

    name: str = Field(..., description="The name of the job.")
    workstation: str = Field(..., description="The workstation where the job runs.")
    status: str = Field(
        ...,
        description="The current status of the job (e.g., 'SUCC', 'ABEND').",
    )
    job_stream: str = Field(..., description="The job stream the job belongs to.")


class CriticalJob(BaseModel):
    """Represents a job that is part of the critical path (TWS 'plan')."""

    job_id: int = Field(..., description="The unique identifier for the job in the plan.")
    job_name: str = Field(..., description="The name of the job.")
    status: str = Field(..., description="The status of the critical job.")
    start_time: str = Field(..., description="The scheduled start time for the job.")


# --- New Data Models for Complete MVP ---


class JobExecution(BaseModel):
    """Represents a single execution of a job."""

    job_id: str = Field(..., description="The unique identifier for the job execution")
    status: str = Field(..., description="The status of this execution")
    start_time: datetime = Field(..., description="When the job execution started")
    end_time: datetime | None = Field(None, description="When the job execution ended")
    duration: str | None = Field(None, description="Duration of the execution")
    error_message: str | None = Field(None, description="Error message if execution failed")


class JobDetails(BaseModel):
    """Detailed information about a TWS job."""

    job_id: str = Field(..., description="The unique identifier for the job")
    name: str = Field(..., description="The name of the job")
    workstation: str = Field(..., description="The workstation where the job runs")
    status: str = Field(..., description="The current status of the job")
    job_stream: str = Field(..., description="The job stream the job belongs to")
    full_definition: dict[str, Any] = Field(..., description="Complete job definition from TWS")
    dependencies: list[str] = Field(default_factory=list, description="List of job dependencies")
    resource_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Resource requirements for the job"
    )
    execution_history: list[JobExecution] = Field(
        default_factory=list, description="Recent execution history"
    )


class PlanDetails(BaseModel):
    """Information about the current TWS plan."""

    plan_id: str = Field(..., description="The unique identifier for the plan")
    creation_date: datetime = Field(..., description="When the plan was created")
    jobs_count: int = Field(..., description="Total number of jobs in the plan")
    estimated_completion: datetime | None = Field(None, description="Estimated completion time")
    status: str = Field(..., description="Current status of the plan")


class ResourceStatus(BaseModel):
    """Information about resource usage in TWS."""

    resource_name: str = Field(..., description="The name of the resource")
    resource_type: str = Field(..., description="The type of resource (CPU, memory, etc.)")
    total_capacity: float | None = Field(None, description="Total capacity of the resource")
    used_capacity: float | None = Field(None, description="Currently used capacity")
    available_capacity: float | None = Field(None, description="Available capacity")
    utilization_percentage: float | None = Field(None, description="Utilization as percentage")


class Event(BaseModel):
    """Represents a TWS event log entry."""

    event_id: str = Field(..., description="The unique identifier for the event")
    timestamp: datetime = Field(..., description="When the event occurred")
    event_type: str = Field(..., description="The type of event")
    severity: str = Field(..., description="The severity level of the event")
    source: str = Field(..., description="The source of the event")
    message: str = Field(..., description="The event message")
    job_id: str | None = Field(None, description="Associated job ID if applicable")
    workstation: str | None = Field(None, description="Associated workstation if applicable")


class PerformanceData(BaseModel):
    """Performance metrics for TWS operations."""

    timestamp: datetime = Field(..., description="When the metrics were collected")
    api_response_times: dict[str, float] = Field(
        default_factory=dict, description="API response times by endpoint"
    )
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percentage: float = Field(..., description="CPU usage percentage")
    active_connections: int = Field(..., description="Number of active connections")
    jobs_per_minute: float = Field(..., description="Jobs processed per minute")


class DependencyTree(BaseModel):
    """Represents the dependency tree for a job."""

    job_id: str = Field(..., description="The job ID this tree represents")
    dependencies: list[str] = Field(default_factory=list, description="Direct dependencies")
    dependents: list[str] = Field(default_factory=list, description="Jobs that depend on this job")
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict, description="Complete dependency graph"
    )


class SystemStatus(BaseModel):
    """A composite model representing the overall status of the TWS environment."""

    workstations: list[WorkstationStatus]
    jobs: list[JobStatus]
    critical_jobs: list[CriticalJob]
