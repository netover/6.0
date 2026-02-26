"""
FastAPI router exposing read‑only TWS endpoints.

This module defines a set of GET routes under the `/tws` prefix for retrieving
information from the TWS engine, model and current plan. All routes depend on
an authenticated operator role by default.
"""

from typing import Any

from fastapi import APIRouter, Depends, Query

from resync.api.security import require_role
from resync.core.factories import get_tws_client as get_client
from resync.services.tws_service import OptimizedTWSClient

__all__ = [
    "get_client",
    "router",
    "engine_info",
    "engine_config",
    "list_users",
    "list_groups",
    "query_jobdefinitions",
    "get_jobdefinition",
    "query_jobstreams",
    "get_jobstream",
    "query_workstations",
    "get_workstation",
    "list_current_plan_jobs",
    "count_current_plan_jobs",
    "current_plan_job_issues",
    "current_plan_job_joblog",
    "get_current_plan_job",
    "get_current_plan_job_predecessors",
    "get_current_plan_job_successors",
    "get_current_plan_job_model",
    "get_current_plan_job_model_description",
    "list_current_plan_jobstreams",
    "count_current_plan_jobstreams",
    "get_current_plan_jobstream",
    "get_current_plan_jobstream_predecessors",
    "get_current_plan_jobstream_successors",
    "get_current_plan_jobstream_model_description",
    "list_current_plan_resources",
    "get_current_plan_resource",
    "current_plan_folder_objects_count",
    "list_consumed_jobs_runs",
]

router = APIRouter(
    prefix="/tws",
    tags=["tws"],
    dependencies=[Depends(require_role("operator"))],
)

@router.get("/engine/info", summary="Get engine information")
async def engine_info(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return basic engine information."""
    return await client.get_engine_info()

@router.get("/engine/config", summary="Get engine configuration")
async def engine_config(
    key: str | None = Query(None),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve the engine configuration. Optionally filter by key."""
    return await client.get_engine_configuration(key)

@router.get("/model/user", summary="List model users")
async def list_users(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """List users defined in the model."""
    return await client.list_users()

@router.get("/model/group", summary="List model groups")
async def list_groups(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """List groups defined in the model."""
    return await client.list_groups()

@router.get("/model/jobdefinition", summary="Query job definitions")
async def query_jobdefinitions(
    q: str | None = Query(None),
    folder: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Search job definitions with optional query, folder and limit."""
    return await client.query_jobdefinitions(q, folder, limit)

@router.get("/model/jobdefinition/{jobdef_id}", summary="Get job definition")
async def get_jobdefinition(
    jobdef_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve a single job definition by its identifier."""
    return await client.get_jobdefinition(jobdef_id)

@router.get("/model/jobstream", summary="Query job streams")
async def query_jobstreams(
    q: str | None = Query(None),
    folder: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Search job streams with optional query, folder and limit."""
    return await client.query_jobstreams(q, folder, limit)

@router.get("/model/jobstream/{jobstream_id}", summary="Get job stream")
async def get_jobstream(
    jobstream_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve a single job stream by its identifier."""
    return await client.get_jobstream(jobstream_id)

@router.get("/model/workstation", summary="Query workstations")
async def query_workstations(
    q: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Search workstations with optional query and limit."""
    return await client.query_workstations(q, limit)

@router.get("/model/workstation/{workstation_id}", summary="Get workstation")
async def get_workstation(
    workstation_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve a specific workstation definition."""
    return await client.get_workstation(workstation_id)

# -------------------------------------------------------------------------
# Current plan routes – Jobs
# -------------------------------------------------------------------------
@router.get("/plan/current/job", summary="List current plan jobs")
async def list_current_plan_jobs(
    q: str | None = Query(None),
    folder: str | None = Query(None),
    status: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """
    List or search jobs in the current plan. Filters include a free form query,
    folder, status and limit.
    """
    return await client.query_current_plan_jobs(q, folder, status, limit)

@router.get("/plan/current/job/count", summary="Count current plan jobs")
async def count_current_plan_jobs(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the number of jobs present in the current plan."""
    return await client.get_current_plan_job_count()

@router.get("/plan/current/job/issues", summary="List current plan job issues")
async def current_plan_job_issues(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return issues detected in the current plan jobs."""
    return await client.get_current_plan_job_issues()

@router.get("/plan/current/job/joblog", summary="Retrieve current plan job logs")
async def current_plan_job_joblog(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve combined job logs for the current plan."""
    return await client.get_current_plan_job_joblog()

@router.get("/plan/current/job/{job_id}", summary="Get current plan job details")
async def get_current_plan_job(
    job_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve details about a specific job in the current plan."""
    return await client.get_current_plan_job(job_id)

@router.get(
    "/plan/current/job/{job_id}/predecessors",
    summary="Get current plan job predecessors",
)
async def get_current_plan_job_predecessors(
    job_id: str,
    depth: int | None = Query(None, ge=1, le=5),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the predecessors of a job in the current plan."""
    return await client.get_current_plan_job_predecessors(job_id, depth)

@router.get(
    "/plan/current/job/{job_id}/successors",
    summary="Get current plan job successors",
)
async def get_current_plan_job_successors(
    job_id: str,
    depth: int | None = Query(None, ge=1, le=5),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the successors of a job in the current plan."""
    return await client.get_current_plan_job_successors(job_id, depth)

@router.get(
    "/plan/current/job/{job_id}/model",
    summary="Get current plan job model",
)
async def get_current_plan_job_model(
    job_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the underlying model of a job in the current plan."""
    return await client.get_current_plan_job_model(job_id)

@router.get(
    "/plan/current/job/{job_id}/model/description",
    summary="Get current plan job model description",
)
async def get_current_plan_job_model_description(
    job_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the model description of a job in the current plan."""
    return await client.get_current_plan_job_model_description(job_id)

# -------------------------------------------------------------------------
# Current plan routes – Job Streams
# -------------------------------------------------------------------------
@router.get("/plan/current/jobstream", summary="List current plan job streams")
async def list_current_plan_jobstreams(
    q: str | None = Query(None),
    folder: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """List or search job streams in the current plan."""
    return await client.query_current_plan_jobstreams(q, folder, limit)

@router.get(
    "/plan/current/jobstream/count",
    summary="Count current plan job streams",
)
async def count_current_plan_jobstreams(
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the number of job streams present in the current plan."""
    return await client.get_current_plan_jobstream_count()

@router.get(
    "/plan/current/jobstream/{jobstream_id}",
    summary="Get current plan job stream details",
)
async def get_current_plan_jobstream(
    jobstream_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve details about a specific job stream in the current plan."""
    return await client.get_current_plan_jobstream(jobstream_id)

@router.get(
    "/plan/current/jobstream/{jobstream_id}/predecessors",
    summary="Get current plan job stream predecessors",
)
async def get_current_plan_jobstream_predecessors(
    jobstream_id: str,
    depth: int | None = Query(None, ge=1, le=5),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the predecessors of a job stream in the current plan."""
    return await client.get_current_plan_jobstream_predecessors(jobstream_id, depth)

@router.get(
    "/plan/current/jobstream/{jobstream_id}/successors",
    summary="Get current plan job stream successors",
)
async def get_current_plan_jobstream_successors(
    jobstream_id: str,
    depth: int | None = Query(None, ge=1, le=5),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the successors of a job stream in the current plan."""
    return await client.get_current_plan_jobstream_successors(jobstream_id, depth)

@router.get(
    "/plan/current/jobstream/{jobstream_id}/model/description",
    summary="Get current plan job stream model description",
)
async def get_current_plan_jobstream_model_description(
    jobstream_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Return the model description of a job stream in the current plan."""
    return await client.get_current_plan_jobstream_model_description(jobstream_id)

# -------------------------------------------------------------------------
# Current plan routes – Resources & Folders
# -------------------------------------------------------------------------
@router.get("/plan/current/resource", summary="List current plan resources")
async def list_current_plan_resources(
    q: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """List or search resources in the current plan."""
    return await client.query_current_plan_resources(q, limit)

@router.get(
    "/plan/current/resource/{resource_id}",
    summary="Get current plan resource details",
)
async def get_current_plan_resource(
    resource_id: str,
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """Retrieve details about a resource in the current plan."""
    return await client.get_current_plan_resource(resource_id)

@router.get(
    "/plan/current/folder/objects-count",
    summary="Count objects in a plan folder",
)
async def current_plan_folder_objects_count(
    folder: str | None = Query(None),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """
    Return the number of objects contained within the specified folder in the
    current plan. If no folder is provided the root folder is used.
    """
    return await client.get_current_plan_folder_objects_count(folder)

# -------------------------------------------------------------------------
# Current plan routes – Consumed jobs
# -------------------------------------------------------------------------
@router.get(
    "/plan/current/consumed-jobs/runs",
    summary="List consumed job runs in current plan",
)
async def list_consumed_jobs_runs(
    job_name: str | None = Query(None),
    limit: int | None = Query(50, ge=1, le=1000),
    client: OptimizedTWSClient = Depends(get_client),
) -> Any:
    """
    Return runs of consumed jobs in the current plan. Optionally filter by job
    name and limit.
    """
    return await client.get_consumed_jobs_runs(job_name, limit)
