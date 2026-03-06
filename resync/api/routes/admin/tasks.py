from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.task_registry import get_task_stats
from resync.core.loop_utils import get_loop_stats


router = APIRouter(prefix="/tasks", tags=["Admin - Tasks"])


class TaskStatsResponse(BaseModel):
    total: int = 0
    running: int = 0
    done: int = 0
    cancelled: int = 0
    sample_names: list[str] = Field(default_factory=list)
    loop_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    note: str = "Counts refer to tracked background tasks created via create_tracked_task."


@router.get(
    "/stats",
    response_model=TaskStatsResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_tasks_stats() -> TaskStatsResponse:
    data = get_task_stats(sample_limit=50)
    data["loop_stats"] = get_loop_stats()
    return TaskStatsResponse(**data)  # type: ignore[arg-type]
