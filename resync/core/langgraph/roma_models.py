"""State and payload models for ROMA orchestration graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """A single decomposed task for ROMA planning."""

    id: str = Field(description="Unique task identifier")
    title: str = Field(description="Short task title")
    description: str = Field(description="Task execution details")
    status: Literal["pending", "done", "failed"] = Field(default="pending")


class RomaState(TypedDict, total=False):
    """State shared by ROMA graph nodes."""

    # input
    user_query: str

    # decomposition
    is_atomic: bool
    plan: list[SubTask]

    # execution
    execution_results: list[dict[str, Any]]

    # output
    final_response: str
    verification_notes: list[str]

    # observability
    execution_logs: Annotated[list[str], operator.add]
