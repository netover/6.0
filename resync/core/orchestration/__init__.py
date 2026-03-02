"""Orchestration module for coordinating multi-agent workflows.

This package exports the public orchestration API.
"""

from __future__ import annotations

from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import EventType, OrchestrationEvent
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    TaskDefinition,
)
from resync.core.orchestration.strategies import (
    OrchestrationStrategy,
    ParallelStrategy,
    SequentialStrategy,
)

__all__ = [
    "AgentAdapter",
    "OrchestrationEvent",
    "EventType",
    "OrchestrationRunner",
    "OrchestrationRequest",
    "OrchestrationResponse",
    "TaskDefinition",
    "OrchestrationStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
]
