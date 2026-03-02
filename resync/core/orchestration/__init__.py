"""Orchestration module for coordinating multi-agent workflows.

This package exports the public orchestration API.
"""

from __future__ import annotations

from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import EventType, OrchestrationEvent
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.schemas import (
    StepConfig,
    StepDependency,
    StepType,
    WorkflowConfig,
)
from resync.core.orchestration.strategies import (
    ExecutionStrategy,
    ParallelStrategy,
    SequentialStrategy,
)

# Backward-compatible alias used by older imports.
OrchestrationStrategy = ExecutionStrategy

__all__ = [
    "AgentAdapter",
    "OrchestrationEvent",
    "EventType",
    "OrchestrationRunner",
    "StepType",
    "StepDependency",
    "StepConfig",
    "WorkflowConfig",
    "OrchestrationStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
]
