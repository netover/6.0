"""Orchestration module for coordinating multi-agent workflows.

This module provides:
- agent_adapter: Adapter for integrating external agents
- events: Event definitions for orchestration
- runner: Execution runner for orchestration tasks
- schemas: Pydantic schemas for orchestration
- strategies: Orchestration strategies
"""

from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import (
    OrchestrationEvent,
    EventType,
)
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    TaskDefinition,
)
from resync.core.orchestration.strategies import (
    OrchestrationStrategy,
    SequentialStrategy,
    ParallelStrategy,
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

This module provides:
- agent_adapter: Adapter for integrating external agents
- events: Event definitions for orchestration
- runner: Execution runner for orchestration tasks
- schemas: Pydantic schemas for orchestration
- strategies: Orchestration strategies
"""

from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import (
    OrchestrationEvent,
    EventType,
)
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    TaskDefinition,
)
from resync.core.orchestration.strategies import (
    OrchestrationStrategy,
    SequentialStrategy,
    ParallelStrategy,
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

This module provides:
- agent_adapter: Adapter for integrating external agents
- events: Event definitions for orchestration
- runner: Execution runner for orchestration tasks
- schemas: Pydantic schemas for orchestration
- strategies: Orchestration strategies
"""

from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import (
    OrchestrationEvent,
    EventType,
)
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
    TaskDefinition,
)
from resync.core.orchestration.strategies import (
    OrchestrationStrategy,
    SequentialStrategy,
    ParallelStrategy,
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

