"""
TWS Specialist Agents Module.

This module provides 4 specialized AI agents for TWS analysis:
- Job Analyst: Return codes, ABENDs, job execution analysis
- Dependency Specialist: Predecessors, successors, critical path
- Resource Specialist: CPU, memory, workstation conflicts
- Knowledge Specialist: Documentation RAG, troubleshooting guides

v5.4.2 Enhancements:
- Parallel executor for concurrent tool execution (PR-8)
- Observable tool run status (PR-9)
- Sub-agent pattern for delegated tasks (PR-10)
- Undo/rollback support (PR-11)
- Risk-based classification (PR-12)

Architecture:
    Query → Orchestrator → [Specialists in parallel] → Synthesizer → Response

Author: Resync Team
Version: 5.4.2
"""

from resync.core.specialists.agents import (
    DependencySpecialist,
    JobAnalystAgent,
    KnowledgeSpecialist,
    ResourceSpecialist,
    TWSSpecialistTeam,
    create_specialist_team,
    get_specialist_team,
)
from resync.core.specialists.models import (
    SpecialistConfig,
    SpecialistResponse,
    SpecialistType,
    TeamExecutionMode,
    TeamResponse,
)
from resync.core.specialists.parallel_executor import (
    ExecutionStrategy,
    ParallelToolExecutor,
    ToolRequest,
    ToolResponse,
    execute_tools_parallel,
    get_parallel_executor,
)
from resync.core.specialists.sub_agent import (
    SubAgent,
    SubAgentConfig,
    SubAgentResult,
    SubAgentStatus,
    dispatch_parallel_sub_agents,
    dispatch_sub_agent,
    register_sub_agent_tools,
)
from resync.core.specialists.tools import (
    ApprovalRequiredError,
    CalendarTool,
    DependencyGraphTool,
    ErrorCodeTool,
    JobLogTool,
    RiskLevel,
    ToolCatalog,
    ToolDefinition,
    ToolExecutionTrace,
    ToolPermission,
    ToolResult,
    ToolRun,
    ToolRunStatus,
    UserRole,
    WorkstationTool,
    calculate_risk_level,
    get_tool_catalog,
)

__all__ = [
    # Agents
    "JobAnalystAgent",
    "DependencySpecialist",
    "ResourceSpecialist",
    "KnowledgeSpecialist",
    "TWSSpecialistTeam",
    "create_specialist_team",
    "get_specialist_team",
    # Tools
    "JobLogTool",
    "ErrorCodeTool",
    "DependencyGraphTool",
    "WorkstationTool",
    "CalendarTool",
    # Tool Framework
    "ToolPermission",
    "UserRole",
    "ToolRunStatus",
    "ToolRun",
    "RiskLevel",
    "ToolResult",
    "ToolExecutionTrace",
    "ToolDefinition",
    "ToolCatalog",
    "ApprovalRequiredError",
    "get_tool_catalog",
    "calculate_risk_level",
    # Parallel Executor (PR-8)
    "ParallelToolExecutor",
    "ToolRequest",
    "ToolResponse",
    "ExecutionStrategy",
    "get_parallel_executor",
    "execute_tools_parallel",
    # Sub-Agent (PR-10)
    "SubAgent",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgentStatus",
    "dispatch_sub_agent",
    "dispatch_parallel_sub_agents",
    "register_sub_agent_tools",
    # Models
    "SpecialistConfig",
    "SpecialistType",
    "TeamExecutionMode",
    "SpecialistResponse",
    "TeamResponse",
]
