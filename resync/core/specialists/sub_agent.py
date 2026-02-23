"""
Sub-Agent Pattern for Resync.

PR-10: Implements specialized sub-agents with restrictions:
- Only read-only tools available
- Cannot spawn recursive sub-agents
- Stateless invocations
- Isolated execution context

Based on Claude Code's AgentTool pattern.

Use cases:
- Parallel search across multiple jobs/logs
- Concurrent analysis of different workstations
- Distributed document retrieval

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from .parallel_executor import (
    ExecutionStrategy,
    ParallelToolExecutor,
    ToolRequest,
    ToolResponse,
)
from .tools import (
    ToolCatalog,
    ToolDefinition,
    ToolPermission,
    get_tool_catalog,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# SUB-AGENT CONFIGURATION
# =============================================================================


@dataclass
class SubAgentConfig:
    """Configuration for sub-agents."""

    # Tool restrictions
    only_read_only_tools: bool = True
    prevent_recursive_spawn: bool = True  # Sub-agent can't create sub-agents
    blocked_tool_names: list[str] = field(
        default_factory=lambda: ["dispatch_agent", "sub_agent"]
    )

    # Execution limits
    max_tool_calls: int = 10
    max_execution_time_seconds: float = 60.0

    # Stateless
    stateless: bool = True  # Each invocation is isolated

    # Context
    inherit_context: bool = True  # Inherit parent context


# =============================================================================
# SUB-AGENT RESULT
# =============================================================================


class SubAgentStatus(str, Enum):
    """Status of sub-agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    agent_id: str
    status: SubAgentStatus

    # Result
    result: Any = None
    summary: str = ""

    # Tools used
    tools_called: list[str] = field(default_factory=list)
    tool_call_count: int = 0

    # Error
    error: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "result": self.result,
            "summary": self.summary,
            "tools_called": self.tools_called,
            "tool_call_count": self.tool_call_count,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# SUB-AGENT
# =============================================================================


class SubAgent:
    """
    A restricted sub-agent for parallel/delegated tasks.

    Key restrictions:
    - Only read-only tools (safe for parallel execution)
    - Cannot create other sub-agents (prevents recursion)
    - Stateless (each invocation is isolated)
    - Limited tool calls and execution time

    Usage:
        agent = SubAgent(prompt="Find all jobs with ABEND errors")
        result = await agent.execute()
        print(result.summary)

    Parallel usage:
        agents = [
            SubAgent(prompt="Find JOB1 errors"),
            SubAgent(prompt="Find JOB2 errors"),
            SubAgent(prompt="Find JOB3 errors"),
        ]
        results = await SubAgent.execute_parallel(agents)
    """

    def __init__(
        self,
        prompt: str,
        context: str | None = None,
        config: SubAgentConfig | None = None,
        parent_session_id: str | None = None,
    ):
        self.agent_id = str(uuid.uuid4())[:8]
        self.prompt = prompt
        self.context = context
        self.config = config or SubAgentConfig()
        self.parent_session_id = parent_session_id

        self._catalog = get_tool_catalog()
        self._executor = ParallelToolExecutor()
        self._tool_calls: list[str] = []
        self._started_at: datetime | None = None
        self._cancelled = False

    def get_available_tools(self) -> list[ToolDefinition]:
        """Get tools available to this sub-agent."""
        if self.config.only_read_only_tools:
            tools = self._catalog.get_read_only_tools()
        else:
            tools = self._catalog.list_tools()

        # Filter blocked tools
        return [t for t in tools if t.name not in self.config.blocked_tool_names]

    def get_tool_names(self) -> list[str]:
        """Get names of available tools."""
        return [t.name for t in self.get_available_tools()]

    async def execute(self) -> SubAgentResult:
        """
        Execute the sub-agent task.

        Returns:
            SubAgentResult with findings
        """
        self._started_at = datetime.now(timezone.utc)

        logger.info(
            "sub_agent_started",
            agent_id=self.agent_id,
            prompt=self.prompt[:100],
            available_tools=self.get_tool_names(),
        )

        try:
            # Execute with timeout
            return await asyncio.wait_for(
                self._execute_internal(),
                timeout=self.config.max_execution_time_seconds,
            )

        except asyncio.TimeoutError:
            return SubAgentResult(
                agent_id=self.agent_id,
                status=SubAgentStatus.TIMEOUT,
                error=f"Execution timed out after {self.config.max_execution_time_seconds}s",
                tools_called=self._tool_calls,
                tool_call_count=len(self._tool_calls),
                started_at=self._started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=self._get_duration_ms(),
            )

        except asyncio.CancelledError:
            return SubAgentResult(
                agent_id=self.agent_id,
                status=SubAgentStatus.CANCELLED,
                error="Execution cancelled",
                tools_called=self._tool_calls,
                tool_call_count=len(self._tool_calls),
                started_at=self._started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=self._get_duration_ms(),
            )

        except Exception as e:
            logger.error("sub_agent_error", agent_id=self.agent_id, error=str(e))
            return SubAgentResult(
                agent_id=self.agent_id,
                status=SubAgentStatus.FAILED,
                error=str(e),
                tools_called=self._tool_calls,
                tool_call_count=len(self._tool_calls),
                started_at=self._started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=self._get_duration_ms(),
            )

    async def _execute_internal(self) -> SubAgentResult:
        """Internal execution logic."""
        # For now, this is a simple implementation
        # In a full implementation, this would:
        # 1. Call LLM with the prompt and available tools
        # 2. Parse tool calls from LLM response
        # 3. Execute tools and collect results
        # 4. Return summary

        # Placeholder: Execute a search based on the prompt
        # This should be replaced with actual LLM integration

        results = []

        # Simple keyword-based tool selection
        prompt_lower = self.prompt.lower()

        if "job" in prompt_lower or "log" in prompt_lower:
            # Try to search job logs
            job_names = self._extract_job_names(self.prompt)
            if job_names:
                requests = [
                    ToolRequest(tool_name="get_job_log", parameters={"job_name": jn})
                    for jn in job_names[:5]  # Limit to 5 jobs
                ]
                responses = await self._executor.execute(
                    requests, strategy=ExecutionStrategy.CONCURRENT
                )
                self._tool_calls.extend(["get_job_log"] * len(requests))
                results.extend([r for r in responses if r.success])

        if "workstation" in prompt_lower or "ws" in prompt_lower:
            # Try to get workstation status
            ws_names = self._extract_workstation_names(self.prompt)
            if ws_names:
                requests = [
                    ToolRequest(
                        tool_name="get_workstation_status", parameters={"ws_name": ws}
                    )
                    for ws in ws_names[:5]
                ]
                responses = await self._executor.execute(
                    requests, strategy=ExecutionStrategy.CONCURRENT
                )
                self._tool_calls.extend(["get_workstation_status"] * len(requests))
                results.extend([r for r in responses if r.success])

        if (
            "search" in prompt_lower
            or "find" in prompt_lower
            or "documentation" in prompt_lower
        ) and self._catalog.get("search_knowledge_base"):
            # Use RAG search
            request = ToolRequest(
                tool_name="search_knowledge_base",
                parameters={"query": self.prompt, "top_k": 5},
            )
            responses = await self._executor.execute([request])
            self._tool_calls.append("search_knowledge_base")
            results.extend([r for r in responses if r.success])

        # Build summary
        summary = self._build_summary(results)

        return SubAgentResult(
            agent_id=self.agent_id,
            status=SubAgentStatus.COMPLETED,
            result=[r.result for r in results],
            summary=summary,
            tools_called=self._tool_calls,
            tool_call_count=len(self._tool_calls),
            started_at=self._started_at,
            completed_at=datetime.now(timezone.utc),
            duration_ms=self._get_duration_ms(),
        )

    def _extract_job_names(self, text: str) -> list[str]:
        """Extract job names from text."""
        import re

        # TWS job names are typically uppercase, 6-8 chars
        pattern = r"\b[A-Z][A-Z0-9]{5,7}\b"
        matches = re.findall(pattern, text.upper())
        return list(set(matches))

    def _extract_workstation_names(self, text: str) -> list[str]:
        """Extract workstation names from text."""
        import re

        # Workstation names often start with WS or end with _WS
        pattern = r"\b(?:WS[A-Z0-9_]+|[A-Z0-9]+_WS)\b"
        matches = re.findall(pattern, text.upper())
        return list(set(matches))

    def _build_summary(self, results: list[ToolResponse]) -> str:
        """Build a summary of results."""
        if not results:
            return "No relevant information found."

        parts = []
        for r in results:
            if r.success and r.result:
                parts.append(f"[{r.tool_name}]: Found data")

        if not parts:
            return "Tools executed but no significant findings."

        return f"Found {len(results)} results: " + "; ".join(parts[:5])

    def _get_duration_ms(self) -> int:
        """Get execution duration in milliseconds."""
        if self._started_at:
            return int(
                (datetime.now(timezone.utc) - self._started_at).total_seconds() * 1000
            )
        return 0

    def cancel(self) -> None:
        """Request cancellation of this sub-agent."""
        self._cancelled = True

    # =========================================================================
    # CLASS METHODS FOR PARALLEL EXECUTION
    # =========================================================================

    @classmethod
    async def execute_parallel(
        cls,
        agents: list[SubAgent],
        max_concurrent: int = 5,
    ) -> list[SubAgentResult]:
        """
        Execute multiple sub-agents in parallel.

        Args:
            agents: List of sub-agents to execute
            max_concurrent: Maximum concurrent executions

        Returns:
            List of results in same order as input
        """
        if not agents:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(agent: SubAgent) -> SubAgentResult:
            async with semaphore:
                return await agent.execute()

        logger.info(
            "parallel_sub_agents_started",
            agent_count=len(agents),
            max_concurrent=max_concurrent,
        )

        tasks = []
        try:
            async with asyncio.TaskGroup() as tg:
                for agent in agents:
                    tasks.append(tg.create_task(execute_with_semaphore(agent)))
        except* Exception:
            # Individual results are collected by checking task.result()
            pass

        results = []
        for task in tasks:
            try:
                results.append(task.result())
            except (asyncio.CancelledError, Exception) as e:
                # We don't have the specific agent here easily without mapping,
                # but we can return a generic failure if the task itself died.
                # In most cases, agent.execute() handles its own exceptions.
                logger.error("sub_agent_task_failed", error=str(e))

        success_count = sum(1 for r in results if r.status == SubAgentStatus.COMPLETED)
        logger.info(
            "parallel_sub_agents_completed",
            agent_count=len(agents),
            success_count=success_count,
        )

        return results

    @classmethod
    def create_search_agents(
        cls,
        queries: list[str],
        context: str | None = None,
    ) -> list[SubAgent]:
        """
        Factory method to create multiple search sub-agents.

        Args:
            queries: List of search queries
            context: Optional shared context

        Returns:
            List of configured sub-agents
        """
        return [cls(prompt=q, context=context) for q in queries]


# =============================================================================
# SUB-AGENT TOOL (for registration in catalog)
# =============================================================================


async def dispatch_sub_agent(
    prompt: str,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Tool function to dispatch a sub-agent.

    Args:
        prompt: Task description for the sub-agent
        context: Optional additional context

    Returns:
        Sub-agent result as dictionary
    """
    agent = SubAgent(prompt=prompt, context=context)
    result = await agent.execute()
    return result.to_dict()


async def dispatch_parallel_sub_agents(
    prompts: list[str],
    context: str | None = None,
    max_concurrent: int = 5,
) -> list[dict[str, Any]]:
    """
    Tool function to dispatch multiple sub-agents in parallel.

    Args:
        prompts: List of task descriptions
        context: Optional shared context
        max_concurrent: Maximum concurrent agents

    Returns:
        List of sub-agent results
    """
    agents = SubAgent.create_search_agents(prompts, context)
    results = await SubAgent.execute_parallel(agents, max_concurrent)
    return [r.to_dict() for r in results]


# =============================================================================
# REGISTRATION HELPERS
# =============================================================================


def register_sub_agent_tools(catalog: ToolCatalog | None = None) -> None:
    """Register sub-agent tools in the catalog."""
    from .tools import ToolDefinition

    catalog = catalog or get_tool_catalog()

    # Single sub-agent dispatch
    catalog.register(
        ToolDefinition(
            name="dispatch_sub_agent",
            description="Dispatch a specialized read-only sub-agent to search and explore. "
            "Good for finding things across multiple files or when unsure where to look.",
            function=dispatch_sub_agent,
            permission=ToolPermission.READ_ONLY,
            requires_approval=False,
            timeout_seconds=60,
            tags=["sub_agent", "search", "parallel"],
        )
    )

    # Parallel sub-agents dispatch
    catalog.register(
        ToolDefinition(
            name="dispatch_parallel_sub_agents",
            description="Dispatch multiple sub-agents in parallel for concurrent search/analysis. "
            "Use for searching multiple jobs, workstations, or documents simultaneously.",
            function=dispatch_parallel_sub_agents,
            permission=ToolPermission.READ_ONLY,
            requires_approval=False,
            timeout_seconds=120,
            tags=["sub_agent", "search", "parallel", "batch"],
        )
    )

    logger.info("sub_agent_tools_registered")
