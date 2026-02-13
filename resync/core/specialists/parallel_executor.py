"""
Parallel Tool Execution Engine.

PR-8: Implements smart parallel execution strategy:
- Read-only tools run concurrently for maximum performance
- Stateful (write/execute) tools run serially for safety
- Results are reordered to match original request sequence

Based on patterns from Claude Code and Amp.

Author: Resync Team
Version: 5.4.2
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .tools import (
    RiskLevel,
    ToolCatalog,
    ToolDefinition,
    ToolResult,
    ToolRunStatus,
    calculate_risk_level,
    get_tool_catalog,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# TOOL REQUEST/RESPONSE
# =============================================================================


@dataclass
class ToolRequest:
    """A request to execute a tool."""

    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    # Context
    user_id: str | None = None
    session_id: str | None = None

    # Original position (for reordering)
    original_index: int = 0


@dataclass
class ToolResponse:
    """Response from tool execution."""

    request_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None

    # Timing
    duration_ms: int = 0

    # Original position
    original_index: int = 0

    # Execution details
    trace_id: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    can_undo: bool = False


class ExecutionStrategy(str, Enum):
    """Strategy for executing tool requests."""

    CONCURRENT = "concurrent"  # All tools run in parallel
    SERIAL = "serial"  # All tools run sequentially
    SMART = "smart"  # Read-only parallel, stateful serial


# =============================================================================
# PARALLEL EXECUTOR
# =============================================================================


class ParallelToolExecutor:
    """
    Executes multiple tools with smart parallelization.

    Key insight: Read operations can run in parallel,
    but write operations need careful coordination.

    Usage:
        executor = ParallelToolExecutor()
        results = await executor.execute([
            ToolRequest("get_job_log", {"job_name": "JOB1"}),
            ToolRequest("get_job_log", {"job_name": "JOB2"}),
            ToolRequest("get_workstation_status", {"ws_name": "WS1"}),
        ])
        # All three run concurrently, results in original order
    """

    def __init__(
        self,
        catalog: ToolCatalog | None = None,
        max_concurrent: int = 10,
        default_timeout: float = 30.0,
    ):
        self.catalog = catalog or get_tool_catalog()
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        requests: list[ToolRequest],
        strategy: ExecutionStrategy = ExecutionStrategy.SMART,
        user_role: str = "operator",
    ) -> list[ToolResponse]:
        """
        Execute multiple tool requests.

        Args:
            requests: List of tool requests
            strategy: Execution strategy
            user_role: User role for permission checking

        Returns:
            List of responses in original request order
        """
        if not requests:
            return []

        # Tag requests with original index
        for i, req in enumerate(requests):
            req.original_index = i
            if not req.request_id:
                req.request_id = f"req_{i}_{int(time.time() * 1000)}"

        start_time = time.time()

        if strategy == ExecutionStrategy.CONCURRENT:
            responses = await self._execute_concurrent(requests)
        elif strategy == ExecutionStrategy.SERIAL:
            responses = await self._execute_serial(requests)
        else:  # SMART
            responses = await self._execute_smart(requests)

        # Sort by original index
        responses.sort(key=lambda r: r.original_index)

        total_duration = int((time.time() - start_time) * 1000)
        logger.info(
            "parallel_execution_complete",
            total_requests=len(requests),
            strategy=strategy.value,
            total_duration_ms=total_duration,
            success_count=sum(1 for r in responses if r.success),
        )

        return responses

    async def _execute_smart(self, requests: list[ToolRequest]) -> list[ToolResponse]:
        """
        Smart execution: read-only parallel, stateful serial.
        """
        read_only_requests = []
        stateful_requests = []

        for req in requests:
            if self.catalog.is_read_only(req.tool_name):
                read_only_requests.append(req)
            else:
                stateful_requests.append(req)

        logger.debug(
            "smart_execution_split",
            read_only_count=len(read_only_requests),
            stateful_count=len(stateful_requests),
        )

        # Execute read-only in parallel
        read_only_responses = await self._execute_concurrent(read_only_requests)

        # Execute stateful serially
        stateful_responses = await self._execute_serial(stateful_requests)

        return read_only_responses + stateful_responses

    async def _execute_concurrent(self, requests: list[ToolRequest]) -> list[ToolResponse]:
        """Execute all requests concurrently."""
        if not requests:
            return []

        tasks = [self._execute_single_with_semaphore(req) for req in requests]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        result = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                result.append(
                    ToolResponse(
                        request_id=requests[i].request_id,
                        tool_name=requests[i].tool_name,
                        success=False,
                        error=str(resp),
                        original_index=requests[i].original_index,
                    )
                )
            else:
                result.append(resp)

        return result

    async def _execute_serial(self, requests: list[ToolRequest]) -> list[ToolResponse]:
        """Execute all requests serially."""
        responses = []
        for req in requests:
            try:
                resp = await self._execute_single(req)
                responses.append(resp)
            except Exception as e:
                responses.append(
                    ToolResponse(
                        request_id=req.request_id,
                        tool_name=req.tool_name,
                        success=False,
                        error=str(e),
                        original_index=req.original_index,
                    )
                )
        return responses

    async def _execute_single_with_semaphore(self, request: ToolRequest) -> ToolResponse:
        """Execute a single tool with semaphore for concurrency control."""
        async with self._semaphore:
            return await self._execute_single(request)

    async def _execute_single(self, request: ToolRequest) -> ToolResponse:
        """Execute a single tool request."""
        start_time = time.time()

        tool = self.catalog.get(request.tool_name)
        if not tool:
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=f"Tool '{request.tool_name}' not found",
                original_index=request.original_index,
            )

        # Create run for tracking
        run = self.catalog.create_run(request.tool_name)
        self.catalog.update_run_status(run.run_id, ToolRunStatus.IN_PROGRESS)

        # Calculate risk
        risk_level = calculate_risk_level(
            request.tool_name,
            request.parameters,
            tool.permission,
        )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_tool(tool, request.parameters),
                timeout=tool.timeout_seconds or self.default_timeout,
            )

            duration = int((time.time() - start_time) * 1000)
            self.catalog.update_run_status(run.run_id, ToolRunStatus.DONE)

            # Check if result is ToolResult with undo
            can_undo = False
            if isinstance(result, ToolResult):
                can_undo = result.undo_fn is not None
                if can_undo:
                    self.catalog.register_undoable(run.run_id, result)
                result = result.result

            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result,
                duration_ms=duration,
                original_index=request.original_index,
                trace_id=run.run_id,
                risk_level=risk_level,
                can_undo=can_undo,
            )

        except asyncio.TimeoutError:
            self.catalog.update_run_status(
                run.run_id,
                ToolRunStatus.ERROR,
                error=f"Timeout after {tool.timeout_seconds}s",
            )
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=f"Timeout after {tool.timeout_seconds} seconds",
                duration_ms=int((time.time() - start_time) * 1000),
                original_index=request.original_index,
                trace_id=run.run_id,
                risk_level=risk_level,
            )

        except Exception as e:
            self.catalog.update_run_status(
                run.run_id,
                ToolRunStatus.ERROR,
                error=str(e),
            )
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
                original_index=request.original_index,
                trace_id=run.run_id,
                risk_level=risk_level,
            )

    async def _call_tool(self, tool: ToolDefinition, params: dict[str, Any]) -> Any:
        """Call a tool function (handles sync and async)."""
        func = tool.function

        if asyncio.iscoroutinefunction(func):
            return await func(**params)
        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**params))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_executor: ParallelToolExecutor | None = None


def get_parallel_executor() -> ParallelToolExecutor:
    """Get the global parallel executor instance."""
    global _executor
    if _executor is None:
        _executor = ParallelToolExecutor()
    return _executor


async def execute_tools_parallel(
    tool_requests: list[dict[str, Any]],
    strategy: str = "smart",
) -> list[dict[str, Any]]:
    """
    Convenience function to execute tools in parallel.

    Args:
        tool_requests: List of {"tool_name": str, "parameters": dict}
        strategy: "concurrent", "serial", or "smart"

    Returns:
        List of result dicts
    """
    executor = get_parallel_executor()

    requests = [
        ToolRequest(
            tool_name=req["tool_name"],
            parameters=req.get("parameters", {}),
        )
        for req in tool_requests
    ]

    strategy_enum = ExecutionStrategy(strategy)
    responses = await executor.execute(requests, strategy=strategy_enum)

    return [
        {
            "tool_name": r.tool_name,
            "success": r.success,
            "result": r.result,
            "error": r.error,
            "duration_ms": r.duration_ms,
            "trace_id": r.trace_id,
            "risk_level": r.risk_level.value,
            "can_undo": r.can_undo,
        }
        for r in responses
    ]
