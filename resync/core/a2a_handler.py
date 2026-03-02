"""A2A (Agent-to-Agent) Protocol Handler.

Manages task lifecycle, JSON-RPC routing, and agent delegation.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog
from resync.models.a2a import JsonRpcRequest, JsonRpcResponse, TaskState

logger = structlog.get_logger(__name__)


class AgentManagerProtocol(Protocol):
    """Protocol for agent manager interface."""
    
    async def get_agent(self, agent_id: str) -> Any:
        """Get agent by ID."""
        ...


class A2ATask:
    """Represents an A2A task in the system."""

    def __init__(
        self, task_id: str, agent_id: str, method: str, params: dict[str, Any]
    ):
        self.task_id = task_id
        self.agent_id = agent_id
        self.method = method
        self.params = params
        self.state = TaskState.SUBMITTED
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.result: Any = None
        self.error: str | None = None

    def update_state(
        self, state: TaskState, result: Any = None, error: str | None = None
    ):
        """Update task state and timestamp."""
        self.state = state
        self.updated_at = datetime.now(timezone.utc)
        if result is not None:
            self.result = result
        if error is not None:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for responses."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "method": self.method,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class A2AHandler:
    """Handles JSON-RPC requests and manages A2A task execution."""

    def __init__(self, agent_manager: AgentManagerProtocol):
        self.agent_manager = agent_manager
        self._tasks: dict[str, A2ATask] = {}
        self._tasks_lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=1000
        )

    async def handle_request(
        self, agent_id: str, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Process an incoming A2A JSON-RPC request."""
        task_id = str(uuid.uuid4())
        task = A2ATask(task_id, agent_id, request.method, request.params or {})
        
        async with self._tasks_lock:
            self._tasks[task_id] = task

        # Log submission
        logger.info(
            "a2a_task_submitted",
            agent_id=agent_id,
            method=request.method,
            task_id=task_id,
        )
        await self._publish_event("task_submitted", task)

        # Execute based on method
        try:
            task.update_state(TaskState.WORKING)
            await self._publish_event("task_working", task)

            result = await self._execute_method(
                agent_id, request.method, request.params or {}
            )

            task.update_state(TaskState.COMPLETED, result=result)
            await self._publish_event("task_completed", task)

            return JsonRpcResponse(result=result, id=request.id)

        except (ValueError, KeyError, AttributeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "a2a_task_failed",
                agent_id=agent_id,
                method=request.method,
                error=str(e),
            )
            task.update_state(TaskState.FAILED, error=str(e))
            await self._publish_event("task_failed", task)

            return JsonRpcResponse(
                error={"code": -32603, "message": str(e)}, id=request.id
            )

    async def _execute_method(
        self, agent_id: str, method: str, params: dict[str, Any]
    ) -> Any:
        """Internal dispatch for agent methods."""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        # Mapping generic methods to Resync agent actions
        if method == "chat":
            message = params.get("message", "")
            return await agent.arun(message)

        # Fallback to direct tool-like call if possible (simplified for now)
        return await agent.arun(f"Execute action: {method} with params {params}")

    async def _publish_event(self, event_type: str, task: A2ATask):
        """Publish event to internal queue for SSE/WebSockets."""
        event = {"type": "a2a_event", "event": event_type, "task": task.to_dict()}
        try:
            await asyncio.wait_for(self._event_queue.put(event), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Event queue full, dropping event", event_type=event_type)

    async def get_event_stream(self):
        """Generator for SSE events."""
        while True:
            event = await self._event_queue.get()
            yield event
