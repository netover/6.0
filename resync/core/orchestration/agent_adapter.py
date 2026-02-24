# ruff: noqa: E501
"""
Orchestration Agent Adapter

Adapts the orchestration engine to the existing AgentManager and HybridRouter.

v6.0: Updated to support skill_manager injection.
"""

import logging
from typing import Any, Dict, Optional

from resync.core.agent_manager import get_agent_manager
from resync.core.agent_router import HybridRouter, create_router
from resync.core.orchestration.schemas import StepConfig, StepType

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter to execute orchestration steps using system agents.
    """

    def __init__(self, skill_manager: Any = None):
        self.agent_manager = get_agent_manager()
        self._skill_manager = skill_manager
        self._router: Optional[HybridRouter] = None

    @property
    def router(self) -> HybridRouter:
        """Lazy load router to avoid circular imports during init."""
        if self._router is None:
            self._router = create_router(
                self.agent_manager, skill_manager=self._skill_manager
            )
        return self._router

    async def execute_step(
        self, step: StepConfig, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single orchestration step.

        Args:
            step: Step configuration
            input_data: Input data for this step (might be previous step output)
            context: Global execution context (trace_id, session_id, etc.)

        Returns:
            Dict containing output and metadata.
        """
        try:
            if step.type == StepType.AGENT:
                return await self._execute_agent_step(step, input_data, context)
            elif step.type == StepType.TOOL:
                return await self._execute_tool_step(step, input_data, context)
            elif step.type == StepType.LLM:
                # LLM step might just be an agent step with a specific specific model/role implied?
                # For now map to generic agent or router
                return await self._execute_agent_step(step, input_data, context)
            else:
                return {
                    "status": "skipped",
                    "output": f"Step type {step.type} not yet implemented",
                    "reason": "not_implemented",
                }

        except Exception as e:
            logger.error(f"Error executing step {step.id}: {e}", exc_info=True)
            raise

    async def _execute_agent_step(
        self, step: StepConfig, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step via an Agent or Router."""

        # Determine input message
        # If input_data has a 'message' or 'query', use it.
        # Otherwise dump the whole input as string (if it's not empty).
        # Or use a prompt template from step config.

        message = (
            input_data.get("message")
            or input_data.get("query")
            or input_data.get("input")
        )
        if not message and step.inputs:
            # If explicit inputs are defined in config, use them
            message = step.inputs.get("message")

        if not message:
            # If no message found, maybe it's just a trigger?
            # Or assume the prompt_template IS the message
            if step.prompt_template:
                message = step.prompt_template.format(**input_data)
            else:
                message = "Process current context"

        # Check if we target a specific agent
        agent_id = step.agent_id

        if agent_id and agent_id != "router":
            # Direct agent call
            agent = await self.agent_manager.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")

            # TODO: Pass context/skill_context if supported by Agent.arun
            # The current Agent.arun takes skill_context as string.
            skill_ctx = context.get("skill_context", "")

            response = await agent.arun(message, skill_context=skill_ctx)

            return {"output": response, "agent_id": agent_id, "model": agent.model}
        else:
            # Route via HybridRouter
            # We use 'chat_with_metadata' logic but calling route directly

            # Enrich context
            router_context = context.copy()
            if "history" in input_data:
                router_context["history"] = input_data["history"]

            result = await self.router.route(message, context=router_context)

            return {
                "output": result.response,
                "router_mode": result.routing_mode,
                "intent": result.intent,
                "confidence": result.confidence,
                "tools_used": result.tools_used,
                "processing_time_ms": result.processing_time_ms,
            }

    async def _execute_tool_step(
        self, step: StepConfig, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool directly."""
        tool_name = step.tool_name
        if not tool_name:
            raise ValueError("Tool name required for TOOL step")

        # Arguments can come from step.inputs (static) mixed with input_data (dynamic)
        tool_args = step.inputs.copy()
        tool_args.update(input_data)

        # We assume agent_manager has a way to call tools, currently it discovers them but doesn't expose public call_tool
        # logic explicitly except inside agents?
        # Actually AgentHandler has _call_tool.
        # But AgentManager has self.tools dict.

        tool_func = self.agent_manager.tools.get(tool_name)
        if not tool_func:
            raise ValueError(f"Tool {tool_name} not found")

        # Execute tool (sync or async?)
        # Most configured tools in AgentManager seem to be sync or async.
        # We need to inspect.
        import asyncio
        import inspect

        try:
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_args)
            else:
                result = await asyncio.to_thread(tool_func, **tool_args)

            return {"output": result, "tool_name": tool_name}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
