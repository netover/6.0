# ruff: noqa: E501
# pylint
"""
LangGraph Node Definitions.

This module provides reusable node classes for building agent graphs.
Each node represents a discrete step in the agent workflow.

Node Types:
- RouterNode: Intent classification and routing
- LLMNode: LLM interaction with prompt management
- ToolNode: Tool execution with validation
- ValidationNode: Output validation and error handling
- HumanApprovalNode: Human-in-the-loop approval flow
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# [FIX] Import Central Config
from resync.core.llm_config import get_llm_config
from resync.core.redis_init import get_redis_client
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# BASE NODE
# =============================================================================

class BaseNode(ABC):
    """
    Base class for graph nodes.

    All nodes must implement the `__call__` method which takes
    the current state and returns the updated state.
    """

    name: str = "base_node"

    def __init__(self, name: str | None = None):
        if name:
            self.name = name

    @abstractmethod
    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the node logic."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"

# =============================================================================
# ROUTER NODE
# =============================================================================

@dataclass
class RouterConfig:
    """Configuration for the router node."""

    # Classification method
    use_llm: bool = True
    fallback_to_keywords: bool = True

    # LLM settings
    # [FIX] Resolved dynamically via LLMConfig if not provided
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 20

    # Prompt
    prompt_id: str = "intent-router-v1"

    # Keyword patterns (for fallback)
    keyword_patterns: dict[str, list[str]] = None

    def __post_init__(self):
        if self.keyword_patterns is None:
            self.keyword_patterns = {
                "status": ["status", "estado", "workstation", "online", "offline"],
                "troubleshoot": ["erro", "error", "falha", "abend", "problema"],
                "action": ["cancelar", "reiniciar", "executar", "submit"],
                "query": ["como", "o que", "qual", "porque", "documentação"],
            }

        # [FIX] v5.9.9: Resolve model dynamically via LLMConfig. NEVER use hardcoded strings.
        if self.model is None:
            self.model = get_llm_config().get_model(task_type="classification")

class RouterNode(BaseNode):
    """
    Router node for intent classification.

    Uses LLM or keyword matching to classify user intent
    and route to the appropriate handler.

    Usage:
        router = RouterNode(config)
        state = await router(state)
        next_node = state["_next"]  # e.g., "status_handler"
    """

    name = "router"

    def __init__(self, config: RouterConfig | None = None):
        super().__init__()
        self.config = config or RouterConfig()

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Classify intent and set routing."""
        message = state.get("message", "")

        if self.config.use_llm:
            try:
                intent, confidence = await self._classify_with_llm(message)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("router_llm_failed", error=str(e))
                if self.config.fallback_to_keywords:
                    intent, confidence = self._classify_with_keywords(message)
                else:
                    intent, confidence = "general", 0.5
        else:
            intent, confidence = self._classify_with_keywords(message)

        state["intent"] = intent
        state["confidence"] = confidence
        state["_next"] = f"{intent}_handler"
        state["current_node"] = self.name

        logger.debug(
            "router_classified",
            intent=intent,
            confidence=confidence,
            next_node=state["_next"],
        )

        return state

    async def _classify_with_llm(self, message: str) -> tuple[str, float]:
        """Use LLM for classification via LiteLLM (project standard)."""
        from resync.core.langfuse import get_prompt_manager
        from resync.core.utils.llm import call_llm

        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.get_prompt(self.config.prompt_id)

        if not prompt:
            raise ValueError(f"Prompt {self.config.prompt_id} not found")

        compiled = prompt.compile(user_message=message)

        # Use LiteLLM via call_llm (project standard with resilience)
        response = await call_llm(
            prompt=compiled,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        intent = response.strip().lower()

        # Map to valid intents
        valid_intents = ["status", "troubleshoot", "query", "action", "general"]
        if intent not in valid_intents:
            intent = "general"

        return intent, 0.85

    def _classify_with_keywords(self, message: str) -> tuple[str, float]:
        """Use keyword matching for classification."""
        message_lower = message.lower()

        for intent, keywords in self.config.keyword_patterns.items():
            if any(kw in message_lower for kw in keywords):
                return intent, 0.7

        return "general", 0.5

# =============================================================================
# LLM NODE
# =============================================================================

@dataclass
class LLMNodeConfig:
    """Configuration for LLM nodes."""

    # Prompt
    prompt_id: str = ""
    prompt_variables: dict[str, str] = None

    # Model settings
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000

    # Context
    include_history: bool = True
    max_history_messages: int = 5

    def __post_init__(self):
        if self.prompt_variables is None:
            self.prompt_variables = {}

class LLMNode(BaseNode):
    """
    LLM interaction node.

    Handles prompt compilation, LLM calls, and response processing.
    Uses LangFuse for prompt management and tracing.

    Usage:
        llm_node = LLMNode(config)
        state = await llm_node(state)
        response = state["llm_response"]
    """

    name = "llm"

    def __init__(self, config: LLMNodeConfig, name: str | None = None):
        super().__init__(name)
        self.config = config

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute LLM call via project standard (LiteLLM + call_llm with resilience)."""
        from resync.core.langfuse import get_prompt_manager, get_tracer
        from resync.core.utils.llm import call_llm

        # Get prompt
        prompt_manager = get_prompt_manager()
        prompt = await prompt_manager.get_prompt(self.config.prompt_id)

        # [FIX] Resolution Hierarchy: Config Override > Prompt Hint > Central Config Default
        # Decoupled from resync.settings to ensure core logic remains provider-agnostic.
        model = (
            self.config.model
            or (prompt.model_hint if prompt else None)
            or get_llm_config().get_model()
        )

        # Build messages
        messages = []
        message = state.get("message", "")
        history = state.get("history", [])

        if prompt:
            # Compile prompt with variables from state and config
            variables = {**self.config.prompt_variables}
            for key in prompt.config.variables:
                if key in state:
                    variables[key] = state[key]

            system_content = prompt.compile(**variables)
            messages.append({"role": "system", "content": system_content})

        # Add history
        if self.config.include_history and history:
            messages.extend(history[-self.config.max_history_messages :])

        # Add user message
        messages.append({"role": "user", "content": message})

        # Call LLM with tracing via project standard (call_llm with LiteLLM + resilience)
        tracer = get_tracer()

        # Build full prompt from messages for call_llm
        full_prompt = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in messages]
        )

        async with tracer.trace(
            self.name, model=model, prompt_id=self.config.prompt_id
        ) as trace:
            # Use call_llm which has circuit breaker, retry, and timeout built-in
            response = await call_llm(
                prompt=full_prompt,
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            trace.output = response

        state["llm_response"] = response
        state["llm_messages"] = messages
        state["current_node"] = self.name

        return state

# =============================================================================
# TOOL NODE
# =============================================================================

@dataclass
class ToolNodeConfig:
    """Configuration for tool nodes."""

    # Tool settings
    tool_name: str = ""
    validate_input: bool = True
    validate_output: bool = True

    # Retry
    max_retries: int = 3
    retry_delay_ms: int = 500

    # Timeout
    timeout_seconds: float = 30.0

class ToolNode(BaseNode):
    """
    Tool execution node.

    Executes registered tools with validation and retry logic.

    Usage:
        tool_node = ToolNode(config, tool_func)
        state = await tool_node(state)
        result = state["tool_output"]
    """

    name = "tool"

    def __init__(
        self,
        config: ToolNodeConfig,
        tool_func: Callable,
        name: str | None = None,
    ):
        super().__init__(name or config.tool_name or "tool")
        self.config = config
        self.tool_func = tool_func

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool."""
        tool_input = state.get("tool_input", {})
        retry_count = 0
        last_error = None

        while retry_count <= self.config.max_retries:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_tool(tool_input),
                    timeout=self.config.timeout_seconds,
                )

                state["tool_name"] = self.config.tool_name
                state["tool_output"] = result
                state["tool_error"] = None
                state["current_node"] = self.name

                logger.debug(
                    "tool_executed",
                    tool=self.config.tool_name,
                    success=True,
                )

                return state

            except asyncio.TimeoutError:
                last_error = f"Tool {self.config.tool_name} timed out after {self.config.timeout_seconds}s"
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                last_error = str(e)

            retry_count += 1
            if retry_count <= self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_ms / 1000)
                logger.warning(
                    "tool_retry",
                    tool=self.config.tool_name,
                    attempt=retry_count,
                    error=last_error,
                )

        # All retries exhausted
        state["tool_error"] = last_error
        state["current_node"] = self.name

        logger.error(
            "tool_failed",
            tool=self.config.tool_name,
            error=last_error,
        )

        return state

    async def _execute_tool(self, tool_input: dict[str, Any]) -> Any:
        """Execute the tool function."""
        if inspect.iscoroutinefunction(self.tool_func):
            return await self.tool_func(**tool_input)

        result = await asyncio.to_thread(self.tool_func, **tool_input)
        if inspect.isawaitable(result):
            return await result
        return result

# =============================================================================
# VALIDATION NODE
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation nodes."""

    # Validation rules
    required_fields: list[str] = None
    min_response_length: int = 1
    max_response_length: int = 10000

    # Error handling
    error_field: str = "error"
    retry_on_error: bool = True

    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = ["response"]

class ValidationNode(BaseNode):
    """
    Validation node for checking outputs.

    Validates that required fields exist and meet criteria.
    Sets error state if validation fails.

    Usage:
        validator = ValidationNode(config)
        state = await validator(state)
        if state.get("validation_error"):
            # Handle error
    """

    name = "validation"

    def __init__(self, config: ValidationConfig | None = None):
        super().__init__()
        self.config = config or ValidationConfig()

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate the state."""
        errors = []

        # Check required fields
        for field in self.config.required_fields:
            if field not in state or state[field] is None:
                errors.append(f"Missing required field: {field}")

        # Check response length
        response = state.get("response", "")
        if isinstance(response, str):
            if len(response) < self.config.min_response_length:
                errors.append(f"Response too short: {len(response)} chars")
            if len(response) > self.config.max_response_length:
                errors.append(f"Response too long: {len(response)} chars")

        # Check for tool errors
        if state.get("tool_error"):
            errors.append(f"Tool error: {state['tool_error']}")

        if errors:
            state["validation_error"] = "; ".join(errors)
            state["_should_retry"] = self.config.retry_on_error
            logger.warning("validation_failed", errors=errors)
        else:
            state["validation_error"] = None
            state["_should_retry"] = False

        state["current_node"] = self.name
        return state

# =============================================================================
# HUMAN APPROVAL NODE
# =============================================================================

@dataclass
class ApprovalRequest:
    """A pending approval request."""

    id: str
    action: str
    description: str
    user_id: str | None
    created_at: datetime
    expires_at: datetime
    status: str = "pending"  # pending, approved, rejected, expired
    approved_by: str | None = None
    approved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequest":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            action=data["action"],
            description=data["description"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            status=data["status"],
            approved_by=data.get("approved_by"),
            approved_at=datetime.fromisoformat(data["approved_at"])
            if data.get("approved_at")
            else None,
        )

class HumanApprovalNode(BaseNode):
    """
    Human-in-the-loop approval node.

    Pauses execution and waits for human approval before
    executing sensitive actions. Uses Redis for persistence.

    Usage:
        approval_node = HumanApprovalNode()
        state = await approval_node(state)

        if state["approval_status"] == "pending":
            # Wait for approval via API
            pass
    """

    name = "human_approval"
    REDIS_PREFIX = "approval:"

    def __init__(
        self,
        timeout_seconds: int = 300,
        require_reason: bool = False,
    ):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.require_reason = require_reason

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create an approval request."""
        action = state.get("tool_name", "unknown_action")
        user_id = state.get("user_id")

        # Create approval request
        approval_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            id=approval_id,
            action=action,
            description=f"Action: {action} requested by user {user_id}",
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(seconds=self.timeout_seconds),
        )

        # Store in Redis
        try:
            redis = get_redis_client()
            key = f"{self.REDIS_PREFIX}{approval_id}"
            await redis.setex(
                key,
                self.timeout_seconds,
                json.dumps(request.to_dict()),
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("approval_persistence_failed", error=str(e))
            raise RuntimeError(f"Failed to persist approval request: {e}")

        state["requires_approval"] = True
        state["approval_id"] = approval_id
        state["approval_status"] = "pending"
        state["current_node"] = self.name

        logger.info(
            "approval_requested",
            approval_id=approval_id,
            action=action,
            user_id=user_id,
        )

        return state

    @classmethod
    async def approve(cls, approval_id: str, approved_by: str) -> bool:
        """Approve a pending request."""
        redis = get_redis_client()
        key = f"{cls.REDIS_PREFIX}{approval_id}"

        data_str = await redis.get(key)
        if not data_str:
            return False

        try:
            data = json.loads(data_str)
            request = ApprovalRequest.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return False

        if request.status != "pending":
            return False

        if datetime.now(timezone.utc) > request.expires_at:
            request.status = "expired"
            return False

        request.status = "approved"
        request.approved_by = approved_by
        request.approved_at = datetime.now(timezone.utc)

        # Update Redis
        ttl = await redis.ttl(key)
        if ttl < 0:
            ttl = 300

        await redis.setex(key, ttl, json.dumps(request.to_dict()))

        logger.info(
            "approval_granted", approval_id=approval_id, approved_by=approved_by
        )
        return True

    @classmethod
    async def reject(cls, approval_id: str, rejected_by: str, reason: str = "") -> bool:
        """Reject a pending request."""
        redis = get_redis_client()
        key = f"{cls.REDIS_PREFIX}{approval_id}"

        data_str = await redis.get(key)
        if not data_str:
            return False

        try:
            data = json.loads(data_str)
            request = ApprovalRequest.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return False

        if request.status != "pending":
            return False

        request.status = "rejected"

        # Update Redis
        ttl = await redis.ttl(key)
        if ttl < 0:
            ttl = 300

        await redis.setex(key, ttl, json.dumps(request.to_dict()))

        logger.info(
            "approval_rejected", approval_id=approval_id, rejected_by=rejected_by
        )
        return True

    @classmethod
    async def get_status(cls, approval_id: str) -> str | None:
        """Get the status of an approval request."""
        redis = get_redis_client()
        key = f"{cls.REDIS_PREFIX}{approval_id}"

        data_str = await redis.get(key)
        if not data_str:
            return None

        try:
            data = json.loads(data_str)
            return data.get("status")
        except json.JSONDecodeError:
            return None

    @classmethod
    async def list_pending(cls, user_id: str | None = None) -> list[ApprovalRequest]:
        """List pending approval requests."""
        # Note: Scanning keys is inefficient. In production, maintain a set of pending IDs.
        # For now, we'll scan (assuming low volume) or just return empty as strict implementation requiressets.
        # But to be useful without Set structure:
        redis = get_redis_client()
        keys = []
        async for key in redis.scan_iter(f"{cls.REDIS_PREFIX}*"):
            keys.append(key)

        pending = []
        for key in keys:
            data_str = await redis.get(key)
            if data_str:
                try:
                    data = json.loads(data_str)
                    # Note: data might be bytes in some redis clients, but get_redis_client typically returns str if decode_responses=True
                    # Assuming decode_responses=True based on other usage. If not, needs decode.
                    if isinstance(data, bytes):
                        data = json.loads(data.decode("utf-8"))
                    else:
                        data = json.loads(data)

                    request = ApprovalRequest.from_dict(data)
                    if request.status == "pending":
                        if user_id is None or request.user_id == user_id:
                            pending.append(request)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
                    continue
        return pending
