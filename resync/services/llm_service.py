# pylint
# ruff: noqa: E501
"""
LLM Service using OpenAI SDK for OpenAI-Compatible APIs.

This service uses the OpenAI Python SDK to connect to any OpenAI-compatible API:
- NVIDIA NIM (tested and confirmed)
- Azure OpenAI
- Local models (vLLM, ollama with OpenAI mode)
- OpenAI directly

NOTE: This does NOT use LiteLLM directly. If multi-provider support with automatic
      fallback is needed, consider migrating to `from litellm import acompletion`.

Now integrated with LangFuse for:
- Prompt management (externalized prompts)
- Observability and tracing
- Cost tracking

Usage:
    from resync.services.llm_service import get_llm_service

    llm = await get_llm_service()
    response = await llm.generate_agent_response(
        agent_id="tws-agent",
        user_message="Quais jobs estão em ABEND?",
    )

Configuration (settings):
    - llm_model: Model name (e.g., "meta/llama-3.1-70b-instruct")
    - llm_endpoint: Base URL (e.g., "https://integrate.api.nvidia.com/v1")
    - llm_api_key: API key (SecretStr supported)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import AsyncGenerator
from typing import Any

from resync.core.exceptions import (
    BaseAppException,
    ConfigurationError,
    IntegrationError,
    ServiceUnavailableError,
)
from resync.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    RetryWithBackoff,
    TimeoutManager,
)
from resync.core.utils.prompt_formatter import OpinionBasedPromptFormatter
from resync.settings import settings

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from openai import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from resync.core.langfuse import PromptType, get_prompt_manager, get_tracer

    LANGFUSE_INTEGRATION = True
except ImportError:
    LANGFUSE_INTEGRATION = False


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken if available, fallback to word estimate."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return int(len(text.split()) * 1.3)


logger = logging.getLogger(__name__)


def _coerce_secret(value: Any) -> str | None:
    """Accept str or pydantic SecretStr; return plain str or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    get_secret = getattr(value, "get_secret_value", None)
    return get_secret() if callable(get_secret) else str(value)


class LLMService:
    """Service for interacting with LLM APIs through OpenAI-compatible endpoints."""

    def __init__(self) -> None:
        """Initialize LLM service with NVIDIA API configuration"""
        if not OPENAI_AVAILABLE:
            raise IntegrationError(
                message="openai package is required but not installed",
                details={"install_command": "pip install openai"},
            )
        model = getattr(settings, "llm_model", None)
        if model is None:
            model = getattr(settings, "agent_model_name", None)
        if not model:
            raise IntegrationError(
                message="No LLM model configured",
                details={
                    "hint": "Define settings.llm_model or settings.agent_model_name"
                },
            )
        self.model: str = str(model)
        self.default_temperature = 0.6
        self.default_top_p = 0.95
        self.default_max_tokens = 1000
        self.default_frequency_penalty = 0.0
        self.default_presence_penalty = 0.0
        api_key = _coerce_secret(getattr(settings, "llm_api_key", None))
        base_url = getattr(settings, "llm_endpoint", None)
        if not base_url:
            raise IntegrationError(
                message="Missing LLM base_url",
                details={
                    "hint": "Configure settings.llm_endpoint (NVIDIA OpenAI-compatible)"
                },
            )
        if api_key:
            masked = api_key[:4] + "..." if len(api_key) > 4 else "***"
            logger.info("Using LLM API key: %s", masked)
        else:
            logger.info("No LLM API key configured")
        logger.info("LLM base URL: %s", base_url)
        try:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(getattr(settings, "llm_timeout", 20.0) or 20.0),
                max_retries=0,
            )
            logger.info("LLM service initialized with model: %s", self.model)
            self._timeout_s: float = float(
                getattr(settings, "llm_timeout", 20.0) or 20.0
            )
            self._max_concurrency: int = int(
                getattr(settings, "llm_max_concurrency", None)
                or os.getenv("LLM_MAX_CONCURRENCY", "8")
            )
            self._sem = asyncio.Semaphore(self._max_concurrency)
            self._retry = RetryWithBackoff(
                RetryConfig(
                    max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
                    base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "0.5")),
                    max_delay=float(os.getenv("LLM_RETRY_MAX_DELAY", "8.0")),
                    jitter=True,
                    expected_exceptions=(ServiceUnavailableError,),
                )
            )
            self._cb = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("LLM_CB_FAILURE_THRESHOLD", "5")),
                    recovery_timeout=int(os.getenv("LLM_CB_RECOVERY_TIMEOUT", "60")),
                    expected_exception=ServiceUnavailableError,
                    exclude_exceptions=(ConfigurationError, IntegrationError),
                    name="llm",
                )
            )
            logger.info(
                "LLM resilience configured: timeout_s=%s, max_concurrency=%s",
                self._timeout_s,
                self._max_concurrency,
            )
            self._prompt_formatter = OpinionBasedPromptFormatter()
            logger.debug(
                "OpinionBasedPromptFormatter initialized for enhanced RAG accuracy"
            )
        except (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            BadRequestError,
            APIError,
            APITimeoutError,
            APIStatusError,
        ) as exc:
            logger.error(
                "Failed to initialize LLM service (OpenAI error): %s",
                exc,
                exc_info=True,
            )
            raise IntegrationError(
                message="Failed to initialize LLM service",
                details={
                    "error": str(exc),
                    "request_id": getattr(exc, "request_id", None),
                },
            ) from exc
        except Exception as exc:
            if isinstance(
                exc,
                (SystemExit, KeyboardInterrupt, ImportError, AttributeError, TypeError),
            ):
                raise
            logger.error("Failed to initialize LLM service: %s", exc, exc_info=True)
            raise IntegrationError(
                message="Failed to initialize LLM service", details={"error": str(exc)}
            ) from exc

    def _extract_retry_after_seconds(self, exc: Exception) -> int | None:
        """Best-effort extraction of Retry-After (seconds) from OpenAI exceptions."""
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if not ra:
            return None
        try:
            return int(ra)
        except ValueError:
            return None

    def _translate_openai_error(
        self, exc: Exception, *, operation: str
    ) -> BaseAppException:
        """Map OpenAI SDK errors into domain exceptions with correct retry semantics."""
        request_id = getattr(exc, "request_id", None)
        status_code = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if isinstance(exc, AuthenticationError):
            return ConfigurationError(
                message=f"LLM authentication failed during {operation}",
                details={
                    "operation": operation,
                    "request_id": request_id,
                    "status_code": status_code,
                },
                original_exception=exc,
            )
        if isinstance(exc, BadRequestError):
            return IntegrationError(
                message=f"LLM request rejected during {operation}",
                details={
                    "operation": operation,
                    "request_id": request_id,
                    "status_code": status_code,
                    "error": str(exc),
                },
                original_exception=exc,
            )
        if isinstance(exc, RateLimitError):
            return ServiceUnavailableError(
                message=f"LLM rate limited during {operation}",
                retry_after=self._extract_retry_after_seconds(exc),
                details={
                    "operation": operation,
                    "request_id": request_id,
                    "status_code": status_code,
                },
                original_exception=exc,
            )
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return ServiceUnavailableError(
                message=f"LLM network/timeout failure during {operation}",
                details={"operation": operation, "request_id": request_id},
                original_exception=exc,
            )
        if isinstance(exc, APIStatusError):
            if status_code == 429 or (
                isinstance(status_code, int) and status_code >= 500
            ):
                return ServiceUnavailableError(
                    message=f"LLM upstream error during {operation}",
                    retry_after=self._extract_retry_after_seconds(exc),
                    details={
                        "operation": operation,
                        "request_id": request_id,
                        "status_code": status_code,
                    },
                    original_exception=exc,
                )
            return IntegrationError(
                message=f"LLM returned non-retriable status during {operation}",
                details={
                    "operation": operation,
                    "request_id": request_id,
                    "status_code": status_code,
                    "error": str(exc),
                },
                original_exception=exc,
            )
        if isinstance(exc, APIError):
            return ServiceUnavailableError(
                message=f"LLM API error during {operation}",
                details={
                    "operation": operation,
                    "request_id": request_id,
                    "status_code": status_code,
                },
                original_exception=exc,
            )
        return IntegrationError(
            message=f"Unexpected LLM error during {operation}",
            details={
                "operation": operation,
                "request_id": request_id,
                "status_code": status_code,
                "error": str(exc),
            },
            original_exception=exc,
        )

    async def _call_openai(
        self, operation: str, coro_factory: Any, *, retry: bool = True
    ) -> Any:
        """Execute an OpenAI SDK call with bulkhead + timeout + retry + circuit breaker."""

        async def _protected() -> Any:
            async with self._sem:
                try:
                    return await TimeoutManager.with_timeout(
                        coro_factory(),
                        timeout_seconds=self._timeout_s,
                        timeout_exception=ServiceUnavailableError(
                            message=f"LLM operation timed out during {operation}",
                            details={
                                "operation": operation,
                                "timeout_s": self._timeout_s,
                            },
                        ),
                    )
                except (
                    AuthenticationError,
                    RateLimitError,
                    APIConnectionError,
                    BadRequestError,
                    APIError,
                    APITimeoutError,
                    APIStatusError,
                ) as exc:
                    raise self._translate_openai_error(
                        exc, operation=operation
                    ) from exc

        async def _cb_call() -> Any:
            return await self._cb.call(_protected)

        if not retry:
            return await _cb_call()
        return await self._retry.execute(_cb_call)

    async def _chat_completion(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]] | list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Helper around ``chat.completions.create`` with resilience."""
        return await self._call_openai(
            operation,
            lambda: self.client.chat.completions.create(
                model=self.model, messages=messages, stream=stream, **kwargs
            ),
            retry=not stream,
        )

    async def generate_response_with_tools(
        self,
        messages: list[dict[str, str]],
        user_role: str = "operator",
        user_id: str | None = None,
        session_id: str | None = None,
        max_tool_iterations: int = 5,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generates a response with tool support (function calling).

        Args:
            messages: Conversation messages
            user_role: User role (for permissions)
            user_id: User ID (for audit)
            session_id: Session ID
            max_tool_iterations: Maximum tool → LLM iterations
            temperature: Temperature (default: 0.3 for tool calls)
            max_tokens: Maximum tokens
        """
        try:
            from resync.tools.llm_tools import execute_tool_call, get_llm_tools
            from resync.tools.registry import UserRole

            role_map = {
                "viewer": UserRole.VIEWER,
                "operator": UserRole.OPERATOR,
                "admin": UserRole.ADMIN,
                "system": UserRole.SYSTEM,
            }
            user_role_enum = role_map.get(user_role.lower(), UserRole.OPERATOR)
            tools = get_llm_tools(user_role=user_role_enum)
            logger.info(
                f"Generating response with {len(tools)} tools available for role {user_role}"
            )
            current_messages = messages.copy()
            temp = temperature if temperature is not None else 0.3
            max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
            for iteration in range(max_tool_iterations):
                response = await self._chat_completion(
                    operation="tool_completion",
                    messages=current_messages,
                    tools=tools if tools else None,
                    temperature=temp,
                    max_tokens=max_tok,
                )
                message = response.choices[0].message
                if not message.tool_calls:
                    return message.content or ""
                logger.debug(
                    f"LLM requested {len(message.tool_calls)} tool calls at iteration {iteration}"
                )
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )
                for tool_call in message.tool_calls:
                    result = await execute_tool_call(
                        tool_call,
                        user_id=user_id,
                        user_role=user_role_enum,
                        session_id=session_id,
                    )
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            logger.warning("Max tool iterations (%s) reached", max_tool_iterations)
            return "Sorry, I reached the tool iteration limit. Please rephrase your question more specifically."
        except Exception as e:
            if isinstance(
                e,
                (
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ),
            ):
                raise
            logger.error("Error in generate_response_with_tools: %s", e, exc_info=True)
            return await self.generate_response(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from LLM

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            stream: Whether to stream response

        Returns:
            Generated text response
        """
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        frequency_penalty = (
            self.default_frequency_penalty
            if frequency_penalty is None
            else frequency_penalty
        )
        presence_penalty = (
            self.default_presence_penalty
            if presence_penalty is None
            else presence_penalty
        )
        logger.info("Generating LLM response with model: %s", self.model)
        try:
            if stream:
                chunks: list[str] = []
                async for piece in self._generate_response_streaming(
                    messages,
                    temperature,
                    top_p,
                    max_tokens,
                    frequency_penalty,
                    presence_penalty,
                ):
                    chunks.append(piece)
                content = "".join(chunks)
            else:
                response = await self._chat_completion(
                    operation="chat_completion",
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=False,
                )
                content = response.choices[0].message.content or ""
            logger.info("Generated LLM response (%d characters)", len(content))
            return content
        except (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            BadRequestError,
            APIError,
            APITimeoutError,
            APIStatusError,
        ) as exc:
            logger.error("Error generating LLM response: %s", exc, exc_info=True)
            raise IntegrationError(
                message="Failed to generate LLM response",
                details={
                    "error": str(exc),
                    "request_id": getattr(exc, "request_id", None),
                    "model": self.model,
                },
            ) from exc
        except Exception as exc:
            if isinstance(
                exc,
                (
                    BaseAppException,
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    AttributeError,
                ),
            ):
                raise
            logger.error(
                "Unexpected error generating LLM response: %s", exc, exc_info=True
            )
            raise IntegrationError(
                message="Failed to generate LLM response",
                details={"error": str(exc), "model": self.model},
            ) from exc

    async def _generate_response_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from LLM

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty

        Yields:
            Chunks of generated response
        """
        logger.info("Generating streaming LLM response with model: %s", self.model)
        try:
            response = await self._chat_completion(
                operation="chat_completion_stream",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    yield delta.content
        except (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            BadRequestError,
            APIError,
            APITimeoutError,
            APIStatusError,
        ) as exc:
            logger.error(
                "Error generating streaming LLM response: %s", exc, exc_info=True
            )
            raise IntegrationError(
                message="Failed to generate streaming LLM response",
                details={
                    "error": str(exc),
                    "request_id": getattr(exc, "request_id", None),
                    "model": self.model,
                },
            ) from exc
        except Exception as exc:
            if isinstance(
                exc,
                (
                    BaseAppException,
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    AttributeError,
                ),
            ):
                raise
            logger.error(
                "Unexpected error generating streaming LLM response: %s",
                exc,
                exc_info=True,
            )
            raise IntegrationError(
                message="Failed to generate streaming LLM response",
                details={"error": str(exc), "model": self.model},
            ) from exc

    async def generate_agent_response(
        self,
        agent_id: str,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a response from an AI agent.

        Now uses LangFuse prompt management for externalized prompts.
        Falls back to hardcoded prompts if LangFuse is unavailable.

        Args:
            agent_id: ID of agent
            user_message: User's message
            conversation_history: Previous conversation history
            agent_config: Agent configuration

        Returns:
            Generated agent response
        """
        agent_type = (agent_config or {}).get("type", "general")
        agent_name = (agent_config or {}).get("name", f"Agente {agent_id}")
        agent_description = (agent_config or {}).get(
            "description", f"Assistente {agent_type}"
        )
        system_message = None
        if LANGFUSE_INTEGRATION:
            try:
                prompt_manager = get_prompt_manager()
                prompt = await prompt_manager.get_prompt(f"{agent_id}-system")
                if not prompt:
                    prompt = await prompt_manager.get_default_prompt(PromptType.AGENT)
                if prompt:
                    context = f"Agente: {agent_name}\nDescrição: {agent_description}"
                    if agent_config:
                        context += f"\nConfiguração: {agent_config}"
                    system_message = prompt.compile(context=context)
                    logger.debug(
                        "prompt_loaded_from_manager: prompt_id=%s, agent_id=%s",
                        prompt.id,
                        agent_id,
                    )
            except Exception as e:
                if isinstance(
                    e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                ):
                    raise
                logger.warning("prompt_manager_fallback: error=%s", str(e))
        if not system_message:
            system_message = f"You are {agent_name}, {agent_description}. Respond in a helpful and professional manner in Brazilian Portuguese. Be concise and provide accurate information."
        messages: list[dict[str, str]] = [{"role": "system", "content": system_message}]
        if conversation_history:
            messages.extend(conversation_history[-5:])
        messages.append({"role": "user", "content": user_message})
        if LANGFUSE_INTEGRATION:
            tracer = get_tracer()
            async with tracer.trace(
                "generate_agent_response", model=self.model
            ) as trace:
                response = await self.generate_response(messages, max_tokens=800)
                trace.output = response
                trace.input_tokens = sum(
                    (_count_tokens(m.get("content", "")) for m in messages)
                )
                trace.output_tokens = _count_tokens(response)
                return response
        return await self.generate_response(messages, max_tokens=800)

    async def generate_rag_response(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
        source_name: str = "the documentation",
        use_opinion_based: bool = True,
    ) -> str:
        """
        Generate a response using RAG (Retrieval-Augmented Generation).

        Now uses Opinion-Based Prompting by default for +30-50% improvement in
        context adherence. Research shows this technique improves accuracy from
        33% → 73% (120% improvement) by reformulating questions to force LLM
        to prioritize provided context over training data.

        Args:
            query: User's query
            context: Retrieved context/documents
            conversation_history: Previous conversation history
            source_name: Name of the knowledge source (e.g., "TWS manual")
            use_opinion_based: Use opinion-based prompting (recommended: True)

        Returns:
            Generated RAG response
        """
        if use_opinion_based:
            formatted = self._prompt_formatter.format_rag_prompt(
                query=query,
                context=context,
                source_name=source_name,
                strict_mode=True,
                language="pt",
            )
            messages: list[dict[str, str]] = [
                {"role": "system", "content": formatted["system"]}
            ]
            if conversation_history:
                messages.extend(conversation_history[-3:])
            messages.append({"role": "user", "content": formatted["user"]})
        else:
            system_message = None
            if LANGFUSE_INTEGRATION:
                try:
                    prompt_manager = get_prompt_manager()
                    prompt = await prompt_manager.get_default_prompt(PromptType.RAG)
                    if prompt:
                        system_message = prompt.compile(rag_context=context)
                        logger.debug(
                            "rag_prompt_loaded_from_manager",
                            extra={"prompt_id": prompt.id},
                        )
                except Exception as e:
                    if isinstance(
                        e, (SystemExit, KeyboardInterrupt, asyncio.CancelledError)
                    ):
                        raise
                    logger.warning(
                        "rag_prompt_manager_fallback", extra={"error": str(e)}
                    )
            if not system_message:
                system_message = f"You are an AI assistant specialized in answering questions based on the provided context. Use the context information to respond accurately and helpfully. If the context does not contain enough information, state that you don't know. Respond in Brazilian Portuguese.\n\nRelevant Context:\n{context}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_message}
            ]
            if conversation_history:
                messages.extend(conversation_history[-3:])
            messages.append({"role": "user", "content": query})
        response = await self.generate_response(messages, max_tokens=1000)
        self_rag_sample_rate = float(os.getenv("SELF_RAG_SAMPLE_RATE", "0.0"))
        is_grounded = True
        reflection = "self_rag_disabled"
        if random.random() < self_rag_sample_rate:
            is_grounded, reflection = await self._check_hallucination(
                query=query, context=context, response=response
            )
        if not is_grounded:
            logger.warning(
                "self_rag_hallucination_detected",
                extra={"query": query[:50], "reflection": reflection},
            )
            try:
                strict_prompt = f"CRITICAL: Answer ONLY using information from the context below. If the context doesn't contain the answer, say 'I don't have enough information.'\n\nContext:\n{context}\n\nQuestion: {query}"
                messages[-1] = {"role": "user", "content": strict_prompt}
                response = await self.generate_response(
                    messages, max_tokens=1000, temperature=0.2
                )
                logger.info("self_rag_regenerated", extra={"query": query[:50]})
            except Exception as e:
                logger.error("self_rag_regeneration_failed", extra={"error": str(e)})
        else:
            logger.debug("self_rag_grounded", extra={"query": query[:50]})
        if LANGFUSE_INTEGRATION:
            tracer = get_tracer()
            async with tracer.trace(
                "generate_rag_response",
                model=self.model,
                prompt_id="rag",
                metadata={
                    "opinion_based": use_opinion_based,
                    "self_rag_grounded": is_grounded,
                    "self_rag_reflection": reflection,
                },
            ) as trace:
                trace.output = response
                trace.input_tokens = sum(
                    (_count_tokens(m.get("content", "")) for m in messages)
                )
                trace.output_tokens = _count_tokens(response)
                return response
        return response

    async def _check_hallucination(
        self, query: str, context: str, response: str
    ) -> tuple[bool, str]:
        """
        Self-RAG hallucination check: verify response is grounded in context.

        Uses a lightweight LLM call to check if the answer references information
        not present in the retrieved context.

        Args:
            query: Original user query
            context: Retrieved context
            response: Generated response

        Returns:
            Tuple of (is_grounded, reflection_message)
        """
        try:
            check_prompt = f'You are a fact-checker. Determine if the ANSWER is fully supported by the CONTEXT.\n\nCONTEXT:\n{context[:1000]}\n\nQUESTION: {query}\n\nANSWER:\n{response}\n\nIs the answer fully grounded in the context? Respond with ONLY:\n- "YES" if all facts in the answer come from the context\n- "NO: [reason]" if the answer includes information not in the context'
            reflection = await self.generate_response(
                messages=[{"role": "user", "content": check_prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            is_grounded = reflection.strip().upper().startswith("YES")
            return (is_grounded, reflection.strip())
        except Exception as e:
            logger.warning("hallucination_check_failed", extra={"error": str(e)})
            return (True, "check_failed")

    async def health_check(self) -> dict[str, Any]:
        """Perform a lightweight health check on LLM service.

        Uses the ``/models`` endpoint (no tokens consumed) instead of a real
        inference request.  Falls back to a tiny ``max_tokens=1`` completion
        only when the models endpoint is unavailable.
        """
        try:
            try:
                models_resp = await self.client.models.list()
                return {
                    "status": "healthy",
                    "model": self.model,
                    "endpoint": getattr(settings, "llm_endpoint", None),
                    "available_models": len(models_resp.data),
                }
            except Exception:
                await self.generate_response(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=1
                )
                return {
                    "status": "healthy",
                    "model": self.model,
                    "endpoint": getattr(settings, "llm_endpoint", None),
                }
        except (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            BadRequestError,
            APIError,
            APITimeoutError,
            APIStatusError,
        ) as exc:
            return {
                "status": "unhealthy",
                "model": self.model,
                "endpoint": getattr(settings, "llm_endpoint", None),
                "error": str(exc),
                "request_id": getattr(exc, "request_id", None),
            }
        except Exception as exc:
            if isinstance(
                exc,
                (
                    SystemExit,
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                ),
            ):
                raise
            logger.error("exception_caught", exc_info=True, extra={"error": str(exc)})
            return {
                "status": "unhealthy",
                "model": self.model,
                "endpoint": getattr(settings, "llm_endpoint", None),
                "error": str(exc),
            }

    async def aclose(self) -> None:
        """Close underlying HTTP resources of the OpenAI client."""
        import inspect

        if hasattr(self.client, "aclose"):
            # AsyncOpenAI >= 1.x exposes aclose() directly
            await self.client.aclose()
        elif hasattr(self.client, "close"):
            close_result = self.client.close()
            if inspect.isawaitable(close_result):
                await close_result

    async def close(self) -> None:
        """Alias for graceful shutdown (used by wiring teardown)."""
        await self.aclose()

    async def shutdown(self) -> None:
        """Alias for graceful shutdown (used by wiring teardown)."""
        await self.aclose()


_llm_service_lock = asyncio.Lock()
_llm_service_instance: LLMService | None = None


async def get_llm_service() -> LLMService:
    """Get or create global LLM service instance (thread-safe)."""
    global _llm_service_instance
    if _llm_service_instance is None:
        async with _llm_service_lock:
            if _llm_service_instance is None:
                _llm_service_instance = LLMService()
    return _llm_service_instance
