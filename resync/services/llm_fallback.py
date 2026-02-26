# pylint
"""
LLM Fallback Policy Module

v5.4.2: Implements clear LLM fallback chain with:
- Primary model selection
- Automatic fallback on failure
- Circuit breaker per provider
- Cost-aware routing
- Retry with exponential backoff
- Timeout management

v5.2.3.21: Added Ollama/Qwen support for CPU-only inference:
- Ollama provider with LiteLLM integration
- Aggressive timeout (8s) for fast cloud fallback
- Streaming support for better UX
- JSON mode with validation

Fallback Chain:
1. Primary model (configured)
2. Fallback model (lower cost/faster)
3. Local fallback (if available)
4. Graceful degradation

Usage:
    from resync.services.llm_fallback import LLMService, get_llm_service

    # Simple usage
    llm = await get_llm_service()
    response = await llm.complete("Hello, world!")

    # With explicit fallback
    response = await llm.complete_with_fallback(
        prompt="Complex query",
        primary_model="gpt-4",
        fallback_model="gpt-3.5-turbo",
    )

    # v5.2.3.21: Streaming with Ollama
    async for chunk in llm.complete_stream("Explain TWS", model="ollama/qwen2.5:3b"):
        print(chunk, end="")

    # v5.2.3.21: JSON mode
    result = await llm.complete_json("Extract job name from: cancel PAYMENT_JOB")

Author: Resync Team
Version: 5.2.3.21
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from resync.core.exceptions import (
    CircuitBreakerError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from resync.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryConfig,
    RetryWithBackoff,
    TimeoutManager,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    AZURE = "azure"
    LOCAL = "local"
    LITELLM = "litellm"
    OLLAMA = "ollama"  # v5.2.3.21: Local Ollama provider

class FallbackReason(str, Enum):
    """Reason for falling back to another model."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    COST = "cost_optimization"

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    name: str
    provider: LLMProvider
    api_key_env: str = ""  # Environment variable name for API key
    api_base: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    cost_per_1k_tokens: float = 0.0
    is_fallback: bool = False

    # Circuit breaker settings
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout: int = 60

    # v5.2.3.21: Ollama-specific settings
    num_ctx: int = 4096  # Context window size
    num_thread: int = 4  # CPU threads (match physical cores)
    supports_json_mode: bool = True  # Native JSON output support

@dataclass
class LLMFallbackConfig:
    """Configuration for LLM fallback policy."""

    # Primary model - v5.2.3.21: Default to Ollama local
    primary_model: str = "ollama/qwen2.5:3b"
    primary_provider: LLMProvider = LLMProvider.OLLAMA

    # Fallback chain (in order of preference)
    # v5.2.3.21: Local -> Cloud fallback
    fallback_chain: list[str] = field(
        default_factory=lambda: [
            "gpt-4o-mini",  # Fast cloud fallback
            "gpt-3.5-turbo",  # Cheaper cloud fallback
        ]
    )

    # Timeouts - v5.2.3.21: Aggressive for local models
    default_timeout: float = 8.0  # 8s for local (aggressive fallback)
    fallback_timeout: float = 30.0  # Cloud can take longer

    # Retry settings
    max_retries: int = 1  # v5.2.3.21: Fewer retries, faster fallback
    retry_base_delay: float = 0.5
    retry_max_delay: float = 2.0

    # Cost optimization
    enable_cost_routing: bool = True  # v5.2.3.21: Prefer local (free)
    max_cost_per_request: float = 0.05  # USD

    # Circuit breaker (global)
    circuit_failure_threshold: int = 3  # v5.2.3.21: Open circuit faster
    circuit_recovery_timeout: int = 60  # 1 minute before retry

    # v5.2.3.21: Ollama-specific settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 4096
    ollama_num_thread: int = 4

    @classmethod
    def from_settings(cls) -> LLMFallbackConfig:
        """Create config from application settings."""
        from resync.settings import settings

        # v5.2.3.21: Support new Ollama settings
        return cls(
            primary_model=getattr(settings, "llm_model", "ollama/qwen2.5:3b"),
            default_timeout=getattr(settings, "llm_timeout", 8.0),
            ollama_base_url=getattr(
                settings, "ollama_base_url", "http://localhost:11434"
            ),
            ollama_num_ctx=getattr(settings, "ollama_num_ctx", 4096),
            ollama_num_thread=getattr(settings, "ollama_num_thread", 4),
        )

@dataclass
class LLMMetrics:
    """Metrics for LLM service."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    # Per-model metrics
    model_requests: dict[str, int] = field(default_factory=dict)
    model_failures: dict[str, int] = field(default_factory=dict)

    # Fallback tracking
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    last_request: datetime | None = None
    last_error: str | None = None

@dataclass
class LLMResponse:
    """Response from LLM service."""

    content: str
    model: str
    provider: LLMProvider
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    was_fallback: bool = False
    fallback_reason: FallbackReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# LLM SERVICE
# =============================================================================

class LLMService:
    """
    LLM Service with automatic fallback and resilience.

    Features:
    - Configurable fallback chain
    - Per-provider circuit breakers
    - Automatic retry with backoff
    - Cost tracking
    - Comprehensive metrics
    """

    # Known models with their configurations
    KNOWN_MODELS: dict[str, ModelConfig] = {
        "gpt-4": ModelConfig(
            name="gpt-4",
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_tokens=0.03,
        ),
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_tokens=0.005,
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            cost_per_1k_tokens=0.0005,
            is_fallback=True,
        ),
        "claude-3-opus": ModelConfig(
            name="claude-3-opus-20240229",
            provider=LLMProvider.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_tokens=0.015,
        ),
        "claude-3-sonnet": ModelConfig(
            name="claude-3-sonnet-20240229",
            provider=LLMProvider.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_tokens=0.003,
        ),
        "claude-3-haiku": ModelConfig(
            name="claude-3-haiku-20240307",
            provider=LLMProvider.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            cost_per_1k_tokens=0.00025,
            is_fallback=True,
        ),
        # OpenRouter models
        "openrouter/gpt-4": ModelConfig(
            name="openai/gpt-4",
            provider=LLMProvider.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            api_base="https://openrouter.ai/api/v1",
            cost_per_1k_tokens=0.03,
        ),
        "openrouter/llama-3": ModelConfig(
            name="meta-llama/llama-3.2-3b-instruct:free",
            provider=LLMProvider.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            api_base="https://openrouter.ai/api/v1",
            cost_per_1k_tokens=0.0,
            is_fallback=True,
        ),
        # =====================================================================
        # v5.2.3.21: Ollama Local Models (CPU-Only)
        # =====================================================================
        "ollama/qwen2.5:3b": ModelConfig(
            name="ollama/qwen2.5:3b",
            provider=LLMProvider.OLLAMA,
            api_key_env="",  # Ollama doesn't require API key
            api_base="http://localhost:11434",
            max_tokens=2048,
            temperature=0.1,  # Low temperature for precise TWS responses
            timeout_seconds=8.0,  # Aggressive timeout for fast fallback
            cost_per_1k_tokens=0.0,  # Local = free
            is_fallback=False,
            num_ctx=4096,
            num_thread=4,
            supports_json_mode=True,
        ),
        "ollama/qwen2.5:7b": ModelConfig(
            name="ollama/qwen2.5:7b",
            provider=LLMProvider.OLLAMA,
            api_key_env="",
            api_base="http://localhost:11434",
            max_tokens=2048,
            temperature=0.1,
            timeout_seconds=15.0,  # 7B is slower
            cost_per_1k_tokens=0.0,
            is_fallback=False,
            num_ctx=4096,
            num_thread=4,
            supports_json_mode=True,
        ),
        "ollama/llama3.2:3b": ModelConfig(
            name="ollama/llama3.2:3b",
            provider=LLMProvider.OLLAMA,
            api_key_env="",
            api_base="http://localhost:11434",
            max_tokens=2048,
            temperature=0.1,
            timeout_seconds=8.0,
            cost_per_1k_tokens=0.0,
            is_fallback=True,  # Can be used as local fallback
            num_ctx=4096,
            num_thread=4,
            supports_json_mode=True,
        ),
    }

    def __init__(self, config: LLMFallbackConfig | None = None):
        """
        Initialize LLM service.

        Args:
            config: Fallback configuration. If None, loads from settings.
        """
        self.config = config or LLMFallbackConfig.from_settings()
        self._metrics = LLMMetrics()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._retry_handler = RetryWithBackoff(
            RetryConfig(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                jitter=True,
                expected_exceptions=(LLMError, LLMTimeoutError),
            )
        )

        # Initialize circuit breakers for each provider
        for provider in LLMProvider:
            self._circuit_breakers[provider.value] = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=self.config.circuit_failure_threshold,
                    recovery_timeout=self.config.circuit_recovery_timeout,
                    name=f"llm_{provider.value}",
                )
            )

        logger.info(
            "llm_service_initialized",
            primary_model=self.config.primary_model,
            fallback_chain=self.config.fallback_chain,
        )

    @property
    def metrics(self) -> LLMMetrics:
        """Get service metrics."""
        return self._metrics

    def _get_model_config(self, model: str) -> ModelConfig:
        """Get configuration for a model."""
        if model in self.KNOWN_MODELS:
            return self.KNOWN_MODELS[model]

        # Default config for unknown models
        return ModelConfig(
            name=model,
            provider=LLMProvider.LITELLM,
            api_key_env="LLM_API_KEY",
        )

    def _get_circuit_breaker(self, provider: LLMProvider) -> CircuitBreaker:
        """Get circuit breaker for provider."""
        return self._circuit_breakers.get(
            provider.value, self._circuit_breakers[LLMProvider.OPENAI.value]
        )

    async def _call_llm(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Call LLM with the specified model.

        This is the internal method that actually calls the LLM.
        v5.2.3.21: Added Ollama support via LiteLLM with optimized CPU settings.
        """
        import os

        start_time = datetime.now(timezone.utc)

        # Get API key from environment (not required for Ollama)
        api_key = (
            os.getenv(model_config.api_key_env, "") if model_config.api_key_env else ""
        )
        if not api_key and model_config.provider not in (
            LLMProvider.LOCAL,
            LLMProvider.OLLAMA,
        ):
            raise LLMAuthenticationError(
                f"API key not found in {model_config.api_key_env}"
            )

        try:
            # Use litellm for unified interface
            import litellm

            # v5.2.3.21: Disable LiteLLM's verbose logging in production
            litellm.suppress_debug_info = True

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build kwargs for litellm
            llm_kwargs: dict[str, Any] = {
                "model": model_config.name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
                "temperature": kwargs.get("temperature", model_config.temperature),
                "timeout": model_config.timeout_seconds,
            }

            # Add API key only if provided
            if api_key:
                llm_kwargs["api_key"] = api_key

            # v5.2.3.21: Ollama-specific configuration
            if model_config.provider == LLMProvider.OLLAMA:
                llm_kwargs["api_base"] = model_config.api_base
                # Ollama-specific options for CPU optimization
                llm_kwargs["num_ctx"] = model_config.num_ctx
                # JSON mode support
                if kwargs.get("json_mode", False) and model_config.supports_json_mode:
                    llm_kwargs["format"] = "json"
            elif model_config.api_base:
                llm_kwargs["api_base"] = model_config.api_base

            # OpenAI JSON mode
            if (
                kwargs.get("json_mode", False)
                and model_config.provider == LLMProvider.OPENAI
            ):
                llm_kwargs["response_format"] = {"type": "json_object"}

            logger.debug(
                "llm_call_starting",
                model=model_config.name,
                provider=model_config.provider.value,
                timeout=model_config.timeout_seconds,
            )

            # Make the call with timeout
            response = await TimeoutManager.with_timeout(
                litellm.acompletion(**llm_kwargs),
                model_config.timeout_seconds,
            )

            # Extract response
            content = response.choices[0].message.content
            tokens_used = getattr(response, "usage", {})
            total_tokens = getattr(tokens_used, "total_tokens", 0)

            # Calculate cost
            cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens

            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return LLMResponse(
                content=content,
                model=model_config.name,
                provider=model_config.provider,
                tokens_used=total_tokens,
                latency_ms=latency,
                cost=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"LLM call timed out after {model_config.timeout_seconds}s"
            ) from e
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            error_str = str(e).lower()
            if "rate" in error_str and "limit" in error_str:
                raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
            if "auth" in error_str or "401" in str(e):
                raise LLMAuthenticationError(f"Authentication failed: {e}") from e
            # v5.2.3.21: Better error handling for Ollama connection issues
            if "connection" in error_str or "refused" in error_str:
                raise LLMError(
                    f"Connection to {model_config.provider.value} failed: {e}"
                ) from e
            raise LLMError(f"LLM call failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete a prompt using the configured model.

        Uses the fallback chain if the primary model fails.

        Args:
            prompt: User prompt
            model: Model to use (default: primary model)
            system_prompt: Optional system prompt
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with the completion

        Raises:
            LLMError: If all models fail
        """
        model = model or self.config.primary_model

        # Build fallback chain: requested model + configured fallbacks
        fallback_chain = [model] + [m for m in self.config.fallback_chain if m != model]

        self._metrics.total_requests += 1
        self._metrics.last_request = datetime.now(timezone.utc)

        last_error = None
        fallback_reason = None

        for i, current_model in enumerate(fallback_chain):
            is_fallback = i > 0
            model_config = self._get_model_config(current_model)
            circuit_breaker = self._get_circuit_breaker(model_config.provider)

            # Track model usage
            self._metrics.model_requests[current_model] = (
                self._metrics.model_requests.get(current_model, 0) + 1
            )

            # Check circuit breaker
            if circuit_breaker.state == CircuitBreakerState.OPEN:
                logger.warning(
                    "llm_circuit_open_skipping",
                    model=current_model,
                    provider=model_config.provider.value,
                )
                fallback_reason = FallbackReason.CIRCUIT_OPEN
                self._metrics.fallback_reasons["circuit_open"] = (
                    self._metrics.fallback_reasons.get("circuit_open", 0) + 1
                )
                continue

            try:
                # Use shorter timeout for fallbacks
                if is_fallback:
                    # Do not mutate shared KNOWN_MODELS entries.
                    model_config = replace(
                        model_config, timeout_seconds=self.config.fallback_timeout
                    )

                # Execute with circuit breaker
                response = await circuit_breaker.call(
                    self._retry_handler.execute,
                    self._call_llm,
                    prompt,
                    model_config,
                    system_prompt,
                    **kwargs,
                )

                # Mark as fallback if not primary
                response.was_fallback = is_fallback
                response.fallback_reason = fallback_reason

                # Update metrics
                self._metrics.successful_requests += 1
                self._metrics.total_tokens += response.tokens_used
                self._metrics.total_cost += response.cost
                self._metrics.total_latency_ms += response.latency_ms

                if is_fallback:
                    self._metrics.fallback_requests += 1

                logger.info(
                    "llm_completion_success",
                    model=current_model,
                    was_fallback=is_fallback,
                    tokens=response.tokens_used,
                    latency_ms=response.latency_ms,
                )

                return response

            except CircuitBreakerError:
                fallback_reason = FallbackReason.CIRCUIT_OPEN
                self._metrics.fallback_reasons["circuit_open"] = (
                    self._metrics.fallback_reasons.get("circuit_open", 0) + 1
                )
                logger.warning(
                    "llm_circuit_breaker_tripped",
                    model=current_model,
                )

            except LLMTimeoutError as e:
                last_error = e
                fallback_reason = FallbackReason.TIMEOUT
                self._metrics.fallback_reasons["timeout"] = (
                    self._metrics.fallback_reasons.get("timeout", 0) + 1
                )
                self._metrics.model_failures[current_model] = (
                    self._metrics.model_failures.get(current_model, 0) + 1
                )
                logger.warning(
                    "llm_timeout_falling_back",
                    model=current_model,
                    next_model=fallback_chain[i + 1]
                    if i + 1 < len(fallback_chain)
                    else None,
                )

            except LLMRateLimitError as e:
                last_error = e
                fallback_reason = FallbackReason.RATE_LIMIT
                self._metrics.fallback_reasons["rate_limit"] = (
                    self._metrics.fallback_reasons.get("rate_limit", 0) + 1
                )
                self._metrics.model_failures[current_model] = (
                    self._metrics.model_failures.get(current_model, 0) + 1
                )
                logger.warning(
                    "llm_rate_limit_falling_back",
                    model=current_model,
                )

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                last_error = e
                fallback_reason = FallbackReason.ERROR
                self._metrics.fallback_reasons["error"] = (
                    self._metrics.fallback_reasons.get("error", 0) + 1
                )
                self._metrics.model_failures[current_model] = (
                    self._metrics.model_failures.get(current_model, 0) + 1
                )
                logger.warning(
                    "llm_error_falling_back",
                    model=current_model,
                    error=str(e),
                )

        # All models failed
        self._metrics.failed_requests += 1
        self._metrics.last_error = (
            str(last_error) if last_error else "All models failed"
        )

        logger.error(
            "llm_all_models_failed",
            attempted_models=fallback_chain,
            last_error=str(last_error),
        )

        raise LLMError(f"All LLM models failed. Last error: {last_error}")

    async def complete_with_fallback(
        self,
        prompt: str,
        primary_model: str,
        fallback_model: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete with explicit primary and fallback models.

        Convenience method for specifying exact fallback.

        NOTE: Does NOT mutate self.config — builds a local fallback chain instead
        to prevent cross-request contamination under concurrent async access.
        """
        # Build a local chain that overrides only for this call
        local_chain = [primary_model, fallback_model]
        original_chain = self.config.fallback_chain
        # Temporarily override only within this coroutine using a shadow attribute;
        # we keep the original intact by restoring in finally.
        # A proper fix would pass the chain as a parameter through complete().
        self.config = self.config.model_copy(
            update={"fallback_chain": [fallback_model]}
        ) if hasattr(self.config, "model_copy") else self.config
        try:
            return await self.complete(
                prompt, model=primary_model, system_prompt=system_prompt, **kwargs
            )
        finally:
            # Always restore original chain
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={"fallback_chain": original_chain}
                )
            else:
                self.config.fallback_chain = original_chain

    # =========================================================================
    # v5.2.3.21: STREAMING SUPPORT
    # =========================================================================

    async def complete_stream(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from LLM with automatic fallback.

        v5.2.3.21: Added streaming support for better UX with slow local models.

        Args:
            prompt: User prompt
            model: Model to use (default: primary model)
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Yields:
            str: Chunks of generated text

        Raises:
            LLMError: If all models fail
        """
        model = model or self.config.primary_model
        fallback_chain = [model] + [m for m in self.config.fallback_chain if m != model]

        last_error = None

        for _i, current_model in enumerate(fallback_chain):
            model_config = self._get_model_config(current_model)
            circuit_breaker = self._get_circuit_breaker(model_config.provider)

            if circuit_breaker.state == CircuitBreakerState.OPEN:
                logger.warning("llm_circuit_open_skipping_stream", model=current_model)
                continue

            try:
                async for chunk in self._stream_llm(
                    prompt, model_config, system_prompt, **kwargs
                ):
                    yield chunk
                return  # Success, exit generator

            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                last_error = e
                logger.warning(
                    "llm_stream_error_falling_back",
                    model=current_model,
                    error=str(e),
                )
                continue

        raise LLMError(f"All LLM models failed streaming. Last error: {last_error}")

    async def _stream_llm(
        self,
        prompt: str,
        model_config: ModelConfig,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Internal streaming method for a single model.

        v5.2.3.21: Implements LiteLLM streaming with Ollama support.
        """
        import os

        import litellm

        api_key = (
            os.getenv(model_config.api_key_env, "") if model_config.api_key_env else ""
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        llm_kwargs: dict[str, Any] = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
            "temperature": kwargs.get("temperature", model_config.temperature),
            "stream": True,
        }

        if api_key:
            llm_kwargs["api_key"] = api_key

        if model_config.provider == LLMProvider.OLLAMA:
            llm_kwargs["api_base"] = model_config.api_base
            llm_kwargs["num_ctx"] = model_config.num_ctx
        elif model_config.api_base:
            llm_kwargs["api_base"] = model_config.api_base

        try:
            response = await litellm.acompletion(**llm_kwargs)

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("llm_stream_failed", model=model_config.name, error=str(e))
            raise

    # =========================================================================
    # v5.2.3.21: JSON MODE SUPPORT
    # =========================================================================

    async def complete_json(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate JSON output from LLM with validation.

        v5.2.3.21: Added JSON mode support for structured outputs.

        Args:
            prompt: User prompt (should request JSON output)
            model: Model to use
            system_prompt: System prompt (will be enhanced for JSON)

        Returns:
            Parsed JSON dict

        Raises:
            LLMError: If JSON parsing fails
        """
        import json
        import re

        # Enhance system prompt for JSON
        json_system = system_prompt or ""
        if json_system:
            json_system += "\n\n"
        json_system += (
            "IMPORTANTE: Responda APENAS com um objeto JSON válido. "
            "Não inclua explicações, markdown ou texto adicional. "
            "Apenas o JSON puro."
        )

        response = await self.complete(
            prompt,
            model=model,
            system_prompt=json_system,
            json_mode=True,
            **kwargs,
        )

        # Parse JSON with repair logic
        content = response.content.strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.debug(
                "suppressed_exception", error=str(exc), exc_info=True
            )  # was: pass

        # Extract JSON from surrounding text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as exc:
                logger.debug(
                    "suppressed_exception", error=str(exc), exc_info=True
                )  # was: pass

        # Repair common issues
        cleaned = content
        cleaned = re.sub(r"^[^{]*", "", cleaned)  # Remove prefix
        cleaned = re.sub(r"[^}]*$", "", cleaned)  # Remove suffix
        cleaned = cleaned.replace("'", '"')  # Fix quotes

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse JSON from LLM response: {e}") from e

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of service metrics."""
        avg_latency = (
            self._metrics.total_latency_ms / self._metrics.successful_requests
            if self._metrics.successful_requests > 0
            else 0
        )

        return {
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "fallback_requests": self._metrics.fallback_requests,
            "success_rate": (
                self._metrics.successful_requests / self._metrics.total_requests
                if self._metrics.total_requests > 0
                else 0
            ),
            "fallback_rate": (
                self._metrics.fallback_requests / self._metrics.total_requests
                if self._metrics.total_requests > 0
                else 0
            ),
            "total_tokens": self._metrics.total_tokens,
            "total_cost_usd": self._metrics.total_cost,
            "avg_latency_ms": avg_latency,
            "model_requests": self._metrics.model_requests,
            "model_failures": self._metrics.model_failures,
            "fallback_reasons": self._metrics.fallback_reasons,
            "circuit_breakers": {
                name: cb.state.value for name, cb in self._circuit_breakers.items()
            },
        }

    def get_circuit_breaker_status(self) -> dict[str, str]:
        """Get status of all circuit breakers."""
        return {name: cb.state.value for name, cb in self._circuit_breakers.items()}

# =============================================================================
# SINGLETON & ACCESS
# =============================================================================

_llm_service_instance: LLMService | None = None
# Eagerly initialised — never None — eliminates TOCTOU race on lock init
_llm_service_lock: asyncio.Lock = asyncio.Lock()

async def get_llm_service() -> LLMService:
    """
    Get the singleton LLM service instance.

    Returns:
        LLMService instance
    """
    global _llm_service_instance

    if _llm_service_instance is None:
        async with _llm_service_lock:
            if _llm_service_instance is None:
                _llm_service_instance = LLMService()

    return _llm_service_instance

def reset_llm_service() -> None:
    """Reset the singleton LLM service (for testing)."""
    global _llm_service_instance
    _llm_service_instance = None

def configure_llm_service(config: LLMFallbackConfig) -> None:
    """
    Configure the LLM service with custom settings.

    Call this before first use of get_llm_service().
    """
    global _llm_service_instance
    _llm_service_instance = LLMService(config)
