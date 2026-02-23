# pylint: disable=all
# mypy: no-rerun
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
import logging
import os
from typing import Any

from resync.core.exceptions import (
    ConfigurationError,
    IntegrationError,
    ServiceUnavailableError,
)
from resync.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    RetryWithBackoff,
)
from resync.core.utils.prompt_formatter import OpinionBasedPromptFormatter
from resync.settings import settings

try:
    # Import specific exceptions from OpenAI v1.x
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
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


def _coerce_secret(value: Any) -> str | None:
    """Accept str or pydantic SecretStr; return plain str or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # pydantic SecretStr compatibility
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

        # --- Model resolution with fallbacks ---
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

        # Defaults
        self.default_temperature = 0.6
        self.default_top_p = 0.95
        self.default_max_tokens = 1000
        self.default_frequency_penalty = 0.0
        self.default_presence_penalty = 0.0

        # --- API key / endpoint (NVIDIA OpenAI-compatible) ---
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
            masked = (api_key[:4] + "...") if len(api_key) > 4 else "***"
            logger.info("Using LLM API key: %s", masked)
        else:
            logger.info("No LLM API key configured")
        logger.info("LLM base URL: %s", base_url)

        try:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(getattr(settings, "llm_timeout", 20.0) or 20.0),
                # Disable SDK retries: we apply consistent,
                # observable retries ourselves.
                max_retries=0,
            )
            logger.info("LLM service initialized with model: %s", self.model)

            # ------------------------------------------------------------------
            # Resilience defaults (production-grade)
            # ------------------------------------------------------------------
            self._timeout_s: float = float(
                getattr(settings, "llm_timeout", 20.0) or 20.0
            )

            # Bulkhead: cap concurrent in-flight LLM calls.
            self._max_concurrency: int = int(
                getattr(settings, "llm_max_concurrency", None)
                or os.getenv("LLM_MAX_CONCURRENCY", "8")
            )
            self._sem = asyncio.Semaphore(self._max_concurrency)

            # Retry with exponential backoff + jitter (only for transient errors).
            self._retry = RetryWithBackoff(
                RetryConfig(
                    max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
                    base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "0.5")),
                    max_delay=float(os.getenv("LLM_RETRY_MAX_DELAY", "8.0")),
                    jitter=True,
                    expected_exceptions=(ServiceUnavailableError,),
                )
            )

            # Circuit breaker: trips on repeated transient failures;
            # prevents thundering herds.
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
                "llm_resilience_configured",
                timeout_s=self._timeout_s,
                max_concurrency=self._max_concurrency,
            )

            # Opinion-Based Prompting for +30-50% context adherence improvement
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Re-raise critical system exceptions and programming errors
            if isinstance(
                exc,
                (SystemExit, KeyboardInterrupt, ImportError, AttributeError, TypeError),
            ):
                raise
            logger.error("Failed to initialize LLM service: %s", exc, exc_info=True)
            raise IntegrationError(
                message="Failed to initialize LLM service",
                details={"error": str(exc)},
            ) from exc
