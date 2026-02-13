"""
LLM Call Factories using Factory Pattern.

This module implements the Factory pattern for creating LLM calls with different providers
and configurations, making the code more modular, testable, and maintainable.
"""

import asyncio
import time

from litellm import acompletion
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InvalidRequestError,
    RateLimitError,
)

from ...settings import settings
from ..exceptions import LLMError
from ..structured_logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory class for creating LLM calls with different providers and configurations."""

    @staticmethod
    async def call_llm(
        prompt: str,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.1,
        max_retries: int = 3,
        _initial_backoff: float = 1.0,
        api_base: str = None,
        api_key: str = None,
        timeout: float = 30.0,
    ) -> str:
        """Factory method to call LLM with appropriate configuration."""
        start_time = time.time()

        # Use provided api_key or settings, handle default placeholder
        effective_api_key = api_key or settings.LLM_API_KEY
        if effective_api_key == "your_default_api_key_here":
            effective_api_key = None

        # Check if LiteLLM is available and has models configured
        try:
            from resync.core.litellm_init import get_litellm_router

            router = get_litellm_router()
            if not router or len(router.model_list) == 0:
                raise ImportError("No models available in LiteLLM router")
        except ImportError:
            # Fallback to simple mock response for development
            logger.warning("LiteLLM not available or no models configured, using mock response")
            return "LLM service is currently unavailable. This is a mock response for development purposes."

        # Use LiteLLM's acompletion for enhanced functionality with timeout
        try:
            response = await asyncio.wait_for(
                acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_base=api_base or getattr(settings, "LLM_ENDPOINT", None),
                    api_key=effective_api_key,
                    # Additional LiteLLM features
                    metadata={
                        "user_id": getattr(settings, "APP_NAME", "resync"),
                        "session_id": f"tws_{int(time.time())}",
                    },
                ),
                timeout=timeout,
            )

            # Validate response
            if not response.choices or len(response.choices) == 0:
                raise LLMError("Empty response received from LLM")

            content = response.choices[0].message.content
            if content is None:
                raise LLMError("LLM returned null content")

            content = content.strip()
            if not content:
                raise LLMError("LLM returned empty content")

            # Extract usage information for cost tracking
            usage = response.usage or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Record successful call metrics
            total_time = time.time() - start_time
            logger.info(
                "llm_call_completed",
                duration_seconds=round(total_time, 2),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return content

        except asyncio.TimeoutError:
            logger.error("llm_timeout", timeout_seconds=timeout)
            raise LLMError(f"LLM call timed out after {timeout} seconds") from None
        except ContentPolicyViolationError as e:
            logger.warning("llm_content_policy_violation", error=str(e))
            raise LLMError(f"Content policy violation: {str(e)}") from e
        except ContextWindowExceededError as e:
            logger.error("llm_context_window_exceeded", error=str(e))
            raise LLMError(f"Context window exceeded: {str(e)}") from e
        except AuthenticationError as e:
            logger.error("llm_authentication_error", error=str(e))
            raise LLMError(f"Authentication error: {str(e)}") from e
        except RateLimitError as e:
            logger.warning("llm_rate_limit_exceeded", error=str(e))
            raise LLMError(f"Rate limit exceeded: {str(e)}") from e
        except InvalidRequestError as e:
            logger.error("llm_invalid_request", error=str(e))
            raise LLMError(f"Invalid request: {str(e)}") from e
        except APIError as e:
            logger.error("llm_api_error", error=str(e))
            raise LLMError(f"API error: {str(e)}") from e
        except Exception as e:
            logger.error("llm_unexpected_error", error=str(e))
            raise LLMError(f"Unexpected error: {str(e)}") from e


class LLMProviderFactory:
    """Factory for creating LLM providers with specific configurations."""

    @staticmethod
    def create_provider(provider: str, **kwargs) -> "LLMProvider":
        """Create an LLM provider based on the provider type."""
        if provider == "openai":
            return OpenAIProvider(**kwargs)
        if provider == "ollama":
            return OllamaProvider(**kwargs)
        if provider == "anthropic":
            return AnthropicProvider(**kwargs)
        return DefaultLLMProvider(**kwargs)


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.api_base = kwargs.get("api_base")
        self.model = kwargs.get("model")
        self.max_tokens = kwargs.get("max_tokens", 200)
        self.temperature = kwargs.get("temperature", 0.1)

    async def call(self, prompt: str, **kwargs) -> str:
        """Call the LLM with the given prompt."""
        return await LLMFactory.call_llm(
            prompt=prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
            **kwargs,
        )


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "gpt-4o")
        self.api_base = kwargs.get("api_base", "https://api.openai.com/v1")


class OllamaProvider(LLMProvider):
    """Provider for Ollama models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "mistral")
        self.api_base = kwargs.get("api_base", "http://localhost:11434/v1")


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "claude-3-opus-20240229")
        self.api_base = kwargs.get("api_base", "https://api.anthropic.com/v1")


class DefaultLLMProvider(LLMProvider):
    """Default provider for LLM calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model", "gpt-4o")
        self.api_base = kwargs.get("api_base", getattr(settings, "LLM_ENDPOINT", None))
