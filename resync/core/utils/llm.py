# resync/core/utils/llm.py

import json
import re
from typing import TypeVar

from pydantic import BaseModel

from ...settings import settings
from ..resilience import circuit_breaker, retry_with_backoff, with_timeout
from ..structured_logger import get_logger
from .common_error_handlers import retry_on_exception
from .llm_deps import get_litellm_exceptions
from .llm_factories import LLMFactory

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

@circuit_breaker(failure_threshold=3, recovery_timeout=60, name="llm_service")
@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0, jitter=True)
@with_timeout(settings.LLM_TIMEOUT)
@retry_on_exception(
    max_retries=3,
    delay=1.0,
    backoff=2.0,
    # Lazy load exceptions to prevent import-time side effects
    exceptions=lambda: (
        tuple(get_litellm_exceptions())
        + (
            ConnectionError,
            TimeoutError,
            ValueError,
            Exception,
        )
    ),
    logger=logger,
)
async def call_llm(
    prompt: str,
    model: str,
    max_tokens: int = 200,
    temperature: float = 0.1,
    max_retries: int = 3,
    _initial_backoff: float = 1.0,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> str:
    """
    Calls an LLM through LiteLLM with support for multiple providers
    (OpenAI, Ollama, etc.).
    Provides enhanced error handling, cost tracking, and model flexibility.

    Args:
        prompt: The prompt to send to the LLM.
        model: The LLM model to use (e.g., "gpt-4o", "ollama/mistral", etc.).
        max_tokens: Maximum number of tokens in the LLM's response.
        temperature: Controls the randomness of the LLM's response.
        max_retries: Maximum number of retry attempts for the LLM call.
        initial_backoff: Initial delay in seconds before the first retry.
        api_base: Optional API base URL (for local models like Ollama).
        api_key: Optional API key (defaults to settings if not provided).
        timeout: Maximum time in seconds to wait for the LLM response.

    Returns:
        The content of the LLM's response.

    Raises:
        LLMError: If the LLM call fails after all retry attempts or times out.
    """
    return await LLMFactory.call_llm(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        _initial_backoff=_initial_backoff,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
    )

async def call_llm_structured(
    prompt: str,
    output_model: type[T],
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> T | None:
    """
    Call LLM with structured output using Pydantic model.

    Uses JSON mode and parses response into Pydantic model.

    Args:
        prompt: The prompt to send
        output_model: Pydantic model class for output
        model: LLM model to use (defaults to settings.llm_model)
        temperature: Temperature for generation
        max_retries: Number of parse retries

    Returns:
        Parsed Pydantic model instance or None on failure

    Example:
        class Intent(BaseModel):
            intent: str
            confidence: float

        result = await call_llm_structured(
            "Classify: check job status",
            Intent
        )
        print(result.intent)  # "status"
    """
    model = model or settings.llm_model or "gpt-4o"

    # Build schema prompt
    schema = output_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    full_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this schema:
{schema_str}

JSON:"""

    for attempt in range(max_retries):
        try:
            response = await call_llm(
                prompt=full_prompt,
                model=model,
                temperature=temperature,
                max_tokens=500,
            )

            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return output_model.model_validate(data)

        except json.JSONDecodeError as e:
            logger.warning(
                "structured_output_parse_error", attempt=attempt + 1, error=str(e)
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.warning("structured_output_error", attempt=attempt + 1, error=str(e))

    return None
