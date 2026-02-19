"""
Lazy loading dependencies for LLM operations.
This module handles imports of heavy libraries (like litellm) only when needed,
preventing side effects (network/telemetry) at module import time.
"""

from typing import Any, Tuple

_litellm_router = None

def get_litellm_exceptions() -> Tuple[type, ...]:
    """
    Lazy load LiteLLM exceptions.
    Returns a tuple of exception classes for try/except blocks.
    """
    from litellm.exceptions import (
        APIError,
        AuthenticationError,
        BadRequestError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )
    return (
        APIError,
        AuthenticationError,
        BadRequestError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

def get_acompletion() -> Any:
    """Lazy load litellm.acompletion function."""
    from litellm import acompletion
    return acompletion

def get_available_models() -> Any:
    """Lazy load litellm.get_available_models function."""
    from litellm import get_available_models
    return get_available_models
