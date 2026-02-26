"""
Secret Scrubbing Utilities

Automatically masks sensitive information in logs, preventing
accidental credential leakage.

Usage:
    from resync.core.utils.secret_scrubber import scrub_secrets, SecretScrubber

    # Scrub dict
    safe_data = scrub_secrets({"password": "secret123", "user": "admin"})
    # Returns: {"password": "***SCRUBBED***", "user": "admin"}

    # Scrub string
    safe_log = scrub_secrets("TWS_PASSWORD=secret123 user=admin")
    # Returns: "TWS_PASSWORD=***SCRUBBED*** user=admin"
"""

import re
from typing import Any

# Patterns that indicate sensitive fields
SENSITIVE_PATTERNS = [
    # Credentials
    r"password",
    r"passwd",
    r"pwd",
    r"secret",
    r"token",
    r"api[_-]?key",
    r"access[_-]?key",
    r"private[_-]?key",
    r"auth",
    r"bearer",
    r"credential",
    # TWS specific
    r"tws[_-]?password",
    r"tws[_-]?user",
    # LLM API keys
    r"anthropic[_-]?api[_-]?key",
    r"openai[_-]?api[_-]?key",
    r"llm[_-]?api[_-]?key",
    # Database
    r"database[_-]?password",
    r"db[_-]?password",
    r"postgres[_-]?password",
    # Redis
    r"redis[_-]?password",
    r"redis[_-]?auth",
    # JWT
    r"jwt[_-]?secret",
    # Session
    r"session[_-]?secret",
    r"session[_-]?key",
    # Encryption
    r"encryption[_-]?key",
    r"signing[_-]?key",
]

# Compile patterns once
SENSITIVE_REGEX = re.compile(r"(" + "|".join(SENSITIVE_PATTERNS) + r")", re.IGNORECASE)

# Pattern to detect value assignment in strings
# Matches: key=value, key:value, key: value, "key": "value"
VALUE_PATTERN = re.compile(
    r"(" + "|".join(SENSITIVE_PATTERNS) + r")"
    r"\s*[=:]\s*"
    r'["\']?([^"\'\s,}]+)["\']?',
    re.IGNORECASE,
)

SCRUBBED_PLACEHOLDER = "***SCRUBBED***"

class SecretScrubber:
    """Secret scrubber with configurable patterns."""

    def __init__(self, additional_patterns: list[str] | None = None):
        """
        Initialize scrubber with optional additional patterns.

        Args:
            additional_patterns: Additional regex patterns to consider sensitive
        """
        patterns = SENSITIVE_PATTERNS.copy()
        if additional_patterns:
            patterns.extend(additional_patterns)

        self.sensitive_regex = re.compile(
            r"(" + "|".join(patterns) + r")", re.IGNORECASE
        )

        self.value_pattern = re.compile(
            r"(" + "|".join(patterns) + r")"
            r"\s*[=:]\s*"
            r'["\']?([^"\'\s,}]+)["\']?',
            re.IGNORECASE,
        )

    def scrub(self, data: Any, max_depth: int = 10) -> Any:
        """
        Scrub secrets from data (recursively).

        Args:
            data: Data to scrub (dict, list, str, or primitive)
            max_depth: Maximum recursion depth

        Returns:
            Scrubbed copy of data
        """
        if max_depth <= 0:
            return data

        if isinstance(data, dict):
            return self._scrub_dict(data, max_depth)
        elif isinstance(data, list):
            return [self.scrub(item, max_depth - 1) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.scrub(item, max_depth - 1) for item in data)
        elif isinstance(data, str):
            return self._scrub_string(data)
        else:
            return data

    def _scrub_dict(self, data: dict, max_depth: int) -> dict:
        """Scrub dict recursively."""
        scrubbed = {}
        for key, value in data.items():
            # Check if key is sensitive
            if self.sensitive_regex.search(str(key)):
                scrubbed[key] = SCRUBBED_PLACEHOLDER
            else:
                scrubbed[key] = self.scrub(value, max_depth - 1)
        return scrubbed

    def _scrub_string(self, text: str) -> str:
        """Scrub secrets from string (e.g., log messages)."""
        # Replace key=value patterns
        return self.value_pattern.sub(r"\1=***SCRUBBED***", text)

# Global instance
_default_scrubber = SecretScrubber()

def scrub_secrets(data: Any, max_depth: int = 10) -> Any:
    """
    Scrub secrets from data using default scrubber.

    Args:
        data: Data to scrub
        max_depth: Maximum recursion depth

    Returns:
        Scrubbed copy of data
    """
    return _default_scrubber.scrub(data, max_depth)

def scrub_dict(data: dict) -> dict:
    """Scrub secrets from dict (shorthand)."""
    return scrub_secrets(data)

def scrub_string(text: str) -> str:
    """Scrub secrets from string (shorthand)."""
    return scrub_secrets(text)

def is_sensitive_key(key: str) -> bool:
    """Check if key name indicates sensitive data."""
    return bool(SENSITIVE_REGEX.search(key))

# Decorator for automatic scrubbing
def scrub_args_and_result(func):
    """
    Decorator to automatically scrub function arguments and result in logs.

    Usage:
        @scrub_args_and_result
        def my_function(password: str, user: str):
            logger.info("called_function", password="***MASKED***", user=user)
            return {"password": password, "user": user}
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Scrub kwargs for logging
        scrub_secrets(kwargs)

        # Call original function
        result = func(*args, **kwargs)

        # Scrub result for logging
        scrub_secrets(result)

        return result  # Return original, not scrubbed

    return wrapper

# AsyncIO version
def scrub_args_and_result_async(func):
    """Async version of scrub_args_and_result."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        return result

    return wrapper

__all__ = [
    "scrub_secrets",
    "scrub_dict",
    "scrub_string",
    "is_sensitive_key",
    "SecretScrubber",
    "scrub_args_and_result",
    "scrub_args_and_result_async",
    "SCRUBBED_PLACEHOLDER",
]
