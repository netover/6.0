# resync/core/utils/json_parser.py
import logging
from typing import Any

from .common_error_handlers import handle_parsing_errors
from .json_commands import JSONParseCommandExecutor

logger = logging.getLogger(__name__)

# Security constants
MAX_JSON_SIZE = 1024 * 1024  # 1MB limit for JSON content
MAX_TEXT_SIZE = 10 * 1024 * 1024  # 10MB limit for input text


@handle_parsing_errors("Failed to parse LLM JSON response")
def parse_llm_json_response(
    text: str,
    required_keys: list[str],
    max_size: int = MAX_JSON_SIZE,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Extracts, parses, and validates a JSON object from a string,
    often from an LLM response.

    This function includes security measures to prevent DoS attacks
    through large input processing.

    Args:
        text: The string potentially containing a JSON object.
        required_keys: A list of keys that must be present in the JSON.
        max_size: Maximum allowed size for JSON string (security limit).
        strict: If True, raise on extra keys not in required_keys.

    Raises:
        ParsingError: If the JSON is malformed, cannot be found, is missing
                      required keys, or exceeds size limits.
        ValueError: If input exceeds max_size.

    Returns:
        The parsed and validated JSON data as a dictionary.
    """
    return JSONParseCommandExecutor.execute_command(
        text, required_keys, max_size, strict
    )
