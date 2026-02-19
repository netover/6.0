"""
JSON Parsing Commands using Command Pattern.

This module implements the Command pattern for parsing and validating JSON responses
from LLMs, making the code more modular, testable, and maintainable.
"""

import json
import logging
from typing import Any

from ..exceptions import ParsingError
from .common_error_handlers import handle_parsing_errors

logger = logging.getLogger(__name__)

# Security constants
MAX_JSON_SIZE = 1024 * 1024  # 1MB limit for JSON content
MAX_TEXT_SIZE = 10 * 1024 * 1024  # 10MB limit for input text


class JSONParseCommand:
    """Base command class for parsing JSON responses."""

    def __init__(
        self,
        text: str,
        required_keys: list[str],
        max_size: int = MAX_JSON_SIZE,
        strict: bool = True,
    ):
        self.text = text
        self.required_keys = required_keys
        self.max_size = max_size
        self.strict = strict
        self.result = None

    def execute(self) -> dict[str, Any]:
        """Execute the parsing command."""
        self._validate_input()
        self._extract_json()
        self._parse_json()
        self._validate_structure()
        self._validate_keys()
        self._validate_nesting()

        return self.result

    def _validate_input(self) -> None:
        """Validate input size limits."""
        if len(self.text) > MAX_TEXT_SIZE:
            raise ParsingError(f"Input text exceeds maximum size of {MAX_TEXT_SIZE} bytes")

        if len(self.text.encode("utf-8")) > MAX_TEXT_SIZE:
            raise ParsingError(f"Input text exceeds maximum size of {MAX_TEXT_SIZE} bytes")

        # Sanitization: Remove null bytes and other potentially harmful characters
        self.text = self.text.replace("\x00", "").replace("\ufeff", "")

    def _extract_json(self) -> None:
        """Extract JSON string from text."""
        start_index = self.text.find("{")
        end_index = self.text.rfind("}")
        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise ParsingError("No valid JSON object found in the text.")

        self.json_str = self.text[start_index : end_index + 1]

        # Security: Check JSON string size
        if len(self.json_str) > self.max_size:
            raise ParsingError(f"JSON content exceeds maximum size of {self.max_size} bytes")

    def _parse_json(self) -> None:
        """Parse JSON string into dictionary."""
        try:
            self.result = json.loads(self.json_str)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error", error=str(e), json_length=len(self.json_str))
            raise ParsingError(f"Invalid JSON format: {str(e)}") from e

    def _validate_structure(self) -> None:
        """Validate that result is a dictionary."""
        if not isinstance(self.result, dict):
            raise ParsingError("Parsed JSON is not an object (dictionary)")

    def _validate_keys(self) -> None:
        """Validate required keys are present."""
        missing_keys = [key for key in self.required_keys if key not in self.result]
        if missing_keys:
            raise ParsingError(f"JSON is missing required keys: {', '.join(missing_keys)}")

        # Optional strict validation: ensure no extra keys
        if self.strict and self.required_keys:
            extra_keys = [key for key in self.result if key not in self.required_keys]
            if extra_keys:
                raise ParsingError(
                    f"JSON contains unexpected keys in strict mode: {', '.join(extra_keys)}"
                )

    def _validate_nesting(self) -> None:
        """Validate JSON nesting depth."""

        def check_nesting(obj: Any, max_depth: int = 10, current_depth: int = 0) -> None:
            if current_depth > max_depth:
                raise ParsingError(f"JSON nesting depth exceeds maximum of {max_depth}")
            if isinstance(obj, dict):
                for value in obj.values():
                    check_nesting(value, max_depth, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_nesting(item, max_depth, current_depth + 1)

        check_nesting(self.result)


class JSONParseCommandFactory:
    """Factory for creating JSON parsing commands."""

    @staticmethod
    def create_command(
        text: str, required_keys: list[str], max_size: int = MAX_JSON_SIZE, strict: bool = True
    ) -> JSONParseCommand:
        """Create a JSON parsing command."""
        return JSONParseCommand(text, required_keys, max_size, strict)


class JSONParseCommandExecutor:
    """Executor for JSON parsing commands."""

    @staticmethod
    @handle_parsing_errors("Failed to parse LLM JSON response")
    def execute_command(
        text: str, required_keys: list[str], max_size: int = MAX_JSON_SIZE, strict: bool = True
    ) -> dict[str, Any]:
        """Execute a JSON parsing command."""
        command = JSONParseCommandFactory.create_command(text, required_keys, max_size, strict)
        return command.execute()
