"""Common validation models and utilities for enhanced input validation."""

import html
import re
import uuid
from datetime import datetime
from enum import Enum
from re import Pattern
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import StringConstraints as PydanticStringConstraints


class ValidationErrorResponse(BaseModel):
    """Standardized validation error response format."""

    error: str = "Validation failed"
    message: str = "Request validation failed. Please check the provided data."
    details: list[dict[str, Any]] = []
    severity: str = "error"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None
    path: str | None = None
    method: str | None = None
    error_code: str = "VALIDATION_ERROR"

    def add_error(
        self,
        field: str,
        message: str,
        error_type: str = "value_error",
        severity: str = "error",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a validation error detail."""
        error_detail = {
            "field": field,
            "message": message,
            "type": error_type,
            "severity": severity,
            "context": context,
        }
        self.details.append(error_detail)

    def has_errors(self) -> bool:
        """Check if there are any error-level validation issues."""
        return any(detail.get("severity") == "error" for detail in self.details)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level validation issues."""
        return any(detail.get("severity") == "warning" for detail in self.details)

    def get_error_count(self) -> int:
        """Get total number of validation errors."""
        return len(self.details)


class BaseValidatedModel(BaseModel):
    """Base model with common validation methods and sanitization."""

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True,  # Use enum values in serialization
        extra="forbid",  # Forbid extra fields
        validate_by_name=True,  # Allow population by field name (Pydantic v2)
    )

    def sanitize_string_fields(self) -> None:
        """Sanitize all string fields in the model."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                sanitized_value = sanitize_input(field_value)
                setattr(self, field_name, sanitized_value)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, v):
        """Strip whitespace from string fields."""
        if isinstance(v, str):
            return v.strip()
        return v


class StringConstraints:
    """Common string validation constraints."""

    # Agent IDs: alphanumeric, underscore, hyphen (3-50 chars)
    AGENT_ID = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9_-]+$", min_length=3, max_length=50, strip_whitespace=True
        ),
    ]

    # Safe text: alphanumeric, spaces, common punctuation
    SAFE_TEXT = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9\s.,!?'\"()\-:;]*$",
            min_length=1,
            max_length=1000,
            strip_whitespace=True,
        ),
    ]

    # Role/Goal text: more permissive but still safe
    ROLE_TEXT = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9\s.,!?'\"()\-:;/]+$",
            min_length=5,
            max_length=500,
            strip_whitespace=True,
        ),
    ]

    # Model names: alphanumeric and common separators
    MODEL_NAME = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9\-:_/]+$",
            min_length=3,
            max_length=100,
            strip_whitespace=True,
        ),
    ]

    # Tool names: alphanumeric and underscore
    TOOL_NAME = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9_]+$", min_length=3, max_length=50, strip_whitespace=True
        ),
    ]

    # File names: alphanumeric, underscore, hyphen, dot
    FILENAME = Annotated[
        str,
        PydanticStringConstraints(
            pattern=r"^[a-zA-Z0-9_.\-]+$",
            min_length=1,
            max_length=255,
            strip_whitespace=True,
        ),
    ]


class UUIDValidator:
    """UUID validation utilities."""

    @staticmethod
    def validate_uuid(uuid_str: str) -> str:
        """Validate UUID format."""
        try:
            uuid.UUID(uuid_str)
            return uuid_str
        except ValueError:
            raise ValueError(f"Invalid UUID format: {uuid_str}") from None

    @staticmethod
    def validate_uuid_list(uuid_list: list[str]) -> list[str]:
        """Validate list of UUIDs."""
        validated_uuids = []
        for uuid_str in uuid_list:
            try:
                uuid.UUID(uuid_str)
                validated_uuids.append(uuid_str)
            except ValueError:
                raise ValueError(f"Invalid UUID in list: {uuid_str}") from None
        return validated_uuids


class NumericConstraints:
    """Numeric validation constraints."""

    # Pagination limits
    MIN_PAGE = 1
    MAX_PAGE = 1000
    MIN_PAGE_SIZE = 1
    MAX_PAGE_SIZE = 100

    # Rate limiting
    MIN_RATE_LIMIT = 1
    MAX_RATE_LIMIT = 10000

    # File size limits (in bytes)
    MIN_FILE_SIZE = 1
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB for chunked uploads

    # Message length limits
    MIN_MESSAGE_LENGTH = 1
    MAX_MESSAGE_LENGTH = 10000

    # Agent configuration limits
    MIN_AGENT_NAME_LENGTH = 3
    MAX_AGENT_NAME_LENGTH = 100
    MIN_AGENT_DESCRIPTION_LENGTH = 10
    MAX_AGENT_DESCRIPTION_LENGTH = 2000


class ValidationPatterns:
    """Common regex patterns for validation."""

    # Email pattern (more restrictive than EmailStr)
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    # URL pattern for API endpoints
    API_ENDPOINT_PATTERN = re.compile(r"^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9._-]*)*$")

    # File extension pattern
    FILE_EXTENSION_PATTERN = re.compile(r"^\.[a-zA-Z0-9]{1,10}$")

    # Alphanumeric with spaces and common punctuation
    SAFE_TEXT_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,!?'\"()\-:;/]+$")

    # Script/XSS detection pattern
    SCRIPT_PATTERN = re.compile(
        r"<script.*?>.*?</script.*?>|<.*?(javascript|onload|onclick|onerror).*?>",
        re.IGNORECASE | re.DOTALL,
    )

    # Command injection pattern
    COMMAND_INJECTION_PATTERN = re.compile(r"(;|\||&&|`|\$|\(|\)|<|>|\\n|\\r|\\t)")

    # Path traversal pattern
    PATH_TRAVERSAL_PATTERN = re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c|%252e%252e%252f)")

    # UUID pattern
    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )


class SanitizationLevel(str, Enum):
    """Levels of input sanitization."""

    STRICT = "strict"  # Only alphanumeric and basic punctuation
    MODERATE = "moderate"  # Allow more punctuation but block scripts
    PERMISSIVE = "permissive"  # Allow most safe characters
    NONE = "none"  # No sanitization (use with caution)


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


def sanitize_input(text: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> str:
    """
    Sanitize input text to prevent XSS and injection attacks.

    Args:
        text: Input text to sanitize
        level: Sanitization level to apply

    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return text

    # Remove null bytes
    text = text.replace("\x00", "")

    # Basic HTML entity encoding for dangerous characters
    text = html.escape(text)

    # Apply level-specific sanitization
    if level == SanitizationLevel.STRICT:
        # Only allow alphanumeric and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    elif level == SanitizationLevel.MODERATE:
        # Remove script tags and dangerous patterns
        text = ValidationPatterns.SCRIPT_PATTERN.sub("", text)
        text = ValidationPatterns.COMMAND_INJECTION_PATTERN.sub("", text)
        text = ValidationPatterns.PATH_TRAVERSAL_PATTERN.sub("", text)
        # Remove SQL injection patterns
        text = re.sub(
            r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script|declare|truncate)\b",
            "",
            text,
        )
        text = re.sub(r"(--|#|/\*|\*/)", "", text)
    elif level == SanitizationLevel.PERMISSIVE:
        # Only remove obvious script tags
        text = ValidationPatterns.SCRIPT_PATTERN.sub("", text)

    # Remove excessive whitespace
    return re.sub(r"\s+", " ", text).strip()


def validate_string_length(text: str, min_length: int, max_length: int) -> str:
    """
    Validate string length within specified bounds.

    Args:
        text: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Validated string

    Raises:
        ValueError: If string length is outside bounds
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    text = text.strip()
    length = len(text)

    if length < min_length:
        raise ValueError(f"String length {length} is below minimum {min_length}")

    if length > max_length:
        raise ValueError(f"String length {length} exceeds maximum {max_length}")

    return text


def validate_numeric_range(
    value: int | float,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> int | float:
    """
    Validate numeric value within specified range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Validated numeric value

    Raises:
        ValueError: If value is outside bounds
    """
    if not isinstance(value, (int, float)):
        raise ValueError("Input must be a number")

    if min_value is not None and value < min_value:
        raise ValueError(f"Value {value} is below minimum {min_value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"Value {value} exceeds maximum {max_value}")

    return value


def validate_pattern(text: str, pattern: str | Pattern, message: str | None = None) -> str:
    """
    Validate text against a regex pattern.

    Args:
        text: Text to validate
        pattern: Regex pattern to match against
        message: Custom error message

    Returns:
        Validated text

    Raises:
        ValueError: If text doesn't match pattern
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    if not pattern.match(text):
        if message:
            raise ValueError(message)
        raise ValueError(f"Text does not match required pattern: {pattern.pattern}")

    return text


def validate_enum_value(value: str, enum_class: type, case_sensitive: bool = True) -> str:
    """
    Validate value against enum members.

    Args:
        value: Value to validate
        enum_class: Enum class to validate against
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Validated enum value

    Raises:
        ValueError: If value is not a valid enum member
    """
    if not case_sensitive:
        value = value.lower()
        enum_values = [member.value.lower() for member in enum_class]
    else:
        enum_values = [member.value for member in enum_class]

    if value not in enum_values:
        valid_values = ", ".join(enum_values)
        raise ValueError(f"Value '{value}' is not one of: {valid_values}")

    return value


class FieldValidationRule(BaseModel):
    """Individual field validation rule."""

    field_name: str
    rule_type: str  # "length", "pattern", "range", "custom"
    constraint: str | int | float | dict[str, Any]
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
