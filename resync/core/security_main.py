# pylint
"""Module for security functions and input validation.

v5.9.4: Critical Fixes:
- Added Unicode support (PT-BR, ES, FR, DE, etc.)
- Non-destructive sanitization: rejects instead of silently modifying
- New validate_* methods returning informative errors
"""

import logging
import os
import re
from typing import Annotated, Any, TypeAlias

from fastapi import Path

# =============================================================================
# INPUT VALIDATION PATTERNS (v5.9.4 - Unicode Support)
# =============================================================================

# Characters explicitly BLOCKED (XSS/Injection prevention)
# < > are blocked to prevent HTML/script injection
DANGEROUS_CHARS_PATTERN = re.compile(r"[<>]")

# v5.9.4: Unicode-safe pattern using \w with UNICODE flag
# Supports: João, São Paulo, Café, Fábrica, Müller, etc.
# Still blocks: < > (XSS prevention)
SAFE_STRING_PATTERN = re.compile(
    r"^[\w\s.,!?'\"()\-:;@&/+=\#%\[\]{}|~`*\\]*$", re.UNICODE
)

# Pattern to KEEP only safe characters (Unicode-aware)
SAFE_CHARS_ONLY = re.compile(r"[\w\s.,!?'\"()\-:;@&/+=\#%\[\]{}|~`*\\]", re.UNICODE)

# Stricter pattern for fields that should NOT have special characters
# Use for: usernames, IDs, slugs
STRICT_ALPHANUMERIC_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]*$")
STRICT_CHARS_ONLY = re.compile(r"[a-zA-Z0-9_\-]")

# Email validation pattern (simplified RFC 5321)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# TWS Job name pattern (alphanumeric, underscore, hyphen, up to 40 chars)
TWS_JOB_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")

# TWS Workstation pattern
TWS_WORKSTATION_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,16}$")


SanitizedValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | dict[str, "SanitizedValue"]
    | list["SanitizedValue"]
)


class ValidationResult:
    """Validation result with error details."""

    def __init__(
        self,
        is_valid: bool,
        value: str = "",
        error: str | None = None,
        invalid_chars: list[str] | None = None,
    ):
        self.is_valid = is_valid
        self.value = value
        self.error = error
        self.invalid_chars = invalid_chars or []

    def __bool__(self) -> bool:
        return self.is_valid


class InputSanitizer:
    """
    Class for sanitizing and validating user inputs.

    v5.9.4: Updated methods to:
    - Support Unicode characters (accents, cedilla, etc.)
    - Reject invalid input with informative error (do not modify silently)
    - Provide ValidationResult with problem details
    """

    @staticmethod
    def sanitize_environment_value(
        env_var_name: str, default_value: Any, value_type: type = str
    ) -> Any:
        """
        Sanitize and validate environment variable values.

        Args:
            env_var_name: Name of the environment variable
            default_value: Default value if env var is not set or invalid
            value_type: Expected type of the value (str, int, float, bool)

        Returns:
            Sanitized value of the specified type
        """
        raw_value = os.getenv(env_var_name, default_value)

        try:
            if value_type is str:
                return str(raw_value)
            if value_type is int:
                return int(raw_value)
            if value_type is float:
                return float(raw_value)
            if value_type is bool:
                # Handle boolean conversion from string
                if isinstance(raw_value, str):
                    return raw_value.lower() in ("true", "1", "yes", "on")
                return bool(raw_value)
            # For other types, try to convert using the type constructor
            return value_type(raw_value)
        except (ValueError, TypeError):
            # If conversion fails, return the default value
            logger = logging.getLogger(__name__)
            logger.warning(
                "Invalid value for environment variable "
                f"{env_var_name}: {raw_value}. "
                f"Using default: {default_value}"
            )
            return default_value

    @staticmethod
    def validate_string(text: str, max_length: int = 1000) -> ValidationResult:
        """
        Validate string and return detailed result (does not modify input).

        v5.9.4: New API returning informative error instead of modifying data.
        Supports Unicode (accents, international characters).

        Args:
            text: User input string.
            max_length: Maximum allowed length.

        Returns:
            ValidationResult with is_valid, value, error, and invalid_chars.
        """
        if not text:
            return ValidationResult(True, "")

        if len(text) > max_length:
            return ValidationResult(
                False,
                text,
                f"Text exceeds maximum length of {max_length} characters",
            )

        # Check for dangerous characters (< >)
        dangerous = DANGEROUS_CHARS_PATTERN.findall(text)
        if dangerous:
            return ValidationResult(
                False,
                text,
                "Text contains potentially dangerous characters",
                invalid_chars=list(set(dangerous)),
            )

        # Check if all characters are safe (Unicode-aware)
        if not SAFE_STRING_PATTERN.match(text):
            # Identify invalid characters
            invalid = [c for c in text if not SAFE_CHARS_ONLY.match(c)]
            return ValidationResult(
                False,
                text,
                "Text contains invalid characters",
                invalid_chars=list(set(invalid)),
            )

        return ValidationResult(True, text)

    @staticmethod
    def sanitize_string(
        text: str, max_length: int = 1000, strip_dangerous: bool = True
    ) -> str:
        """
        Remove potentially dangerous characters from an input string.

        v5.9.4: Updated to support Unicode. Behavior:
        - strip_dangerous=True: Removes only < > (XSS), keeps accents
        - strip_dangerous=False: Returns empty if contains invalid characters

        NOTE: For strict validation, use validate_string() which returns detailed error.

        Args:
            text: User input string.
            max_length: Maximum length allowed for the string.
            strip_dangerous: If True, removes dangerous chars.
                If False, returns empty on dangerous input.

        Returns:
            Sanitized string.
        """
        if not text:
            return ""

        # Truncate to max length
        text = text[:max_length]

        if strip_dangerous:
            # Remove only dangerous characters (< >) but keep Unicode
            text = DANGEROUS_CHARS_PATTERN.sub("", text)
            # Keep only safe characters (now includes Unicode)
            return "".join(SAFE_CHARS_ONLY.findall(text))
        # Strict mode: return empty if any dangerous character present
        if SAFE_STRING_PATTERN.match(text):
            return text
        return ""

    @staticmethod
    def sanitize_string_strict(text: str, max_length: int = 100) -> str:
        """
        Strict sanitization for IDs, usernames, and slugs.
        Allows only alphanumeric, underscore, and hyphen.

        Args:
            text: Input string.
            max_length: Maximum allowed length.

        Returns:
            Sanitized string (only [a-zA-Z0-9_-]).
        """
        if not text:
            return ""
        text = text[:max_length]
        # Keep only strict alphanumeric characters
        return "".join(STRICT_CHARS_ONLY.findall(text))

    @staticmethod
    def sanitize_tws_job_name(job_name: str) -> str:
        """
        Sanitize TWS job name.
        Allows only valid characters for HWA/TWS job names.

        Args:
            job_name: Job name to sanitize.

        Returns:
            Sanitized job name or empty string if invalid.
        """
        if not job_name:
            return ""
        job_name = job_name.strip().upper()[:40]
        if TWS_JOB_PATTERN.match(job_name):
            return job_name
        # Strip invalid chars
        return "".join(re.findall(r"[A-Za-z0-9_\-]", job_name))[:40]

    @staticmethod
    def sanitize_tws_workstation(workstation: str) -> str:
        """
        Sanitize TWS workstation name.

        Args:
            workstation: Workstation name.

        Returns:
            Sanitized workstation.
        """
        if not workstation:
            return ""
        workstation = workstation.strip().upper()[:16]
        if TWS_WORKSTATION_PATTERN.match(workstation):
            return workstation
        return "".join(re.findall(r"[A-Za-z0-9_\-]", workstation))[:16]

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format.

        Args:
            email: Email string to validate.

        Returns:
            True if the email is valid, False otherwise.
        """
        if not email or len(email) > 254:
            return False
        return bool(EMAIL_PATTERN.match(email))

    @staticmethod
    def sanitize_email(email: str) -> str:
        """
        Sanitize and validate an email.

        Args:
            email: Email string.

        Returns:
            Sanitized email or empty string if invalid.
        """
        if not email:
            return ""
        email = email.strip().lower()[:254]
        if InputSanitizer.validate_email(email):
            return email
        return ""

    @staticmethod
    def sanitize_dict(
        data: dict[str, object], max_depth: int = 3, current_depth: int = 0
    ) -> dict[str, "SanitizedValue"]:
        """
        Recursively sanitize a dictionary.

        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            Sanitized dictionary
        """
        if current_depth >= max_depth:
            return {}

        sanitized: dict[str, SanitizedValue] = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = InputSanitizer.sanitize_string(str(key), 100)

            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = InputSanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = InputSanitizer.sanitize_dict(
                    value, max_depth, current_depth + 1
                )
            elif isinstance(value, list):
                sanitized[clean_key] = InputSanitizer.sanitize_list(
                    value, max_depth, current_depth + 1
                )
            elif isinstance(value, (int, float, bool)):
                sanitized[clean_key] = value
            else:
                # Convert other types to string and sanitize
                sanitized[clean_key] = InputSanitizer.sanitize_string(str(value))

        return sanitized

    @staticmethod
    def sanitize_list(
        data: list[object], max_depth: int = 3, current_depth: int = 0
    ) -> list["SanitizedValue"]:
        """
        Recursively sanitize a list.

        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            Sanitized list
        """
        if current_depth >= max_depth:
            return []

        sanitized: list[SanitizedValue] = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(InputSanitizer.sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(
                    InputSanitizer.sanitize_dict(item, max_depth, current_depth + 1)
                )
            elif isinstance(item, list):
                sanitized.append(
                    InputSanitizer.sanitize_list(item, max_depth, current_depth + 1)
                )
            elif isinstance(item, (int, float, bool)):
                sanitized.append(item)
            else:
                # Convert other types to string and sanitize
                sanitized.append(InputSanitizer.sanitize_string(str(item)))

        return sanitized


def sanitize_input(text: str, strip_dangerous: bool = True) -> str:
    """
    Remove potentially dangerous characters from an input string.

    v5.9.4: Now supports Unicode characters (accents, cedilla, etc.).

    Args:
        text: User input string.
        strip_dangerous: If True, strips dangerous chars. If False, rejects entirely.

    Returns:
        The sanitized string.
    """
    return InputSanitizer.sanitize_string(text, strip_dangerous=strip_dangerous)


def validate_input(text: str, max_length: int = 1000) -> ValidationResult:
    """
    Validate string and return detailed result.

    v5.9.4: New function for non-destructive validation with informative error.

    Args:
        text: String to validate.
        max_length: Maximum allowed length.

    Returns:
        ValidationResult with is_valid, value, error and invalid_chars.
    """
    return InputSanitizer.validate_string(text, max_length)


def sanitize_input_strict(text: str) -> str:
    """
    Strict sanitization - only alphanumeric, underscore, and hyphen.
    Use for: IDs, usernames, slugs.

    Args:
        text: The input string.

    Returns:
        Sanitized string.
    """
    return InputSanitizer.sanitize_string_strict(text)


def sanitize_tws_job_name(job_name: str) -> str:
    """
    Sanitize TWS/HWA job name.

    Args:
        job_name: Job name.

    Returns:
        Sanitized name.
    """
    return InputSanitizer.sanitize_tws_job_name(job_name)


def sanitize_tws_workstation(workstation: str) -> str:
    """
    Sanitize TWS/HWA workstation name.

    Args:
        workstation: Workstation name.

    Returns:
        Sanitized workstation.
    """
    return InputSanitizer.sanitize_tws_workstation(workstation)


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email string.

    Returns:
        True if valid.
    """
    return InputSanitizer.validate_email(email)


# Annotated type for IDs, ensuring they follow a safe format.
SafeAgentID = Annotated[
    str, Path(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
]

# Annotated type for emails
SafeEmail = Annotated[
    str,
    Path(max_length=254, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
]

# Annotated type for TWS job names (v5.4.0)
SafeTWSJobName = Annotated[
    str, Path(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
]

# Annotated type for TWS workstations (v5.4.0)
SafeTWSWorkstation = Annotated[
    str, Path(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9_\-]+$")
]


__all__ = [
    "InputSanitizer",
    "ValidationResult",
    "sanitize_input",
    "validate_input",
    "sanitize_input_strict",
    "sanitize_tws_job_name",
    "sanitize_tws_workstation",
    "validate_email",
    "SafeAgentID",
    "SafeEmail",
    "SafeTWSJobName",
    "SafeTWSWorkstation",
    "SAFE_STRING_PATTERN",
    "STRICT_ALPHANUMERIC_PATTERN",
    "TWS_JOB_PATTERN",
    "TWS_WORKSTATION_PATTERN",
]
