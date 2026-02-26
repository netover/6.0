"""
Security module for input validation and sanitization.

Provides comprehensive security utilities for:
- Input validation with detailed error reporting
- String sanitization with Unicode support
- Environment variable handling
- Type-safe validation results

Version: 6.1.0
Changes from v6.0.0 (Hybrid 360° Audit — Feb 2026):
- [P0-01] validate_string() agora normaliza NFKC e allow_unicode tem efeito real
- [P1-01] SafeEmail/SafeTWSJobName/SafeTWSWorkstation migrados Path() → Field()
- [P1-02] sanitize_environment_value: raw_value não logado (log injection fix)
- [P1-03] validate_tws_job_name: re.match inline substituído por STRICT_CHARS_ONLY
- [P1-04] sanitize_environment_value: docstring corrigida + raise_on_error adicionado
- [P1-05] sanitize_string strip_dangerous=False: validação antes de qualquer mutação
- [P2-01] @classmethod sem cls convertidos para @staticmethod
- [P2-02] invalid_chars property retorna tuple imutável
- [P2-03] _logger anotado com ClassVar
- [P2-04] SanitizedValue migrado para type statement (PEP 695)
- [P2-05] Exceções integradas via raise_on_error em todos validate_*
- [P2-06] \\# removido de SAFE_STRING_PATTERN e SAFE_CHARS_ONLY (ruff W605)

SECURITY NOTE: Este módulo provê defesa em profundidade para inputs.
Sanitização NÃO substitui escaping contextual na camada de apresentação
(Jinja |e, React JSX, parameterized SQL, etc.).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from fastapi import Path
from pydantic import Field

# =============================================================================
# TYPE ALIASES (v6.1.0 — PEP 695, Python 3.12+)
# =============================================================================

if TYPE_CHECKING:
    # SupportsDunderLT removido — era import morto (P3 fix)
    pass

# [P2-04] type statement (PEP 695) suporta recursão nativa em Python 3.12+
# Em Python 3.14 é a forma canônica e resolve corretamente em runtime
type SanitizedValue = (
    str | int | float | bool | dict[str, SanitizedValue] | list[SanitizedValue]
)


# =============================================================================
# CUSTOM EXCEPTIONS (v6.1.0 — agora integradas ao fluxo via raise_on_error)
# =============================================================================


class SecurityValidationError(Exception):
    """Base exception for security validation failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        field_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.field_name = field_name

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.error_code,
            "field": self.field_name,
        }


class InvalidInputError(SecurityValidationError):
    """Raised when input contains invalid characters."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        invalid_chars: list[str] | None = None,
    ) -> None:
        super().__init__(message, "INVALID_INPUT", field_name)
        self.invalid_chars = invalid_chars or []


class InputTooLongError(SecurityValidationError):
    """Raised when input exceeds maximum length."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        max_length: int | None = None,
        actual_length: int | None = None,
    ) -> None:
        super().__init__(message, "INPUT_TOO_LONG", field_name)
        self.max_length = max_length
        self.actual_length = actual_length


class DangerousInputError(SecurityValidationError):
    """Raised when input contains potentially dangerous characters."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        dangerous_chars: list[str] | None = None,
    ) -> None:
        super().__init__(message, "DANGEROUS_INPUT", field_name)
        self.dangerous_chars = dangerous_chars or []


# =============================================================================
# INPUT VALIDATION PATTERNS (v6.1.0)
# =============================================================================

# Characters explicitly BLOCKED (XSS/Injection prevention)
DANGEROUS_CHARS_PATTERN: re.Pattern[str] = re.compile(r"[<>]")

# [P2-06] \# removido — # não precisa de escape em character class sem re.VERBOSE
# [P3]    re.UNICODE removido — redundante para str patterns em Python 3
SAFE_STRING_PATTERN: re.Pattern[str] = re.compile(
    r"^[\w\s.,!?'\"()\-:;@&/+=#%\[\]{}|~`*\\]*$"
)

SAFE_CHARS_ONLY: re.Pattern[str] = re.compile(
    r"[\w\s.,!?'\"()\-:;@&/+=#%\[\]{}|~`*\\]"
)

# Padrão ASCII-only para allow_unicode=False (P0-01 fix)
ASCII_SAFE_STRING_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z0-9_\s.,!?'\"()\-:;@&/+=#%\[\]{}|~`*\\]*$"
)

ASCII_SAFE_CHARS_ONLY: re.Pattern[str] = re.compile(
    r"[A-Za-z0-9_\s.,!?'\"()\-:;@&/+=#%\[\]{}|~`*\\]"
)

# Stricter pattern for IDs, usernames, slugs
STRICT_ALPHANUMERIC_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]*$")
STRICT_CHARS_ONLY: re.Pattern[str] = re.compile(r"[a-zA-Z0-9_-]")

# Email (simplified RFC 5321 — documented limitation)
EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# TWS patterns
TWS_JOB_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
TWS_WORKSTATION_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]{1,16}$")


# =============================================================================
# UTILITY FUNCTIONS (v6.1.0)
# =============================================================================


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text using NFKC form.

    Converts compatibility equivalents and composes combining characters.
    Example: 'e' + combining accent → 'é' (single code point).

    Args:
        text: Input text to normalize.

    Returns:
        NFKC-normalized text, or empty string if input is falsy.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Safely truncate text to specified length.

    Never exceeds max_length. If suffix fits, appends it to indicate truncation.

    Args:
        text: Input text.
        max_length: Maximum allowed length (inclusive).
        suffix: Suffix to append when truncated (default: "...").

    Returns:
        Text guaranteed to be <= max_length characters.
    """
    if not text or len(text) <= max_length:
        return text
    if len(suffix) < max_length:
        return text[: max_length - len(suffix)] + suffix
    return text[:max_length]


# =============================================================================
# VALIDATION RESULT (v6.1.0)
# =============================================================================


class ValidationResult:
    """
    Immutable validation result with error details.

    Attributes:
        is_valid: Whether the validation passed.
        value: The (possibly normalized) input value.
        error: Human-readable error message.
        invalid_chars: Immutable tuple of invalid characters found.
        error_code: Machine-readable error code.
    """

    __slots__ = (
        "_is_valid",
        "_value",
        "_error",
        "_invalid_chars",
        "_error_code",
    )

    def __init__(
        self,
        is_valid: bool,
        value: str = "",
        error: str | None = None,
        invalid_chars: list[str] | tuple[str, ...] | None = None,
        error_code: str | None = None,
    ) -> None:
        self._is_valid = is_valid
        self._value = value
        self._error = error
        # [P2-02] Armazenado como tuple — imutável, caller não pode mutar estado
        self._invalid_chars: tuple[str, ...] = tuple(invalid_chars or [])
        self._error_code = error_code

    @property
    def is_valid(self) -> bool:
        """Whether validation passed."""
        return self._is_valid

    @property
    def value(self) -> str:
        """The (possibly normalized) input value."""
        return self._value

    @property
    def error(self) -> str | None:
        """Human-readable error message."""
        return self._error

    @property
    def invalid_chars(self) -> tuple[str, ...]:
        """[P2-02] Immutable tuple of invalid characters found."""
        return self._invalid_chars

    @property
    def error_code(self) -> str | None:
        """Machine-readable error code."""
        return self._error_code

    def __bool__(self) -> bool:
        """Support bool() conversion for easy truthiness checks."""
        return self._is_valid

    def __repr__(self) -> str:
        return (
            f"ValidationResult(is_valid={self._is_valid!r}, "
            f"value={self._value!r}, error={self._error!r}, "
            f"invalid_chars={self._invalid_chars!r}, "
            f"error_code={self._error_code!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self._is_valid,
            "value": self._value,
            "error": self._error,
            "invalid_chars": list(self._invalid_chars),  # JSON-serializable
            "error_code": self._error_code,
        }


# =============================================================================
# INPUT SANITIZER (v6.1.0)
# =============================================================================


class InputSanitizer:
    """
    Class for sanitizing and validating user inputs.

    Two validation approaches:
    1. validate_*: non-destructive, returns ValidationResult with details.
       Use raise_on_error=True to raise typed exceptions instead.
    2. sanitize_*: returns cleaned strings (may normalize/truncate input).

    Best practice: use validate_* for security-critical checks where you need
    to know exactly what went wrong. Use sanitize_* when you just need a safe
    string to work with.
    """

    # [P2-03] ClassVar annotation — mypy strict compliance
    _logger: ClassVar[logging.Logger | None] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """Get or create logger (lazy initialization)."""
        if cls._logger is None:
            cls._logger = logging.getLogger(f"{__name__}.InputSanitizer")
        return cls._logger

    @staticmethod
    def sanitize_environment_value(
        env_var_name: str,
        default_value: Any,
        value_type: type[str] | type[int] | type[float] | type[bool] = str,
        *,
        raise_on_error: bool = False,
    ) -> Any:
        """
        Sanitize and validate environment variable values.

        Args:
            env_var_name: Name of the environment variable.
            default_value: Default value if env var is not set or invalid.
            value_type: Expected type (str, int, float, bool).
            raise_on_error: If True, raises InvalidInputError on conversion
                failure instead of returning default silently.

        Returns:
            Sanitized value of the specified type, or default_value on failure
            (when raise_on_error=False).

        Raises:
            InvalidInputError: If conversion fails and raise_on_error=True.
        """
        raw_value = os.getenv(env_var_name)

        if raw_value is None:
            return default_value

        if value_type is str:
            return raw_value

        try:
            if value_type is bool:
                if isinstance(raw_value, str):
                    return raw_value.lower() in ("true", "1", "yes", "on")
                return bool(raw_value)
            if value_type is int:
                return int(raw_value)
            if value_type is float:
                return float(raw_value)
            return value_type(raw_value)  # type: ignore[call-arg]
        except (ValueError, TypeError) as exc:
            # [P1-02] NUNCA logar raw_value — pode conter segredos (DB_PASSWORD, JWT_SECRET)
            # repr() escapa \n, \r, ANSI codes; [:200] limita tamanho
            InputSanitizer._get_logger().warning(
                "Invalid value for environment variable %s "
                "(value redacted). Using default: %r. Error: %s",
                env_var_name[:100],
                default_value,
                type(exc).__name__,
            )
            # [P1-04] raise_on_error ativa exceção tipada em vez de silêncio
            if raise_on_error:
                raise InvalidInputError(
                    message=f"Invalid value for env var '{env_var_name}'",
                    field_name=env_var_name,
                ) from exc
            return default_value

    @staticmethod
    def validate_string(
        text: str,
        max_length: int = 1000,
        allow_unicode: bool = True,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate string and return detailed result (non-destructive).

        [P0-01] Agora aplica NFKC antes de validar para consistência com
        sanitize_string(). allow_unicode=False restringe ao conjunto ASCII.

        Args:
            text: User input string.
            max_length: Maximum allowed length.
            allow_unicode: If True, accepts Unicode (NFKC normalized).
                If False, only ASCII printable characters allowed.
            raise_on_error: If True, raises typed exception on failure.

        Returns:
            ValidationResult with is_valid, value (normalized), error,
            invalid_chars, and error_code.

        Raises:
            InputTooLongError: If text exceeds max_length and raise_on_error=True.
            DangerousInputError: If dangerous chars found and raise_on_error=True.
            InvalidInputError: If invalid chars found and raise_on_error=True.
        """
        if not text:
            return ValidationResult(is_valid=True, value="", error_code="EMPTY")

        # Length check on original (before normalization — NFKC rarely changes length)
        actual_length = len(text)
        if actual_length > max_length:
            result = ValidationResult(
                is_valid=False,
                value=text,
                error=(
                    f"Text exceeds maximum length of {max_length} "
                    f"characters (got {actual_length})"
                ),
                error_code="TOO_LONG",
            )
            if raise_on_error:
                raise InputTooLongError(
                    result.error or "",  # type: ignore[arg-type]
                    max_length=max_length,
                    actual_length=actual_length,
                )
            return result

        # [P0-01] Normalizar NFKC para consistência com sanitize_string()
        # Garante que "e + combining accent" e "é" sejam tratados igualmente
        normalized = normalize_unicode(text) if allow_unicode else text

        # Dangerous chars check em ambos original e normalizado (defense-in-depth)
        dangerous_orig = set(DANGEROUS_CHARS_PATTERN.findall(text))
        dangerous_norm = set(DANGEROUS_CHARS_PATTERN.findall(normalized))
        dangerous = list(dangerous_orig | dangerous_norm)
        if dangerous:
            result = ValidationResult(
                is_valid=False,
                value=text,
                error=(
                    "Text contains potentially dangerous characters (< >) "
                    "which may indicate XSS attack"
                ),
                invalid_chars=dangerous,
                error_code="DANGEROUS_CHARS",
            )
            if raise_on_error:
                raise DangerousInputError(
                    result.error or "",
                    dangerous_chars=dangerous,
                )
            return result

        # [P0-01] Seleciona padrão baseado em allow_unicode
        str_pattern = SAFE_STRING_PATTERN if allow_unicode else ASCII_SAFE_STRING_PATTERN
        char_pattern = SAFE_CHARS_ONLY if allow_unicode else ASCII_SAFE_CHARS_ONLY

        if not str_pattern.match(normalized):
            invalid = list({c for c in normalized if not char_pattern.match(c)})
            result = ValidationResult(
                is_valid=False,
                value=text,
                error=f"Text contains {len(invalid)} invalid character(s)",
                invalid_chars=invalid,
                error_code="INVALID_CHARS",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", invalid_chars=invalid)
            return result

        # Retorna valor normalizado — consistente com sanitize_string()
        return ValidationResult(is_valid=True, value=normalized, error_code="VALID")

    @staticmethod  # [P2-01] era @classmethod sem uso de cls
    def validate_string_strict(
        text: str,
        max_length: int = 100,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate strict alphanumeric input (IDs, usernames, slugs).

        Args:
            text: User input string.
            max_length: Maximum allowed length.
            raise_on_error: If True, raises typed exception on failure.

        Returns:
            ValidationResult with detailed error information.
        """
        if not text:
            return ValidationResult(is_valid=True, value="", error_code="EMPTY")

        if len(text) > max_length:
            result = ValidationResult(
                is_valid=False,
                value=text,
                error=f"Text exceeds maximum length of {max_length} characters",
                error_code="TOO_LONG",
            )
            if raise_on_error:
                raise InputTooLongError(
                    result.error or "",
                    max_length=max_length,
                    actual_length=len(text),
                )
            return result

        if not STRICT_ALPHANUMERIC_PATTERN.match(text):
            invalid = list({c for c in text if not STRICT_CHARS_ONLY.match(c)})
            result = ValidationResult(
                is_valid=False,
                value=text,
                error=(
                    f"Text contains invalid characters. "
                    f"Only [a-zA-Z0-9_-] allowed, found: {invalid}"
                ),
                invalid_chars=invalid,
                error_code="INVALID_CHARS",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", invalid_chars=invalid)
            return result

        return ValidationResult(is_valid=True, value=text, error_code="VALID")

    @staticmethod  # [P2-01] era @classmethod sem uso de cls
    def validate_email(
        email: str,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate email format with detailed error reporting.

        NOTE: Uses simplified RFC 5321 pattern. Does not support IDN domains,
        IP literals, or quoted local parts. Use email-validator library if
        full RFC compliance is required.

        Args:
            email: Email string to validate.
            raise_on_error: If True, raises typed exception on failure.

        Returns:
            ValidationResult with detailed error information.
            On success, value is lowercased email.
        """
        if not email:
            result = ValidationResult(
                is_valid=False,
                value="",
                error="Email is required",
                error_code="EMPTY",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", field_name="email")
            return result

        if len(email) > 254:
            result = ValidationResult(
                is_valid=False,
                value=email,
                error="Email exceeds maximum length of 254 characters",
                error_code="TOO_LONG",
            )
            if raise_on_error:
                raise InputTooLongError(
                    result.error or "",
                    field_name="email",
                    max_length=254,
                    actual_length=len(email),
                )
            return result

        if not EMAIL_PATTERN.match(email):
            result = ValidationResult(
                is_valid=False,
                value=email,
                error="Email format is invalid (expected: user@domain.tld)",
                error_code="INVALID_FORMAT",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", field_name="email")
            return result

        return ValidationResult(is_valid=True, value=email.lower(), error_code="VALID")

    @staticmethod  # [P2-01] era @classmethod sem uso de cls
    def validate_tws_job_name(
        job_name: str,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate TWS job name format.

        Args:
            job_name: Job name to validate.
            raise_on_error: If True, raises typed exception on failure.

        Returns:
            ValidationResult. On success, value is uppercased job name.
        """
        if not job_name:
            result = ValidationResult(
                is_valid=False,
                value="",
                error="Job name is required",
                error_code="EMPTY",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", field_name="job_name")
            return result

        if len(job_name) > 40:
            result = ValidationResult(
                is_valid=False,
                value=job_name,
                error="Job name exceeds maximum length of 40 characters",
                error_code="TOO_LONG",
            )
            if raise_on_error:
                raise InputTooLongError(
                    result.error or "",
                    field_name="job_name",
                    max_length=40,
                    actual_length=len(job_name),
                )
            return result

        if not TWS_JOB_PATTERN.match(job_name):
            # [P1-03] STRICT_CHARS_ONLY pré-compilado em vez de re.match inline
            invalid = list({c for c in job_name if not STRICT_CHARS_ONLY.match(c)})
            result = ValidationResult(
                is_valid=False,
                value=job_name,
                error=(
                    f"Job name contains invalid characters. "
                    f"Only [A-Za-z0-9_-] allowed, found: {invalid}"
                ),
                invalid_chars=invalid,
                error_code="INVALID_CHARS",
            )
            if raise_on_error:
                raise InvalidInputError(result.error or "", invalid_chars=invalid)
            return result

        return ValidationResult(
            is_valid=True, value=job_name.upper(), error_code="VALID"
        )

    @staticmethod
    def sanitize_string(
        text: str,
        max_length: int = 1000,
        strip_dangerous: bool = True,
    ) -> str:
        """
        Remove potentially dangerous characters from an input string.

        [P1-05] strip_dangerous=False agora valida ANTES de qualquer mutação,
        preservando o contrato "non-destructive" documentado.

        Behavior:
        - strip_dangerous=True: normalizes NFKC, removes < >, keeps accents
        - strip_dangerous=False: validates original text first; only normalizes
          and returns if valid (non-destructive contract honored)

        Args:
            text: User input string.
            max_length: Maximum length allowed for the string.
            strip_dangerous: If True, strips dangerous chars.
                If False, returns empty string if any dangerous char present.

        Returns:
            Sanitized string.
        """
        if not text:
            return ""

        if strip_dangerous:
            # Modo permissivo: normaliza → trunca → remove < > → filtra
            text = normalize_unicode(text)
            text = truncate_text(text, max_length)
            text = DANGEROUS_CHARS_PATTERN.sub("", text)
            return "".join(SAFE_CHARS_ONLY.findall(text))

        # [P1-05] Modo estrito: valida PRIMEIRO o texto original sem mutar
        # Trunca apenas para limite de comprimento (operação reversível)
        candidate = text[:max_length]
        if SAFE_STRING_PATTERN.match(candidate):
            # Só normaliza após validação — mantém contrato non-destructive
            return normalize_unicode(candidate)
        return ""

    @staticmethod
    def sanitize_string_strict(
        text: str,
        max_length: int = 100,
    ) -> str:
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
        text = normalize_unicode(text)
        text = text[:max_length]
        return "".join(STRICT_CHARS_ONLY.findall(text))

    @staticmethod
    def sanitize_tws_job_name(job_name: str) -> str:
        """
        Sanitize TWS job name.

        Args:
            job_name: Job name to sanitize.

        Returns:
            Sanitized, uppercased job name or empty string if input is empty.
        """
        if not job_name:
            return ""
        job_name = normalize_unicode(job_name)
        job_name = job_name.strip().upper()[:40]
        if TWS_JOB_PATTERN.match(job_name):
            return job_name
        # [P1-03] STRICT_CHARS_ONLY pré-compilado
        return "".join(STRICT_CHARS_ONLY.findall(job_name))[:40]

    @staticmethod
    def sanitize_tws_workstation(workstation: str) -> str:
        """
        Sanitize TWS workstation name.

        Args:
            workstation: Workstation name.

        Returns:
            Sanitized, uppercased workstation name.
        """
        if not workstation:
            return ""
        workstation = normalize_unicode(workstation)
        workstation = workstation.strip().upper()[:16]
        if TWS_WORKSTATION_PATTERN.match(workstation):
            return workstation
        return "".join(STRICT_CHARS_ONLY.findall(workstation))[:16]

    @staticmethod
    def sanitize_email(email: str) -> str:
        """
        Sanitize and validate an email.

        Args:
            email: Email string.

        Returns:
            Lowercased sanitized email or empty string if invalid.
        """
        if not email:
            return ""
        email = email.strip().lower()[:254]
        result = InputSanitizer.validate_email(email)
        return result.value if result.is_valid else ""

    @staticmethod
    def sanitize_dict(
        data: dict[str, Any],
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> dict[str, SanitizedValue]:
        """
        Recursively sanitize a dictionary.

        Args:
            data: Dictionary to sanitize.
            max_depth: Maximum recursion depth.
            current_depth: Current recursion depth.

        Returns:
            Sanitized dictionary. Empty dict if max_depth reached (logged).
        """
        if current_depth >= max_depth:
            InputSanitizer._get_logger().warning(
                "sanitize_dict: max_depth=%d reached at depth=%d, "
                "discarding %d keys",
                max_depth,
                current_depth,
                len(data),
            )
            return {}

        sanitized: dict[str, SanitizedValue] = {}
        for key, value in data.items():
            clean_key = InputSanitizer.sanitize_string(str(key), 100)

            if not clean_key:
                InputSanitizer._get_logger().warning(
                    "sanitize_dict: key %r sanitized to empty string, skipping",
                    str(key)[:50],
                )
                continue

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
                sanitized[clean_key] = InputSanitizer.sanitize_string(str(value))

        return sanitized

    @staticmethod
    def sanitize_list(
        data: list[Any],
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> list[SanitizedValue]:
        """
        Recursively sanitize a list.

        Args:
            data: List to sanitize.
            max_depth: Maximum recursion depth.
            current_depth: Current recursion depth.

        Returns:
            Sanitized list. Empty list if max_depth reached (logged).
        """
        if current_depth >= max_depth:
            InputSanitizer._get_logger().warning(
                "sanitize_list: max_depth=%d reached at depth=%d, "
                "discarding %d items",
                max_depth,
                current_depth,
                len(data),
            )
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
                sanitized.append(InputSanitizer.sanitize_string(str(item)))

        return sanitized


# =============================================================================
# CONVENIENCE FUNCTIONS (v6.1.0)
# =============================================================================


def sanitize_input(text: str, strip_dangerous: bool = True) -> str:
    """Sanitize input string. See InputSanitizer.sanitize_string()."""
    return InputSanitizer.sanitize_string(text, strip_dangerous=strip_dangerous)


def validate_input(
    text: str,
    max_length: int = 1000,
    allow_unicode: bool = True,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Validate string. See InputSanitizer.validate_string()."""
    return InputSanitizer.validate_string(
        text, max_length, allow_unicode=allow_unicode, raise_on_error=raise_on_error
    )


def validate_input_strict(
    text: str,
    max_length: int = 100,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Strictly validate alphanumeric input. See InputSanitizer.validate_string_strict()."""
    return InputSanitizer.validate_string_strict(
        text, max_length, raise_on_error=raise_on_error
    )


def sanitize_input_strict(text: str, max_length: int = 100) -> str:
    """Strict sanitization. See InputSanitizer.sanitize_string_strict()."""
    return InputSanitizer.sanitize_string_strict(text, max_length)


def sanitize_tws_job_name(job_name: str) -> str:
    """Sanitize TWS/HWA job name."""
    return InputSanitizer.sanitize_tws_job_name(job_name)


def validate_tws_job_name(
    job_name: str,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Validate TWS/HWA job name."""
    return InputSanitizer.validate_tws_job_name(job_name, raise_on_error=raise_on_error)


def sanitize_tws_workstation(workstation: str) -> str:
    """Sanitize TWS/HWA workstation name."""
    return InputSanitizer.sanitize_tws_workstation(workstation)


def validate_email(
    email: str,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Validate email format. See InputSanitizer.validate_email()."""
    return InputSanitizer.validate_email(email, raise_on_error=raise_on_error)


def sanitize_email(email: str) -> str:
    """Sanitize and validate email."""
    return InputSanitizer.sanitize_email(email)


# =============================================================================
# FASTAPI TYPE ANNOTATIONS (v6.1.0)
# =============================================================================

# [P1-01] Field() para uso em Pydantic models, Body, Query, Schema
# Valida em qualquer contexto Pydantic v2, não apenas em path params

# [P1-01] Field() para uso em Pydantic models, Body, Query, Schema
# Valida em qualquer contexto Pydantic v2, não apenas em path params
# Python 3.12+ não precisa de : type = annotation - Annotated já é type alias

SafeAgentID = Annotated[
    str, Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
]

SafeEmail = Annotated[
    str,
    Field(
        max_length=254,
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    ),
]

SafeTWSJobName = Annotated[
    str, Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
]

SafeTWSWorkstation = Annotated[
    str, Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9_-]+$")
]

# Variante EXCLUSIVA para uso como path parameter em rotas FastAPI:
# @app.get("/agents/{agent_id}")
# async def get_agent(agent_id: SafeAgentIDPath): ...
SafeAgentIDPath = Annotated[
    str, Path(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
]


# =============================================================================
# EXPORTS (v6.1.0)
# =============================================================================

__all__: list[str] = [
    # Classes
    "InputSanitizer",
    "ValidationResult",
    "SecurityValidationError",
    "InvalidInputError",
    "InputTooLongError",
    "DangerousInputError",
    # Functions — sanitize
    "sanitize_input",
    "sanitize_input_strict",
    "sanitize_tws_job_name",
    "sanitize_tws_workstation",
    "sanitize_email",
    # Functions — validate
    "validate_input",
    "validate_input_strict",
    "validate_email",
    "validate_tws_job_name",
    # Type aliases — Field() (Pydantic models, Body, Query)
    "SafeAgentID",
    "SafeEmail",
    "SafeTWSJobName",
    "SafeTWSWorkstation",
    # Type alias — Path() (somente path params de rota)
    "SafeAgentIDPath",
    # Patterns (exported for testing)
    "SAFE_STRING_PATTERN",
    "ASCII_SAFE_STRING_PATTERN",
    "STRICT_ALPHANUMERIC_PATTERN",
    "TWS_JOB_PATTERN",
    "TWS_WORKSTATION_PATTERN",
    # Utility functions
    "normalize_unicode",
    "truncate_text",
]
