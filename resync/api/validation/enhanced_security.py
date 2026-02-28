# pylint
"""Enhanced security validation with async context managers and improved type hints."""

from __future__ import annotations

import hmac
import ipaddress
import re
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from re import Pattern
from typing import (
    Any,
    TypeVar,
)

import structlog

# Use unified JWT module (PyJWT)
from resync.core.jwt_utils import JWTError, jwt

try:
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    HAS_PASSLIB = True
except ImportError:
    # Fallback if passlib is not available
    pwd_context = None
    HAS_PASSLIB = False

from pydantic import BaseModel, Field

from resync.api.validation.common import SanitizationLevel, sanitize_input
from resync.settings import settings

# Type aliases for better readability
IPAddress = IPv4Address | IPv6Address
SecurityEvent = dict[str, Any]
ValidationResult = tuple[bool, str | None]

# Generic type for validation models
T = TypeVar("T", bound=BaseModel)

# Logger setup
logger = structlog.get_logger(__name__)

# Constants for security thresholds
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900  # 15 minutes
SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
CSRF_TOKEN_BYTES = 32
PASSWORD_RESET_TOKEN_EXPIRY_HOURS = 24

# Regular expressions for enhanced validation
SECURE_PASSWORD_PATTERN: Pattern = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$"
)
IP_ADDRESS_PATTERN: Pattern = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)
DOMAIN_PATTERN: Pattern = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)
JWT_PATTERN: Pattern = re.compile(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$")

# Trusted IP ranges for production environments
TRUSTED_IP_RANGES: list[str] = [
    "127.0.0.0/8",  # localhost
    "10.0.0.0/8",  # private network
    "172.16.0.0/12",  # private network
    "192.168.0.0/16",  # private network
]

# Suspicious patterns to detect in inputs
SUSPICIOUS_PATTERNS: list[Pattern] = [
    re.compile(r"(?i)<script[^>]*>.*?</script>", re.DOTALL),
    re.compile(r"(?i)javascript\s*:"),
    re.compile(r"(?i)vbscript\s*:"),
    re.compile(r"(?i)on\w+\s*="),
    re.compile(r"(?i)expression\s*\("),
    re.compile(r"(?i)data\s*:"),
    re.compile(r"(?i)eval\s*\("),
    re.compile(r"(?i)alert\s*\("),
    re.compile(r"(?i)document\.cookie"),
    re.compile(r"(?i)document\.write"),
]

class SecurityLevel(str, Enum):
    """Security levels for different contexts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, Enum):
    """Types of security threats detected."""

    BRUTE_FORCE = "brute_force"
    XSS = "xss"
    SQL_INJECTION = "sql_injection"
    CSRF = "csrf"
    RECONNAISSANCE = "reconnaissance"
    PRIVILEGE_ESCALATION = "privilege_escalation"

class SecurityEventSeverity(str, Enum):
    """Severity levels for security events."""

    # Using numeric values for proper comparison
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SecurityEventType(str, Enum):
    """Types of security events."""

    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    CSRF_DETECTED = "csrf_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class SecurityContext(BaseModel):
    """Context for security operations."""

    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    session_id: str | None = None
    threat_level: SecurityLevel = SecurityLevel.LOW
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

class SecurityEventLog(BaseModel):
    """Structure for logging security events."""

    event_type: SecurityEventType
    severity: SecurityEventSeverity
    source_ip: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    threat_type: ThreatType | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None

class TokenPayload(BaseModel):
    """Structure for JWT token payloads."""

    sub: str = Field(..., description="Subject (user identifier)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    jti: str = Field(..., description="JWT ID")
    scopes: list[str] = Field(default_factory=list, description="Token scopes")
    session_id: str | None = Field(None, description="Session identifier")
    ip_address: str | None = Field(None, description="IP address at token issuance")

class SecurityToken(BaseModel):
    """Structure for security tokens."""

    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Seconds until expiration")
    refresh_token: str | None = Field(None, description="Refresh token")
    csrf_token: str | None = Field(None, description="CSRF protection token")

class RateLimitInfo(BaseModel):
    """Information about rate limiting."""

    limit: int = Field(..., description="Request limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Time when limit resets")
    window_seconds: int = Field(..., description="Window duration in seconds")

class InputValidationResult(BaseModel):
    """Result of input validation."""

    is_valid: bool = Field(..., description="Whether input is valid")
    sanitized_value: str | None = Field(None, description="Sanitized input value")
    error_message: str | None = Field(None, description="Error message if invalid")
    threat_detected: ThreatType | None = Field(None, description="Detected threat type")
    security_context: SecurityContext = Field(default_factory=SecurityContext)

class SecurityValidator(ABC):
    """Abstract base class for security validators."""

    @abstractmethod
    async def validate(self, data: Any, context: SecurityContext) -> ValidationResult:
        """Validate data with security context."""

class AsyncSecurityContextManager:
    """Async context manager for security operations."""

    def __init__(self, context: SecurityContext):
        self.context = context
        self.start_time: float | None = None

    async def __aenter__(self) -> AsyncSecurityContextManager:
        """Enter the async context."""
        self.start_time = time.time()
        logger.info("security_context_entered", context=self.context.model_dump())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the async context."""
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(
            "security_context_exited",
            duration=duration,
            context=self.context.model_dump(),
        )
        return False  # Don't suppress exceptions

class EnhancedSecurityValidator:
    """Enhanced security validator with comprehensive validation capabilities."""

    def __init__(self, settings_module: Any = None):
        """Initialize the security validator."""
        self.settings = settings_module or settings
        self.failed_attempts: dict[str, int] = {}
        self.lockout_times: dict[str, float] = {}
        self.session_store: dict[str, SecurityContext] = {}
        # Performance optimization: cache for repeated validations
        self._validation_cache: dict[str, InputValidationResult] = {}
        self._threat_cache: dict[str, ThreatType | None] = {}

    @asynccontextmanager
    async def security_context(
        self, context: SecurityContext
    ) -> AsyncGenerator[SecurityContext, None]:
        """Async context manager for security operations."""
        async with AsyncSecurityContextManager(context) as manager:
            yield manager.context

    async def validate_password_strength(
        self, password: str, security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> InputValidationResult:
        """
        Validate password strength with enhanced security checks.

        Args:
            password: Password to validate
            security_level: Required security level

        Returns:
            InputValidationResult with validation outcome
        """
        context = SecurityContext(threat_level=security_level)

        # Truncate password to avoid bcrypt limitation (72 bytes max)
        truncated_password = password[:72]

        # Basic length check
        if len(truncated_password) < 8:
            return InputValidationResult(
                is_valid=False,
                error_message="Password must be at least 8 characters long",
                threat_detected=ThreatType.BRUTE_FORCE,
                security_context=context,
            )

        # High security level requires stronger passwords
        if security_level == SecurityLevel.HIGH:
            if not SECURE_PASSWORD_PATTERN.match(truncated_password):
                return InputValidationResult(
                    is_valid=False,
                    error_message=(
                        "Password must contain at least 12 characters including "
                        "uppercase, lowercase, digit, and special character"
                    ),
                    threat_detected=ThreatType.BRUTE_FORCE,
                    security_context=context,
                )

        # Medium security level
        elif security_level == SecurityLevel.MEDIUM:
            if len(truncated_password) < 10:
                return InputValidationResult(
                    is_valid=False,
                    error_message=(
                        "Password must be at least 10 characters long "
                        "for medium security"
                    ),
                    threat_detected=ThreatType.BRUTE_FORCE,
                    security_context=context,
                )

            # Check for character variety
            has_upper = any(c.isupper() for c in truncated_password)
            has_lower = any(c.islower() for c in truncated_password)
            has_digit = any(c.isdigit() for c in truncated_password)
            has_special = any(
                c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in truncated_password
            )

            if not (has_upper and has_lower and has_digit and has_special):
                return InputValidationResult(
                    is_valid=False,
                    error_message=(
                        (
                            "Password must contain uppercase, lowercase, digit, "
                            "and special character"
                        )
                    ),
                    threat_detected=ThreatType.BRUTE_FORCE,
                    security_context=context,
                )

        # Check for common weak passwords
        weak_passwords = {
            "password",
            "123456",
            "12345678",
            "qwerty",
            "abc123",
            "password123",
            "admin",
            "root",
            "guest",
            "test",
        }

        if truncated_password.lower() in weak_passwords:
            return InputValidationResult(
                is_valid=False,
                error_message=(
                    "Password is too common, please choose a stronger "
                    "password"
                ),
                threat_detected=ThreatType.BRUTE_FORCE,
                security_context=context,
            )

        # Sanitize and return result
        sanitized = sanitize_input(truncated_password, SanitizationLevel.STRICT)
        return InputValidationResult(
            is_valid=True, sanitized_value=sanitized, security_context=context
        )

    async def validate_email_security(
        self, email: str, security_level: SecurityLevel = SecurityLevel.MEDIUM
    ) -> InputValidationResult:
        """
        Validate email with security checks.

        Args:
            email: Email to validate
            security_level: Required security level

        Returns:
            InputValidationResult with validation outcome
        """
        context = SecurityContext(threat_level=security_level)

        try:
            # Basic email validation using Pydantic
            from pydantic import validate_email

            validated_email = validate_email(email)[1]  # Returns (name, email)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("exception_caught", error=str(e), exc_info=True)
            return InputValidationResult(
                is_valid=False,
                error_message="Invalid email format",
                security_context=context,
            )

        # Check for suspicious patterns
        if self._detect_threats(validated_email):
            return InputValidationResult(
                is_valid=False,
                error_message="Email contains suspicious content",
                threat_detected=ThreatType.XSS,
                security_context=context,
            )

        # Domain validation for high security
        if security_level == SecurityLevel.HIGH:
            domain = validated_email.split("@")[1]
            if not DOMAIN_PATTERN.match(domain):
                return InputValidationResult(
                    is_valid=False,
                    error_message="Email domain is not valid",
                    security_context=context,
                )

        # Sanitize and return result
        sanitized = sanitize_input(validated_email, SanitizationLevel.MODERATE)
        return InputValidationResult(
            is_valid=True, sanitized_value=sanitized, security_context=context
        )

    async def validate_csrf_token(
        self,
        token: str,
        expected_token: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> InputValidationResult:
        """
        Validate CSRF token securely with optional user/session binding.

        Args:
            token: Provided CSRF token (may include binding: user_id:session_id:token)
            expected_token: Expected CSRF token (may include binding)
            user_id: Optional user ID for binding validation
            session_id: Optional session ID for binding validation

        Returns:
            InputValidationResult with validation outcome
        """
        context = SecurityContext(threat_level=SecurityLevel.HIGH)

        # Extract actual token from binding format if present
        actual_token = token.split(":")[-1] if ":" in token else token
        expected_actual_token = (
            expected_token.split(":")[-1] if ":" in expected_token else expected_token
        )

        # Validate binding if provided
        if user_id or session_id:
            token_parts = token.split(":")
            expected_parts = expected_token.split(":")

            if len(token_parts) != 3 or len(expected_parts) != 3:
                return InputValidationResult(
                    is_valid=False,
                    error_message="Invalid CSRF token binding format",
                    threat_detected=ThreatType.CSRF,
                    security_context=context,
                )

            # Validate user and session binding
            token_user, token_session, actual_token = token_parts
            expected_user, expected_session, expected_actual_token = expected_parts

            if user_id and token_user != user_id:
                return InputValidationResult(
                    is_valid=False,
                    error_message="CSRF token user binding mismatch",
                    threat_detected=ThreatType.CSRF,
                    security_context=context,
                )

            if session_id and token_session != session_id:
                return InputValidationResult(
                    is_valid=False,
                    error_message="CSRF token session binding mismatch",
                    threat_detected=ThreatType.CSRF,
                    security_context=context,
                )

        # Secure comparison to prevent timing attacks
        try:
            if not hmac.compare_digest(actual_token, expected_actual_token):
                return InputValidationResult(
                    is_valid=False,
                    error_message="Invalid CSRF token",
                    threat_detected=ThreatType.CSRF,
                    security_context=context,
                )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("exception_caught", error=str(e), exc_info=True)
            return InputValidationResult(
                is_valid=False,
                error_message="CSRF token validation failed",
                threat_detected=ThreatType.CSRF,
                security_context=context,
            )

        return InputValidationResult(is_valid=True, security_context=context)

    async def validate_jwt_token(
        self, token: str, secret_key: str, algorithms: list[str] = None
    ) -> tuple[bool, TokenPayload | None, str | None]:
        """
        Validate JWT token with enhanced security.

        Args:
            token: JWT token to validate
            secret_key: Secret key for verification
            algorithms: Allowed algorithms

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        if not algorithms:
            algorithms = ["HS256"]

        # Basic format validation
        if not JWT_PATTERN.match(token):
            return False, None, "Invalid JWT token format"

        try:
            # Decode and verify token
            payload = jwt.decode(token, secret_key, algorithms=algorithms)

            # Validate required fields
            required_fields = ["sub", "exp", "iat", "jti"]
            for field in required_fields:
                if field not in payload:
                    return False, None, f"Missing required field: {field}"

            # Check expiration
            if payload["exp"] < int(time.time()):
                return False, None, "Token has expired"

            # Create payload model
            token_payload = TokenPayload(**payload)
            return True, token_payload, None

        except JWTError as e:
            return False, None, f"Token validation failed: {str(e)}"
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("exception_caught", error=str(e), exc_info=True)
            return False, None, f"Unexpected error: {str(e)}"

    async def validate_ip_address(
        self, ip: str, trusted_ranges: list[str] = None
    ) -> InputValidationResult:
        """
        Validate IP address against trusted ranges.

        Args:
            ip: IP address to validate
            trusted_ranges: List of trusted IP ranges

        Returns:
            InputValidationResult with validation outcome
        """
        context = SecurityContext(threat_level=SecurityLevel.MEDIUM)

        if not trusted_ranges:
            trusted_ranges = TRUSTED_IP_RANGES

        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return InputValidationResult(
                is_valid=False,
                error_message="Invalid IP address format",
                security_context=context,
            )

        # Check against trusted ranges
        is_trusted = False
        for range_str in trusted_ranges:
            try:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip_obj in network:
                    is_trusted = True
                    break
            except ValueError:
                continue

        context.metadata["is_trusted"] = is_trusted
        return InputValidationResult(
            is_valid=True, sanitized_value=str(ip_obj), security_context=context
        )

    async def validate_input_security(
        self,
        input_data: str,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        _allowed_patterns: list[Pattern] | None = None,
    ) -> InputValidationResult:
        """
        Comprehensive input validation with threat detection.

        Args:
            input_data: Input data to validate
            security_level: Required security level
            allowed_patterns: Patterns explicitly allowed

        Returns:
            InputValidationResult with validation outcome
        """
        context = SecurityContext(threat_level=security_level)

        # Null byte check
        if "\x00" in input_data:
            return InputValidationResult(
                is_valid=False,
                error_message="Null bytes not allowed in input",
                threat_detected=ThreatType.SQL_INJECTION,
                security_context=context,
            )

        # Length limits based on security level
        max_lengths = {
            SecurityLevel.LOW: 1000,
            SecurityLevel.MEDIUM: 500,
            SecurityLevel.HIGH: 255,
            SecurityLevel.CRITICAL: 100,
        }

        max_length = max_lengths.get(security_level, 500)
        if len(input_data) > max_length:
            return InputValidationResult(
                is_valid=False,
                error_message=(
                    f"Input exceeds maximum length of {max_length} "
                    "characters"
                ),
                security_context=context,
            )

        # Detect threats
        threat_type = self._detect_threats(input_data)
        if threat_type:
            return InputValidationResult(
                is_valid=False,
                error_message="Suspicious content detected",
                threat_detected=threat_type,
                security_context=context,
            )

        # Sanitize based on security level
        sanitization_levels = {
            SecurityLevel.LOW: SanitizationLevel.PERMISSIVE,
            SecurityLevel.MEDIUM: SanitizationLevel.MODERATE,
            SecurityLevel.HIGH: SanitizationLevel.STRICT,
            SecurityLevel.CRITICAL: SanitizationLevel.STRICT,
        }

        level = sanitization_levels.get(security_level, SanitizationLevel.MODERATE)
        sanitized = sanitize_input(input_data, level)

        return InputValidationResult(
            is_valid=True, sanitized_value=sanitized, security_context=context
        )

    def _detect_threats(self, input_data: str) -> ThreatType | None:
        """
        Detect security threats in input data with caching for performance.

        Args:
            input_data: Input data to scan

        Returns:
            Detected threat type or None
        """
        # Performance optimization: cache results for repeated inputs
        cache_key = hash(input_data)
        if cache_key in self._threat_cache:
            return self._threat_cache[cache_key]

        input_lower = input_data.lower()

        # Check for suspicious patterns
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.search(input_data):
                return ThreatType.XSS

        # Check for SQL injection patterns (compiled once for performance)
        if not hasattr(self, "_sql_patterns_compiled"):
            self._sql_patterns_compiled = [
                re.compile(
                    r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b"
                ),
                re.compile(r"(?i)--|#|/\*|\*/"),
                re.compile(r"(?i)'(\s*)or(\s*)'1'='1"),
            ]

        for compiled_pattern in self._sql_patterns_compiled:
            if compiled_pattern.search(input_lower):
                return ThreatType.SQL_INJECTION

        # Check for path traversal
        if ".." in input_data or "%2e%2e" in input_lower:
            threat = ThreatType.RECONNAISSANCE
        else:
            threat = None

        # Cache the result
        self._threat_cache[cache_key] = threat
        return threat

    async def rate_limit_check(
        self, identifier: str, limit: int = 100, window_seconds: int = 60
    ) -> RateLimitInfo:
        """
        Check rate limiting for an identifier.

        Args:
            identifier: Identifier to check (IP, user ID, etc.)
            limit: Request limit per window
            window_seconds: Time window in seconds

        Returns:
            RateLimitInfo with current status
        """
        # In a real implementation, this would use Redis or similar
        # For now, we'll simulate with in-memory storage
        current_time = time.time()

        # Simulate rate limiting logic
        # In production, this would use atomic operations in Redis
        remaining = max(0, limit - 1)  # Simplified for example

        return RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=datetime.fromtimestamp(
                current_time + window_seconds, tz=timezone.utc
            ),
            window_seconds=window_seconds,
        )

    async def log_security_event(self, event: SecurityEventLog) -> None:
        """
        Log a security event.

        Args:
            event: Security event to log
        """
        # In production, this would send to a security information and
        # event management (SIEM) system
        # Convert enums to their values for logging
        event_data = {
            "event_type": (
                event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type)
            ),
            "severity": (
                event.severity.value
                if hasattr(event.severity, "value")
                else str(event.severity)
            ),
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "threat_type": (
                event.threat_type.value
                if event.threat_type and hasattr(event.threat_type, "value")
                else (str(event.threat_type) if event.threat_type else None)
            ),
            "details": event.details,
            "correlation_id": event.correlation_id,
        }

        # Use the string representation of the severity for logging
        # This avoids the comparison issue with structlog
        logger.info("security_event", **event_data)

    async def generate_csrf_token(
        self, user_id: str | None = None, session_id: str | None = None
    ) -> str:
        """
        Generate a secure CSRF token with optional user/session binding.

        Args:
            user_id: Optional user identifier for token binding
            session_id: Optional session identifier for token binding

        Returns:
            Generated CSRF token (includes binding if provided)
        """
        # Generate base token
        token = secrets.token_urlsafe(CSRF_TOKEN_BYTES)

        # Add binding for additional security (user_id:session_id:token)
        if user_id or session_id:
            binding_parts = [user_id or "anonymous", session_id or "no_session", token]
            return ":".join(binding_parts)

        return token

    async def generate_session_id(self) -> str:
        """
        Generate a secure session ID.

        Returns:
            Generated session ID
        """
        return str(uuid.uuid4())

    async def hash_password(self, password: str) -> str:
        """
        Hash a password securely using passlib (bcrypt).

        CRITICAL SECURITY: This method requires passlib to be available.
        If passlib is not installed or fails, the method will raise RuntimeError
        to prevent insecure password storage.

        Args:
            password: Password to hash (will be truncated to 72 chars for bcrypt)

        Returns:
            Securely hashed password using bcrypt

        Raises:
            RuntimeError: If passlib is not available or hashing fails
        """
        # Truncate password to avoid any hashing limitations
        # CRITICAL: Passwords longer than 72 chars will be truncated - inform user
        truncated_password = password[:72]  # Use 72 to stay within bcrypt limit
        if len(password) > 72:
            logger.warning(
                "Password truncated from %s to 72 characters for hashing", len(password)
            )

        if HAS_PASSLIB:
            try:
                # Try to hash with passlib
                return pwd_context.hash(truncated_password)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                # CRITICAL SECURITY: Never fall back to plain text
                logger.error("Password hashing failed: %s", e)
                raise RuntimeError(
                    "Password hashing failed - cannot proceed with insecure storage"
                ) from None
        else:
            # CRITICAL SECURITY: Never fall back to plain text in any environment
            logger.error("Passlib not available - secure password hashing is required")
            raise RuntimeError(
                "Secure password hashing library required but not available"
            )

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its secure hash using passlib (bcrypt).

        CRITICAL SECURITY: This method only accepts secure hashes.
        Plain text passwords with warning prefixes are rejected to prevent
        downgrade attacks.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Securely hashed password (bcrypt format)

        Returns:
            True if password matches the hash, False otherwise

        Raises:
            RuntimeError: If passlib is not available (security requirement)
        """
        # Truncate password to avoid any hashing limitations
        truncated_password = plain_password[:72]  # Use 72 to stay within bcrypt limit

        # CRITICAL SECURITY: Reject plain text passwords (no fallback allowed)
        if hashed_password.startswith("$plaintext_warning$"):
            logger.error("Plain text password detected - secure hashing required")
            return False

        # Handle secure hashes with passlib only
        if HAS_PASSLIB:
            try:
                return pwd_context.verify(truncated_password, hashed_password)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Password verification failed: %s", e)
                return False

        # CRITICAL SECURITY: No insecure fallback allowed
        logger.error("Secure password verification library required but not available")
        return False

# Dependency function for FastAPI
async def get_security_validator() -> EnhancedSecurityValidator:
    """
    FastAPI dependency to get security validator instance.

    Returns:
        EnhancedSecurityValidator instance
    """
    return EnhancedSecurityValidator()

# Utility functions for validation
async def validate_password(
    password: str, security_level: SecurityLevel, validator: EnhancedSecurityValidator
) -> InputValidationResult:
    """
    Validate password using security validator.

    Args:
        password: Password to validate
        security_level: Required security level
        validator: Security validator instance

    Returns:
        InputValidationResult with validation outcome
    """
    return await validator.validate_password_strength(password, security_level)

async def validate_email(
    email: str, security_level: SecurityLevel, validator: EnhancedSecurityValidator
) -> InputValidationResult:
    """
    Validate email using security validator.

    Args:
        email: Email to validate
        security_level: Required security level
        validator: Security validator instance

    Returns:
        InputValidationResult with validation outcome
    """
    return await validator.validate_email_security(email, security_level)

async def validate_input(
    input_data: str, security_level: SecurityLevel, validator: EnhancedSecurityValidator
) -> InputValidationResult:
    """
    Validate input using security validator.

    Args:
        input_data: Input data to validate
        security_level: Required security level
        validator: Security validator instance

    Returns:
        InputValidationResult with validation outcome
    """
    return await validator.validate_input_security(input_data, security_level)

# Security Headers Middleware
class SecurityHeadersMiddleware:
    """Middleware to add security headers to all responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                # Add security headers
                headers = [(k, v) for k, v in message.get("headers", [])]
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"x-xss-protection", b"1; mode=block"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)

# Export all public classes and functions
__all__ = [
    "EnhancedSecurityValidator",
    "SecurityHeadersMiddleware",
    "get_security_validator",
    "validate_password",
    "validate_email",
    "validate_input",
    "SecurityLevel",
    "ThreatType",
    "SecurityEventSeverity",
    "SecurityEventType",
    "SecurityContext",
    "SecurityEventLog",
    "TokenPayload",
    "SecurityToken",
    "RateLimitInfo",
    "InputValidationResult",
    "SecurityValidator",
    "AsyncSecurityContextManager",
]
