"""
Security hardening configuration for Resync application.

This module provides comprehensive security configurations including
headers, rate limiting, input validation, and threat protection.
"""

import hashlib
import hmac
import logging
import secrets
from typing import Any

logger = logging.getLogger(__name__)


class SecurityHardeningConfig:
    """Centralized security configuration and hardening."""

    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }

    # CSP Configuration
    CONTENT_SECURITY_POLICY = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "media-src 'none'; "
        "object-src 'none'; "
        "frame-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_BURST_MULTIPLIER = 2
    RATE_LIMIT_WINDOW_SECONDS = 60

    # Input Validation
    MAX_REQUEST_SIZE_MB = 10
    MAX_FIELD_LENGTH = 1000
    MAX_ARRAY_LENGTH = 100
    MAX_NESTING_DEPTH = 10

    # Authentication Security
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL_CHARS = True

    # Encryption
    ENCRYPTION_ALGORITHM = "AES-256-GCM"
    KEY_DERIVATION_ITERATIONS = 100000

    # Threat Protection
    BLOCKED_USER_AGENTS = {
        "sqlmap",
        "nmap",
        "masscan",
        "dirbuster",
        "gobuster",
        "nikto",
        "wpscan",
        "joomlavs",
        "drupalvs",
    }

    BLOCKED_PATHS = {
        "/.env",
        "/.git/",
        "/wp-admin/",
        "/wp-login.php",
        "/phpmyadmin/",
        "/adminer.php",
        "/wp-content/",
        "/wp-includes/",
    }

    @classmethod
    def get_security_headers(cls) -> dict[str, str]:
        """Get comprehensive security headers."""
        return cls.SECURITY_HEADERS.copy()

    @classmethod
    def get_csp_policy(cls) -> str:
        """Get Content Security Policy string."""
        return cls.CONTENT_SECURITY_POLICY

    @classmethod
    def get_rate_limit_config(cls) -> dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "requests_per_minute": cls.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "burst_multiplier": cls.RATE_LIMIT_BURST_MULTIPLIER,
            "window_seconds": cls.RATE_LIMIT_WINDOW_SECONDS,
        }

    @classmethod
    def get_input_validation_config(cls) -> dict[str, Any]:
        """Get input validation configuration."""
        return {
            "max_request_size_mb": cls.MAX_REQUEST_SIZE_MB,
            "max_field_length": cls.MAX_FIELD_LENGTH,
            "max_array_length": cls.MAX_ARRAY_LENGTH,
            "max_nesting_depth": cls.MAX_NESTING_DEPTH,
        }

    @classmethod
    def get_password_policy(cls) -> dict[str, Any]:
        """Get password security policy."""
        return {
            "min_length": cls.PASSWORD_MIN_LENGTH,
            "require_uppercase": cls.PASSWORD_REQUIRE_UPPERCASE,
            "require_lowercase": cls.PASSWORD_REQUIRE_LOWERCASE,
            "require_digits": cls.PASSWORD_REQUIRE_DIGITS,
            "require_special_chars": cls.PASSWORD_REQUIRE_SPECIAL_CHARS,
        }

    @classmethod
    def is_user_agent_blocked(cls, user_agent: str) -> bool:
        """Check if user agent is blocked."""
        if not user_agent:
            return False

        ua_lower = user_agent.lower()
        return any(blocked in ua_lower for blocked in cls.BLOCKED_USER_AGENTS)

    @classmethod
    def is_path_blocked(cls, path: str) -> bool:
        """Check if path is blocked."""
        if not path:
            return False

        path_lower = path.lower()
        return any(blocked in path_lower for blocked in cls.BLOCKED_PATHS)

    @classmethod
    def generate_secure_token(cls, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)

    @classmethod
    def hash_password(cls, password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
        """Hash a password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(16)

        # Use PBKDF2 with SHA-256
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, cls.KEY_DERIVATION_ITERATIONS
        )

        return key, salt

    @classmethod
    def verify_password(cls, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify a password against its hash."""
        expected_key, _ = cls.hash_password(password, salt)
        return hmac.compare_digest(hashed, expected_key)

    @classmethod
    def sanitize_input(cls, input_str: str, max_length: int | None = None) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not input_str:
            return ""

        # Remove null bytes and other dangerous characters
        sanitized = input_str.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

        # Limit length if specified
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    @classmethod
    def validate_request_size(cls, content_length: int | None) -> bool:
        """Validate request size against limits."""
        if content_length is None:
            return True

        max_size_bytes = cls.MAX_REQUEST_SIZE_MB * 1024 * 1024
        return content_length <= max_size_bytes

    @classmethod
    def get_threat_detection_config(cls) -> dict[str, Any]:
        """Get threat detection configuration."""
        return {
            "blocked_user_agents": list(cls.BLOCKED_USER_AGENTS),
            "blocked_paths": list(cls.BLOCKED_PATHS),
            "max_request_size_mb": cls.MAX_REQUEST_SIZE_MB,
            "rate_limit_requests_per_minute": cls.RATE_LIMIT_REQUESTS_PER_MINUTE,
        }

    @classmethod
    def validate_security_config(cls) -> bool:
        """Validate security configuration for consistency."""
        issues = []

        # Check CSP policy
        if not cls.CONTENT_SECURITY_POLICY.strip():
            issues.append("CSP policy cannot be empty")

        # Check rate limiting
        if cls.RATE_LIMIT_REQUESTS_PER_MINUTE <= 0:
            issues.append("Rate limit must be positive")

        # Check password policy
        if cls.PASSWORD_MIN_LENGTH < 8:
            issues.append("Minimum password length should be at least 8")

        if issues:
            for issue in issues:
                logger.error(f"Security configuration issue: {issue}")
            return False

        logger.info("Security configuration validated successfully")
        return True


# Validate security config on import
if not SecurityHardeningConfig.validate_security_config():
    logger.error("Security configuration validation failed")
    raise ValueError("Invalid security configuration")

# Export common security utilities
security_headers = SecurityHardeningConfig.get_security_headers()
csp_policy = SecurityHardeningConfig.get_csp_policy()
rate_limit_config = SecurityHardeningConfig.get_rate_limit_config()
input_validation_config = SecurityHardeningConfig.get_input_validation_config()
password_policy = SecurityHardeningConfig.get_password_policy()
threat_detection_config = SecurityHardeningConfig.get_threat_detection_config()
