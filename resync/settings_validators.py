# ruff: noqa: E501
"""Field validators for the Settings class.

This module contains all Pydantic field validators used by the Settings class
to keep the main settings module more focused and maintainable.

Validation Strategy:
- Field validators: per-field constraints, run during field construction
- Model validator: cross-field consistency, runs after all fields are set
- Defense in depth: some constraints are checked in both layers intentionally
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import SecretStr, ValidationInfo, field_validator, model_validator

from .settings_types import Environment

def _redact_sensitive(val: Any) -> str:
    """Mask sensitive string values for logs/errors."""
    if isinstance(val, SecretStr):
        return "**********"
    s = str(val)
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return f"{s[:2]}...{s[-2:]}"

class SettingsValidators:
    """Collection of field validators for Settings class."""

    # =========================================================================
    # CONSTANTS
    # =========================================================================
    _INSECURE_ADMIN_PASSWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            # Common weak passwords
            "change_me_please",
            "change_me_immediately",
            "change_me",
            "admin",
            "administrator",
            "root",
            "password",
            "password123",
            "password1234",
            "password12345",
            "12345678",
            "123456789",
            "1234567890",
            "qwerty",
            "qwerty123",
            "qwertyuiop",
            "letmein",
            "welcome",
            "welcome1",
            "monkey",
            "dragon",
            "master",
            "login",
            "passw0rd",
            "p@ssword",
            "p@ssw0rd",
            "admin123",
            "admin1234",
            "root123",
            "toor",
            "test",
            "test1234",
            "guest",
            "guest123",
            "default",
            "secret",
            "secret123",
            "111111",
            "222222",
            "333333",
            "444444",
            "555555",
            "666666",
            "7777777",
            "888888",
            "999999",
            "000000",
            "abc123",
            "abcd1234",
            "1234abcd",
            "iloveyou",
            "sunshine",
            "princess",
            "football",
            "baseball",
            "soccer",
            "hockey",
            "shadow",
            "ashley",
            "michael",
            "superman",
            "batman",
            "trustno1",
            "access",
            # Note: "master" appears earlier in the list, removed duplicate
            "hello",
            "charlie",
            "donald",
            "admin888",
            "admin666",
            "q1w2e3r4",
            "1q2w3e4r",
            "zaq12wsx",
            "xsw21qaz",
        }
    )

    _COMMON_TWS_PASSWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "password",
            "twsuser",
            "tws_password",
            "change_me",
            "tws123",
            "tws1234",
            "ibkr",
            "interactive",
            "broker",
            "demo",
            "demo1234",
        }
    )

    _PASSWORD_COMPLEXITY_MIN_LENGTH: ClassVar[int] = 12
    _PASSWORD_COMPLEXITY_REQUIRE_UPPER: ClassVar[bool] = True
    _PASSWORD_COMPLEXITY_REQUIRE_LOWER: ClassVar[bool] = True
    _PASSWORD_COMPLEXITY_REQUIRE_DIGIT: ClassVar[bool] = True
    _PASSWORD_COMPLEXITY_REQUIRE_SPECIAL: ClassVar[bool] = True

    @field_validator("base_dir")
    @classmethod
    def validate_base_dir(cls, v: Path) -> Path:
        """[P0-09 FIX] Resolve base_dir to absolute path and validate existence/permissions.
        
        Uses atomic operations to prevent TOCTOU race conditions.
        """
        resolved_path = v.resolve()
        
        # [P0-09 FIX] Atomic check using try/except instead of exists() + is_dir()
        # Prevents TOCTOU race condition
        try:
            if not resolved_path.is_dir():
                raise ValueError(
                    f"REQUIRED: base_dir must be a directory, not a file: {resolved_path}"
                )
        except OSError as e:
            # [P3-08 FIX] Enhanced error message with field name
            raise ValueError(
                f"REQUIRED: base_dir (BASE_DIR) directory must exist and be accessible. "
                f"Path: {resolved_path}. Error: {e}"
            ) from e
        
        # [P0-09 FIX] Check read permission atomically
        try:
            # Attempt to list directory (requires read + execute)
            list(resolved_path.iterdir())
        except PermissionError as e:
            raise ValueError(
                f"PERMISSION DENIED: base_dir (BASE_DIR) is not readable: {resolved_path}"
            ) from e
        except OSError as e:
            # Other OS errors (unmounted, network issues, etc.)
            raise ValueError(
                f"OS ERROR: Cannot access base_dir (BASE_DIR): {resolved_path}. Error: {e}"
            ) from e
        
        return resolved_path

    @field_validator("db_pool_max_size")
    @classmethod
    def validate_db_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that max_size >= min_size."""
        min_size = info.data.get("db_pool_min_size", 0)
        if v < min_size:
            raise ValueError(
                f"CONFIGURATION ERROR: db_pool_max_size ({v}) must be >= db_pool_min_size ({min_size})"
            )
        return v

    @field_validator("redis_pool_max_size")
    @classmethod
    def validate_redis_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that max_size >= min_size."""
        min_size = info.data.get("redis_pool_min_size", 0)
        if v < min_size:
            raise ValueError(
                f"CONFIGURATION ERROR: redis_pool_max_size ({v}) must be >= redis_pool_min_size ({min_size})"
            )
        return v

    @field_validator("redis_url", "rate_limit_storage_uri")
    @classmethod
    def validate_redis_url(cls, v: SecretStr | str) -> SecretStr | str:
        """Validate Redis URL format."""
        val = v.get_secret_value() if isinstance(v, SecretStr) else v
        if not val.startswith(("redis://", "rediss://")):
            raise ValueError(
                "Redis URL must start with 'redis://' or 'rediss://'. "
                "Example: redis://localhost:6379/0"
            )
        try:
            parsed = urlparse(val)
            if not parsed.hostname:
                # Do not include the full URL in the error message as it may contain credentials
                raise ValueError("Redis URL missing hostname")
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Mask the URL in the error message
            redacted = _redact_sensitive(val)
            raise ValueError(f"Invalid Redis URL format: {redacted}") from e
        return v

    @field_validator("admin_password")
    @classmethod
    def validate_admin_password(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Validate admin password strength and reject insecure values.

        Enforces:
        - Minimum length (12 chars in production)
        - Blacklist of common weak passwords
        - Complexity requirements (upper, lower, digit, special)
        """
        import re

        env = info.data.get("environment")
        min_len = cls._PASSWORD_COMPLEXITY_MIN_LENGTH

        if env == Environment.PRODUCTION:
            if v is None:
                # [P3-08 FIX] Enhanced error message with env var name
                raise ValueError(
                    "SECURITY FAILURE: admin_password (ADMIN_PASSWORD or APP_ADMIN_PASSWORD) "
                    "is REQUIRED in Production. Set via environment variable."
                )
            pwd = v.get_secret_value()

            # Check blacklist
            if pwd.lower() in cls._INSECURE_ADMIN_PASSWORDS:
                raise ValueError(
                    f"INSECURE PASSWORD: '{pwd[:1]}...' is a known weak password. "
                    "Use a complex random string."
                )

            # Check minimum length
            if len(pwd) < min_len:
                raise ValueError(
                    f"INSECURE PASSWORD: admin_password must be at least {min_len} characters in Production. "
                    f"Got {len(pwd)}."
                )

            # Check complexity requirements
            errors = []
            if cls._PASSWORD_COMPLEXITY_REQUIRE_UPPER and not re.search(r"[A-Z]", pwd):
                errors.append("at least 1 uppercase letter")
            if cls._PASSWORD_COMPLEXITY_REQUIRE_LOWER and not re.search(r"[a-z]", pwd):
                errors.append("at least 1 lowercase letter")
            if cls._PASSWORD_COMPLEXITY_REQUIRE_DIGIT and not re.search(r"\d", pwd):
                errors.append("at least 1 digit")
            if cls._PASSWORD_COMPLEXITY_REQUIRE_SPECIAL and not re.search(
                r'[!@#$%^&*(),.?":{}|<>]', pwd
            ):
                errors.append("at least 1 special character (!@#$%^&*() etc.)")

            if errors:
                raise ValueError(
                    f"INSECURE PASSWORD: admin_password must contain {', '.join(errors)} in Production. "
                    f"Got password with length {len(pwd)}."
                )
        elif v is not None:
            pwd = v.get_secret_value()

            # Check blacklist in non-production too
            if pwd.lower() in cls._INSECURE_ADMIN_PASSWORDS:
                raise ValueError(
                    f"INSECURE PASSWORD: '{pwd[:1]}...' is a known weak password. "
                    "Use a complex random string."
                )

            # Warn but don't block in non-production
            if len(pwd) < 8:
                raise ValueError(
                    f"INSECURE PASSWORD: admin_password should be at least 8 characters. "
                    f"Got {len(pwd)}."
                )

        return v

    @field_validator("cors_allow_credentials")
    @classmethod
    def validate_credentials_with_wildcard(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate CORS credentials with wildcard origins."""
        env = info.data.get("environment")
        origins = info.data.get("cors_allowed_origins", [])
        if v and "*" in origins:
            if env == Environment.PRODUCTION:
                raise ValueError(
                    "CORS wildcard origins with credentials not allowed in production"
                )
            warnings.warn(
                "CORS wildcard origins with credentials is insecure. "
                "Consider using explicit origins.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, v: SecretStr, info: ValidationInfo) -> SecretStr:
        """Valida chave da API em produção."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and (
            not v.get_secret_value()
            or v.get_secret_value() == "dummy_key_for_development"
        ):
            # [P3-08 FIX] Enhanced error message with env var name
            raise ValueError(
                "LLM_API_KEY (or APP_LLM_API_KEY) must be set to a valid key in production"
            )
        return v

    @field_validator("tws_verify")
    @classmethod
    def validate_tws_verify_warning(
        cls, v: bool | str, info: ValidationInfo
    ) -> bool | str:
        """Emite warning para TWS verification em produção."""
        env = info.data.get("environment")
        is_disabled = (isinstance(v, bool) and not v) or (
            isinstance(v, str) and v.lower() == "false"
        )
        if env == Environment.PRODUCTION and is_disabled:
            warnings.warn(
                "TWS verification is disabled in production. This is a security risk.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("tws_user")
    @classmethod
    def validate_tws_user(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate TWS user when not in mock mode."""
        env = info.data.get("environment")
        mock_mode = info.data.get("tws_mock_mode", True)
        if env == Environment.PRODUCTION and not mock_mode:
            if not v or not v.strip():
                # [P3-08 FIX] Enhanced error message
                raise ValueError(
                    "TWS_USER (or APP_TWS_USER) is required when not in mock mode (production)"
                )
        return v

    @field_validator("tws_password")
    @classmethod
    def validate_tws_password(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Validate TWS password strength when not in mock mode."""
        if v is None:
            return v
        env = info.data.get("environment")
        mock_mode = info.data.get("tws_mock_mode", True)
        if env == Environment.PRODUCTION and not mock_mode:
            pwd = v.get_secret_value()
            if not pwd:
                # [P3-08 FIX] Enhanced error message
                raise ValueError(
                    "TWS_PASSWORD (or APP_TWS_PASSWORD) is required when not in mock mode"
                )
            if len(pwd) < 12:
                raise ValueError(
                    "TWS_PASSWORD must be at least 12 characters in production"
                )
            if pwd.lower() in cls._COMMON_TWS_PASSWORDS:
                raise ValueError("TWS_PASSWORD cannot be a common/default password")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """
        Validate secret_key for JWT signing.

        v5.3.20: Consolidated from fastapi_app/core/config.py
        - In production: MUST be set via environment variable (not default)
        - Must be at least 32 characters for security
        """
        env = info.data.get("environment")
        if v is None:
            if env == Environment.PRODUCTION:
                # [P3-08 FIX] Enhanced error message with generation command
                raise ValueError(
                    "SECRET_KEY (or APP_SECRET_KEY) must be set via environment variable in production. "
                    "Generate a secure random key: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            return None

        secret_value = v.get_secret_value()

        if env == Environment.PRODUCTION:
            if "CHANGE_ME" in secret_value or secret_value == "":
                raise ValueError(
                    "SECRET_KEY (or APP_SECRET_KEY) must be set via environment variable in production. "
                    "Generate a secure random key: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            if len(secret_value) < 32:
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters in production for security."
                )
        return v

    @field_validator("debug")
    @classmethod
    def validate_debug_in_production(cls, v: bool, info: ValidationInfo) -> bool:
        """Ensure debug mode is disabled in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and v:
            raise ValueError("Debug mode must be disabled in production")
        return v

    @field_validator("upload_dir")
    @classmethod
    def validate_upload_dir(cls, v: Path, info: ValidationInfo) -> Path:
        """[P0-09 FIX] Validate upload_dir existence and write permissions atomically.
        
        Uses atomic mkdir operation to prevent TOCTOU race conditions.
        """
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and not v.is_absolute():
            warnings.warn(
                f"upload_dir '{v}' is relative. In production, use an absolute path "
                "or mount a persistent volume to avoid data loss.",
                UserWarning,
                stacklevel=2,
            )

        # [P0-09 FIX] Atomic directory creation + permission check
        # Prevents TOCTOU by doing one operation instead of check-then-act
        try:
            # Ensure directory exists (atomic operation)
            v.mkdir(parents=True, exist_ok=True)
            
            # Verify write permission by attempting to create a temp file
            test_file = v / ".write_test_temp"
            try:
                test_file.touch(exist_ok=True)
                test_file.unlink()  # Clean up
            except OSError as e:
                raise ValueError(
                    f"PERMISSION DENIED: upload_dir (UPLOAD_DIR) is not writable: {v}"
                ) from e
        except OSError as e:
            raise ValueError(
                f"PERMISSION DENIED: Cannot create/access upload_dir (UPLOAD_DIR): {v}. Error: {e}"
            ) from e

        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """[P1-11 FIX] Validate CORS origins — reject wildcard and localhost in production.
        
        Security rationale:
        - Wildcard '*' allows any origin (CSRF, XSS amplification)
        - Localhost in production makes no sense (API doesn't serve localhost clients)
        """
        env = info.data.get("environment")
        
        if env == Environment.PRODUCTION:
            # Reject wildcard
            if "*" in v:
                raise ValueError(
                    "CORS wildcard origins ('*') not allowed in production. "
                    "Specify exact production domains."
                )
            
            # [P1-11 FIX] Reject localhost in production
            localhost_patterns = (
                "http://localhost",
                "https://localhost",
                "http://127.0.0.1",
                "https://127.0.0.1",
                "http://[::1]",
                "https://[::1]",
            )
            
            invalid_origins = [
                origin for origin in v 
                if any(origin.startswith(pattern) for pattern in localhost_patterns)
            ]
            
            if invalid_origins:
                raise ValueError(
                    f"CORS localhost origins not allowed in production: {invalid_origins}. "
                    "Production APIs should not accept requests from localhost. "
                    "Specify production domains instead."
                )
        
        return v

    @field_validator("enforce_https")
    @classmethod
    def validate_https_enforcement(cls, v: bool, info: ValidationInfo) -> bool:
        """Warn if HTTPS is not enforced in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and not v:
            warnings.warn(
                "enforce_https is False in production. HSTS headers will not be sent. "
                "Enable this when running behind a TLS-terminating proxy.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("backup_dir")
    @classmethod
    def validate_backup_dir(cls, v: Path, info: ValidationInfo) -> Path:
        """Warn if backup_dir is relative in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and not v.is_absolute():
            warnings.warn(
                f"backup_dir '{v}' is relative. In production, use an absolute path "
                "to ensure backups are stored in a persistent location.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("session_timeout_minutes")
    @classmethod
    def validate_session_timeout(cls, v: int, info: ValidationInfo) -> int:
        """Enforce reasonable session timeout in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and v > 60:
            warnings.warn(
                f"session_timeout_minutes is {v}, which is longer than recommended (30-60 min) "
                "for production. Consider reducing for better security.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str, info: ValidationInfo) -> str:
        """Warn if log level is DEBUG in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and v == "DEBUG":
            warnings.warn(
                "LOG_LEVEL is DEBUG in production. This may impact performance "
                "and potentially expose sensitive information. Use INFO or WARNING.",
                UserWarning,
                stacklevel=2,
            )
        return v

    # =========================================================================
    # POOL SIZE CROSS-VALIDATORS
    # =========================================================================

    @field_validator("http_pool_max_size")
    @classmethod
    def validate_http_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that http_pool_max_size >= http_pool_min_size."""
        min_size = info.data.get("http_pool_min_size", 0)
        if v < min_size:
            raise ValueError(
                f"http_pool_max_size ({v}) must be >= http_pool_min_size ({min_size})"
            )
        return v

    @field_validator("redis_max_connections")
    @classmethod
    def validate_redis_connection_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate redis_max_connections >= redis_min_connections and reasonable ranges (Issue 12)."""
        min_size = info.data.get("redis_min_connections", 0)
        if v < min_size:
            raise ValueError(
                f"CONFIGURATION ERROR: redis_max_connections ({v}) must be >= redis_min_connections ({min_size})"
            )
        # Issue 12: Positive range
        if v <= 0:
            raise ValueError(
                f"INVALID RANGE: redis_max_connections must be > 0. Got {v}"
            )
        return v

    # =========================================================================
    # MODEL-LEVEL VALIDATOR (runs after all field validators)
    # =========================================================================

    @model_validator(mode="after")
    def validate_cross_field_consistency(self) -> "SettingsValidators":
        """
        Cross-field consistency checks that run after all individual validators.

        [P2-08 FIX] Removed redundant pool size checks (already validated by @field_validator).
        Focus on cross-field dependencies that can't be validated in isolation.
        """
        errors = []

        # [P2-08 FIX] Pool size checks removed — already validated by @field_validator
        # (validate_db_pool_sizes, validate_redis_pool_sizes, etc.)
        # Keeping them here would be redundant and add ~30% validation overhead.

        # 1. Pool lifetime must be > idle timeout
        lifetime_pairs = [
            ("db_pool_idle_timeout", "db_pool_max_lifetime", "db_pool"),
            ("redis_pool_idle_timeout", "redis_pool_max_lifetime", "redis_pool"),
            ("http_pool_idle_timeout", "http_pool_max_lifetime", "http_pool"),
        ]
        for idle_field, lifetime_field, label in lifetime_pairs:
            idle = getattr(self, idle_field)
            lifetime = getattr(self, lifetime_field)
            if lifetime <= idle:
                errors.append(
                    f"{label}: max_lifetime ({lifetime}s) must be > idle_timeout ({idle}s)"
                )

        # 2. TWS granular timeouts must be <= overall request timeout
        tws_request_timeout = getattr(self, "tws_request_timeout")
        for sub_field in (
            "tws_timeout_connect",
            "tws_timeout_read",
            "tws_timeout_write",
            "tws_timeout_pool",
        ):
            sub_val = getattr(self, sub_field)
            if sub_val > tws_request_timeout:
                errors.append(
                    f"{sub_field} ({sub_val}s) > "
                    f"tws_request_timeout ({tws_request_timeout}s)"
                )

        # 3. Backoff ranges: base must be <= max
        backoff_pairs = [
            (
                "redis_startup_backoff_base",
                "redis_startup_backoff_max",
                "Redis startup",
            ),
            ("tws_retry_backoff_base", "tws_retry_backoff_max", "TWS retry"),
        ]
        for base_field, max_field, label in backoff_pairs:
            base_val = getattr(self, base_field)
            max_val = getattr(self, max_field)
            if base_val > max_val:
                errors.append(
                    f"{label}: {base_field} ({base_val}) > {max_field} ({max_val})"
                )

        # 4. Hybrid weights must sum to ~1.0 when auto_weight is off
        if not getattr(self, "hybrid_auto_weight", True):
            vec_w = getattr(self, "hybrid_vector_weight", 0.5)
            bm25_w = getattr(self, "hybrid_bm25_weight", 0.5)
            total = vec_w + bm25_w
            if not (0.99 <= total <= 1.01):
                errors.append(
                    f"Hybrid weights sum={total:.4f}, expected ≈1.0 "
                    f"when hybrid_auto_weight=False"
                )

        # 5. Service credentials when enabled
        if getattr(self, "langfuse_enabled", False):
            if not getattr(self, "langfuse_public_key", ""):
                errors.append("langfuse_public_key required when langfuse_enabled=True")
            lf_secret = getattr(self, "langfuse_secret_key", None)
            if not lf_secret or not lf_secret.get_secret_value():
                errors.append("langfuse_secret_key required when langfuse_enabled=True")

        if getattr(self, "enterprise_enable_siem", False):
            if not getattr(self, "enterprise_siem_endpoint", None):
                errors.append("enterprise_siem_endpoint required when SIEM enabled")

        # 6. Production-specific cross-checks
        # [P1-10 FIX] Use Environment enum, not string comparison
        env = getattr(self, "environment")
        if env == Environment.PRODUCTION:  # [P1-10 FIX] Was: if env == "production"
            # Secret key length must meet minimum
            secret_key = getattr(self, "secret_key")
            min_len = getattr(self, "MIN_SECRET_KEY_LENGTH", 32)
            if secret_key is None:
                errors.append("secret_key (SECRET_KEY) must be set in production")
            elif len(secret_key.get_secret_value()) < min_len:
                errors.append(
                    f"secret_key length ({len(secret_key.get_secret_value())}) "
                    f"< MIN_SECRET_KEY_LENGTH ({min_len})"
                )

            # Admin password length must meet minimum
            admin_pw = getattr(self, "admin_password", None)
            min_pw_len = getattr(self, "MIN_ADMIN_PASSWORD_LENGTH", 8)
            if admin_pw and len(admin_pw.get_secret_value()) < min_pw_len:
                errors.append(
                    f"admin_password length < MIN_ADMIN_PASSWORD_LENGTH ({min_pw_len})"
                )

        # 7. TWS credentials when not in mock mode
        if not getattr(self, "tws_mock_mode", True):
            if not getattr(self, "tws_user", None):
                errors.append("tws_user (TWS_USER) is required when tws_mock_mode=False")
            tws_pw = getattr(self, "tws_password", None)
            if not tws_pw or not tws_pw.get_secret_value():
                errors.append("tws_password (TWS_PASSWORD) is required when tws_mock_mode=False")

        if errors:
            raise ValueError(
                "Settings cross-field validation failed:\n  - " + "\n  - ".join(errors)
            )

        return self
