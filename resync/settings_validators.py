"""Field validators for the Settings class.

This module contains all Pydantic field validators used by the Settings class
to keep the main settings module more focused and maintainable.

Validation Strategy:
- Field validators: per-field constraints, run during field construction
- Model validator: cross-field consistency, runs after all fields are set
- Defense in depth: some constraints are checked in both layers intentionally
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

from pydantic import SecretStr, ValidationInfo, field_validator, model_validator

from .settings_types import Environment


class SettingsValidators:
    """Collection of field validators for Settings class."""

    # =========================================================================
    # CONSTANTS
    # =========================================================================
    _INSECURE_ADMIN_PASSWORDS: ClassVar[frozenset[str]] = frozenset({
        "change_me_please",
        "change_me_immediately",
        "admin",
        "password",
        "12345678",
    })

    _COMMON_TWS_PASSWORDS: ClassVar[frozenset[str]] = frozenset({
        "password",
        "twsuser",
        "tws_password",
        "change_me",
    })

    @field_validator("base_dir")
    @classmethod
    def validate_base_dir(cls, v: Path) -> Path:
        """Resolve base_dir para path absoluto e valida existência."""
        resolved_path = v.resolve()
        if not resolved_path.exists():
            raise ValueError(f"base_dir ({resolved_path}) does not exist")
        if not resolved_path.is_dir():
            raise ValueError(f"base_dir ({resolved_path}) is not a directory")
        return resolved_path

    @field_validator("db_pool_max_size")
    @classmethod
    def validate_db_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Valida que max_size >= min_size."""
        min_size = info.data.get("db_pool_min_size", 0)
        if v < min_size:
            raise ValueError(f"db_pool_max_size ({v}) must be >= db_pool_min_size ({min_size})")
        return v

    @field_validator("redis_pool_max_size")
    @classmethod
    def validate_redis_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Valida que max_size >= min_size."""
        min_size = info.data.get("redis_pool_min_size", 0)
        if v < min_size:
            raise ValueError(
                f"redis_pool_max_size ({v}) must be >= redis_pool_min_size ({min_size})"
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
        except Exception as e:
            # Mask the URL in the error message
            raise ValueError("Invalid Redis URL format") from e
        return v

    @field_validator("admin_password")
    @classmethod
    def validate_admin_password(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Validate admin password strength and reject insecure values.
        
        Note: Uses hardcoded minimum (8) because MIN_ADMIN_PASSWORD_LENGTH
        is declared after this field in the model. The model_validator
        performs the authoritative check with the configurable minimum.
        """
        env = info.data.get("environment")
        min_len = 8  # Hardcoded; model_validator uses MIN_ADMIN_PASSWORD_LENGTH

        if env == Environment.PRODUCTION:
            if v is None:
                raise ValueError(
                    "Admin password is required in production"
                )
            pwd = v.get_secret_value()
            if len(pwd) < min_len:
                raise ValueError(
                    f"Admin password must be at least {min_len} characters in production"
                )
            if pwd.lower() in cls._INSECURE_ADMIN_PASSWORDS:
                raise ValueError(
                    "Insecure admin password not allowed in production"
                )
        elif v is not None:
            pwd = v.get_secret_value()
            if len(pwd) < min_len:
                raise ValueError(
                    f"Admin password must be at least {min_len} characters"
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
        if (
            env == Environment.PRODUCTION
            and (not v.get_secret_value() or v.get_secret_value() == "dummy_key_for_development")
        ):
            raise ValueError("LLM_API_KEY must be set to a valid key in production")
        return v

    @field_validator("tws_verify")
    @classmethod
    def validate_tws_verify_warning(cls, v: bool | str, info: ValidationInfo) -> bool | str:
        """Emite warning para TWS verification em produção."""
        env = info.data.get("environment")
        is_disabled = (v is False) or (isinstance(v, str) and v.lower() == "false")
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
                raise ValueError(
                    "TWS_USER is required when not in mock mode (production)"
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
                raise ValueError(
                    "TWS_PASSWORD is required when not in mock mode"
                )
            if len(pwd) < 12:
                raise ValueError(
                    "TWS_PASSWORD must be at least 12 characters in production"
                )
            if pwd.lower() in cls._COMMON_TWS_PASSWORDS:
                raise ValueError(
                    "TWS_PASSWORD cannot be a common/default password"
                )
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr, info: ValidationInfo) -> SecretStr:
        """
        Validate secret_key for JWT signing.

        v5.3.20: Consolidated from fastapi_app/core/config.py
        - In production: MUST be set via environment variable (not default)
        - Must be at least 32 characters for security
        """
        env = info.data.get("environment")
        secret_value = v.get_secret_value()

        if env == Environment.PRODUCTION:
            # Check for default/placeholder values
            if "CHANGE_ME" in secret_value or secret_value == "":
                raise ValueError(
                    "SECRET_KEY must be set via environment variable in production. "
                    "Generate a secure random key: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            # Enforce minimum length for security
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
        if env == Environment.PRODUCTION and v is True:
            raise ValueError("Debug mode must be disabled in production")
        return v

    @field_validator("upload_dir")
    @classmethod
    def validate_upload_dir(cls, v: Path, info: ValidationInfo) -> Path:
        """Warn if upload_dir is relative in production."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and not v.is_absolute():
            warnings.warn(
                f"upload_dir '{v}' is relative. In production, use an absolute path "
                "or mount a persistent volume to avoid data loss.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Validate CORS origins — reject wildcard in production."""
        env = info.data.get("environment")
        # [DECISION] Allow localhost even in production as per user request
        if env == Environment.PRODUCTION and "*" in v:
            raise ValueError(
                "CORS wildcard origins ('*') not allowed in production. "
                "Specify exact production domains (localhost is allowed)."
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
        """Validate redis_max_connections >= redis_min_connections."""
        min_size = info.data.get("redis_min_connections", 0)
        if v < min_size:
            raise ValueError(
                f"redis_max_connections ({v}) must be >= redis_min_connections ({min_size})"
            )
        return v

    # =========================================================================
    # MODEL-LEVEL VALIDATOR (runs after all field validators)
    # =========================================================================

    @model_validator(mode="after")
    def validate_cross_field_consistency(self) -> "SettingsValidators":
        """
        Cross-field consistency checks that run after all individual validators.

        Ensures the overall configuration is coherent, not just individual fields.
        """
        errors = []

        # 1. Pool sizes: max must be >= min for all pool pairs
        pool_pairs = [
            ("db_pool_min_size", "db_pool_max_size"),
            ("redis_pool_min_size", "redis_pool_max_size"),
            ("redis_min_connections", "redis_max_connections"),
            ("http_pool_min_size", "http_pool_max_size"),
        ]
        for min_field, max_field in pool_pairs:
            min_val = getattr(self, min_field)
            max_val = getattr(self, max_field)
            if max_val < min_val:
                errors.append(f"{max_field} ({max_val}) < {min_field} ({min_val})")

        # 2. Pool lifetime must be > idle timeout
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

        # 3. TWS granular timeouts must be <= overall request timeout
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

        # 4. Backoff ranges: base must be <= max
        backoff_pairs = [
            ("redis_startup_backoff_base", "redis_startup_backoff_max", "Redis startup"),
            ("tws_retry_backoff_base", "tws_retry_backoff_max", "TWS retry"),
        ]
        for base_field, max_field, label in backoff_pairs:
            base_val = getattr(self, base_field)
            max_val = getattr(self, max_field)
            if base_val > max_val:
                errors.append(
                    f"{label}: {base_field} ({base_val}) > {max_field} ({max_val})"
                )

        # 5. Hybrid weights must sum to ~1.0 when auto_weight is off
        if not getattr(self, "hybrid_auto_weight", True):
            vec_w = getattr(self, "hybrid_vector_weight", 0.5)
            bm25_w = getattr(self, "hybrid_bm25_weight", 0.5)
            total = vec_w + bm25_w
            if not (0.99 <= total <= 1.01):
                errors.append(
                    f"Hybrid weights sum={total:.4f}, expected ≈1.0 "
                    f"when hybrid_auto_weight=False"
                )

        # 6. Service credentials when enabled
        if getattr(self, "langfuse_enabled", False):
            if not getattr(self, "langfuse_public_key", ""):
                errors.append(
                    "langfuse_public_key required when langfuse_enabled=True"
                )
            lf_secret = getattr(self, "langfuse_secret_key", None)
            if not lf_secret or not lf_secret.get_secret_value():
                errors.append(
                    "langfuse_secret_key required when langfuse_enabled=True"
                )

        if getattr(self, "enterprise_enable_siem", False):
            if not getattr(self, "enterprise_siem_endpoint", None):
                errors.append(
                    "enterprise_siem_endpoint required when SIEM enabled"
                )

        # 7. Production-specific cross-checks (FIXED: use direct enum comparison)
        env = getattr(self, "environment")
        if env == Environment.PRODUCTION:
            # Secret key length must meet minimum
            secret_key = getattr(self, "secret_key")
            min_len = getattr(self, "MIN_SECRET_KEY_LENGTH", 32)
            if len(secret_key.get_secret_value()) < min_len:
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

        # 8. TWS credentials when not in mock mode
        if not getattr(self, "tws_mock_mode", True):
            if not getattr(self, "tws_user", None):
                errors.append(
                    "tws_user is required when tws_mock_mode=False"
                )
            tws_pw = getattr(self, "tws_password", None)
            if not tws_pw or not tws_pw.get_secret_value():
                errors.append(
                    "tws_password is required when tws_mock_mode=False"
                )

        if errors:
            raise ValueError(
                "Settings cross-field validation failed:\n  - " + "\n  - ".join(errors)
            )

        return self
