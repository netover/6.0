"""Field validators for the Settings class.

This module contains all Pydantic field validators used by the Settings class
to keep the main settings module more focused and maintainable.
"""

import warnings
from pathlib import Path

from pydantic import SecretStr, ValidationInfo, field_validator, model_validator

from .settings_types import Environment


class SettingsValidators:
    """Collection of field validators for Settings class."""

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
        """Valida formato da URL Redis."""
        val = v.get_secret_value() if isinstance(v, SecretStr) else v
        if not (val.startswith(("redis://", "rediss://"))):
            raise ValueError(
                "Redis URL deve começar com 'redis://' ou 'rediss://'. "
                "Exemplo: redis://localhost:6379 ou rediss://localhost:6379"
            )
        return v

    @field_validator("admin_password")
    @classmethod
    def validate_password_strength(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Valida força mínima da senha."""
        env = info.data.get("environment")

        # Em produção: senha obrigatória com 8+ caracteres
        if env == Environment.PRODUCTION:
            if v is None or len(v.get_secret_value()) < 8:
                raise ValueError("Senha do admin deve ter no mínimo 8 caracteres (produção)")
        # Em desenvolvimento: permitir None, mas se definida, exigir 8+ caracteres
        else:
            if v is not None and len(v.get_secret_value()) < 8:
                raise ValueError("Senha deve ter no mínimo 8 caracteres")
        return v

    @field_validator("admin_password")
    @classmethod
    def validate_insecure_in_prod(
        cls, v: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        """Bloqueia senhas inseguras em produção."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and v is not None:
            insecure = {
                "change_me_please",
                "change_me_immediately",
                "admin",
                "password",
                "12345678",
            }
            if v.get_secret_value().lower() in insecure:
                raise ValueError("Insecure admin password not allowed in production")
        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_production_cors(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Valida CORS em produção."""
        env = info.data.get("environment")
        if env == Environment.PRODUCTION and "*" in v:
            raise ValueError("Wildcard CORS origins not allowed in production")
        return v

    @field_validator("cors_allow_credentials")
    @classmethod
    def validate_credentials_with_wildcard(cls, v: bool, info: ValidationInfo) -> bool:
        """Valida credenciais com wildcard origins."""
        origins = info.data.get("cors_allowed_origins", [])
        if v and "*" in origins:
            warnings.warn(
                "CORS wildcard origins with credentials allowed is insecure. "
                "Consider using explicit origins instead of wildcard.",
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

    @field_validator("tws_user", "tws_password")
    @classmethod
    def validate_tws_credentials(cls, v: SecretStr | None, info: ValidationInfo) -> SecretStr | None:
        """Valida credenciais TWS quando não está em mock mode."""
        if info.field_name == "tws_password" and v is not None:
            env = info.data.get("environment")
            mock_mode = info.data.get("tws_mock_mode")
            if env == Environment.PRODUCTION and not mock_mode:
                # SecretStr esperado; valida conteúdo
                if not v.get_secret_value():
                    raise ValueError("TWS_PASSWORD is required when not in mock mode")
                if len(v.get_secret_value()) < 12:
                    raise ValueError("TWS_PASSWORD must be at least 12 characters in production")
                common_passwords = {
                    "password",
                    "twsuser",
                    "tws_password",
                    "change_me",
                }
                if v.get_secret_value().lower() in common_passwords:
                    raise ValueError("TWS_PASSWORD cannot be a common/default password")
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
        """Ensure CORS origins are properly configured in production."""
        env = info.data.get("environment")
        # [DECISION] Allow localhost even in production as per user request
        if env == Environment.PRODUCTION and "*" in v:
            raise ValueError(
                "CORS_ALLOW_ORIGINS cannot contain '*' in production. "
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

    # =========================================================================
    # MODEL-LEVEL VALIDATOR (runs after all field validators)
    # =========================================================================

    @model_validator(mode="after")
    def validate_cross_field_consistency(self):
        """
        Cross-field consistency checks that run after all individual validators.

        Ensures the overall configuration is coherent, not just individual fields.
        """
        errors = []

        # 1. Pool sizes: max must be >= min for all pool pairs
        pool_pairs = [
            ("db_pool_min_size", "db_pool_max_size"),
            ("redis_pool_min_size", "redis_pool_max_size"),
            ("http_pool_min_size", "http_pool_max_size"),
        ]
        for min_field, max_field in pool_pairs:
            min_val = getattr(self, min_field, None)
            max_val = getattr(self, max_field, None)
            if min_val is not None and max_val is not None and max_val < min_val:
                errors.append(f"{max_field} ({max_val}) < {min_field} ({min_val})")

        # 2. Pool lifetime must be > idle timeout
        lifetime_pairs = [
            ("db_pool_idle_timeout", "db_pool_max_lifetime", "db_pool"),
            ("redis_pool_idle_timeout", "redis_pool_max_lifetime", "redis_pool"),
            ("http_pool_idle_timeout", "http_pool_max_lifetime", "http_pool"),
        ]
        for idle_field, lifetime_field, label in lifetime_pairs:
            idle = getattr(self, idle_field, None)
            lifetime = getattr(self, lifetime_field, None)
            if idle is not None and lifetime is not None and lifetime <= idle:
                errors.append(
                    f"{label}: max_lifetime ({lifetime}s) must be > idle_timeout ({idle}s)"
                )

        # 3. TWS granular timeouts must be <= overall request timeout
        tws_request_timeout = getattr(self, "tws_request_timeout", None)
        if tws_request_timeout is not None:
            for sub_field in ("tws_timeout_connect", "tws_timeout_read",
                              "tws_timeout_write", "tws_timeout_pool"):
                sub_val = getattr(self, sub_field, None)
                if sub_val is not None and sub_val > tws_request_timeout:
                    errors.append(
                        f"{sub_field} ({sub_val}s) > tws_request_timeout ({tws_request_timeout}s)"
                    )

        # 4. Production-specific cross-checks
        env = getattr(self, "environment", None)
        if env and str(env).lower() in ("production", "prod"):
            # Secret key length must meet minimum
            secret_key = getattr(self, "secret_key", None)
            min_len = getattr(self, "MIN_SECRET_KEY_LENGTH", 32)
            if secret_key and len(secret_key.get_secret_value()) < min_len:
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

        if errors:
            raise ValueError(
                "Settings cross-field validation failed:\n  - " + "\n  - ".join(errors)
            )

        return self
