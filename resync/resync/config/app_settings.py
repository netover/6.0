"""
Application settings module.

DEPRECATION NOTICE (v5.4.1):
This module is deprecated. Use `from resync.settings import settings` instead.
AppSettings now delegates to the main settings system which includes:
- Production validators
- Secure defaults (no hardcoded passwords)
- Environment-specific rules

Migration:
    # OLD (deprecated):
    from resync.config.app_settings import AppSettings
    settings = AppSettings()

    # NEW (recommended):
    from resync.settings import settings
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY: No hardcoded default credentials
# =============================================================================
_INSECURE_DEFAULTS = frozenset(
    {
        "admin",
        "password",
        "change-me",
        "change-me-in-production",
        "secret",
        "123456",
    }
)


def _get_env_or_fail(key: str, fallback_for_dev: str | None = None) -> str:
    """
    Get environment variable, with optional dev fallback.

    v5.9.4: Em produção, variáveis obrigatórias SEMPRE levantam erro se ausentes.
    Nunca retorna string vazia silenciosamente.

    In production (ENVIRONMENT=production), missing required vars raise errors.
    In development, returns fallback if provided.
    """
    value = os.getenv(key)
    env = os.getenv("ENVIRONMENT", "development").lower()

    if value:
        # Warn if using insecure value in production
        if env == "production" and value.lower() in _INSECURE_DEFAULTS:
            logger.warning(
                "insecure_config_value",
                key=key,
                hint=f"Set a secure value for {key} in production",
            )
        return value

    # No value set
    if env == "production":
        # v5.9.4: SEMPRE levantar erro em produção para variáveis sem valor
        # Nunca retornar string vazia silenciosamente
        raise ValueError(
            f"Environment variable {key} must be set in production. "
            f"No default values or empty strings allowed for security."
        )

    # Development: use fallback if provided
    if fallback_for_dev:
        logger.debug("using_dev_fallback", key=key)
        return fallback_for_dev

    # Development without fallback: return empty but log warning
    logger.warning(
        "missing_env_var_in_dev",
        key=key,
        hint=f"Consider setting {key} in your .env file",
    )
    return ""


@dataclass
class AppSettings:
    """
    Application settings for TWS integration.

    DEPRECATED: Use `from resync.settings import settings` instead.

    This class now delegates to the main settings system for security.
    Direct instantiation will emit a deprecation warning.

    Security changes in v5.4.1:
    - No hardcoded default passwords
    - Production validation enforced
    - Fails closed on missing required config
    """

    # TWS Connection - no default passwords
    tws_host: str = field(default_factory=lambda: _get_env_or_fail("TWS_HOST", "localhost"))
    tws_port: int = field(default_factory=lambda: int(os.getenv("TWS_PORT", "8080")))
    tws_username: str = field(default_factory=lambda: _get_env_or_fail("TWS_USERNAME", "admin"))
    tws_password: str = field(
        default_factory=lambda: _get_env_or_fail("TWS_PASSWORD")
    )  # NO DEFAULT
    tws_engine_name: str = field(
        default_factory=lambda: _get_env_or_fail("TWS_ENGINE_NAME", "ENGINE")
    )
    tws_engine_owner: str = field(
        default_factory=lambda: _get_env_or_fail("TWS_ENGINE_OWNER", "owner")
    )

    # JWT - no default secret
    jwt_secret_key: str = field(
        default_factory=lambda: _get_env_or_fail("JWT_SECRET_KEY")
    )  # NO DEFAULT
    jwt_algorithm: str = field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))

    def __post_init__(self) -> None:
        """Emit deprecation warning and validate settings."""
        warnings.warn(
            "AppSettings is deprecated. Use 'from resync.settings import settings' instead. "
            "See resync/config/app_settings.py for migration guide.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Validate in production
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            self._validate_production()

    def _validate_production(self) -> None:
        """Validate settings for production use."""
        errors = []

        # Check for missing required fields
        if not self.tws_password:
            errors.append("TWS_PASSWORD must be set in production")

        if not self.jwt_secret_key:
            errors.append("JWT_SECRET_KEY must be set in production")
        elif len(self.jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")

        # Check for insecure values
        if self.tws_password and self.tws_password.lower() in _INSECURE_DEFAULTS:
            errors.append("TWS_PASSWORD cannot be a default/insecure value in production")

        if self.jwt_secret_key and self.jwt_secret_key.lower() in _INSECURE_DEFAULTS:
            errors.append("JWT_SECRET_KEY cannot be a default/insecure value in production")

        if errors:
            error_msg = "Production configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.critical("app_settings_validation_failed", errors=errors)
            raise ValueError(error_msg)


def get_app_settings() -> AppSettings:
    """
    Get application settings.

    DEPRECATED: Use `from resync.settings import get_settings` instead.
    """
    warnings.warn(
        "get_app_settings() is deprecated. Use get_settings() from resync.settings instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return AppSettings()


# =============================================================================
# Compatibility layer - redirect to main settings when possible
# =============================================================================
def get_unified_settings() -> Any:
    """
    Get settings from the main unified settings system.

    This is the recommended way to access settings.
    Returns the main Settings object which includes all validators.
    """
    try:
        from resync.settings import get_settings

        return get_settings()
    except ImportError:
        logger.warning("main_settings_unavailable", fallback="AppSettings")
        return AppSettings()


__all__ = [
    "AppSettings",
    "get_app_settings",
    "get_unified_settings",
]
