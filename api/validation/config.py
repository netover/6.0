"""Validation configuration for the resync application."""

import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValidationMode(str, Enum):
    """Validation strictness modes."""

    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    DISABLED = "disabled"


class SanitizationLevel(str, Enum):
    """Input sanitization levels."""

    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    NONE = "none"


class ValidationConfigModel(BaseModel):
    """Main validation configuration model."""

    enabled: bool = Field(default=True, description="Enable validation globally")

    mode: ValidationMode = Field(
        default=ValidationMode.STRICT, description="Validation strictness mode"
    )

    sanitization_level: SanitizationLevel = Field(
        default=SanitizationLevel.MODERATE, description="Input sanitization level"
    )

    max_validation_errors: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of validation errors to return",
    )

    enable_logging: bool = Field(default=True, description="Enable validation logging")

    log_level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Validation logging level",
    )

    enable_metrics: bool = Field(default=True, description="Enable validation metrics collection")

    rate_limit_validation: bool = Field(default=True, description="Enable validation rate limiting")

    validation_rate_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Validation requests per minute rate limit",
    )

    skip_paths: list[str] = Field(
        default_factory=lambda: [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static",
            "/assets",
            "/favicon.ico",
        ],
        description="Paths to skip validation for",
    )

    custom_validators: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom validator configurations",
        max_length=50,
    )

    error_response_format: str = Field(
        default="detailed",
        pattern=r"^(basic|detailed|verbose)$",
        description="Validation error response format",
    )

    enable_field_suggestions: bool = Field(
        default=True, description="Enable field suggestions in error messages"
    )

    max_string_length: int = Field(
        default=10000, ge=1, le=100000, description="Maximum allowed string length"
    )

    max_nested_depth: int = Field(
        default=10, ge=1, le=50, description="Maximum nested object depth"
    )

    max_array_items: int = Field(
        default=1000, ge=1, le=10000, description="Maximum items in arrays"
    )

    allowed_file_types: list[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "text/csv",
            "application/json",
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ],
        description="Allowed file MIME types",
    )

    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,  # 1KB minimum
        le=100 * 1024 * 1024,  # 100MB maximum
        description="Maximum file size in bytes",
    )

    enable_circuit_breaker: bool = Field(
        default=True, description="Enable circuit breaker for validation failures"
    )

    circuit_breaker_threshold: int = Field(
        default=10, ge=1, le=100, description="Circuit breaker failure threshold"
    )

    circuit_breaker_timeout: int = Field(
        default=60, ge=10, le=3600, description="Circuit breaker timeout in seconds"
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("skip_paths")
    @classmethod
    def validate_skip_paths(cls, v):
        """Validate skip paths."""
        if not v:
            return v
        # Validate path format
        for path in v:
            if not path.startswith("/"):
                raise ValueError(f"Skip path must start with '/': {path}")
            if ".." in path or "//" in path:
                raise ValueError(f"Invalid skip path format: {path}")
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in v:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        return unique_paths

    @field_validator("custom_validators")
    @classmethod
    def validate_custom_validators(cls, v):
        """Validate custom validator configurations."""
        if not v:
            return v
        for name, config in v.items():
            if not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid custom validator name: {name}")
            if not isinstance(config, dict):
                raise ValueError(f"Custom validator '{name}' must be a dictionary")
            # Validate config structure
            required_keys = {"type", "enabled"}
            if not all(key in config for key in required_keys):
                raise ValueError(
                    f"Custom validator '{name}' missing required keys: {required_keys}"
                )
            if not isinstance(config["enabled"], bool):
                raise ValueError(f"Custom validator '{name}' enabled must be boolean")
        return v

    @field_validator("allowed_file_types")
    @classmethod
    def validate_allowed_file_types(cls, v):
        """Validate allowed file types."""
        if not v:
            raise ValueError("At least one allowed file type must be specified")
        # Validate MIME type format
        for mime_type in v:
            if not re.match(r"^[a-zA-Z0-9\-]+\/[a-zA-Z0-9\-\+]+(;.*)?$", mime_type):
                raise ValueError(f"Invalid MIME type format: {mime_type}")
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for mime_type in v:
            if mime_type not in seen:
                seen.add(mime_type)
                unique_types.append(mime_type)
        return unique_types


class AgentValidationConfig(BaseModel):
    """Agent-specific validation configuration."""

    max_name_length: int = Field(default=100, ge=1, le=500, description="Maximum agent name length")

    max_description_length: int = Field(
        default=500, ge=1, le=2000, description="Maximum agent description length"
    )

    allowed_models: list[str] = Field(
        default_factory=lambda: ["gpt-3.5-turbo", "gpt-4", "claude-3", "llama2"],
        description="Allowed AI models",
    )

    max_tools: int = Field(
        default=20, ge=0, le=100, description="Maximum number of tools per agent"
    )

    require_unique_name: bool = Field(default=True, description="Require unique agent names")

    validate_model_compatibility: bool = Field(
        default=True, description="Validate model compatibility"
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class ChatValidationConfig(BaseModel):
    """Chat-specific validation configuration."""

    max_message_length: int = Field(
        default=10000, ge=1, le=100000, description="Maximum message length"
    )

    max_session_name_length: int = Field(
        default=200, ge=1, le=500, description="Maximum session name length"
    )

    max_context_messages: int = Field(
        default=100, ge=1, le=1000, description="Maximum context messages"
    )

    enable_content_filtering: bool = Field(default=True, description="Enable content filtering")

    blocked_keywords: list[str] = Field(
        default_factory=list, description="Keywords to block in messages", max_length=100
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class SecurityValidationConfig(BaseModel):
    """Security-specific validation configuration."""

    enable_xss_protection: bool = Field(default=True, description="Enable XSS protection")

    enable_sql_injection_protection: bool = Field(
        default=True, description="Enable SQL injection protection"
    )

    enable_command_injection_protection: bool = Field(
        default=True, description="Enable command injection protection"
    )

    enable_path_traversal_protection: bool = Field(
        default=True, description="Enable path traversal protection"
    )

    max_request_size: int = Field(
        default=1 * 1024 * 1024,  # 1MB
        ge=1024,  # 1KB minimum
        le=10 * 1024 * 1024,  # 10MB maximum
        description="Maximum request size in bytes",
    )

    rate_limit_per_minute: int = Field(
        default=100, ge=1, le=1000, description="Requests per minute rate limit"
    )

    max_login_attempts: int = Field(default=5, ge=1, le=20, description="Maximum login attempts")

    enable_captcha_validation: bool = Field(default=False, description="Enable CAPTCHA validation")

    model_config = ConfigDict(
        extra="forbid",
    )


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")

    requests_per_minute: int = Field(default=60, ge=1, le=1000, description="Requests per minute")

    burst_size: int = Field(default=10, ge=1, le=100, description="Burst size for rate limiting")

    window_size: int = Field(
        default=60, ge=1, le=3600, description="Rate limiting window size in seconds"
    )

    key_prefix: str = Field(default="validation_rate_limit", description="Rate limiting key prefix")

    enable_ip_based_limiting: bool = Field(
        default=True, description="Enable IP-based rate limiting"
    )

    enable_user_based_limiting: bool = Field(
        default=False, description="Enable user-based rate limiting"
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class ValidationSettings:
    """Validation settings manager."""

    def __init__(self, config_file: str | Path | None = None):
        """
        Initialize validation settings.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config: ValidationConfigModel | None = None
        self._agent_config: AgentValidationConfig | None = None
        self._chat_config: ChatValidationConfig | None = None
        self._security_config: SecurityValidationConfig | None = None
        self._rate_limit_config: RateLimitConfig | None = None

        # Load configuration
        self.load_config()

    def load_config(self) -> None:
        """Load validation configuration."""
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config_data = f.read()

                # Try JSON first, then fallback to other formats
                if self.config_file.suffix.lower() == ".json":
                    import json

                    config_dict = json.loads(config_data)
                else:
                    # Try TOML
                    try:
                        import toml

                        config_dict = toml.loads(config_data)
                    except ImportError:
                        # Try YAML
                        try:
                            import yaml

                            config_dict = yaml.safe_load(config_data)
                        except ImportError:
                            # Fallback to simple key=value format
                            config_dict = self._parse_key_value_config(config_data)

                self._config = ValidationConfigModel(**config_dict)

            except Exception as e:
                raise ValueError(
                    f"Failed to load validation config from {self.config_file}: {e}"
                ) from e
        else:
            # Use default configuration
            self._config = ValidationConfigModel()

    def _parse_key_value_config(self, config_text: str) -> dict:
        """Parse simple key=value configuration format."""
        config_dict = {}
        for line in config_text.splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Try to parse as different types
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
                elif value.startswith("[") and value.endswith("]"):
                    # Simple list parsing
                    value = [item.strip() for item in value[1:-1].split(",")]

                config_dict[key] = value

        return config_dict

    def save_config(self, config_file: str | Path | None = None) -> None:
        """
        Save validation configuration.

        Args:
            config_file: Path to save configuration file
        """
        if config_file:
            self.config_file = Path(config_file)

        if not self.config_file:
            raise ValueError("No configuration file specified")

        # Create directory if it doesn't exist
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON by default
        with open(self.config_file, "w") as f:
            import json

            json.dump(self._config.model_dump(), f, indent=2)

    def update_config(self, **kwargs) -> None:
        """
        Update configuration values.

        Args:
            **kwargs: Configuration values to update
        """
        if not self._config:
            self._config = ValidationConfigModel()

        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def get_config(self) -> ValidationConfigModel:
        """Get main validation configuration."""
        if not self._config:
            self._config = ValidationConfigModel()
        return self._config

    def get_agent_config(self) -> AgentValidationConfig:
        """Get agent validation configuration."""
        if not self._agent_config:
            self._agent_config = AgentValidationConfig()
        return self._agent_config

    def get_chat_config(self) -> ChatValidationConfig:
        """Get chat validation configuration."""
        if not self._chat_config:
            self._chat_config = ChatValidationConfig()
        return self._chat_config

    def get_security_config(self) -> SecurityValidationConfig:
        """Get security validation configuration."""
        if not self._security_config:
            self._security_config = SecurityValidationConfig()
        return self._security_config

    def get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limiting configuration."""
        if not self._rate_limit_config:
            self._rate_limit_config = RateLimitConfig()
        return self._rate_limit_config

    def is_validation_enabled(self) -> bool:
        """Check if validation is enabled."""
        return self._config.enabled if self._config else True

    def get_skip_paths(self) -> list[str]:
        """Get paths to skip validation for."""
        return self._config.skip_paths if self._config else []

    def get_max_validation_errors(self) -> int:
        """Get maximum validation errors to return."""
        return self._config.max_validation_errors if self._config else 50


# Global validation settings instance
_validation_settings: ValidationSettings | None = None


def get_validation_settings(
    config_file: str | Path | None = None,
) -> ValidationSettings:
    """
    Get global validation settings instance.

    Args:
        config_file: Path to configuration file

    Returns:
        ValidationSettings instance
    """
    global _validation_settings

    if _validation_settings is None:
        _validation_settings = ValidationSettings(config_file)

    return _validation_settings


def set_validation_settings(settings: ValidationSettings) -> None:
    """
    Set global validation settings instance.

    Args:
        settings: ValidationSettings instance
    """
    global _validation_settings
    _validation_settings = settings


__all__ = [
    "ValidationMode",
    "SanitizationLevel",
    "ValidationConfigModel",
    "AgentValidationConfig",
    "ChatValidationConfig",
    "SecurityValidationConfig",
    "RateLimitConfig",
    "ValidationSettings",
    "get_validation_settings",
    "set_validation_settings",
]
