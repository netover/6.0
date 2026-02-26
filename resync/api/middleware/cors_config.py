import ipaddress
import logging
import re
import socket
from enum import Enum
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types for CORS configuration."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"

class CORSMethods(str, Enum):
    """Allowed HTTP methods for CORS."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"

class CORSPolicy(BaseModel):
    """
    CORS policy configuration model with validation.

    This model defines the CORS policy for different environments with
    strict validation to ensure security best practices.
    """

    environment: Environment = Field(
        description="Environment type (development, production, test)"
    )
    allowed_origins: list[str] = Field(
        default=[],
        description="List of allowed origins. Use specific domains in production, wildcards only in development.",
    )
    allowed_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="List of allowed HTTP methods.",
    )
    allowed_headers: list[str] = Field(
        default=["Content-Type", "Authorization", "X-Requested-With"],
        description="List of allowed headers.",
    )
    allow_credentials: bool = Field(
        default=False, description="Whether to allow credentials in CORS requests."
    )
    max_age: int = Field(
        default=86400, description="Maximum age in seconds for preflight cache."
    )
    allow_all_origins: bool = Field(
        default=False, description="Whether to allow all origins (development only)."
    )
    log_violations: bool = Field(
        default=True,
        description="Whether to log CORS violations for security monitoring.",
    )
    origin_regex_patterns: list[str] = Field(
        default=[], description="Regex patterns for dynamic origin validation."
    )

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            v = v.lower()
            if v in ["dev", "development"]:
                return Environment.DEVELOPMENT
            if v in ["prod", "production"]:
                return Environment.PRODUCTION
            if v in ["test", "testing"]:
                return Environment.TEST
        return v

    @field_validator("allowed_origins", mode="after")
    @classmethod
    def validate_origin(cls, v, info):
        """Validate each origin in the allowed_origins list."""
        if not v:
            return v
        environment = info.data.get("environment")
        validated_origins = []
        for origin in v:
            if environment == Environment.PRODUCTION and "*" in origin:
                raise ValueError(
                    "Wildcard origins are not allowed in production. Use specific domain names instead."
                )
            if origin != "*" and (not cls._is_valid_origin_format(origin)):
                raise ValueError(
                    f"Invalid origin format: {origin}. Expected format: http(s)://domain.com or http(s)://domain.com:port"
                )
            validated_origins.append(origin)
        return validated_origins

    @field_validator("allowed_methods", mode="after")
    @classmethod
    def validate_method(cls, v):
        """Validate HTTP methods."""
        if not v:
            return v
        allowed_methods = {method.value for method in CORSMethods}
        validated_methods = []
        for method in v:
            if method not in allowed_methods:
                raise ValueError(
                    f"Invalid HTTP method: {method}. Allowed methods: {', '.join(allowed_methods)}"
                )
            validated_methods.append(method)
        return validated_methods

    @field_validator("max_age")
    @classmethod
    def validate_max_age(cls, v):
        """Validate max age is reasonable."""
        if v < 0:
            raise ValueError("max_age must be non-negative")
        if v > 86400 * 7:
            raise ValueError("max_age should not exceed 7 days (604800 seconds)")
        return v

    @field_validator("origin_regex_patterns", mode="after")
    @classmethod
    def validate_regex_pattern(cls, v, info):
        """Validate regex patterns are compilable and not allowed in production."""
        if not v:
            return v
        environment = info.data.get("environment")

        # Check environment FIRST (fail fast) - avoid unnecessary regex compilation
        if environment == Environment.PRODUCTION:
            raise ValueError(
                "Regex patterns are not allowed in production. Use explicit domain names in allowed_origins instead."
            )

        # Only compile regex if not production
        validated_patterns = []
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
            validated_patterns.append(pattern)
        return validated_patterns

    @staticmethod
    def _is_valid_origin_format(origin: str) -> bool:
        """
        Validate origin format comprehensively.

        Args:
            origin: Origin string to validate

        Returns:
            True if format is valid, False otherwise
        """
        if origin == "*":
            return True
        try:
            parsed = urlparse(origin)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("exception_caught", exc_info=True, extra={"error": str(e)})
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        netloc = parsed.netloc
        if ".." in netloc or "//" in netloc[1:]:
            return False
        host = parsed.hostname
        if host:
            if host.lower() == "localhost":
                return True
            if ":" in host:
                # Use ipaddress module for robust IP validation
                try:
                    ipaddress.ip_address(host.strip("[]"))
                    return True
                except ValueError:
                    pass
            elif "." in host:
                try:
                    socket.inet_aton(host)
                    return True
                except OSError:
                    domain_pattern = "^[a-zA-Z0-9]([a-zA-Z0-9\\-]{0,61}[a-zA-Z0-9])?(\\.[a-zA-Z0-9]([a-zA-Z0-9\\-]{0,61}[a-zA-Z0-9])?)*$"
                    return bool(re.match(domain_pattern, host))
        return False

    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if an origin is allowed based on the policy.

        Args:
            origin: Origin to check

        Returns:
            True if origin is allowed, False otherwise
        """
        if self.allow_all_origins or "*" in self.allowed_origins:
            return True
        if origin in self.allowed_origins:
            return True
        for pattern in self.origin_regex_patterns:
            try:
                if re.match(pattern, origin):
                    return True
            except re.error:
                logger.warning("Invalid regex pattern in CORS config: %s", pattern)
                continue
        return False

    def get_cors_config_dict(self) -> dict:
        """
        Get CORS configuration as a dictionary for FastAPI middleware.

        Returns:
            Dictionary with CORS configuration parameters
        """
        return {
            "allow_origins": self.allowed_origins
            if not self.allow_all_origins
            else ["*"],
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age,
        }

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

class CORSConfig(BaseModel):
    """
    Main CORS configuration model that contains policies for different environments.
    """

    development: CORSPolicy = Field(
        default_factory=lambda: CORSPolicy(
            environment=Environment.DEVELOPMENT,
            allowed_origins=["*"],
            allow_all_origins=True,
            allow_credentials=False,
            log_violations=True,
        ),
        description="CORS policy for development environment",
    )
    production: CORSPolicy = Field(
        default_factory=lambda: CORSPolicy(
            environment=Environment.PRODUCTION,
            allowed_origins=[],
            allow_all_origins=False,
            allow_credentials=False,
            log_violations=True,
            origin_regex_patterns=[],
        ),
        description="CORS policy for production environment",
    )
    test: CORSPolicy = Field(
        default_factory=lambda: CORSPolicy(
            environment=Environment.TEST,
            allowed_origins=["http://localhost:3000", "http://localhost:8000"],
            allow_all_origins=False,
            allow_credentials=True,
            log_violations=True,
        ),
        description="CORS policy for test environment",
    )

    def get_policy(self, environment: str | Environment) -> CORSPolicy:
        """
        Get CORS policy for a specific environment.

        Args:
            environment: Environment name or Environment enum value

        Returns:
            CORSPolicy for the specified environment
        """
        if isinstance(environment, str):
            environment = Environment(environment.lower())
        if environment == Environment.DEVELOPMENT:
            return self.development
        if environment == Environment.PRODUCTION:
            return self.production
        if environment == Environment.TEST:
            return self.test
        raise ValueError(f"Unknown environment: {environment}")

    def update_policy(self, environment: str | Environment, policy: CORSPolicy) -> None:
        """
        Update CORS policy for a specific environment.

        Args:
            environment: Environment name or Environment enum value
            policy: New CORS policy to apply
        """
        if isinstance(environment, str):
            environment = Environment(environment.lower())
        if environment == Environment.DEVELOPMENT:
            self.development = policy
        elif environment == Environment.PRODUCTION:
            self.production = policy
        elif environment == Environment.TEST:
            self.test = policy
        else:
            raise ValueError(f"Unknown environment: {environment}")

cors_config = CORSConfig()
