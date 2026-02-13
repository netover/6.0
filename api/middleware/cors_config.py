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

    # Environment-specific settings
    environment: Environment = Field(description="Environment type (development, production, test)")

    # Allowed origins configuration
    allowed_origins: list[str] = Field(
        default=[],
        description="List of allowed origins. Use specific domains in production, wildcards only in development.",
    )

    # Allowed methods configuration
    allowed_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="List of allowed HTTP methods.",
    )

    # Allowed headers configuration
    allowed_headers: list[str] = Field(
        default=["Content-Type", "Authorization", "X-Requested-With"],
        description="List of allowed headers.",
    )

    # CORS behavior settings
    allow_credentials: bool = Field(
        default=False, description="Whether to allow credentials in CORS requests."
    )

    max_age: int = Field(
        default=86400,  # 24 hours
        description="Maximum age in seconds for preflight cache.",
    )

    # Security settings
    allow_all_origins: bool = Field(
        default=False, description="Whether to allow all origins (development only)."
    )

    # Logging settings
    log_violations: bool = Field(
        default=True,
        description="Whether to log CORS violations for security monitoring.",
    )

    # Dynamic validation settings
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

    # Pydantic V2: Validate entire list instead of each_item
    @field_validator("allowed_origins", mode="after")
    @classmethod
    def validate_origin(cls, v, info):
        """Validate each origin in the allowed_origins list."""
        if not v:
            return v

        # Get environment from context
        environment = info.data.get("environment")

        validated_origins = []
        for origin in v:
            # Check for wildcard in production
            if environment == Environment.PRODUCTION and "*" in origin:
                raise ValueError(
                    "Wildcard origins are not allowed in production. "
                    "Use specific domain names instead."
                )

            # Validate origin format
            if origin != "*" and not cls._is_valid_origin_format(origin):
                raise ValueError(
                    f"Invalid origin format: {origin}. "
                    "Expected format: http(s)://domain.com or http(s)://domain.com:port"
                )

            validated_origins.append(origin)

        return validated_origins

    # Pydantic V2: Validate entire list instead of each_item
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
        if v > 86400 * 7:  # 7 days
            raise ValueError("max_age should not exceed 7 days (604800 seconds)")
        return v

    # Pydantic V2: Validate entire list instead of each_item
    @field_validator("origin_regex_patterns", mode="after")
    @classmethod
    def validate_regex_pattern(cls, v, info):
        """Validate regex patterns are compilable and not allowed in production."""
        if not v:
            return v

        environment = info.data.get("environment")
        validated_patterns = []

        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

            # Check if regex patterns are used in production
            if environment == Environment.PRODUCTION:
                raise ValueError(
                    "Regex patterns are not allowed in production. "
                    "Use explicit domain names in allowed_origins instead."
                )

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

        # Parse the origin using urlparse for proper validation
        try:
            parsed = urlparse(origin)
        except Exception as e:
            logger.error("exception_caught", error=str(e), exc_info=True)
            return False

        # Check that the origin has a valid scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Check that the origin has a netloc (network location)
        if not parsed.netloc:
            return False

        # Validate netloc components to prevent common attacks
        netloc = parsed.netloc

        # Check for invalid characters or patterns
        if (
            ".." in netloc or "//" in netloc[1:]
        ):  # Prevent directory traversal and duplicate slashes
            return False

        # Extract host and port
        host = parsed.hostname

        if host:
            # Special case: localhost is always valid
            if host.lower() == "localhost":
                return True

            # IPv4/IPv6 validation
            # Check if it's a valid domain name or IP address
            # For domain validation, check basic pattern
            if ":" in host:  # Could be IPv6
                try:
                    socket.inet_pton(socket.AF_INET6, host.strip("[]"))
                    return True
                except OSError as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

            elif "." in host:  # Likely IPv4 or domain
                try:
                    socket.inet_aton(host)  # Valid IPv4
                    return True
                except OSError:
                    # Not IPv4, check if it's a valid domain name
                    # Simple domain validation using regex
                    domain_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
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
        # Quick check for wildcard
        if self.allow_all_origins or "*" in self.allowed_origins:
            return True

        # Check explicit allowed origins
        if origin in self.allowed_origins:
            return True

        # Check regex patterns for dynamic validation
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
            "allow_origins": (self.allowed_origins if not self.allow_all_origins else ["*"]),
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age,
        }

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )


class CORSConfig(BaseModel):
    """
    Main CORS configuration model that contains policies for different environments.
    """

    # Default policies for each environment
    development: CORSPolicy = Field(
        default_factory=lambda: CORSPolicy(
            environment=Environment.DEVELOPMENT,
            allowed_origins=["*"],
            allow_all_origins=True,
            allow_credentials=False,  # MUST be False with wildcard origins (CORS spec)
            log_violations=True,
        ),
        description="CORS policy for development environment",
    )

    production: CORSPolicy = Field(
        default_factory=lambda: CORSPolicy(
            environment=Environment.PRODUCTION,
            allowed_origins=[],  # Must be configured explicitly
            allow_all_origins=False,
            allow_credentials=False,  # More restrictive for production
            log_violations=True,
            origin_regex_patterns=[],  # No regex patterns in production for security
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


# NOTE: Avoid module-level singletons here.
# The application factory constructs a CORSConfig instance during startup so
# configuration is always derived from the active settings/environment.


# NOTE: Backward-compatible module-level singleton. Prefer creating a
# CORSConfig instance in app_factory or DI. This may be removed in v7.0.
cors_config = CORSConfig()
