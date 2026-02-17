"""
Startup Validation Module

This module provides comprehensive validation for application startup,
including environment variables, dependencies, and configuration checks.
All validation functions follow a fail-fast approach to ensure the
application only starts with valid configuration.
"""

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from resync.settings import Settings

# Configure startup logger
startup_logger = structlog.get_logger("resync.startup.validation")

# Required environment variables for different components
REQUIRED_ENV_VARS = {
    "redis": ["REDIS_URL"],
    "tws": ["TWS_HOST", "TWS_PORT", "TWS_USER", "TWS_PASSWORD"],
    "llm": ["LLM_ENDPOINT", "LLM_API_KEY"],
    "security": ["ADMIN_USERNAME", "ADMIN_PASSWORD", "SECRET_KEY"],
}

# Optional but recommended environment variables
RECOMMENDED_ENV_VARS = {
    "monitoring": ["LOG_LEVEL"],
    "server": ["SERVER_HOST", "SERVER_PORT"],
}


class ConfigurationValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DependencyUnavailableError(Exception):
    """Raised when required dependencies are unavailable."""

    def __init__(self, message: str, dependency: str, details: dict[str, Any] | None = None):
        self.message = message
        self.dependency = dependency
        self.details = details or {}
        super().__init__(self.message)


class StartupError(Exception):
    """Raised for general startup errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


def validate_environment_variables() -> dict[str, str]:
    """
    Validate required environment variables are present and valid.

    Returns:
        Dict containing validated environment variables

    Raises:
        ConfigurationValidationError: If required variables are missing or invalid
    """
    startup_logger.info("validating_environment_variables_started")

    missing_vars: list[str] = []
    invalid_vars: list[str] = []
    validated_vars: dict[str, str] = {}

    # Check required variables
    for _component, vars_list in REQUIRED_ENV_VARS.items():
        for var_name in vars_list:
            value = os.getenv(var_name)
            if not value:
                missing_vars.append(var_name)
            else:
                # Basic validation for known variable types
                if var_name == "TWS_PORT":
                    try:
                        port = int(value)
                        if not (1 <= port <= 65535):
                            invalid_vars.append(f"{var_name}={value} (must be 1-65535)")
                        else:
                            validated_vars[var_name] = value
                    except ValueError:
                        invalid_vars.append(f"{var_name}={value} (must be integer)")
                elif var_name in ["REDIS_URL", "LLM_ENDPOINT"]:
                    # Use urllib.parse to perform a more flexible URL validation. Accept any
                    # scheme as long as it contains a scheme and either a network location or
                    # path component. This allows for redis+sentinel:// and other extended
                    # schemes while still rejecting malformed values. See SECURITY_CONFIGURATION_GUIDE.md
                    # for details on acceptable URL formats.
                    from urllib.parse import urlparse

                    parsed = urlparse(value)
                    # A valid URL must have a scheme and either a netloc (e.g. host:port)
                    # or a path (for Unix socket schemes). Some schemes like redis+sentinel
                    # include a plus in the scheme, which urlparse handles correctly.
                    if not parsed.scheme or not (parsed.netloc or parsed.path):
                        invalid_vars.append(
                            f"{var_name}={value} (must be a valid URL with scheme and host or path)"
                        )
                    else:
                        validated_vars[var_name] = value
                else:
                    # Basic non-empty validation for other required vars
                    if len(value.strip()) == 0:
                        invalid_vars.append(f"{var_name} (cannot be empty)")
                    else:
                        validated_vars[var_name] = value

    # Check recommended variables and warn if missing
    for component, vars_list in RECOMMENDED_ENV_VARS.items():
        for var_name in vars_list:
            value = os.getenv(var_name)
            if not value:
                startup_logger.warning(
                    "recommended_env_var_missing",
                    component=component,
                    variable=var_name,
                    default_behavior="using default values",
                )

    # Report errors
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        startup_logger.error(
            "required_env_vars_missing", missing_vars=missing_vars, error_message=error_msg
        )
        raise ConfigurationValidationError(error_msg, {"missing_vars": missing_vars})

    if invalid_vars:
        error_msg = f"Invalid environment variables: {', '.join(invalid_vars)}"
        startup_logger.error("invalid_env_vars", invalid_vars=invalid_vars, error_message=error_msg)
        raise ConfigurationValidationError(error_msg, {"invalid_vars": invalid_vars})

    startup_logger.info(
        "environment_variables_validated",
        validated_count=len(validated_vars),
        total_required=sum(len(vars_list) for vars_list in REQUIRED_ENV_VARS.values()),
    )

    return validated_vars


async def validate_redis_connection(max_retries: int = 3, timeout: float = 5.0) -> bool:
    """
    Validate Redis connection with retry logic.

    Args:
        max_retries: Maximum number of connection attempts
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful

    Raises:
        DependencyUnavailableError: If Redis is unavailable after retries
    """
    startup_logger.info(
        "validating_redis_connection_started", max_retries=max_retries, timeout=timeout
    )

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ConfigurationValidationError("REDIS_URL environment variable not set")

    for attempt in range(max_retries):
        start_time = time.time()

        try:
            # Import here to avoid circular dependencies
            from redis.asyncio import ConnectionError as RedisConnectionError
            from redis.asyncio import Redis
            from redis.asyncio import TimeoutError as RedisTimeoutError

            client = Redis.from_url(
                redis_url, socket_connect_timeout=timeout, socket_timeout=timeout
            )

            # Test connection
            await client.ping()

            # Clean up
            await client.close()
            await client.connection_pool.disconnect()

            duration = time.time() - start_time
            startup_logger.info(
                "redis_connection_validated",
                attempt=attempt + 1,
                duration_ms=int(duration * 1000),
                redis_host=redis_url.split("@")[-1] if "@" in redis_url else redis_url,
            )
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            duration = time.time() - start_time
            startup_logger.warning(
                "redis_connection_attempt_failed",
                attempt=attempt + 1,
                max_retries=max_retries,
                duration_ms=int(duration * 1000),
                error_type=type(e).__name__,
                error_message=str(e),
            )

            if attempt < max_retries - 1:
                # Exponential backoff
                backoff = min(10.0, 0.1 * (2**attempt))
                await asyncio.sleep(backoff)
            else:
                # Final failure
                raise DependencyUnavailableError(
                    f"Redis unavailable after {max_retries} attempts: {type(e).__name__}",
                    "redis",
                    {
                        "attempts": max_retries,
                        "last_error": str(e),
                        "redis_url": redis_url.split("@")[-1] if "@" in redis_url else redis_url,
                    },
                ) from e

        except Exception as e:
            # Unexpected error
            startup_logger.error(
                "redis_unexpected_error",
                error_type=type(e).__name__,
                error_message=str(e) if os.getenv("ENVIRONMENT") != "production" else None,
            )
            raise StartupError(
                f"Unexpected Redis validation error: {type(e).__name__}", {"error": str(e)}
            ) from e

    return False


def validate_tws_configuration() -> dict[str, str]:
    """
    Validate TWS configuration parameters.

    Returns:
        Dict containing validated TWS configuration

    Raises:
        ConfigurationValidationError: If TWS configuration is invalid
    """
    startup_logger.info("validating_tws_configuration_started")

    tws_config = {}
    validation_errors = []

    # Validate TWS host
    tws_host = os.getenv("TWS_HOST")
    if not tws_host or len(tws_host.strip()) == 0:
        validation_errors.append("TWS_HOST cannot be empty")
    else:
        tws_config["host"] = tws_host.strip()

    # Validate TWS port
    tws_port_str = os.getenv("TWS_PORT")
    if not tws_port_str:
        validation_errors.append("TWS_PORT not set")
    else:
        try:
            tws_port = int(tws_port_str)
            # Accept any valid TCP port between 1 and 65535. Previously the lower
            # bound was restricted to 1024, which prevented the use of privileged
            # ports (e.g. 7500) in development and testing environments. See
            # CACHE_GUIDE.md for rationale. This change aligns port validation
            # with the full range of allowed TCP ports.
            if not (1 <= tws_port <= 65535):
                validation_errors.append(
                    f"TWS_PORT {tws_port} must be between 1-65535"
                )
            else:
                tws_config["port"] = str(tws_port)
        except ValueError:
            validation_errors.append(f"TWS_PORT '{tws_port_str}' must be a valid integer")

    # Validate TWS credentials
    tws_user = os.getenv("TWS_USER")
    tws_password = os.getenv("TWS_PASSWORD")

    if not tws_user or len(tws_user.strip()) == 0:
        validation_errors.append("TWS_USER cannot be empty")
    else:
        tws_config["user"] = tws_user.strip()

    if not tws_password or len(tws_password.strip()) == 0:
        validation_errors.append("TWS_PASSWORD cannot be empty")
    else:
        tws_config["password"] = tws_password.strip()

    if validation_errors:
        error_msg = f"TWS configuration validation failed: {'; '.join(validation_errors)}"
        startup_logger.error(
            "tws_configuration_invalid",
            validation_errors=validation_errors,
            error_message=error_msg,
        )
        raise ConfigurationValidationError(error_msg, {"validation_errors": validation_errors})

    startup_logger.info(
        "tws_configuration_validated", tws_host=tws_config["host"], tws_port=tws_config["port"]
    )

    return tws_config


def validate_security_settings() -> dict[str, str]:
    """
    Validate security-related configuration.

    Returns:
        Dict containing validated security settings

    Raises:
        ConfigurationValidationError: If security settings are invalid
    """
    startup_logger.info("validating_security_settings_started")

    security_config = {}
    validation_errors = []

    # Validate admin credentials
    admin_username = os.getenv("ADMIN_USERNAME")
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_username or len(admin_username.strip()) < 3:
        validation_errors.append("ADMIN_USERNAME must be at least 3 characters")
    else:
        security_config["admin_username"] = admin_username.strip()

    if not admin_password or len(admin_password.strip()) < 8:
        validation_errors.append("ADMIN_PASSWORD must be at least 8 characters")
    else:
        security_config["admin_password"] = admin_password.strip()

    # Validate secret key
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key or len(secret_key.strip()) < 32:
        validation_errors.append("SECRET_KEY must be at least 32 characters for security")
    else:
        security_config["secret_key"] = secret_key.strip()

    if validation_errors:
        error_msg = f"Security settings validation failed: {'; '.join(validation_errors)}"
        startup_logger.error(
            "security_settings_invalid",
            validation_errors=validation_errors,
            error_message=error_msg,
        )
        raise ConfigurationValidationError(error_msg, {"validation_errors": validation_errors})

    startup_logger.info("security_settings_validated")

    return security_config


async def validate_all_settings() -> "Settings":
    """
    Comprehensive validation of all application settings and dependencies.

    This function performs all necessary validations in the correct order:
    1. Environment variables
    2. Security settings
    3. TWS configuration
    4. Redis connection

    Returns:
        Validated Settings object

    Raises:
        ConfigurationValidationError: For configuration issues
        DependencyUnavailableError: For unavailable dependencies
        StartupError: For other startup issues
    """
    startup_logger.info("comprehensive_validation_started")

    start_time = time.time()
    validation_results = {}

    try:
        # Step 1: Validate environment variables
        validation_results["env_vars"] = validate_environment_variables()

        # Step 2: Validate security settings
        validation_results["security"] = validate_security_settings()

        # Step 3: Validate TWS configuration
        validation_results["tws"] = validate_tws_configuration()

        # Step 4: Validate Redis connection (async operation)
        await validate_redis_connection()

        # Step 5: Load and return settings
        from resync.settings import load_settings

        settings = load_settings()

        total_duration = time.time() - start_time
        startup_logger.info(
            "comprehensive_validation_completed",
            total_duration_ms=int(total_duration * 1000),
            components_validated=list(validation_results.keys()),
            settings_loaded=True,
        )

        return settings

    except (ConfigurationValidationError, DependencyUnavailableError, StartupError):
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected validation error: {type(e).__name__}"
        startup_logger.error(
            "validation_unexpected_error",
            error_type=type(e).__name__,
            error_message=str(e) if os.getenv("ENVIRONMENT") != "production" else None,
            partial_results=list(validation_results.keys()),
        )
        raise StartupError(
            error_msg, {"error": str(e), "partial_results": validation_results}
        ) from e
