"""
CORS configuration module for Resync application.
"""

import logging
from urllib.parse import urlparse

from fastapi import FastAPI

from resync.api.middleware.cors_config import CORSPolicy, Environment
from resync.api.middleware.cors_middleware import add_cors_middleware
from resync.settings import settings

logger = logging.getLogger(__name__)


def validate_origin(origin: str) -> bool:
    """
    Validate a single origin string.

    Args:
        origin: Origin string to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(origin, str) or not origin.strip():
        return False

    try:
        parsed = urlparse(origin.strip())
        if parsed.scheme in ("http", "https") and parsed.netloc or origin.strip() == "*":
            return True
        logger.warning(f"Invalid origin format: {origin}")
        return False
    except Exception as e:
        logger.warning(f"Error parsing origin '{origin}': {e}")
        return False


def parse_cors_origins(cors_origins_setting: str | list[str] | None) -> list[str]:
    """
    Parse CORS origins from settings, handling both string and list formats.

    Args:
        cors_origins_setting: Raw CORS origins setting from config

    Returns:
        List of valid origin strings
    """
    if not cors_origins_setting:
        return []

    if isinstance(cors_origins_setting, str):
        # Parse comma-separated string
        origins = [origin.strip() for origin in cors_origins_setting.split(",") if origin.strip()]
    elif isinstance(cors_origins_setting, list):
        # Use as-is if it's already a list
        origins = cors_origins_setting
    else:
        logger.warning(
            f"Unexpected CORS origins type: {type(cors_origins_setting)}, returning empty list"
        )
        return []

    # Validate each origin and filter out invalid ones
    valid_origins = []
    for origin in origins:
        if validate_origin(origin):
            valid_origins.append(origin.strip())

    return valid_origins


def configure_cors(app: FastAPI, settings_module: object | None = None) -> None:
    """
    Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings_module: Settings object (defaults to global settings)
    """
    settings_obj = settings_module or settings

    cors_enabled = getattr(settings_obj, "CORS_ENABLED", True)
    if not cors_enabled:
        logger.info("CORS middleware disabled")
        return

    cors_environment = getattr(settings_obj, "CORS_ENVIRONMENT", "development")

    # Get allowed origins from settings - it could be a list or a comma-separated string
    cors_origins_setting = getattr(settings_obj, "CORS_ALLOWED_ORIGINS", [])

    # Parse and validate origins
    valid_origins = parse_cors_origins(cors_origins_setting)

    # Create and add CORS middleware with environment-specific configuration
    try:
        if valid_origins:
            custom_policy = CORSPolicy(
                environment=Environment(cors_environment),
                allowed_origins=valid_origins,
                allow_all_origins=False,
                allow_credentials=getattr(settings_obj, "CORS_ALLOW_CREDENTIALS", False),
                log_violations=getattr(settings_obj, "CORS_LOG_VIOLATIONS", True),
            )
            add_cors_middleware(app=app, environment=cors_environment, custom_policy=custom_policy)
        else:
            add_cors_middleware(app=app, environment=cors_environment)

        logger.info(f"CORS middleware added for environment: {cors_environment}")
    except Exception as e:
        logger.error(f"Failed to add CORS middleware: {e}")
        logger.info("Continuing without CORS middleware")
