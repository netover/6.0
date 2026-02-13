"""
Admin Environment Variables Management.

Provides a web interface to view and edit environment configuration.
Configurations are persisted to a .env file for system-wide settings.

SECURITY NOTES:
- Sensitive values are masked in GET responses
- Changes require admin role
- All changes are logged for audit
- System restart may be required for some changes
"""

import logging
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from resync.api.routes.admin.main import verify_admin_credentials
import aiofiles

logger = logging.getLogger(__name__)

# v5.9.5: Added authentication
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


# Configuration
ENV_FILE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "config" / ".env"
)
ENV_EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    / "config"
    / "database.env.example"
)


class VariableCategory(str, Enum):
    """Categories of environment variables."""

    DATABASE = "database"
    SECURITY = "security"
    API = "api"
    CACHE = "cache"
    MONITORING = "monitoring"
    NOTIFICATION = "notification"
    TWS = "tws"
    RAG = "rag"
    SYSTEM = "system"


class EnvironmentVariable(BaseModel):
    """Model for an environment variable."""

    name: str
    value: str | None = None
    category: VariableCategory
    description: str
    is_sensitive: bool = False
    is_required: bool = False
    default_value: str | None = None
    validation_pattern: str | None = None


class EnvironmentVariableUpdate(BaseModel):
    """Model for updating an environment variable."""

    value: str


class EnvironmentConfig(BaseModel):
    """Complete environment configuration."""

    variables: dict[str, EnvironmentVariable]
    last_modified: str | None = None
    env_file_exists: bool = False


# Define all configurable environment variables
ENVIRONMENT_SCHEMA: dict[str, EnvironmentVariable] = {
    # Database
    "DATABASE_DRIVER": EnvironmentVariable(
        name="DATABASE_DRIVER",
        category=VariableCategory.DATABASE,
        description="Database driver (postgresql, sqlite, mysql)",
        default_value="postgresql",
        is_required=True,
    ),
    "DATABASE_HOST": EnvironmentVariable(
        name="DATABASE_HOST",
        category=VariableCategory.DATABASE,
        description="Database server hostname",
        default_value="localhost",
        is_required=True,
    ),
    "DATABASE_PORT": EnvironmentVariable(
        name="DATABASE_PORT",
        category=VariableCategory.DATABASE,
        description="Database server port",
        default_value="5432",
        validation_pattern=r"^\d+$",
    ),
    "DATABASE_NAME": EnvironmentVariable(
        name="DATABASE_NAME",
        category=VariableCategory.DATABASE,
        description="Database name",
        default_value="resync",
        is_required=True,
    ),
    "DATABASE_USER": EnvironmentVariable(
        name="DATABASE_USER",
        category=VariableCategory.DATABASE,
        description="Database username",
        default_value="resync",
        is_required=True,
    ),
    "DATABASE_PASSWORD": EnvironmentVariable(
        name="DATABASE_PASSWORD",
        category=VariableCategory.DATABASE,
        description="Database password",
        is_sensitive=True,
        is_required=True,
    ),
    "DATABASE_POOL_SIZE": EnvironmentVariable(
        name="DATABASE_POOL_SIZE",
        category=VariableCategory.DATABASE,
        description="Connection pool size",
        default_value="10",
        validation_pattern=r"^\d+$",
    ),
    "DATABASE_MAX_OVERFLOW": EnvironmentVariable(
        name="DATABASE_MAX_OVERFLOW",
        category=VariableCategory.DATABASE,
        description="Max overflow connections",
        default_value="20",
        validation_pattern=r"^\d+$",
    ),
    "DATABASE_SSL_MODE": EnvironmentVariable(
        name="DATABASE_SSL_MODE",
        category=VariableCategory.DATABASE,
        description="SSL mode (disable, prefer, require)",
        default_value="prefer",
    ),
    # Security
    "SECRET_KEY": EnvironmentVariable(
        name="SECRET_KEY",
        category=VariableCategory.SECURITY,
        description="JWT secret key for authentication",
        is_sensitive=True,
        is_required=True,
    ),
    "ACCESS_TOKEN_EXPIRE_MINUTES": EnvironmentVariable(
        name="ACCESS_TOKEN_EXPIRE_MINUTES",
        category=VariableCategory.SECURITY,
        description="Access token expiration in minutes",
        default_value="30",
        validation_pattern=r"^\d+$",
    ),
    "ALGORITHM": EnvironmentVariable(
        name="ALGORITHM",
        category=VariableCategory.SECURITY,
        description="JWT algorithm",
        default_value="HS256",
    ),
    # API
    "API_HOST": EnvironmentVariable(
        name="API_HOST",
        category=VariableCategory.API,
        description="API server host",
        default_value="0.0.0.0",  # noqa: S104 - Container deployment default
    ),
    "API_PORT": EnvironmentVariable(
        name="API_PORT",
        category=VariableCategory.API,
        description="API server port",
        default_value="8000",
        validation_pattern=r"^\d+$",
    ),
    "API_DEBUG": EnvironmentVariable(
        name="API_DEBUG",
        category=VariableCategory.API,
        description="Enable debug mode (true/false)",
        default_value="false",
    ),
    "API_WORKERS": EnvironmentVariable(
        name="API_WORKERS",
        category=VariableCategory.API,
        description="Number of API workers",
        default_value="4",
        validation_pattern=r"^\d+$",
    ),
    "CORS_ORIGINS": EnvironmentVariable(
        name="CORS_ORIGINS",
        category=VariableCategory.API,
        description="Allowed CORS origins (comma-separated)",
        default_value="*",
    ),
    # Cache (Redis)
    "REDIS_HOST": EnvironmentVariable(
        name="REDIS_HOST",
        category=VariableCategory.CACHE,
        description="Redis server hostname",
        default_value="localhost",
    ),
    "REDIS_PORT": EnvironmentVariable(
        name="REDIS_PORT",
        category=VariableCategory.CACHE,
        description="Redis server port",
        default_value="6379",
        validation_pattern=r"^\d+$",
    ),
    "REDIS_PASSWORD": EnvironmentVariable(
        name="REDIS_PASSWORD",
        category=VariableCategory.CACHE,
        description="Redis password",
        is_sensitive=True,
    ),
    "REDIS_DB": EnvironmentVariable(
        name="REDIS_DB",
        category=VariableCategory.CACHE,
        description="Redis database number",
        default_value="0",
        validation_pattern=r"^\d+$",
    ),
    # TWS/HWA
    "TWS_HOST": EnvironmentVariable(
        name="TWS_HOST",
        category=VariableCategory.TWS,
        description="TWS/HWA server hostname",
        default_value="localhost",
    ),
    "TWS_PORT": EnvironmentVariable(
        name="TWS_PORT",
        category=VariableCategory.TWS,
        description="TWS/HWA server port",
        default_value="31116",
        validation_pattern=r"^\d+$",
    ),
    "TWS_USERNAME": EnvironmentVariable(
        name="TWS_USERNAME",
        category=VariableCategory.TWS,
        description="TWS/HWA username",
    ),
    "TWS_PASSWORD": EnvironmentVariable(
        name="TWS_PASSWORD",
        category=VariableCategory.TWS,
        description="TWS/HWA password",
        is_sensitive=True,
    ),
    "TWS_SSL_ENABLED": EnvironmentVariable(
        name="TWS_SSL_ENABLED",
        category=VariableCategory.TWS,
        description="Enable SSL for TWS connection",
        default_value="true",
    ),
    "TWS_TIMEOUT": EnvironmentVariable(
        name="TWS_TIMEOUT",
        category=VariableCategory.TWS,
        description="TWS connection timeout in seconds",
        default_value="30",
        validation_pattern=r"^\d+$",
    ),
    # RAG/pgvector (uses DATABASE_URL for PostgreSQL connection)
    "RAG_COLLECTION": EnvironmentVariable(
        name="RAG_COLLECTION",
        category=VariableCategory.RAG,
        description="pgvector collection name for RAG documents",
        default_value="resync_documents",
    ),
    "RAG_USE_MOCK": EnvironmentVariable(
        name="RAG_USE_MOCK",
        category=VariableCategory.RAG,
        description="Use mock RAG service (true/false)",
        default_value="false",
    ),
    "EMBEDDING_MODEL": EnvironmentVariable(
        name="EMBEDDING_MODEL",
        category=VariableCategory.RAG,
        description="Embedding model name",
        default_value="text-embedding-3-small",
    ),
    # LLM
    "OPENAI_API_KEY": EnvironmentVariable(
        name="OPENAI_API_KEY",
        category=VariableCategory.API,
        description="OpenAI API key",
        is_sensitive=True,
    ),
    "ANTHROPIC_API_KEY": EnvironmentVariable(
        name="ANTHROPIC_API_KEY",
        category=VariableCategory.API,
        description="Anthropic API key",
        is_sensitive=True,
    ),
    "LLM_MODEL": EnvironmentVariable(
        name="LLM_MODEL",
        category=VariableCategory.API,
        description="Default LLM model",
        default_value="gpt-4",
    ),
    "LLM_TEMPERATURE": EnvironmentVariable(
        name="LLM_TEMPERATURE",
        category=VariableCategory.API,
        description="LLM temperature (0.0-2.0)",
        default_value="0.7",
    ),
    # Notifications
    "SMTP_HOST": EnvironmentVariable(
        name="SMTP_HOST",
        category=VariableCategory.NOTIFICATION,
        description="SMTP server hostname",
    ),
    "SMTP_PORT": EnvironmentVariable(
        name="SMTP_PORT",
        category=VariableCategory.NOTIFICATION,
        description="SMTP server port",
        default_value="587",
        validation_pattern=r"^\d+$",
    ),
    "SMTP_USER": EnvironmentVariable(
        name="SMTP_USER",
        category=VariableCategory.NOTIFICATION,
        description="SMTP username",
    ),
    "SMTP_PASSWORD": EnvironmentVariable(
        name="SMTP_PASSWORD",
        category=VariableCategory.NOTIFICATION,
        description="SMTP password",
        is_sensitive=True,
    ),
    "SLACK_WEBHOOK_URL": EnvironmentVariable(
        name="SLACK_WEBHOOK_URL",
        category=VariableCategory.NOTIFICATION,
        description="Slack webhook URL",
        is_sensitive=True,
    ),
    "TEAMS_WEBHOOK_URL": EnvironmentVariable(
        name="TEAMS_WEBHOOK_URL",
        category=VariableCategory.NOTIFICATION,
        description="Microsoft Teams webhook URL",
        is_sensitive=True,
    ),
    # System
    "LOG_LEVEL": EnvironmentVariable(
        name="LOG_LEVEL",
        category=VariableCategory.SYSTEM,
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
        default_value="INFO",
    ),
    "ENVIRONMENT": EnvironmentVariable(
        name="ENVIRONMENT",
        category=VariableCategory.SYSTEM,
        description="Environment name (development, staging, production)",
        default_value="production",
    ),
    # LangFuse - AI Observability
    "LANGFUSE_ENABLED": EnvironmentVariable(
        name="LANGFUSE_ENABLED",
        category=VariableCategory.API,
        description="Enable LangFuse observability (true/false)",
        default_value="false",
    ),
    "LANGFUSE_PUBLIC_KEY": EnvironmentVariable(
        name="LANGFUSE_PUBLIC_KEY",
        category=VariableCategory.API,
        description="LangFuse public key",
    ),
    "LANGFUSE_SECRET_KEY": EnvironmentVariable(
        name="LANGFUSE_SECRET_KEY",
        category=VariableCategory.API,
        description="LangFuse secret key",
        is_sensitive=True,
    ),
    "LANGFUSE_HOST": EnvironmentVariable(
        name="LANGFUSE_HOST",
        category=VariableCategory.API,
        description="LangFuse host URL",
        default_value="https://cloud.langfuse.com",
    ),
    "LANGFUSE_SAMPLE_RATE": EnvironmentVariable(
        name="LANGFUSE_SAMPLE_RATE",
        category=VariableCategory.API,
        description="LangFuse trace sampling rate (0.0-1.0)",
        default_value="1.0",
    ),
    # Evidently - ML Monitoring
    "EVIDENTLY_ENABLED": EnvironmentVariable(
        name="EVIDENTLY_ENABLED",
        category=VariableCategory.SYSTEM,
        description="Enable Evidently drift monitoring (true/false)",
        default_value="false",
    ),
    "EVIDENTLY_REPORTS_DIR": EnvironmentVariable(
        name="EVIDENTLY_REPORTS_DIR",
        category=VariableCategory.SYSTEM,
        description="Directory for Evidently drift reports",
        default_value="/var/log/resync/evidently",
    ),
    # Backup
    "BACKUP_DIR": EnvironmentVariable(
        name="BACKUP_DIR",
        category=VariableCategory.SYSTEM,
        description="Directory for storing backups",
        default_value="/var/backups/resync",
    ),
}


def _mask_sensitive_value(value: str) -> str:
    """Mask sensitive values for display."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _load_env_file() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}

    if ENV_FILE_PATH.exists():
        with open(ENV_FILE_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip("\"'")
                    env_vars[key.strip()] = value

    return env_vars


def _save_env_file(env_vars: dict[str, str]) -> None:
    """Save environment variables to .env file."""
    # Ensure config directory exists
    ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if file exists
    if ENV_FILE_PATH.exists():
        backup_path = ENV_FILE_PATH.with_suffix(
            f".env.backup.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        ENV_FILE_PATH.rename(backup_path)
        logger.info("Created backup: %s", backup_path)

    # Write new file
    with open(ENV_FILE_PATH, "w") as f:
        f.write("# Resync Environment Configuration\n")
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("# WARNING: This file contains sensitive information\n\n")

        # Group by category
        categories = {}
        for key, value in env_vars.items():
            schema = ENVIRONMENT_SCHEMA.get(key)
            category = schema.category.value if schema else "other"
            if category not in categories:
                categories[category] = {}
            categories[category][key] = value

        for category, vars in sorted(categories.items()):
            f.write(f"\n# {category.upper()}\n")
            for key, value in sorted(vars.items()):
                # Quote values with spaces
                if " " in value or '"' in value:
                    value = f'"{value}"'
                f.write(f"{key}={value}\n")

    logger.info("Environment file saved: %s", ENV_FILE_PATH)


@router.get("/environment", tags=["Admin Environment"])
async def get_environment_config():
    """
    Get all environment configuration.

    Sensitive values are masked for security.
    """
    # Load current values from environment and .env file
    file_vars = _load_env_file()

    result = {}
    for name, schema in ENVIRONMENT_SCHEMA.items():
        # Get value from: 1) OS environment, 2) .env file, 3) default
        value = os.getenv(name) or file_vars.get(name) or schema.default_value

        result[name] = {
            "name": name,
            "value": _mask_sensitive_value(value) if schema.is_sensitive and value else value,
            "category": schema.category.value,
            "description": schema.description,
            "is_sensitive": schema.is_sensitive,
            "is_required": schema.is_required,
            "default_value": schema.default_value,
            "is_set": bool(os.getenv(name) or file_vars.get(name)),
            "source": "environment"
            if os.getenv(name)
            else ("file" if file_vars.get(name) else "default"),
        }

    return {
        "variables": result,
        "env_file_path": str(ENV_FILE_PATH),
        "env_file_exists": ENV_FILE_PATH.exists(),
        "last_modified": datetime.fromtimestamp(ENV_FILE_PATH.stat().st_mtime).isoformat()
        if ENV_FILE_PATH.exists()
        else None,
        "categories": [c.value for c in VariableCategory],
    }


@router.get("/environment/category/{category}", tags=["Admin Environment"])
async def get_environment_by_category(category: VariableCategory):
    """Get environment variables by category."""
    file_vars = _load_env_file()

    result = {}
    for name, schema in ENVIRONMENT_SCHEMA.items():
        if schema.category == category:
            value = os.getenv(name) or file_vars.get(name) or schema.default_value
            result[name] = {
                "name": name,
                "value": _mask_sensitive_value(value) if schema.is_sensitive and value else value,
                "description": schema.description,
                "is_sensitive": schema.is_sensitive,
                "is_required": schema.is_required,
                "default_value": schema.default_value,
                "is_set": bool(os.getenv(name) or file_vars.get(name)),
            }

    return {
        "category": category.value,
        "variables": result,
    }


@router.get("/environment/{variable_name}", tags=["Admin Environment"])
async def get_environment_variable(variable_name: str):
    """Get a specific environment variable."""
    if variable_name not in ENVIRONMENT_SCHEMA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown variable: {variable_name}",
        )

    schema = ENVIRONMENT_SCHEMA[variable_name]
    file_vars = _load_env_file()
    value = os.getenv(variable_name) or file_vars.get(variable_name) or schema.default_value

    return {
        "name": variable_name,
        "value": _mask_sensitive_value(value) if schema.is_sensitive and value else value,
        "category": schema.category.value,
        "description": schema.description,
        "is_sensitive": schema.is_sensitive,
        "is_required": schema.is_required,
        "default_value": schema.default_value,
        "is_set": bool(os.getenv(variable_name) or file_vars.get(variable_name)),
        "source": "environment"
        if os.getenv(variable_name)
        else ("file" if file_vars.get(variable_name) else "default"),
    }


@router.put("/environment/{variable_name}", tags=["Admin Environment"])
async def update_environment_variable(
    variable_name: str,
    update: EnvironmentVariableUpdate,
):
    """
    Update an environment variable.

    The value is saved to the .env file and will be loaded on next restart.
    Some changes may require a system restart to take effect.
    """
    if variable_name not in ENVIRONMENT_SCHEMA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown variable: {variable_name}",
        )

    schema = ENVIRONMENT_SCHEMA[variable_name]

    # Validate pattern if specified
    if schema.validation_pattern and not re.match(schema.validation_pattern, update.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid value format for {variable_name}",
        )

    # Load current vars
    file_vars = _load_env_file()

    # Update value
    old_value = file_vars.get(variable_name)
    file_vars[variable_name] = update.value

    # Save to file
    _save_env_file(file_vars)

    # Also set in current process environment
    os.environ[variable_name] = update.value

    logger.info(
        f"Environment variable updated: {variable_name}",
        extra={
            "variable": variable_name,
            "category": schema.category.value,
            "had_previous_value": old_value is not None,
        },
    )

    # Determine if restart is needed
    restart_required = variable_name in [
        "DATABASE_DRIVER",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USER",
        "DATABASE_PASSWORD",
        "API_PORT",
        "API_WORKERS",
    ]

    return {
        "success": True,
        "variable": variable_name,
        "message": f"Variable {variable_name} updated successfully",
        "restart_required": restart_required,
        "restart_message": "Please restart the application for this change to take effect"
        if restart_required
        else None,
    }


@router.post("/environment/bulk-update", tags=["Admin Environment"])
async def bulk_update_environment(updates: dict[str, str]):
    """
    Update multiple environment variables at once.

    Useful for initial setup or configuration changes.
    """
    file_vars = _load_env_file()
    updated = []
    errors = []

    for variable_name, value in updates.items():
        if variable_name not in ENVIRONMENT_SCHEMA:
            errors.append({"variable": variable_name, "error": "Unknown variable"})
            continue

        schema = ENVIRONMENT_SCHEMA[variable_name]

        # Validate pattern
        if schema.validation_pattern and not re.match(schema.validation_pattern, value):
            errors.append({"variable": variable_name, "error": "Invalid format"})
            continue

        file_vars[variable_name] = value
        os.environ[variable_name] = value
        updated.append(variable_name)

    # Save all changes
    if updated:
        _save_env_file(file_vars)

    return {
        "success": len(errors) == 0,
        "updated": updated,
        "errors": errors,
        "restart_required": any(
            v in updated
            for v in [
                "DATABASE_DRIVER",
                "DATABASE_HOST",
                "DATABASE_PORT",
                "DATABASE_NAME",
                "DATABASE_USER",
                "DATABASE_PASSWORD",
            ]
        ),
    }


@router.delete("/environment/{variable_name}", tags=["Admin Environment"])
async def delete_environment_variable(variable_name: str):
    """
    Remove an environment variable from the .env file.

    The variable will revert to its default value.
    """
    if variable_name not in ENVIRONMENT_SCHEMA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown variable: {variable_name}",
        )

    file_vars = _load_env_file()

    if variable_name not in file_vars:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Variable {variable_name} is not set in .env file",
        )

    del file_vars[variable_name]
    _save_env_file(file_vars)

    # Remove from process environment
    if variable_name in os.environ:
        del os.environ[variable_name]

    logger.info("Environment variable removed: %s", variable_name)

    return {
        "success": True,
        "variable": variable_name,
        "message": f"Variable {variable_name} removed from .env file",
    }


@router.get("/environment/file/raw", tags=["Admin Environment"])
async def get_raw_env_file():
    """
    Get the raw contents of the .env file.

    WARNING: This may expose sensitive values.
    """
    if not ENV_FILE_PATH.exists():
        return {
            "exists": False,
            "content": "",
            "path": str(ENV_FILE_PATH),
        }

    async with aiofiles.open(ENV_FILE_PATH) as f:
        content = f.read()

    return {
        "exists": True,
        "content": content,
        "path": str(ENV_FILE_PATH),
        "size_bytes": ENV_FILE_PATH.stat().st_size,
        "last_modified": datetime.fromtimestamp(ENV_FILE_PATH.stat().st_mtime).isoformat(),
    }


@router.post("/environment/file/raw", tags=["Admin Environment"])
async def save_raw_env_file(content: dict[str, str]):
    """
    Save raw content to the .env file.

    WARNING: This overwrites the entire file.
    """
    # Ensure config directory exists
    ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create backup
    if ENV_FILE_PATH.exists():
        backup_path = ENV_FILE_PATH.with_suffix(
            f".env.backup.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        ENV_FILE_PATH.rename(backup_path)

    # Write new content
    async with aiofiles.open(ENV_FILE_PATH, "w") as f:
        f.write(content.get("content", ""))

    logger.info("Raw .env file saved")

    return {
        "success": True,
        "path": str(ENV_FILE_PATH),
        "message": ".env file saved successfully",
    }


@router.get("/environment/export", tags=["Admin Environment"])
async def export_environment():
    """
    Export current configuration as downloadable .env format.

    Sensitive values are NOT masked in export.
    """
    file_vars = _load_env_file()

    lines = [
        "# Resync Environment Configuration",
        f"# Exported: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for category in VariableCategory:
        category_vars = {
            k: v
            for k, v in file_vars.items()
            if k in ENVIRONMENT_SCHEMA and ENVIRONMENT_SCHEMA[k].category == category
        }

        if category_vars:
            lines.append(f"\n# {category.value.upper()}")
            for key, value in sorted(category_vars.items()):
                if " " in value:
                    value = f'"{value}"'
                lines.append(f"{key}={value}")

    return {
        "content": "\n".join(lines),
        "filename": f"resync_config_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.env",
    }


@router.post("/environment/validate", tags=["Admin Environment"])
async def validate_environment():
    """
    Validate current environment configuration.

    Checks for required variables and validates formats.
    """
    file_vars = _load_env_file()
    issues = []
    warnings = []

    for name, schema in ENVIRONMENT_SCHEMA.items():
        value = os.getenv(name) or file_vars.get(name)

        # Check required
        if schema.is_required and not value:
            issues.append(
                {
                    "variable": name,
                    "category": schema.category.value,
                    "issue": "Required variable not set",
                    "severity": "error",
                }
            )

        # Check pattern
        if value and schema.validation_pattern and not re.match(schema.validation_pattern, value):
            issues.append(
                {
                    "variable": name,
                    "category": schema.category.value,
                    "issue": "Invalid value format",
                    "severity": "error",
                }
            )

        # Check sensitive with default
        if schema.is_sensitive and value == schema.default_value:
            warnings.append(
                {
                    "variable": name,
                    "category": schema.category.value,
                    "issue": "Sensitive variable using default value",
                    "severity": "warning",
                }
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_variables": len(ENVIRONMENT_SCHEMA),
        "configured_variables": sum(
            1 for name in ENVIRONMENT_SCHEMA if os.getenv(name) or file_vars.get(name)
        ),
    }
