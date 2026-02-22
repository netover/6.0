"""
Environment Validation Script for Resync v6.0

This script validates that all required environment variables are present
and properly configured BEFORE the Docker container starts.

Usage:
    python -m resync.scripts.check_env

Exit codes:
    0 - All validations passed
    1 - Validation failed (missing or invalid variables)

This script should be run as part of the Docker entrypoint or
CI/CD pipeline to fail fast if configuration is missing.
"""

import os
import sys
from pathlib import Path


def get_required_vars() -> list[tuple[str, str]]:
    """
    Return list of (variable_name, description) tuples.
    These are CRITICAL variables that must be present for the app to start.

    Note: Settings.py uses APP_ prefix, but we check both with and without prefix
    for flexibility in different environments.
    """
    return [
        ("SECRET_KEY", "JWT signing key (use SecretStr in production)"),
        ("APP_SECRET_KEY", "JWT signing key (alternative - with APP_ prefix)"),
        ("REDIS_URL", "Redis connection string (redis://host:port/db)"),
        ("APP_REDIS_URL", "Redis connection string (alternative - with APP_ prefix)"),
        ("DATABASE_URL", "PostgreSQL connection string"),
        (
            "APP_DATABASE_URL",
            "PostgreSQL connection string (alternative - with APP_ prefix)",
        ),
    ]


def get_optional_warn_vars() -> list[tuple[str, str]]:
    """
    Return list of (variable_name, description) tuples.
    These are OPTIONAL variables that will warn if missing but won't fail startup.
    """
    return [
        ("OPENAI_API_KEY", "OpenAI API key for LLM features"),
        ("TWS_API_ENDPOINT", "TWS API endpoint URL"),
        ("LANGFUSE_SECRET_KEY", "Langfuse observability secret"),
        ("GRAPHRAG_ENABLED", "Enable GraphRAG (true/false)"),
    ]


def validate_env_file_exists() -> bool:
    """Check if .env file exists in expected locations."""
    possible_locations = [
        Path(".env"),
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for location in possible_locations:
        if location.exists():
            return True
    return False


def check_required_vars() -> tuple[bool, list[str]]:
    """
    Check all required environment variables.
    Returns (success, list_of_errors).

    For each variable, we check both with and without APP_ prefix.
    """
    errors = []

    # Group variables by their base name (with or without APP_ prefix)
    var_groups = {
        "SECRET_KEY": ["SECRET_KEY", "APP_SECRET_KEY"],
        "REDIS_URL": ["REDIS_URL", "APP_REDIS_URL"],
        "DATABASE_URL": ["DATABASE_URL", "APP_DATABASE_URL"],
    }

    descriptions = {
        "SECRET_KEY": "JWT signing key (use SecretStr in production)",
        "REDIS_URL": "Redis connection string (redis://host:port/db)",
        "DATABASE_URL": "PostgreSQL connection string",
    }

    for base_name, alt_names in var_groups.items():
        # Check if ANY of the variants is set
        value = None
        found_var = None

        for var_name in alt_names:
            val = os.getenv(var_name)
            if val is not None and val.strip():
                value = val
                found_var = var_name
                break

        description = descriptions[base_name]

        if value is None:
            # Check for placeholder values in any variant
            for var_name in alt_names:
                val = os.getenv(var_name)
                if val and val.upper() in ["", "CHANGE_ME", "YOUR_KEY_HERE", "TODO"]:
                    errors.append(
                        f"[ERROR] {var_name}: Contains placeholder value '{val}' - must be set for production"
                    )
                    break
            else:
                errors.append(
                    f"[ERROR] {base_name}: Required variable is missing - {description}"
                )
        elif not value.strip():
            errors.append(f"[ERROR] {found_var}: Variable is empty - {description}")
        else:
            # Check for placeholder values
            if value.upper() in ["CHANGE_ME", "YOUR_KEY_HERE", "TODO"]:
                errors.append(
                    f"[ERROR] {found_var}: Contains placeholder value '{value}' - must be set for production"
                )

    return len(errors) == 0, errors


def check_optional_vars() -> list[str]:
    """
    Check optional environment variables.
    Returns list of warnings.
    """
    warnings = []

    for var_name, description in get_optional_warn_vars():
        value = os.getenv(var_name)

        if value is None or (isinstance(value, str) and not value.strip()):
            warnings.append(
                f"[WARN] {var_name}: Optional variable missing - {description}"
            )

    return warnings


def main() -> int:
    """Main entry point. Returns 0 on success, 1 on failure."""
    print("Resync Environment Validation")
    print("=" * 50)

    # Check .env file
    has_env_file = validate_env_file_exists()
    if has_env_file:
        print("[OK] .env file found")
    else:
        print("[WARN] No .env file found (using environment variables directly)")

    print("\nChecking required variables...")
    required_ok, required_errors = check_required_vars()

    for error in required_errors:
        print(error)

    print("\nChecking optional variables...")
    optional_warnings = check_optional_vars()

    for warning in optional_warnings:
        print(warning)

    print("\n" + "=" * 50)

    if required_ok:
        print("[OK] All required variables validated successfully!")
        print("\nReady to start Resync application")
        return 0
    else:
        print("[ERROR] Environment validation FAILED")
        print("\nTo fix:")
        print("   1. Set required environment variables")
        print("   2. Or create .env file with required values")
        print("   3. For Docker: add --env-file or set variables in docker-compose.yml")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Unexpected error during validation: {e}")
        sys.exit(1)
