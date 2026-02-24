# pylint
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog


def setup_logging() -> None:
    """
    Configures structured logging for the application with JSON format.
    """
    import os
    from logging.handlers import RotatingFileHandler

    # Get log level from environment or settings
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create logs directory with YYYYMMDD format if it doesn't exist
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_dir = Path("logs") / today
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove default handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler with rotation (10MB max size, 5 backups)
    file_handler = RotatingFileHandler(
        log_dir / "resync.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,  # 10MB
    )
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)


def log_with_correlation(
    level: int,
    message: str,
    correlation_id: str | None = None,
    component: str = "main",
    operation: str | None = None,
    error: Exception | None = None,
    **extra_fields: Any,
) -> None:
    """
    Structured logging with correlation ID and contextual information.

    Args:
        level: Logging level (logging.INFO, logging.ERROR, etc.)
        message: Log message
        correlation_id: Correlation ID for tracing
        component: Component name (main, tws_client, etc.)
        operation: Operation being performed
        error: Exception object for error logging
        extra_fields: Additional structured fields
    """
    # Use structlog for structured logging
    logger = structlog.get_logger()

    # Prepare structured log entry
    log_entry = {
        "message": message,
        "component": component,
        "operation": operation,
        "correlation_id": correlation_id,
        **extra_fields,
    }

    if error:
        log_entry["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    # Log based on level
    if level == logging.DEBUG:
        logger.debug("LOG_EVENT", **log_entry)
    elif level == logging.INFO:
        logger.info("LOG_EVENT", **log_entry)
    elif level == logging.WARNING:
        logger.warning("LOG_EVENT", **log_entry)
    elif level == logging.ERROR:
        logger.error("LOG_EVENT", **log_entry)
    elif level == logging.CRITICAL:
        logger.critical("LOG_EVENT", **log_entry)


def log_audit_event(
    action: str,
    user_id: str,
    details: dict[str, Any],
    correlation_id: str | None = None,
    severity: str = "INFO",
) -> None:
    """
    Structured logging function for audit events.

    Args:
        action: The audited action
        user_id: The ID of the user performing the action
        details: Additional details about the action
        correlation_id: Optional correlation ID for distributed tracing
        severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
    """
    # Redact sensitive data from details
    sanitized_details = _sanitize_audit_details(details)

    logger = structlog.get_logger()
    logger.info(
        "AUDIT_EVENT",
        action=action,
        user_id=user_id,
        details=sanitized_details,
        event_type="audit",
        correlation_id=correlation_id,
        severity=severity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Also persist the audit event to the database for long-term storage
    try:
        from resync.core.audit_log import get_audit_log_manager

        audit_manager = get_audit_log_manager()
        audit_manager.log_audit_event(
            action=action,
            user_id=user_id,
            details=sanitized_details,
            correlation_id=correlation_id,
            source_component="main",
            severity=severity,
        )
    except Exception as e:
        logger.error("Failed to persist audit event to database: %s", e, exc_info=True)


def _sanitize_audit_details(details: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize audit details by redacting sensitive information.

    Args:
        details: The details dictionary to sanitize

    Returns:
        A sanitized dictionary with sensitive data redacted
    """
    if not isinstance(details, dict):
        return details

    sanitized: dict[str, Any] = {}
    # Fields that should be redacted
    sensitive_fields = {
        "password",
        "secret",
        "key",
        "token",
        "api_key",
        "auth",
        "credentials",
        "credit_card",
        "ssn",
        "social_security",
        "cvv",
        "card_number",
        "pin",
        "cvv2",
    }

    for key, value in details.items():
        if isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = _sanitize_audit_details(value)
        elif isinstance(key, str) and key.lower() in sensitive_fields:
            sanitized[key] = "REDACTED"
        else:
            sanitized[key] = value

    return sanitized
