"""
Database Security Middleware for SQL Injection Prevention

This middleware provides comprehensive protection against SQL injection attacks:
- Request parameter validation
- Query string sanitization
- Database operation monitoring
- Automatic audit logging
"""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from resync.core.database_security import DatabaseAuditor, log_database_access

logger = logging.getLogger(__name__)


class DatabaseSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detecting and preventing SQL injection attacks.

    Monitors all HTTP requests for potential SQL injection patterns
    and blocks suspicious requests before they reach the application.
    """

    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = [
        # High-confidence SQLi patterns only (avoid false positives on natural language).
        r"(?i)\bunion\b\s+\bselect\b",
        r"(?i)(?:\bor\b|\band\b)\s+\d+\s*=\s*\d+\s*(?:--|/\*)?",
        r"(?i)\bsleep\s*\(",
        r"(?i)\bbenchmark\s*\(",
        r"(?i)\bwaitfor\b\s+\bdelay\b",
        r"(?i)\bxp_\w+\b",
        r"(?i)\bsp_\w+\b",
        r"(?i)\binformation_schema\b",
        r"(?i);\s*(?:drop|alter|create|truncate)\b",
    ]

    def __init__(self, app: Callable, enabled: bool = True):
        """
        Initialize database security middleware.

        Args:
            app: ASGI application
            enabled: Whether middleware is active
        """
        super().__init__(app)
        self.enabled = enabled
        self.blocked_requests = 0
        self.total_requests = 0

        # Pre-compile patterns once (avoid per-request regex compilation)
        import re as _re
        self._compiled_patterns = [_re.compile(p) for p in self.SQL_INJECTION_PATTERNS]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through security middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response or passes to next middleware

        Raises:
            HTTPException: If SQL injection is detected
        """
        if not self.enabled:
            return await call_next(request)

        self.total_requests += 1

        try:
            # Analyze request for SQL injection patterns
            await self._analyze_request_for_sql_injection(request)

            # Process request through next middleware
            response = await call_next(request)

            # Log successful database operations
            self._log_request_outcome(request, True)

            return response

        except HTTPException:
            raise
        except Exception as e:
            # Log failed request
            self._log_request_outcome(request, False, str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error"
            ) from e

    async def _analyze_request_for_sql_injection(self, request: Request) -> None:
        """
        Analyzes request for potential SQL injection attacks.

        Args:
            request: HTTP request to analyze

        Raises:
            HTTPException: If SQL injection is detected
        """
        # Get request data to analyze
        request_data = await self._extract_request_data(request)

        # Check each value against injection patterns
        for key, value in request_data.items():
            if self._contains_sql_injection(value):
                DatabaseAuditor.log_security_violation(
                    "sql_injection_detected",
                    f"{key}={value}",
                    getattr(request.state, "user_id", None),
                )

                self.blocked_requests += 1

                logger.warning(
                    "sql_injection_blocked",
                    key=key,
                    value_preview=str(value)[:100],
                    client_host=request.client.host if request.client else "unknown",
                )

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Potential SQL injection detected. Request blocked.",
                )

    async def _extract_request_data(self, request: Request) -> dict[str, Any]:
        """
        Extracts all relevant data from request for analysis.

        Args:
            request: HTTP request

        Returns:
            Dictionary of request data
        """
        data = {}

        # Add query parameters
        for key, value in request.query_params.items():
            data[f"query.{key}"] = value

        # Add path parameters
        for key, value in request.path_params.items():
            data[f"path.{key}"] = value

        # Add headers that might contain SQL
        suspicious_headers = ["user-agent", "referer", "x-forwarded-for"]
        for header in suspicious_headers:
            if header in request.headers:
                data[f"header.{header}"] = request.headers[header]

        # Try to get body data for POST/PUT requests
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")

                if "application/json" in content_type:
                    body = await request.json()
                    if isinstance(body, dict):
                        for key, value in body.items():
                            data[f"body.{key}"] = value
                    else:
                        data["body"] = body

                elif "application/x-www-form-urlencoded" in content_type:
                    form = await request.form()
                    for key, value in form.items():
                        data[f"form.{key}"] = value

        except Exception as e:
            logger.debug("failed_to_extract_request_body", error=str(e))

        return data

    def _contains_sql_injection(self, value: Any) -> bool:
        """
        Checks if value contains SQL injection patterns.

        Args:
            value: Value to check

        Returns:
            True if SQL injection is detected
        """
        if value is None:
            return False

        # Convert to string for pattern matching
        str_value = str(value)

        # Check against compiled injection patterns
        for pattern in self._compiled_patterns:
            if pattern.search(str_value):
                return True

        # Do NOT block on generic characters (quotes, #, etc.). SQLAlchemy parameterization
        # is the primary defense; middleware should only catch high-confidence patterns.
        return False

    def _log_request_outcome(self, request: Request, success: bool, error: str = None) -> None:
        """
        Logs the outcome of request processing.

        Args:
            request: HTTP request
            success: Whether request was processed successfully
            error: Error message if request failed
        """
        try:
            operation = f"{request.method} {request.url.path}"

            log_database_access(
                operation=operation,
                table="unknown",  # Table might not be known at middleware level
                success=success,
                user_id=getattr(request.state, "user_id", None),
                error=error,
            )
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("failed_to_log_request_outcome", error=str(e))

    def get_security_stats(self) -> dict[str, Any]:
        """
        Gets security statistics for monitoring.

        Returns:
            Dictionary of security statistics
        """
        block_rate = (
            (self.blocked_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate_percent": round(block_rate, 2),
            "middleware_enabled": self.enabled,
            "patterns_monitored": len(self.SQL_INJECTION_PATTERNS),
        }


class DatabaseConnectionSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for securing database connections and operations.

    Provides additional security for database-specific endpoints.
    """

    # Database operation endpoints that require extra scrutiny
    DATABASE_ENDPOINTS = ["/admin/audit", "/admin/logs", "/api/v1/database/", "/api/db/", "/sql/"]

    def __init__(self, app: Callable, enabled: bool = True):
        """
        Initialize database connection security middleware.

        Args:
            app: ASGI application
            enabled: Whether middleware is active
        """
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through database security middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response or passes to next middleware
        """
        if not self.enabled:
            return await call_next(request)

        # Check if this is a database endpoint
        if self._is_database_endpoint(request.url.path):
            # Add security headers
            response = await call_next(request)
            response.headers["X-Database-Security-Enabled"] = "true"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response

        return await call_next(request)

    def _is_database_endpoint(self, path: str) -> bool:
        """
        Checks if request path is a database endpoint.

        Args:
            path: Request path

        Returns:
            True if this is a database endpoint
        """
        return any(path.startswith(endpoint) for endpoint in self.DATABASE_ENDPOINTS)


# Factory functions for easy middleware setup
def create_database_security_middleware(
    app: Callable, enabled: bool = True
) -> DatabaseSecurityMiddleware:
    """
    Creates database security middleware instance.

    Args:
        app: ASGI application
        enabled: Whether middleware should be enabled

    Returns:
        DatabaseSecurityMiddleware instance
    """
    return DatabaseSecurityMiddleware(app, enabled=enabled)


def create_database_connection_security_middleware(
    app: Callable, enabled: bool = True
) -> DatabaseConnectionSecurityMiddleware:
    """
    Creates database connection security middleware instance.

    Args:
        app: ASGI application
        enabled: Whether middleware should be enabled

    Returns:
        DatabaseConnectionSecurityMiddleware instance
    """
    return DatabaseConnectionSecurityMiddleware(app, enabled=enabled)
