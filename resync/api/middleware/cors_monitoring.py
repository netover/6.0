# pylint
# mypy
"""
CORS Monitoring Middleware

This module provides functionality to monitor and log CORS-related activities
for security and compliance purposes.
"""

import logging
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse
from fastapi import Request
from pydantic import BaseModel
from resync.core.logger import log_with_correlation

logger = logging.getLogger(__name__)

class CORSOperation(str, Enum):
    """Enum for different CORS operations."""

    PREFLIGHT = "preflight"
    REQUEST = "request"
    VIOLATION = "violation"

class CORSMonitor:
    """Monitor and log CORS-related activities."""

    def __init__(self):
        self.violations = deque(maxlen=1000)
        self.allowed_origins = set()
        self.blocked_origins = set()
        self.requests = deque(maxlen=2000)

    def monitor_request(
        self, request: Request, operation: CORSOperation = CORSOperation.REQUEST
    ) -> dict[str, Any]:
        """
        Monitor an incoming request for CORS-related information.

        Args:
            request: The incoming request object
            operation: The type of CORS operation

        Returns:
            Dictionary with CORS monitoring information
        """
        origin = request.headers.get("origin")
        method = request.method
        requested_headers = request.headers.get("access-control-request-headers")
        access_control_method = request.headers.get("access-control-request-method")
        if origin:
            if operation == CORSOperation.VIOLATION:
                self.blocked_origins.add(origin)
            else:
                self.allowed_origins.add(origin)
        cors_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": origin,
            "method": method,
            "requested_headers": requested_headers,
            "access_control_method": access_control_method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent"),
            "operation": operation.value,
        }
        self.requests.append(cors_info)
        return cors_info

    def log_violation(
        self,
        origin: str,
        path: str,
        details: str = "",
        method: str = "",
        requested_headers: str = "",
    ) -> None:
        """
        Log a CORS violation for security monitoring.

        Args:
            origin: Origin that caused the violation
            path: Path where the violation occurred
            details: Additional details about the violation
            method: The HTTP method of the request
            requested_headers: Headers that were requested
        """
        violation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": origin,
            "path": path,
            "method": method,
            "requested_headers": requested_headers,
            "details": details,
        }
        self.violations.append(violation)
        self.monitor_request(
            type(
                "MockRequest",
                (),
                {
                    "headers": {"origin": origin, "user-agent": "CORS-Monitor"},
                    "url": type("URL", (), {"path": path})(),
                    "method": method,
                },
            )(),
            CORSOperation.VIOLATION,
        )
        log_with_correlation(
            logging.WARNING,
            f"CORS violation detected: {origin} accessing {path}",
            origin=origin,
            path=path,
            details=details,
            method=method,
        )
        logger.warning(
            "CORS violation: %s -> %s (%s). Details: %s", origin, path, method, details
        )

    def get_violations(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent CORS violations.

        Args:
            limit: Maximum number of violations to return

        Returns:
            List of recent violations
        """
        return self.violations[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get CORS monitoring statistics.

        Returns:
            Dictionary with CORS monitoring statistics
        """
        total_requests = len(self.requests)
        total_violations = len(self.violations)
        unique_origins = len(self.allowed_origins)
        blocked_origins_count = len(self.blocked_origins)
        return {
            "total_requests": total_requests,
            "total_violations": total_violations,
            "unique_origins": unique_origins,
            "blocked_origins_count": blocked_origins_count,
            "violation_rate": total_violations / total_requests
            if total_requests > 0
            else 0,
            "last_violation": self.violations[-1] if self.violations else None,
        }

    def reset_violations(self) -> None:
        """Reset the violations log."""
        self.violations.clear()

    def reset_all(self) -> None:
        """Reset all logs."""
        self.violations.clear()
        self.requests.clear()
        self.allowed_origins.clear()
        self.blocked_origins.clear()

class CORSLogEntry(BaseModel):
    """Model for CORS log entries."""

    timestamp: str
    origin: str
    method: str
    requested_headers: str | None = None
    access_control_method: str | None = None
    path: str
    user_agent: str | None = None
    operation: str
    is_violation: bool = False
    details: str | None = ""

def monitor_cors() -> CORSMonitor:
    """
    Create and return a CORS monitor instance.

    Returns:
        CORSMonitor instance for monitoring CORS activities
    """
    return CORSMonitor()

class CORSMonitoringMiddleware:
    """
    Middleware to monitor and log CORS-related activities.
    """

    def __init__(self, app, allowed_origins: list[str] = None):
        self.app = app
        self.cors_monitor = monitor_cors()
        self.allowed_origins = set(allowed_origins or [])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and request.method == "OPTIONS":
                origin = request.headers.get("origin")
                access_method = request.headers.get("access-control-request-method")
                if origin:
                    self.cors_monitor.monitor_request(request, CORSOperation.PREFLIGHT)
                    is_allowed = any(
                        (
                            allowed_origin == "*"
                            or origin == allowed_origin
                            or (
                                allowed_origin.startswith(".")
                                and (
                                    origin == allowed_origin[1:]
                                    or origin.endswith("." + allowed_origin[1:])
                                )
                            )
                            for allowed_origin in self.allowed_origins
                        )
                    )
                    if not is_allowed:
                        self.cors_monitor.log_violation(
                            origin,
                            request.url.path,
                            f"Origin not in allowed list during preflight. Allowed: {list(self.allowed_origins)}",
                            access_method or request.method,
                        )
            return await send(message)

        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlparse(origin)
                if not parsed.scheme or not parsed.netloc:
                    self.cors_monitor.log_violation(
                        origin,
                        request.url.path,
                        "Invalid origin format",
                        request.method,
                        request.headers.get("access-control-request-headers", ""),
                    )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("exception_caught", exc_info=True, extra={"error": str(e)})
                self.cors_monitor.log_violation(
                    origin,
                    request.url.path,
                    f"Malformed origin URL: {str(e)}",
                    request.method,
                    request.headers.get("access-control-request-headers", ""),
                )
        return await self.app(scope, receive, send_wrapper)
