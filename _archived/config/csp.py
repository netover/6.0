"""
CSP configuration module for Resync application.
"""

import logging

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from resync.settings import settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

        return response


def configure_csp(app: FastAPI, settings_module: object | None = None) -> None:
    """
    Configure CSP and other security headers for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings_module: Settings object (defaults to global settings)
    """
    settings_obj = settings_module or settings

    csp_enabled = getattr(settings_obj, "CSP_ENABLED", True)
    if not csp_enabled:
        logger.info("CSP and security middleware disabled")
        return

    # Determine report-only mode based on environment (report-only in non-production)
    environment = getattr(settings_obj, "ENVIRONMENT", "development")
    report_only = environment.lower() != "production"

    # Import and add CSP middleware
    from resync.api.middleware.csp_middleware import CSPMiddleware

    app.add_middleware(CSPMiddleware, report_only=report_only)

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info(f"CSP middleware added (report_only={report_only}, environment={environment})")
    logger.info("Security headers middleware added")
