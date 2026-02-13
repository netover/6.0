"""
Enhanced security configuration for Resync application with production-ready settings.
"""

import logging

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from resync.api.validation.enhanced_security import SecurityHeadersMiddleware
from resync.settings import settings

logger = logging.getLogger(__name__)


class EnhancedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced security middleware with comprehensive protection mechanisms.
    """

    def __init__(
        self,
        app: FastAPI,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_referrer_policy: bool = True,
        enable_permissions_policy: bool = True,
        enable_feature_policy: bool = True,
        enable_xss_protection: bool = True,
        enable_content_type_options: bool = True,
        enable_frame_options: bool = True,
        strict_transport_security_max_age: int = 31536000,  # 1 year
        content_security_policy: str | None = None,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str | None = None,
        feature_policy: str | None = None,
    ):
        """
        Initialize enhanced security middleware.

        Args:
            app: FastAPI application instance
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            enable_referrer_policy: Enable Referrer Policy
            enable_permissions_policy: Enable Permissions Policy
            enable_feature_policy: Enable Feature Policy (deprecated)
            enable_xss_protection: Enable XSS Protection
            enable_content_type_options: Enable Content-Type Options
            enable_frame_options: Enable Frame Options
            strict_transport_security_max_age: HSTS max age in seconds
            content_security_policy: Custom CSP policy
            referrer_policy: Referrer policy header value
            permissions_policy: Permissions policy header value
            feature_policy: Feature policy header value
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts and settings.is_production
        self.enable_csp = enable_csp
        self.enable_referrer_policy = enable_referrer_policy
        self.enable_permissions_policy = enable_permissions_policy
        self.enable_feature_policy = enable_feature_policy
        self.enable_xss_protection = enable_xss_protection
        self.enable_content_type_options = enable_content_type_options
        self.enable_frame_options = enable_frame_options
        self.strict_transport_security_max_age = strict_transport_security_max_age
        self.content_security_policy = content_security_policy or self._default_csp()
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or self._default_permissions_policy()
        self.feature_policy = feature_policy or self._default_feature_policy()

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with enhanced security headers.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response with security headers
        """
        # Skip security headers for certain paths
        if self._should_skip_security_headers(request):
            return await call_next(request)

        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        return response

    def _should_skip_security_headers(self, request: Request) -> bool:
        """
        Determine if security headers should be skipped for this request.

        Args:
            request: Incoming request

        Returns:
            True if security headers should be skipped
        """
        # Skip for health check endpoints
        if request.url.path in ["/health", "/health/", "/ping", "/ping/"]:
            return True

        # Skip for static files in production (handled by web server)
        return bool(settings.is_production and request.url.path.startswith("/static/"))

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.

        Args:
            response: Response to add headers to
        """
        headers = response.headers

        # HTTP Strict Transport Security (HSTS)
        if self.enable_hsts:
            headers["Strict-Transport-Security"] = (
                f"max-age={self.strict_transport_security_max_age}; includeSubDomains; preload"
            )

        # Content Security Policy (CSP)
        if self.enable_csp:
            # Use Report-Only in non-production environments
            csp_header = (
                "Content-Security-Policy-Report-Only"
                if not settings.is_production
                else "Content-Security-Policy"
            )
            headers[csp_header] = self.content_security_policy

        # Referrer Policy
        if self.enable_referrer_policy:
            headers["Referrer-Policy"] = self.referrer_policy

        # Permissions Policy
        if self.enable_permissions_policy:
            headers["Permissions-Policy"] = self.permissions_policy

        # Feature Policy (deprecated but still supported)
        if self.enable_feature_policy:
            headers["Feature-Policy"] = self.feature_policy

        # XSS Protection
        if self.enable_xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"

        # Content-Type Options
        if self.enable_content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"

        # Frame Options
        if self.enable_frame_options:
            headers["X-Frame-Options"] = "DENY"

        # Additional security headers
        headers["X-Permitted-Cross-Domain-Policies"] = "none"
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"

    def _default_csp(self) -> str:
        """
        Generate default Content Security Policy.

        Returns:
            CSP policy string
        """
        # Base directives
        directives = {
            "default-src": ["'self'"],
            "script-src": [
                "'self'",
                "'unsafe-inline'",
            ],  # Allow inline for now, should be restricted
            "style-src": [
                "'self'",
                "'unsafe-inline'",
            ],  # Allow inline for now, should be restricted
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'", "https:", "data:"],
            "connect-src": ["'self'"],
            "media-src": ["'self'"],
            "object-src": ["'none'"],
            "child-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "form-action": ["'self'"],
            "base-uri": ["'self'"],
            "manifest-src": ["'self'"],
        }

        # Add report URI in non-production for monitoring
        if not settings.is_production:
            directives["report-uri"] = ["/csp-violation-report"]

        # Build policy string
        policy_parts = []
        for directive, sources in directives.items():
            policy_parts.append(f"{directive} {' '.join(sources)}")

        return "; ".join(policy_parts)

    def _default_permissions_policy(self) -> str:
        """
        Generate default Permissions Policy.

        Returns:
            Permissions policy string
        """
        return (
            "geolocation=(), midi=(), notifications=(), push=(), sync-xhr=(), "
            "microphone=(), camera=(), magnetometer=(), gyroscope=(), speaker=(), "
            "vibrate=(), fullscreen=(), payment=(), usb=(), xr-spatial-tracking=()"
        )

    def _default_feature_policy(self) -> str:
        """
        Generate default Feature Policy.

        Returns:
            Feature policy string
        """
        return (
            "geolocation 'none'; midi 'none'; notifications 'none'; push 'none'; "
            "sync-xhr 'none'; microphone 'none'; camera 'none'; magnetometer 'none'; "
            "gyroscope 'none'; speaker 'none'; vibrate 'none'; fullscreen 'none'; "
            "payment 'none'; usb 'none'"
        )


def configure_enhanced_security(
    app: FastAPI,
    security_middleware_config: dict | None = None,
) -> None:
    """
    Configure enhanced security for the FastAPI application.

    Args:
        app: FastAPI application instance
        security_middleware_config: Configuration for security middleware
    """
    if security_middleware_config is None:
        security_middleware_config = {}

    # Add enhanced security middleware
    app.add_middleware(EnhancedSecurityMiddleware, **security_middleware_config)

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    logger.info("Enhanced security middleware configured")
    if settings.is_production:
        logger.info("Production security settings enabled")
    else:
        logger.info("Development security settings enabled")


# Production security recommendations
PRODUCTION_SECURITY_RECOMMENDATIONS = {
    "hsts": True,
    "csp_report_only": False,
    "referrer_policy": "strict-origin-when-cross-origin",
    "permissions_policy": "geolocation=(), microphone=(), camera=()",
    "xss_protection": True,
    "content_type_options": True,
    "frame_options": True,
    "cross_domain_policies": "none",
    "coep": "require-corp",
    "coop": "same-origin",
    "corp": "same-origin",
}

# Development security settings (more permissive for debugging)
DEVELOPMENT_SECURITY_SETTINGS = {
    "hsts": False,
    "csp_report_only": True,
    "referrer_policy": "no-referrer-when-downgrade",
    "permissions_policy": "geolocation=*, microphone=*, camera=*",
    "xss_protection": True,
    "content_type_options": True,
    "frame_options": True,
    "cross_domain_policies": "master-only",
    "coep": "unsafe-none",
    "coop": "unsafe-none",
    "corp": "cross-origin",
}

# Security headers checklist
SECURITY_HEADERS_CHECKLIST = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "Referrer-Policy",
    "Permissions-Policy",
    "X-XSS-Protection",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-Permitted-Cross-Domain-Policies",
    "Cross-Origin-Embedder-Policy",
    "Cross-Origin-Opener-Policy",
    "Cross-Origin-Resource-Policy",
]


def validate_security_headers(response_headers: dict) -> list[str]:
    """
    Validate that all recommended security headers are present.

    Args:
        response_headers: Dictionary of response headers

    Returns:
        List of missing security headers
    """
    missing_headers = []
    headers_lower = {k.lower(): v for k, v in response_headers.items()}

    for header in SECURITY_HEADERS_CHECKLIST:
        if header.lower() not in headers_lower:
            missing_headers.append(header)

    return missing_headers


# Export configuration
__all__ = [
    "EnhancedSecurityMiddleware",
    "configure_enhanced_security",
    "PRODUCTION_SECURITY_RECOMMENDATIONS",
    "DEVELOPMENT_SECURITY_SETTINGS",
    "SECURITY_HEADERS_CHECKLIST",
    "validate_security_headers",
]
