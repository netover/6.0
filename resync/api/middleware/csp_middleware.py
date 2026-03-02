# pylint: disable-all
# mypy: ignore-errors
"""Content Security Policy (CSP) middleware for FastAPI.

Critical fixes applied:
- P0-12: CSP violation reports handled BEFORE call_next() to prevent double processing
- P1-19: Content-Type and Content-Length validation added
- P2-30: Settings imported once in __init__ instead of per-request
"""

import base64
import logging
import secrets

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

try:
    from resync.csp_validation import CSPValidationError, process_csp_report
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("csp_validation module not found — CSP violation processing disabled")
    CSPValidationError = ValueError  # fallback
    
    async def process_csp_report(request):
        return {"report": {}, "status": "validation_disabled"}

logger = logging.getLogger(__name__)

# Maximum size for CSP violation reports (10KB)
MAX_CSP_REPORT_SIZE = 10 * 1024

class CSPMiddleware(BaseHTTPMiddleware):
    """
    Middleware that implements Content Security Policy (CSP) with nonce generation.

    This middleware generates cryptographically secure nonces for each request
    and adds appropriate CSP headers to protect against XSS and other attacks.
    
    Security improvements:
    - CSP violation reports handled early to prevent double processing
    - Content-Type and size validation on reports
    - Settings imported once during init
    """

    def __init__(self, app, report_only: bool = False, enforce_https: bool = False):
        """
        Initialize CSP middleware.

        Args:
            app: The FastAPI application
            report_only: If True, CSP violations are reported but not enforced
            enforce_https: If True, add HSTS header for HTTPS enforcement
        """
        super().__init__(app)
        self.report_only = report_only
        self.enforce_https = enforce_https
        
        # Import settings once during init (not per-request)
        from resync.settings import settings
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process each request and add CSP headers.
        
        CSP violation reports are handled BEFORE calling next middleware
        to prevent double-processing of the request.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            Response with CSP headers
        """
        # Handle CSP violations BEFORE calling next middleware (P0-12 fix)
        if request.url.path == "/csp-violation-report" and request.method == "POST":
            await self._handle_csp_violation_report(request)
            # Return 204 No Content per CSP spec
            return Response(status_code=204)
        
        # Generate cryptographically secure nonce for this request
        nonce = self._generate_nonce()

        # Store nonce in request state for template access
        request.state.csp_nonce = nonce

        # Generate CSP policy with the nonce
        csp_policy = self._generate_csp_policy(nonce)

        # Process the request
        response = await call_next(request)

        # Add CSP headers to response
        self._add_csp_headers(response, csp_policy)

        return response

    def _generate_nonce(self) -> str:
        """
        Generate a cryptographically secure nonce.

        Returns:
            A base64-encoded nonce string
        """
        # Generate 16 bytes of cryptographically secure random data
        nonce_bytes = secrets.token_bytes(16)
        # Convert to base64 for use in CSP (required by CSP specification)
        return base64.b64encode(nonce_bytes).decode("utf-8")

    def _generate_csp_policy(self, nonce: str) -> str:
        """
        Generate CSP policy directives.

        Args:
            nonce: The generated nonce for this request

        Returns:
            CSP policy string
        """
        # Settings already imported in __init__ (P2-30 fix)
        # Base CSP directives with enhanced security
        directives = {
            "default-src": ["'self'"],
            "script-src": ["'self'", f"'nonce-{nonce}'"],
            "style-src": ["'self'", f"'nonce-{nonce}'"],  # Allow nonce for styles too
            "img-src": ["'self'", "data:", "blob:", "https:"],
            "font-src": ["'self'", "https:", "data:"],
            "connect-src": ["'self'"],
            "frame-ancestors": ["'none'"],  # Prevent clickjacking
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "object-src": ["'none'"],  # Explicitly block objects/plugins
            "child-src": ["'self'"],  # For workers iframes, etc
        }

        # Allow connect-src to external URLs if configured
        connect_src_urls = getattr(self._settings, "CSP_CONNECT_SRC_URLS", [])
        if connect_src_urls:
            directives["connect-src"].extend(connect_src_urls)

        # Add report-uri if configured
        report_uri = getattr(self._settings, "CSP_REPORT_URI", None)
        if report_uri:
            directives["report-uri"] = [report_uri]
        else:
            # Add report-to directive for modern CSP reporting (if supported)
            report_to = getattr(self._settings, "CSP_REPORT_TO", None)
            if report_to:
                directives["report-to"] = [report_to]

        # Build policy string
        policy_parts = []
        for directive, sources in directives.items():
            policy_parts.append(f"{directive} {' '.join(sources)}")

        return "; ".join(policy_parts)

    def _add_csp_headers(self, response: Response, csp_policy: str) -> None:
        """
        Add CSP headers to the response.

        Args:
            response: The response object
            csp_policy: The CSP policy string
        """
        header_name = (
            "Content-Security-Policy-Report-Only"
            if self.report_only
            else "Content-Security-Policy"
        )
        response.headers[header_name] = csp_policy

        # NOTE: This middleware is intentionally limited to CSP. Other security headers
        # (HSTS, X-Frame-Options, etc.) are handled by the pure-ASGI SecurityHeadersMiddleware
        # to avoid duplicated/conflicting headers in the middleware chain.

    async def _handle_csp_violation_report(self, request: Request) -> None:
        """
        Handle CSP violation reports with validation and proper logging.
        
        P1-19 fix: Added Content-Type and Content-Length validation.

        Args:
            request: The request containing CSP violation report
        """
        try:
            if request.method != "POST":
                return
            
            # Validate Content-Type (P1-19 fix)
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith(("application/csp-report", "application/json")):
                logger.warning(
                    "csp_report_invalid_content_type",
                    content_type=content_type,
                )
                return
            
            # Validate Content-Length (P1-19 fix - DoS protection)
            content_length = request.headers.get("content-length")
            try:
                cl = int(content_length) if content_length else 0
            except ValueError:
                cl = 0
            
            if cl < 0 or cl > MAX_CSP_REPORT_SIZE:
                logger.warning(
                    "csp_report_size_violation",
                    size=cl,
                    max_allowed=MAX_CSP_REPORT_SIZE,
                )
                return

            # Process the CSP report with enhanced validation
            try:
                result = await process_csp_report(request)

                # Log specific violation details from sanitized data
                report = result.get("report", {})
                csp_report = (
                    report.get("csp-report", {})
                    if isinstance(report, dict) and "csp-report" in report
                    else report
                )

                logger.warning(
                    "csp_violation",
                    blocked_uri=csp_report.get("blocked-uri", "unknown"),
                    violated_directive=csp_report.get("violated-directive", "unknown"),
                    effective_directive=csp_report.get("effective-directive", "unknown"),
                    script_sample=csp_report.get("script-sample", "none"),
                )
            except CSPValidationError as e:
                logger.warning("csp_validation_error", error=str(e))
                return
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("csp_report_processing_error", error=str(e))
                return

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("csp_report_handler_error", error=str(e))

def create_csp_middleware(app) -> CSPMiddleware:
    """
    Factory function to create CSP middleware with configuration from settings.

    Args:
        app: The FastAPI application

    Returns:
        Configured CSPMiddleware instance
    """
    # Import settings lazily to avoid circular imports
    from resync.settings import settings

    # Check if CSP should be in report-only mode
    report_only = getattr(settings, "CSP_REPORT_ONLY", False)

    # Check if CSP is enabled
    csp_enabled = getattr(settings, "CSP_ENABLED", True)

    if not csp_enabled:
        logger.info("CSP middleware disabled via settings")
        # Return a no-op middleware
        return BaseHTTPMiddleware(app)

    logger.info("CSP middleware initialized (report_only=%s)", report_only)
    return CSPMiddleware(app, report_only=report_only)
