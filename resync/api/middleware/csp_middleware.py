# pylint: disable-all
# mypy: ignore-errors
"""Content Security Policy (CSP) middleware for FastAPI."""

import base64
import logging
import secrets

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class CSPMiddleware(BaseHTTPMiddleware):
    """Injects CSP headers/nonces and delegates report handling to route layer."""

    def __init__(self, app, report_only: bool = False, enforce_https: bool = False):
        super().__init__(app)
        self.report_only = report_only
        self.enforce_https = enforce_https

        from resync.settings import settings

        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Do not intercept /csp-violation-report body handling here.
        # Route-level handler in app_factory enforces stream limits and rate limiting.
        if request.url.path == "/csp-violation-report" and request.method == "POST":
            return await call_next(request)

        nonce = self._generate_nonce()
        request.state.csp_nonce = nonce
        csp_policy = self._generate_csp_policy(nonce)

        response = await call_next(request)
        self._add_csp_headers(response, csp_policy)

        return response

    def _generate_nonce(self) -> str:
        nonce_bytes = secrets.token_bytes(16)
        return base64.b64encode(nonce_bytes).decode("utf-8")

    def _generate_csp_policy(self, nonce: str) -> str:
        directives = {
            "default-src": ["'self'"],
            "script-src": ["'self'", f"'nonce-{nonce}'"],
            "style-src": ["'self'", f"'nonce-{nonce}'"],
            "img-src": ["'self'", "data:", "blob:", "https:"],
            "font-src": ["'self'", "https:", "data:"],
            "connect-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "object-src": ["'none'"],
            "child-src": ["'self'"],
        }

        connect_src_urls = getattr(self._settings, "CSP_CONNECT_SRC_URLS", [])
        if connect_src_urls:
            directives["connect-src"].extend(connect_src_urls)

        report_uri = getattr(self._settings, "CSP_REPORT_URI", None)
        if report_uri:
            directives["report-uri"] = [report_uri]
        else:
            report_to = getattr(self._settings, "CSP_REPORT_TO", None)
            if report_to:
                directives["report-to"] = [report_to]

        policy_parts = []
        for directive, sources in directives.items():
            policy_parts.append(f"{directive} {' '.join(sources)}")

        return "; ".join(policy_parts)

    def _add_csp_headers(self, response: Response, csp_policy: str) -> None:
        header_name = (
            "Content-Security-Policy-Report-Only"
            if self.report_only
            else "Content-Security-Policy"
        )
        response.headers[header_name] = csp_policy


def create_csp_middleware(app) -> CSPMiddleware:
    """Factory function to create CSP middleware with settings-backed config."""
    from resync.settings import settings

    report_only = getattr(settings, "CSP_REPORT_ONLY", False)
    csp_enabled = getattr(settings, "CSP_ENABLED", True)

    if not csp_enabled:
        logger.info("CSP middleware disabled via settings")
        return BaseHTTPMiddleware(app)

    logger.info("CSP middleware initialized (report_only=%s)", report_only)
    return CSPMiddleware(app, report_only=report_only)
