"""
Security Headers Middleware - OWASP Best Practices 2024/2025

Implementa headers de segurança obrigatórios para produção:
- Strict-Transport-Security (HSTS)
- X-Frame-Options (Clickjacking protection)
- X-Content-Type-Options (MIME sniffing protection)
- X-XSS-Protection (Legacy XSS protection)
- Referrer-Policy (Privacy)
- Permissions-Policy (Feature restrictions)

PERFORMANCE: Pure ASGI implementation (1.5-2x faster than BaseHTTPMiddleware)

Referências:
- OWASP Secure Headers Project
- Mozilla Observatory
- Security Headers.com
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

class SecurityHeadersMiddleware:
    """
    Pure ASGI middleware que adiciona headers de segurança a todas as respostas.

    Headers implementados conforme OWASP Security Headers Project 2024.

    PERFORMANCE: 1.5-2x faster than BaseHTTPMiddleware due to:
    - No coroutine overhead
    - Direct ASGI interface
    - Minimal memory allocations
    """

    def __init__(
        self,
        app: ASGIApp,
        enforce_https: bool = True,
        hsts_max_age: int = 63072000,  # 2 anos
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = False,  # Só ative se registrar no preload list
    ):
        """
        Inicializa middleware de security headers.

        Args:
            app: ASGI application
            enforce_https: Se True, adiciona HSTS header
            hsts_max_age: Tempo em segundos para HSTS (padrão: 2 anos)
            hsts_include_subdomains: Incluir subdomínios no HSTS
            hsts_preload: Adicionar preload directive (requer registro)
        """
        self.app = app
        self.enforce_https = enforce_https
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload

        # Pre-build security headers (avoid building on every request)
        self._security_headers = self._build_security_headers()

    def _build_security_headers(self) -> list[tuple[bytes, bytes]]:
        """
        Build security headers once during initialization.

        Returns:
            List of (header_name, header_value) tuples as bytes
        """
        headers = [
            # Prevent clickjacking
            (b"x-frame-options", b"DENY"),
            # Prevent MIME sniffing
            (b"x-content-type-options", b"nosniff"),
            # XSS protection (legacy, still useful)
            (b"x-xss-protection", b"1; mode=block"),
            # Referrer policy for privacy
            (b"referrer-policy", b"strict-origin-when-cross-origin"),
            # Permissions policy (feature restrictions)
            (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
            # Content-Security-Policy (CSP)
            #
            # CSP is handled by ``CSPMiddleware`` (nonce-based; report-only in
            # non-production). Keeping CSP in a separate middleware avoids
            # duplicate CSP headers and prevents breaking Swagger UI in dev.
        ]

        # Add HSTS if HTTPS is enforced
        if self.enforce_https:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.hsts_preload:
                hsts_value += "; preload"
            headers.append((b"strict-transport-security", hsts_value.encode()))

        return headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        ASGI interface implementation.

        Args:
            scope: ASGI connection scope
            receive: ASGI receive channel
            send: ASGI send channel
        """
        if scope["type"] != "http":
            # Pass through non-HTTP requests (WebSocket, lifespan, etc)
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            """Intercept response and add security headers."""
            if message["type"] == "http.response.start":
                # Get headers and add security headers
                headers = MutableHeaders(scope=message)

                # Add all pre-built security headers
                for header_name, header_value in self._security_headers:
                    headers.append(header_name.decode(), header_value.decode())

            await send(message)

        await self.app(scope, receive, send_with_security_headers)
