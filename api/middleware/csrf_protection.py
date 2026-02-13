import hmac
import secrets

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection with:
    - Double-submit cookie pattern
    - HMAC validation
    - SameSite cookies
    """

    def __init__(self, app, secret_key: str):
        super().__init__(app)
        self.secret_key = secret_key.encode()
        self.cookie_name = "csrf_token"
        self.header_name = "X-CSRF-Token"

    async def dispatch(self, request: Request, call_next):
        """
        Dispatch method to handle CSRF protection.
        """
        # Skip for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return await call_next(request)

        # Skip for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)

        # Validate CSRF token
        cookie_token = request.cookies.get(self.cookie_name)
        header_token = request.headers.get(self.header_name)

        if not cookie_token or not header_token:
            raise HTTPException(status_code=403, detail="CSRF token missing")

        # Validate tokens match and are valid
        if not self._validate_csrf_tokens(cookie_token, header_token):
            logger.warning(
                "CSRF token validation failed",
                extra={
                    "ip": request.client.host,
                    "path": request.url.path,
                    "user_agent": request.headers.get("user-agent"),
                },
            )
            raise HTTPException(status_code=403, detail="CSRF token validation failed")

        response = await call_next(request)

        # Rotate token for high-security operations
        if request.url.path in self._high_security_endpoints():
            new_token = self._generate_csrf_token()
            response.set_cookie(
                key=self.cookie_name,
                value=new_token,
                httponly=True,
                secure=True,
                samesite="strict",
                max_age=3600,
            )

        return response

    def _generate_csrf_token(self) -> str:
        """Generate cryptographically secure CSRF token."""
        random_bytes = secrets.token_bytes(32)
        signature = hmac.new(self.secret_key, random_bytes, digestmod="sha256").digest()

        return (random_bytes + signature).hex()

    def _validate_csrf_tokens(self, cookie_token: str, header_token: str) -> bool:
        """Validate CSRF tokens using constant-time comparison.

        SECURITY: All comparisons use ``secrets.compare_digest`` to prevent
        timing side-channel attacks.  Never use plain ``==`` or ``!=`` for
        token comparison.
        """
        try:
            cookie_bytes = bytes.fromhex(cookie_token)
            header_bytes = bytes.fromhex(header_token)
        except (ValueError, TypeError):
            return False

        # Validate lengths first (prevents length-based timing differences and
        # avoids unnecessary work on malformed tokens).
        if len(cookie_bytes) != 64 or len(header_bytes) != 64:  # 32 bytes random + 32 bytes HMAC-SHA256
            return False

        # Constant-time comparison of the two submitted tokens.
        if not secrets.compare_digest(cookie_bytes, header_bytes):
            return False

        random_part = cookie_bytes[:32]
        signature_part = cookie_bytes[32:]

        expected_signature = hmac.new(self.secret_key, random_part, digestmod="sha256").digest()

        return secrets.compare_digest(signature_part, expected_signature)

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no CSRF needed)."""
        public_endpoints = [
            "/token",
            "/login",
            "/health",
            "/metrics",
            "/api/health",
            "/api/metrics",
        ]
        return any(path.startswith(ep) for ep in public_endpoints)

    def _high_security_endpoints(self) -> list[str]:
        """Endpoints requiring token rotation."""
        return ["/api/workflow/delete", "/api/admin/", "/api/settings/", "/api/secure/"]
