import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from resync.api.middleware.cors_config import CORSPolicy, Environment, cors_config
from resync.settings import settings

logger = logging.getLogger(__name__)


def _get_secure_cors_origins(allow_origins: list[str] | None) -> list[str]:
    """
    Get CORS origins with security validation.

    SECURITY (v5.4.1):
    - Production: Wildcard "*" is NOT allowed
    - Development: Wildcard allowed with warning

    Args:
        allow_origins: Provided origins list

    Returns:
        Safe origins list

    Raises:
        ValueError: If wildcard used in production without explicit origins
    """
    env = os.getenv("ENVIRONMENT", "development").lower()

    # If origins provided explicitly
    if allow_origins:
        if "*" in allow_origins and env == "production":
            logger.error(
                "cors_wildcard_in_production - Wildcard CORS origins not allowed in production"
            )
            raise ValueError(
                "CORS wildcard '*' is not allowed in production. "
                "Set explicit CORS_ALLOWED_ORIGINS environment variable."
            )
        if "*" in allow_origins:
            logger.warning(
                "cors_wildcard_enabled env=%s - Using wildcard CORS - not recommended for production",
                env,
            )
        return allow_origins

    # No origins provided - use secure defaults
    if env == "production":
        # In production, fail closed - no CORS by default
        logger.warning(
            "cors_no_origins_configured - No CORS origins configured - defaulting to same-origin only"
        )
        return []  # Empty = same-origin only

    # Development - allow localhost variants
    dev_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    logger.info("cors_using_dev_defaults origins=%s", dev_origins)
    return dev_origins


class LoggingCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware with enhanced logging and security monitoring.

    This middleware extends the standard FastAPI CORS middleware to provide:
    - Detailed logging of CORS violations for security monitoring
    - Dynamic origin validation with regex patterns
    - Environment-specific CORS policies
    - Performance metrics for CORS operations

    SECURITY (v5.4.1):
    - No wildcard fallback in production
    - Secure defaults for development
    """

    def __init__(
        self,
        app: ASGIApp,
        policy: CORSPolicy,
        *,
        allow_origins: list[str] = None,
        allow_methods: list[str] = None,
        allow_headers: list[str] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,
        allow_origin_regex: str = None,
    ):
        """
        Initialize the logging CORS middleware.

        Args:
            app: ASGI application
            policy: CORS policy configuration
            allow_origins: List of allowed origins (secure default if None)
            allow_methods: List of allowed methods
            allow_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            max_age: Maximum age for preflight cache
            allow_origin_regex: Regex pattern for origin validation
        """
        super().__init__(app)
        self.policy = policy

        # SECURITY: Use secure CORS origin defaults
        self.allow_origins = _get_secure_cors_origins(allow_origins)

        self.allow_methods = allow_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]
        self.allow_headers = allow_headers or [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.allow_origin_regex = allow_origin_regex

        # Initialize internal CORS middleware for actual CORS handling
        self._cors_middleware = CORSMiddleware(
            app=self.app,
            allow_origins=self.allow_origins,
            allow_methods=self.allow_methods,
            allow_headers=self.allow_headers,
            allow_credentials=self.allow_credentials,
            max_age=self.max_age,
        )

        # Statistics for monitoring
        self._cors_violations = 0
        self._cors_requests = 0
        self._preflight_requests = 0

    async def dispatch(self, request: Request, call_next):
        """
        Process incoming requests with CORS validation and logging.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response with CORS headers applied
        """
        origin = request.headers.get("origin")
        method = request.method

        # Increment total CORS requests counter
        self._cors_requests += 1

        # Check if this is a preflight request
        is_preflight = bool(
            method == "OPTIONS"
            and (
                request.headers.get("access-control-request-method")
                or request.headers.get("access-control-request-headers")
            )
        )

        if is_preflight:
            self._preflight_requests += 1

        # If no origin header, proceed normally (same-origin request)
        if not origin:
            return await call_next(request)

        # Validate origin against policy
        is_allowed = self.policy.is_origin_allowed(origin)

        if not is_allowed and self.policy.log_violations:
            self._log_cors_violation(request, origin, method, is_preflight)
            self._cors_violations += 1

        # Apply CORS headers using the internal CORS middleware
        # Note: We let the standard CORS middleware handle the actual header setting
        # but we log violations for security monitoring
        response = await call_next(request)

        # Add CORS headers if origin is allowed
        if is_allowed:
            response = self._add_cors_headers(response, origin, method, is_preflight)

        return response

    def _log_cors_violation(
        self, request: Request, origin: str, method: str, is_preflight: bool
    ) -> None:
        """
        Log CORS violations for security monitoring.

        Args:
            request: Incoming request
            origin: Origin that was rejected
            method: HTTP method
            is_preflight: Whether this was a preflight request
        """
        user_agent = request.headers.get("user-agent", "Unknown")
        referrer = request.headers.get("referer", "None")
        remote_ip = request.client.host if request.client else "Unknown"

        logger.warning(
            f"CORS violation detected: origin='{origin}' method='{method}' "
            f"path='{request.url.path}' preflight={is_preflight} "
            f"remote_ip='{remote_ip}' user_agent='{user_agent}' referrer='{referrer}'"
        )

    def _add_cors_headers(
        self, response: Response, origin: str, method: str, is_preflight: bool
    ) -> Response:
        """
        Add appropriate CORS headers to the response.

        Args:
            response: Response to modify
            origin: Allowed origin
            method: HTTP method
            is_preflight: Whether this is a preflight request

        Returns:
            Modified response with CORS headers
        """
        # Add basic CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin

        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        # Add preflight-specific headers if needed
        if is_preflight:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            response.headers["Access-Control-Max-Age"] = str(self.max_age)

        return response

    def get_stats(self) -> dict:
        """
        Get CORS middleware statistics for monitoring.

        Returns:
            Dictionary with CORS statistics
        """
        return {
            "total_requests": self._cors_requests,
            "preflight_requests": self._preflight_requests,
            "violations": self._cors_violations,
            "violation_rate": (
                self._cors_violations / self._cors_requests * 100 if self._cors_requests > 0 else 0
            ),
        }


def add_cors_middleware(
    app: FastAPI,
    environment: str | Environment = None,
    custom_policy: CORSPolicy = None,
) -> None:
    """
    Add CORS middleware to the FastAPI application.

    This is a convenience function that creates and adds the CORS middleware
    to the application in one step.

    Args:
        app: FastAPI application
        environment: Environment name (development, production, test) or Environment enum
        custom_policy: Custom CORS policy (overrides environment-based policy)
    """
    # Determine environment if not provided
    if environment is None:
        # Try to get from settings or default to development
        environment = getattr(settings, "ENVIRONMENT", "development")

    # Ensure environment is properly converted to Environment enum
    if isinstance(environment, str):
        environment = Environment(environment.lower())

    # Get CORS policy
    policy = custom_policy or cors_config.get_policy(environment)

    logger.info(
        f"Adding CORS middleware for environment: {policy.environment} "
        f"(allow_all_origins={policy.allow_all_origins}, "
        f"allowed_origins={len(policy.allowed_origins)}, "
        f"allow_credentials={policy.allow_credentials})"
    )

    # Add the middleware to the app with proper parameters
    app.add_middleware(
        CORSMiddleware,
        allow_origins=policy.allowed_origins if not policy.allow_all_origins else ["*"],
        allow_methods=policy.allowed_methods,
        allow_headers=policy.allowed_headers,
        allow_credentials=policy.allow_credentials,
        max_age=policy.max_age,
    )


# Environment-specific CORS configuration functions
def get_development_cors_config() -> CORSPolicy:
    """Get CORS configuration for development environment."""
    return CORSPolicy(
        environment=Environment.DEVELOPMENT,
        allowed_origins=["*"],
        allow_all_origins=True,
        allow_credentials=True,
        log_violations=True,
    )


def get_production_cors_config(
    allowed_origins: list[str] = None, allow_credentials: bool = False
) -> CORSPolicy:
    """
    Get CORS configuration for production environment.

    Args:
        allowed_origins: List of specific allowed origins
        allow_credentials: Whether to allow credentials

    Returns:
        Production CORS policy
    """
    if allowed_origins is None:
        allowed_origins = []

    return CORSPolicy(
        environment=Environment.PRODUCTION,
        allowed_origins=allowed_origins,
        allow_all_origins=False,  # Strict: no wildcards in production
        allow_credentials=allow_credentials,
        log_violations=True,
        origin_regex_patterns=[],  # Can add regex patterns for dynamic validation
    )


def get_test_cors_config() -> CORSPolicy:
    """Get CORS configuration for test environment."""
    return CORSPolicy(
        environment=Environment.TEST,
        allowed_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_all_origins=False,
        allow_credentials=True,
        log_violations=True,
    )
