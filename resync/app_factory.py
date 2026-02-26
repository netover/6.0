"""
Application factory for creating and configuring the FastAPI application.

This module is imported by resync/main.py and provides the actual application
initialization logic following the factory pattern.

No Global State:
    The ``create_app()`` function creates a new ApplicationFactory instance
    per call to avoid state contamination between tests. All runtime state
    lives on ``app.state``.

    ``ApplicationFactory.__init__`` accepts an optional ``settings`` argument
    to enable full DI in tests without monkey-patching the module.
"""

import hashlib
import importlib  # [P3-01] Modern Python 3 API for dynamic imports
import os
from typing import TYPE_CHECKING, Any

import orjson  # [P2-02] top-level import — not per-request
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from slowapi.errors import RateLimitExceeded
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response  # [P3-01] FileResponse from starlette.responses
from starlette.staticfiles import StaticFiles as StarletteStaticFiles
from starlette.types import Scope

from resync.core.structured_logger import configure_structured_logging, get_logger

if TYPE_CHECKING:
    from resync.core.startup import lifespan as app_lifespan
    from resync.settings import Settings


def _get_lifespan() -> "app_lifespan":
    """Lazy loader for lifespan to avoid circular imports at module load time."""
    from resync.core.startup import lifespan

    return lifespan


def _get_settings() -> "Settings":
    """[P1-02] Lazy settings accessor — allows test-time DI via ApplicationFactory(settings=...)."""
    from resync.settings import get_settings

    return get_settings()


# =============================================================================
# MODULE CONSTANTS (production security limits)
# =============================================================================
# [P2-05 FIX] All other configuration values centralized in resync.settings.Settings
# Use self.settings.<field_name> instead of module-level _DEFAULT_* constants.

#: Maximum CSP report payload size (bytes) - prevents DoS via large payloads.
#: Hard-coded security limit (intentionally not configurable via settings).
_MAX_CSP_PAYLOAD_SIZE = 4096

#: Minimum CSP nonce length for production (hex characters).
#: Hard-coded security requirement (intentionally not configurable via settings).
_MIN_CSP_NONCE_LENGTH = 16

# Configure app factory logger
app_logger = get_logger("resync.app_factory")

# Use app_logger as the primary logger for this module
logger = app_logger


# [P2-01] Public replacement for the private slowapi._rate_limit_exceeded_handler.
# Uses the public RateLimitExceeded API only — safe across slowapi upgrades.
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle 429 Too Many Requests with Retry-After header."""
    retry_after = str(getattr(exc, "retry_after", 60))
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": str(exc.detail) if hasattr(exc, "detail") else "Too Many Requests",
            "retry_after": retry_after,
        },
        headers={"Retry-After": retry_after},
    )


class CachedStaticFiles(StarletteStaticFiles):
    """Static files handler with ETag and Cache-Control headers.

    [P3-03] Settings resolved once at instantiation time, not per request.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize static files handler and cache settings."""
        super().__init__(**kwargs)
        # [P2-05 FIX] Direct access to Settings fields eliminates duplicate defaults.
        # All fields are guaranteed to exist with validated defaults from Pydantic.
        settings = _get_settings()
        self._cache_max_age = settings.static_cache_max_age
        self._etag_hash_length = settings.ETAG_HASH_LENGTH

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Return response with cache-friendly headers."""
        response = await super().get_response(path, scope)

        if response.status_code == 200:
            response.headers["Cache-Control"] = f"public, max-age={self._cache_max_age}"

            # Generate ETag from file metadata for cache validation
            try:
                # Reuse stat_result from FileResponse if available (avoid double I/O)
                if isinstance(response, FileResponse) and getattr(
                    response, "stat_result", None
                ):
                    st = response.stat_result
                    file_metadata = f"{st.st_size}-{int(st.st_mtime)}"
                else:
                    # Fallback: file stat not available, use path hash
                    file_metadata = None

                if file_metadata is not None:
                    digest = hashlib.sha256(file_metadata.encode()).hexdigest()[
                        : self._etag_hash_length
                    ]
                    response.headers["ETag"] = f'"{digest}"'
            except OSError as exc:
                logger.warning("failed_to_generate_etag", error=str(exc))
                # Don't generate ETag if we can't get file metadata
                # This prevents serving stale content indefinitely

        return response


class ApplicationFactory:
    """
    Factory for creating and configuring FastAPI applications.

    This class encapsulates all application initialization logic,
    providing a clean separation of concerns and modular architecture.

    Args:
        settings: Optional settings instance for test-time DI. When None,
                  ``_get_settings()`` is called lazily on first use.
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        """Initialize the application factory.

        [P1-02] Accept settings via constructor to avoid global singleton
        contamination between test runs. Production code calls create_app()
        which passes no settings (lazy resolution via get_settings()).
        """
        self._settings = settings  # None = lazy resolution
        self.app: FastAPI
        self.templates: Jinja2Templates | None = None
        self.template_env: Environment | None = None

    @property
    def settings(self) -> "Settings":
        """[P1-02] Lazy settings accessor — resolves once and caches on instance."""
        if self._settings is None:
            self._settings = _get_settings()
        return self._settings

    def create_application(self) -> FastAPI:
        """
        Create and configure the FastAPI application.

        Returns:
            Fully configured FastAPI application instance
        """
        # Configure logging EARLY - must be before any other imports that use logging
        configure_structured_logging(
            log_level=self.settings.log_level,
            json_logs=self.settings.log_format == "json",
            development_mode=self.settings.is_development,
        )

        # Validate settings first
        self._validate_critical_settings()

        # Get lifespan via lazy loader to avoid circular imports
        app_lifespan = _get_lifespan()

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=self.settings.project_name,
            version=self.settings.project_version,
            description=self.settings.description,
            lifespan=app_lifespan,
            docs_url="/api/docs" if not self.settings.is_production else None,
            redoc_url="/api/redoc" if not self.settings.is_production else None,
            openapi_url="/api/openapi.json" if not self.settings.is_production else None,
        )

        # Configure all components in order
        self._setup_templates()
        self._configure_middleware()
        self._configure_exception_handlers()
        self._setup_dependency_injection()
        self._register_routers()
        self._mount_static_files()
        self._register_special_endpoints()

        logger.info(
            "application_created",
            environment=self.settings.environment.value,
            debug_mode=self.settings.is_development,
        )

        return self.app

    def _validate_critical_settings(self) -> None:
        """Validate critical settings before application startup."""
        errors = []

        # Redis configuration
        if self.settings.redis_pool_min_size > self.settings.redis_pool_max_size:
            max_sz = self.settings.redis_pool_max_size
            min_sz = self.settings.redis_pool_min_size
            raise ValueError(
                "redis_pool_max_age "
                f"({max_sz}) must be >= redis_pool_min_size ({min_sz})"
            )

        # Production-specific validations
        if self.settings.is_production:
            # [P2-05 FIX] Direct field access - Pydantic ensures defaults exist
            min_pw_len = self.settings.MIN_ADMIN_PASSWORD_LENGTH
            min_sk_len = self.settings.MIN_SECRET_KEY_LENGTH

            # Admin credentials
            admin_pw = (
                self.settings.admin_password.get_secret_value()
                if self.settings.admin_password
                else ""
            )
            if not admin_pw or len(admin_pw.strip()) < min_pw_len:
                errors.append(
                    f"ADMIN_PASSWORD must be set (>= {min_pw_len} chars) in production"
                )
            elif admin_pw.strip().lower() in {
                "change_me_please",
                "admin",
                "password",
            }:
                errors.append("Insecure admin password in production")

            # JWT secret key
            secret = (
                self.settings.secret_key.get_secret_value() if self.settings.secret_key else ""
            )
            if (
                not secret
                or len(secret.strip()) < min_sk_len
                or ("CHANGE_ME" in secret.upper())
            ):
                errors.append(
                    f"SECRET_KEY must be set (>= {min_sk_len} chars) in production"
                )

            # Debug must remain disabled
            if getattr(self.settings, "debug", False):
                errors.append("Debug mode must be disabled in production")

            # CORS hardening
            if any(origin.strip() == "*" for origin in self.settings.cors_allowed_origins):
                errors.append("Wildcard CORS origins not allowed in production")

            # LLM configuration hardening
            llm_key = getattr(self.settings, "llm_api_key", None)
            if llm_key is not None:
                try:
                    if llm_key.get_secret_value() == "dummy_key_for_development":
                        errors.append("Invalid LLM API key in production")
                except (AttributeError, TypeError):
                    # If type drifts, fail closed in production
                    errors.append("Invalid LLM API key in production")

        # Raise if any critical errors
        if errors:
            for error in errors:
                logger.error("configuration_error", error=error)
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        logger.info("settings_validation_passed")

    def _setup_templates(self) -> None:
        """Configure Jinja2 template engine."""
        templates_dir = self.settings.base_dir / "templates"

        if not templates_dir.exists():
            logger.warning("templates_directory_not_found", path=str(templates_dir))
            return

        # Create Jinja2 environment with security settings
        # [P2-05 FIX] JINJA2_TEMPLATE_CACHE_SIZE is a computed @property in Settings
        self.template_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(
                enabled_extensions=("html", "xml"),
                default_for_string=True,
                default=True,
            ),
            auto_reload=self.settings.is_development,
            cache_size=self.settings.JINJA2_TEMPLATE_CACHE_SIZE
            if self.settings.is_production
            else 0,
            enable_async=True,
            # extensions=['resync.core.csp_jinja_extension.CSPNonceExtension']
        )

        self.templates = Jinja2Templates(directory=str(templates_dir))
        self.templates.env = self.template_env

        # Store in app state for access in routes
        self.app.state.template_env = self.template_env
        self.app.state.templates = self.templates

        logger.info("templates_configured", directory=str(templates_dir))

    def _configure_middleware(self) -> None:
        """Configure all middleware in a safe, production-grade order.

        Starlette/FastAPI middleware execution order:
            - The *last* `add_middleware()` call is executed *first* (outermost).

        Desired outer-to-inner order:
            1) CorrelationIdMiddleware (sets contextvars for all downstream logs)
            2) Security headers (applied even to CORS preflight)
            3) CSP (production-only)
            4) CORS (proper preflight handling)
            5) Rate limiting (can be bypassed by preflight; intentional)
        """
        from resync.api.middleware.cors_config import CORSConfig
        from resync.api.middleware.cors_middleware import LoggingCORSMiddleware

        # 0) Reverse-proxy hardening (enterprise default)
        # - ProxyHeadersMiddleware: trust X-Forwarded-* only from known proxies
        # - TrustedHostMiddleware: protect against Host header attacks
        #
        # Configure via:
        #   TRUSTED_HOSTS="api.example.com,*.example.com"
        #   PROXY_TRUSTED_HOSTS="10.0.0.0/8,192.168.0.0/16"  (or "*" ONLY in dev)
        try:
            if self.settings.is_production:
                trusted_hosts_env = os.getenv("TRUSTED_HOSTS", "").strip()
                if trusted_hosts_env:
                    from fastapi.middleware.trustedhost import TrustedHostMiddleware

                    allowed_hosts = [
                        h.strip()
                        for h in trusted_hosts_env.split(",")
                        if h.strip()
                    ]
                    self.app.add_middleware(
                        TrustedHostMiddleware,
                        allowed_hosts=allowed_hosts,
                    )

                proxy_enabled = os.getenv("PROXY_HEADERS", "false").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if proxy_enabled:
                    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

                    # Comma-separated list of proxy IPs/CIDRs. Defaults to FORWARDED_ALLOW_IPS.
                    proxy_trusted = os.getenv(
                        "PROXY_TRUSTED_HOSTS",
                        os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
                    )
                    self.app.add_middleware(
                        ProxyHeadersMiddleware,
                        trusted_hosts=proxy_trusted,
                    )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Fail closed in production: better to crash than accept spoofed headers.
            if self.settings.is_production:
                logger.critical("proxy_middleware_setup_failed_prod", error=str(e))
                raise
            logger.warning("proxy_middleware_setup_failed", error=str(e))

        # 1) Rate limiting (startup-time wiring)
        # [P1-01] BaseException replaced with specific infra exceptions.
        #         CancelledError / SystemExit / KeyboardInterrupt must propagate freely.
        try:
            from resync.core.security.rate_limiter_v2 import setup_rate_limiting

            setup_rate_limiting(self.app)

            # Set up slowapi for unified_config_api endpoints
            from slowapi.middleware import SlowAPIMiddleware
            from resync.api.unified_config_api import limiter

            self.app.state.limiter = limiter
            self.app.add_middleware(SlowAPIMiddleware)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
            # [P1-01] Rate limiting must not silently fail open in production.
            if self.settings.is_production:
                logger.critical("rate_limiting_setup_failed_prod", error=str(e))
                raise
            logger.warning("rate_limiting_setup_failed", error=str(e))

        # 2) CORS configuration (delegate to Starlette CORSMiddleware)
        cors_config = CORSConfig()
        cors_policy = cors_config.get_policy(self.settings.environment.value)

        # Convert list of regex patterns to a single regex string when provided.
        allow_origin_regex = None
        if getattr(cors_policy, "origin_regex_patterns", None):
            allow_origin_regex = "|".join(
                f"(?:{p})" for p in cors_policy.origin_regex_patterns
            )

        self.app.add_middleware(
            LoggingCORSMiddleware,
            allow_origins=cors_policy.allowed_origins,
            allow_methods=cors_policy.allowed_methods,
            allow_headers=cors_policy.allowed_headers,
            allow_credentials=cors_policy.allow_credentials,
            max_age=cors_policy.max_age,
            allow_origin_regex=allow_origin_regex,
            log_violations=cors_policy.log_violations,
        )

        # 3) CSP (production-only)
        if self.settings.is_production:
            from resync.api.middleware.csp_middleware import CSPMiddleware

            self.app.add_middleware(CSPMiddleware, report_only=False)

        # 4) Additional security headers
        from resync.config.security import add_additional_security_headers

        add_additional_security_headers(self.app)

        # 5) Correlation IDs MUST be outermost (added last)
        from resync.api.middleware.correlation_id import CorrelationIdMiddleware

        self.app.add_middleware(CorrelationIdMiddleware, header_name="X-Correlation-ID")

        logger.info("middleware_configured")

    def _configure_exception_handlers(self) -> None:
        """Register global exception handlers."""
        from resync.api.exception_handlers import register_exception_handlers

        register_exception_handlers(self.app)

        # [P2-01] Use public _rate_limit_handler instead of private
        #         slowapi._rate_limit_exceeded_handler (prefixed _ = private API).
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

        logger.info("exception_handlers_registered")

    def _setup_dependency_injection(self) -> None:
        """Configure explicit FastAPI dependencies (canonical DI).

        The HTTP request path must not depend on a global DI container.
        Providers live in resync.core.wiring and are consumed via FastAPI Depends.
        """
        logger.info("dependency_injection_configured", mode="fastapi_depends")

    def _register_routers(self) -> None:
        """
        Register all API routers with fail-fast logic in dev mode.

        v5.8.0: Unified API structure - all routes under resync/api/routes/
        v6.3.0: Fail-fast in development mode for broken routes.
        """
        # =================================================================
        # CORE ROUTERS (v7.0: canonical paths under resync/api/routes/)
        # =================================================================

        # Essential routers - must load (fail-fast always)
        # [P2-04] unified_config_api added here — critical infra since v6.1.0
        essential_routers = [
            ("resync.api.routes.core.health", "router", "health_router"),
            ("resync.api.routes.admin.main", "admin_router", "admin_router"),
            ("resync.api.agents", "agents_router", "agents_router"),
            ("resync.api.chat", "chat_router", "chat_router"),
            ("resync.api.unified_config_api", "router", "unified_config_router"),  # [P2-04]
        ]

        for module_path, router_name, log_name in essential_routers:
            try:
                # [P3-01] Modern importlib API replaces legacy __import__
                mod = importlib.import_module(module_path)
                router = getattr(mod, router_name)
                self.app.include_router(router)
            except ImportError as e:
                logger.critical(
                    "essential_router_import_failed",
                    module=module_path,
                    router=router_name,
                    error=str(e),
                )
                raise

        # Optional routers - graceful degradation in prod, fail-fast in dev
        optional_routers = [
            ("resync.api.routes.admin.prompts", "prompt_router", "prompt_router"),
            ("resync.api.routes.admin.routing", "router", "routing_router"),
            ("resync.api.routes.audit", "router", "audit_router"),
            ("resync.api.routes.cache", "router", "cache_router"),
            ("resync.api.routes.cors_monitoring", "router", "cors_monitor_router"),
            ("resync.api.routes.performance", "router", "performance_router"),
        ]

        for module_path, router_name, log_name in optional_routers:
            try:
                # [P3-01] Modern importlib API replaces legacy __import__
                mod = importlib.import_module(module_path)
                router = getattr(mod, router_name)
                self.app.include_router(router)
            except ImportError as e:
                if self.settings.is_development:
                    logger.error(
                        "route_import_failed_dev", module=module_path, error=str(e)
                    )
                    raise
                else:
                    logger.warning(
                        "route_import_failed_prod", module=module_path, error=str(e)
                    )

        # Additional routers from main_improved
        try:
            from resync.api.routes.core.health import config_router
            from resync.api.routes.endpoints import router as api_router
            from resync.api.routes.rag.upload import router as rag_upload_router

            self.app.include_router(api_router, prefix="/api")
            self.app.include_router(config_router, prefix="/api/v1")
            self.app.include_router(rag_upload_router, prefix="/api/v1")
        except ImportError as e:
            if self.settings.is_development:
                logger.error("optional_routers_not_available", error=str(e))
                raise
            logger.warning("optional_routers_not_available", error=str(e))

        # v5.9.9: Enhanced endpoints (orchestrator-based)
        try:
            from resync.api.enhanced_endpoints import enhanced_router
            from resync.api.routes.orchestration import router as orchestration_router

            self.app.include_router(enhanced_router)
            self.app.include_router(orchestration_router, prefix="/api/v1")
            logger.info("enhanced_endpoints_registered", prefix="/api/v2")
            logger.info(
                "orchestration_router_registered", prefix="/api/v1/orchestration"
            )
        except ImportError as e:
            if self.settings.is_development:
                logger.error("enhanced_endpoints_not_available", error=str(e))
                raise
            logger.warning("enhanced_endpoints_not_available", error=str(e))

        # v5.9.9: GraphRAG admin endpoints
        try:
            from resync.api.graphrag_admin import router as graphrag_admin_router

            self.app.include_router(graphrag_admin_router)
            logger.info(
                "graphrag_admin_endpoints_registered", prefix="/api/admin/graphrag"
            )
        except ImportError as e:
            if self.settings.is_development:
                logger.error("graphrag_admin_not_available", error=str(e))
                raise
            logger.warning("graphrag_admin_not_available", error=str(e))

        # v6.1: Document Knowledge Graph (DKG) admin endpoints
        try:
            from resync.api.document_kg_admin import router as dkg_admin_router

            self.app.include_router(dkg_admin_router)
            logger.info(
                "document_kg_admin_endpoints_registered", prefix="/api/admin/kg"
            )
        except ImportError as e:
            if self.settings.is_development:
                logger.error("document_kg_admin_not_available", error=str(e))
                raise
            logger.warning("document_kg_admin_not_available", error=str(e))

        # Register monitoring routers
        try:
            from resync.api.routes.monitoring.routes import monitoring_router
            from resync.api.routes.monitoring.prometheus_exporter import (
                router as prometheus_exporter_router,
            )
            from resync.core.monitoring_integration import register_dashboard_route

            self.app.include_router(monitoring_router, tags=["Monitoring"])
            self.app.include_router(
                prometheus_exporter_router,
                tags=["Monitoring - Prometheus"],
            )
            register_dashboard_route(self.app)
            logger.info("monitoring_routers_registered")
        except ImportError as e:
            if self.settings.is_development:
                logger.error("monitoring_routers_not_available", error=str(e))
                raise
            logger.warning("monitoring_routers_not_available", error=str(e))

        # =================================================================
        # v5.8.0: UNIFIED ROUTES (migrated from fastapi_app/)
        # =================================================================
        try:
            # Admin routes (migrated from fastapi_app)
            from resync.api.routes.admin.backup import router as backup_router
            from resync.api.routes.admin.config import router as admin_config_router
            from resync.api.routes.admin.connectors import router as connectors_router
            from resync.api.routes.admin.environment import router as environment_router
            from resync.api.routes.admin.feedback_curation import (
                router as feedback_curation_router,
            )
            from resync.api.routes.admin.notification_admin import (
                router as notification_admin_router,
            )
            from resync.api.routes.admin.rag_reranker import (
                router as rag_reranker_router,
            )
            from resync.api.routes.admin.semantic_cache import (
                router as semantic_cache_router,
            )
            from resync.api.routes.admin.settings_manager import (
                router as settings_manager_router,
            )
            from resync.api.routes.admin.skills import router as skills_router
            from resync.api.routes.admin.teams import router as teams_router
            from resync.api.routes.admin.teams_notifications_admin import (
                router as teams_notifications_admin_router,
            )
            from resync.api.routes.admin.teams_webhook_admin import (
                router as teams_webhook_admin_router,
            )
            from resync.api.routes.admin.threshold_tuning import (
                router as threshold_tuning_router,
            )
            from resync.api.routes.admin.tws_instances import (
                router as tws_instances_router,
            )
            from resync.api.routes.admin.users import router as admin_users_router
            from resync.api.routes.admin.v2 import router as admin_v2_router

            # Other routes (migrated from fastapi_app)
            from resync.api.routes.core.status import router as status_router
            from resync.api.routes.knowledge.ingest_api import (
                router as knowledge_ingest_router,
            )
            from resync.api.routes.monitoring.admin_monitoring import (
                router as admin_monitoring_router,
            )

            # Monitoring routes (migrated from fastapi_app)
            from resync.api.routes.monitoring.ai_monitoring import (
                router as ai_monitoring_router,
            )
            # prometheus_exporter_router already imported and registered above
            from resync.api.routes.monitoring.metrics_dashboard import (
                router as metrics_dashboard_router,
            )
            from resync.api.routes.monitoring.observability import (
                router as observability_router,
            )

            # learning_router removed in v5.9.3 (drift/eval features unused)
            from resync.api.routes.rag.query import router as rag_query_router

            # Teams webhook public endpoint
            from resync.api.routes.teams_webhook import router as teams_webhook_router

            # API Key Management
            from resync.api.v1.admin import admin_api_keys_router

            # Register unified admin routes
            unified_admin_routers = [
                (backup_router, "/api/v1/admin", ["Admin - Backup"]),
                (admin_config_router, "/api/v1/admin", ["Admin - Config"]),
                (
                    settings_manager_router,
                    "/api/v1/admin",
                    ["Admin - Settings Manager"],
                ),
                (teams_router, "/api/v1/admin", ["Admin - Teams"]),
                (teams_webhook_admin_router, "/api", ["Admin - Teams Webhook Users"]),
                (notification_admin_router, "/api/v1/admin", ["Admin - Notifications"]),
                (
                    teams_notifications_admin_router,
                    "/api",
                    ["Admin - Teams Notifications"],
                ),
                (tws_instances_router, "/api/v1/admin", ["Admin - TWS Instances"]),
                (admin_users_router, "/api/v1/admin", ["Admin - Users"]),
                (semantic_cache_router, "/api/v1/admin", ["Admin - Semantic Cache"]),
                (rag_reranker_router, "/api/v1", ["Admin - RAG Reranker"]),
                (
                    threshold_tuning_router,
                    "/api/v1/admin",
                    ["Admin - Threshold Tuning"],
                ),
                (environment_router, "/api/v1/admin", ["Admin - Environment"]),
                (connectors_router, "/api/v1/admin", ["Admin - Connectors"]),
                (feedback_curation_router, "", ["Admin - Feedback Curation"]),
                (admin_api_keys_router, "/api/v1/admin", ["Admin - API Keys"]),
                (admin_v2_router, "/api/v2/admin", ["Admin V2"]),
                (skills_router, "/api/v1/admin", ["Admin - Skills"]),
            ]

            # Register unified monitoring routes (with admin auth)
            from resync.api.routes.core.auth import (
                verify_admin_credentials as _verify_admin,
            )

            unified_monitoring_routers = [
                (ai_monitoring_router, "/api/v1/monitoring", ["Monitoring - AI"]),
                (
                    observability_router,
                    "/api/v1/monitoring",
                    ["Monitoring - Observability"],
                ),
                (
                    metrics_dashboard_router,
                    "/api/v1/monitoring",
                    ["Monitoring - Metrics"],
                ),
                (admin_monitoring_router, "/api/v1/monitoring", ["Monitoring - Admin"]),
            ]

            # Register other unified routes
            unified_other_routers = [
                (status_router, "/api/v1", ["Status"]),
                # learning_router removed in v5.9.3
                (rag_query_router, "/api/v1/rag", ["RAG"]),
                (knowledge_ingest_router, "/api/v1", ["Knowledge Ingestion"]),
                (teams_webhook_router, "", ["Teams Webhook"]),
            ]

            for router, prefix, tags in unified_admin_routers + unified_other_routers:
                self.app.include_router(router, prefix=prefix, tags=tags)

            # Monitoring routes get admin auth at registration time
            for router, prefix, tags in unified_monitoring_routers:
                self.app.include_router(
                    router,
                    prefix=prefix,
                    tags=tags,
                    dependencies=[Depends(_verify_admin)],
                )

            logger.info(
                "unified_routers_registered",
                admin_count=len(unified_admin_routers),
                monitoring_count=len(unified_monitoring_routers),
                other_count=len(unified_other_routers),
            )
        except ImportError as e:
            if self.settings.is_development:
                logger.error("unified_routers_not_available", error=str(e))
                raise
            logger.warning("unified_routers_not_available", error=str(e))

        logger.info("routers_registered")

    def _mount_static_files(self) -> None:
        """Mount static file directory with caching.

        All static assets are served from /static prefix.
        Legacy submounts (/css, /js, /img, /fonts, /assets) removed -
        use /static/{subdir}/... instead.
        """
        static_dir = self.settings.base_dir / "static"

        if not static_dir.exists():
            logger.warning("static_directory_not_found", path=str(static_dir))
            return

        # Mount main static directory with caching
        self.app.mount(
            "/static", CachedStaticFiles(directory=str(static_dir)), name="static"
        )

        logger.info("static_files_mounted", directory=str(static_dir))

    def _register_special_endpoints(self) -> None:
        """Register special endpoints (frontend, CSP, etc.)."""
        # [P2-03] Import process_csp_report once at registration time, not per-request
        from resync.csp_validation import process_csp_report
        from resync.core.security.rate_limiter_v2 import rate_limit

        # Root redirect
        @self.app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            """Redirect root to admin dashboard."""
            return RedirectResponse(url="/admin", status_code=307)

        # Admin panel is now handled by the admin router

        # Revision page
        @self.app.get("/revisao", include_in_schema=False, response_class=HTMLResponse)
        async def revisao_page(request: Request) -> HTMLResponse:
            """Serve the revision page."""
            return self._render_template("revisao.html", request)

        # CSP violation report endpoint with rate limiting to prevent
        # DoS from browser extensions.
        @self.app.post("/csp-violation-report", include_in_schema=False)
        @rate_limit("30/minute")
        async def csp_violation_report(request: Request) -> JSONResponse:
            """
            Handle CSP violation reports with stream-based payload validation.

            Security:
                - Stream-based validation prevents DoS via Content-Length spoofing
                - Transfer-Encoding: chunked bypass protection
                - Size limit prevents memory exhaustion
            """
            body_size = 0
            chunks: list[bytes] = []

            try:
                async for chunk in request.stream():
                    body_size += len(chunk)
                    if body_size > _MAX_CSP_PAYLOAD_SIZE:
                        logger.warning(
                            "csp_report_payload_rejected",
                            size=body_size,
                            max_allowed=_MAX_CSP_PAYLOAD_SIZE,
                            client_host=request.client.host
                            if request.client
                            else "unknown",
                        )
                        return JSONResponse(
                            {"status": "ignored", "reason": "payload_too_large"},
                            status_code=413,
                        )
                    chunks.append(chunk)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:  # noqa: BLE001
                logger.error("csp_report_stream_error", error=str(e))
                return JSONResponse(
                    {"status": "ignored", "reason": "stream_error"},
                    status_code=400,
                )

            # [P2-02] orjson imported at module top-level — no per-request import cost
            try:
                report_data: object = orjson.loads(b"".join(chunks))
            except (ValueError, orjson.JSONDecodeError) as e:
                logger.warning(
                    "csp_invalid_json",
                    error=str(e),
                    client_host=request.client.host if request.client else "unknown",
                )
                return JSONResponse(
                    {"status": "ignored", "reason": "invalid_json"},
                    status_code=400,
                )

            if not isinstance(report_data, dict):
                logger.warning(
                    "csp_report_invalid_structure",
                    received_type=type(report_data).__name__,
                    client_host=request.client.host if request.client else "unknown",
                )
                return JSONResponse(
                    {"status": "ignored", "reason": "invalid_structure"},
                    status_code=400,
                )

            # [P2-03] process_csp_report captured at registration time (closure)
            return await process_csp_report(request, report_data=report_data)

        logger.info("special_endpoints_registered")

    def _render_template(self, template_name: str, request: Request) -> HTMLResponse:
        """
        Render a template with CSP nonce support.

        Args:
            template_name: Name of the template file
            request: FastAPI request object

        Returns:
            Rendered HTML response

        Raises:
            HTTPException: 500 if CSP nonce is missing in production (fail-closed)
        """
        if not self.templates:
            raise HTTPException(
                status_code=500, detail="Template engine not configured"
            )

        nonce = getattr(request.state, "csp_nonce", None)

        if self.settings.is_production:
            if not nonce:
                logger.critical("csp_nonce_missing_prod", template=template_name)
                raise HTTPException(
                    status_code=500, detail="Security middleware failed"
                )
            if not isinstance(nonce, str) or len(nonce) < _MIN_CSP_NONCE_LENGTH:
                logger.critical(
                    "csp_nonce_invalid_prod",
                    template=template_name,
                    nonce_present=nonce is not None,
                    nonce_type=type(nonce).__name__ if nonce else "None",
                    nonce_length=len(nonce) if isinstance(nonce, str) else 0,
                    required_length=_MIN_CSP_NONCE_LENGTH,
                )
                raise HTTPException(
                    status_code=500, detail="Security middleware failed"
                )

        try:
            nonce_value = nonce or ""
            context = {
                "request": request,
                "nonce": nonce_value,
                "csp_nonce": nonce_value,  # backward compatible
                "settings": {
                    "project_name": self.settings.project_name,
                    "version": self.settings.project_version,
                    # Security: don't expose environment to prevent fingerprinting
                },
            }
            return self.templates.TemplateResponse(request, template_name, context)
        except FileNotFoundError:
            logger.error("template_not_found", template=template_name)
            raise HTTPException(
                status_code=404, detail=f"Template {template_name} not found"
            ) from None
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:  # noqa: BLE001
            logger.error("template_render_error", template=template_name, error=str(e))
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from None


def create_app() -> FastAPI:
    """Entry point para Uvicorn e Pytest.

    Creates a new ApplicationFactory instance per call to avoid
    state contamination between tests.

    Returns:
        Configured FastAPI application
    """
    return ApplicationFactory().create_application()
