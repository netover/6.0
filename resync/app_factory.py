# mypy: ignore-errors
"""
Application factory for creating and configuring the FastAPI application.

This module is imported by resync/main.py and provides the actual application
initialization logic following the factory pattern.

No Global State:
    The ``create_app()`` function creates a new ApplicationFactory instance
    per call to avoid state contamination between tests. All runtime state
    lives on ``app.state``.
"""

import hashlib

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.responses import HTMLResponse
from starlette.staticfiles import FileResponse, StaticFiles as StarletteStaticFiles

from resync.core.structured_logger import get_logger
from resync.settings import settings

# =============================================================================
# MODULE CONSTANTS (configurable via settings / admin UI where noted)
# =============================================================================

#: Default max-age for static file Cache-Control header (seconds).
#: Overridden at runtime by ``settings.static_cache_max_age``.
_DEFAULT_STATIC_CACHE_MAX_AGE = 86400  # 1 day

#: Number of hex chars from SHA-256 kept for ETag values.
#: 16 hex chars = 64-bit collision space — sufficient for cache validation.
#: Configurable via ``settings.ETAG_HASH_LENGTH``.
_DEFAULT_ETAG_HASH_LENGTH = 16

#: Jinja2 template bytecode cache size (number of templates).
#: Configurable via ``settings.JINJA2_TEMPLATE_CACHE_SIZE`` (computed property).
_DEFAULT_TEMPLATE_CACHE_SIZE = 400

#: Timeout for cancelling background tasks during shutdown (seconds).
#: Configurable via ``settings.SHUTDOWN_TASK_CANCEL_TIMEOUT``.
_DEFAULT_TASK_CANCEL_TIMEOUT = 5.0

#: Minimum admin password length in production.
_DEFAULT_MIN_PASSWORD_LENGTH = 8

#: Minimum SECRET_KEY length in production.
_DEFAULT_MIN_SECRET_KEY_LENGTH = 32

# Configure app factory logger
app_logger = get_logger("resync.app_factory")

# Use app_logger as the primary logger for this module
logger = app_logger


class CachedStaticFiles(StarletteStaticFiles):
    """Static files handler with ETag and Cache-Control headers."""

    async def get_response(self, path: str, scope):
        """Return response with cache-friendly headers."""
        response = await super().get_response(path, scope)

        if response.status_code == 200:
            cache_max_age = getattr(
                settings, "static_cache_max_age", _DEFAULT_STATIC_CACHE_MAX_AGE
            )
            response.headers["Cache-Control"] = f"public, max-age={cache_max_age}"

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
                    hash_len = getattr(
                        settings, "ETAG_HASH_LENGTH", _DEFAULT_ETAG_HASH_LENGTH
                    )
                    digest = hashlib.sha256(file_metadata.encode()).hexdigest()[
                        :hash_len
                    ]
                    response.headers["ETag"] = f'"{digest}"'
            except Exception as exc:
                logger.warning("failed_to_generate_etag", error=str(exc))
                # Don't generate ETag if we can't get file metadata
                # This prevents serving stale content indefinitely

        return response


class ApplicationFactory:
    """
    Factory for creating and configuring FastAPI applications.

    This class encapsulates all application initialization logic,
    providing a clean separation of concerns and modular architecture.
    """

    def __init__(self):
        """Initialize the application factory."""
        self.app: FastAPI | None = None
        self.templates: Jinja2Templates | None = None
        self.template_env: Environment | None = None

    def create_application(self) -> FastAPI:
        """
        Create and configure the FastAPI application.

        Returns:
            Fully configured FastAPI application instance
        """
        # Configure logging EARLY
        from resync.core.structured_logger import configure_structured_logging

        configure_structured_logging(
            log_level=settings.log_level,
            json_logs=settings.log_format == "json",
            development_mode=settings.is_development,
        )

        # Validate settings first
        self._validate_critical_settings()

        from resync.core.startup import lifespan as app_lifespan

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=settings.project_name,
            version=settings.project_version,
            description=settings.description,
            lifespan=app_lifespan,
            docs_url="/api/docs" if not settings.is_production else None,
            redoc_url="/api/redoc" if not settings.is_production else None,
            openapi_url="/api/openapi.json" if not settings.is_production else None,
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
            environment=settings.environment.value,
            debug_mode=settings.is_development,
        )

        return self.app

    def _validate_critical_settings(self) -> None:
        """Validate critical settings before application startup."""
        errors = []

        # Redis configuration
        if settings.redis_pool_min_size > settings.redis_pool_max_size:
            raise ValueError(
                f"redis_pool_max_size ({settings.redis_pool_max_size}) must be >= redis_pool_min_size ({settings.redis_pool_min_size})"
            )

        # Production-specific validations
        if settings.is_production:
            min_pw_len = getattr(
                settings, "MIN_ADMIN_PASSWORD_LENGTH", _DEFAULT_MIN_PASSWORD_LENGTH
            )
            min_sk_len = getattr(
                settings, "MIN_SECRET_KEY_LENGTH", _DEFAULT_MIN_SECRET_KEY_LENGTH
            )

            # Admin credentials
            admin_pw = (
                settings.admin_password.get_secret_value()
                if settings.admin_password
                else ""
            )
            if not admin_pw or len(admin_pw.strip()) < min_pw_len:
                errors.append(
                    f"ADMIN_PASSWORD must be set (>= {min_pw_len} chars) in production"
                )
            elif admin_pw.strip().lower() in {"change_me_please", "admin", "password"}:
                errors.append("Insecure admin password in production")

            # JWT secret key
            secret = settings.secret_key.get_secret_value()
            if (
                not secret
                or len(secret.strip()) < min_sk_len
                or ("CHANGE_ME" in secret.upper())
            ):
                errors.append(
                    f"SECRET_KEY must be set (>= {min_sk_len} chars) in production"
                )

            # Debug must remain disabled
            if getattr(settings, "debug", False):
                errors.append("Debug mode must be disabled in production")

            # CORS hardening
            if any(origin.strip() == "*" for origin in settings.cors_allowed_origins):
                errors.append("Wildcard CORS origins not allowed in production")

            # LLM configuration hardening
            llm_key = getattr(settings, "llm_api_key", None)
            if llm_key is not None:
                try:
                    if llm_key.get_secret_value() == "dummy_key_for_development":
                        errors.append("Invalid LLM API key in production")
                except Exception:
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
        templates_dir = settings.base_dir / "templates"

        if not templates_dir.exists():
            logger.warning("templates_directory_not_found", path=str(templates_dir))
            return

        # Create Jinja2 environment with security settings
        self.template_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(
                enabled_extensions=("html", "xml"),
                default_for_string=True,
                default=True,
            ),
            auto_reload=settings.is_development,
            cache_size=getattr(
                settings, "JINJA2_TEMPLATE_CACHE_SIZE", _DEFAULT_TEMPLATE_CACHE_SIZE
            )
            if settings.is_production
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

        # 1) Rate limiting (startup-time wiring)
        try:
            from resync.core.security.rate_limiter_v2 import setup_rate_limiting

            setup_rate_limiting(self.app)
        except Exception as e:
            # Rate limiting must not silently fail open in production.
            if settings.is_production:
                logger.critical("rate_limiting_setup_failed_prod", error=str(e))
                raise
            logger.warning("rate_limiting_setup_failed", error=str(e))

        # 2) CORS configuration (delegate to Starlette CORSMiddleware)
        cors_config = CORSConfig()
        cors_policy = cors_config.get_policy(settings.environment.value)

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
        if settings.is_production:
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

        # Essential routers - must load (fail-fast)
        essential_routers = [
            ("resync.api.routes.core.health", "router", "health_router"),
            ("resync.api.routes.admin.main", "admin_router", "admin_router"),
            ("resync.api.agents", "agents_router", "agents_router"),
            ("resync.api.chat", "chat_router", "chat_router"),
        ]

        for module_path, router_name, log_name in essential_routers:
            try:
                mod = __import__(module_path, fromlist=[router_name])
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
                mod = __import__(module_path, fromlist=[router_name])
                router = getattr(mod, router_name)
                self.app.include_router(router)
            except ImportError as e:
                if settings.is_development:
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
            from resync.api.routes.endpoints import router as api_router
            from resync.api.routes.core.health import config_router
            from resync.api.routes.rag.upload import router as rag_upload_router

            self.app.include_router(api_router, prefix="/api")
            self.app.include_router(config_router, prefix="/api/v1")
            self.app.include_router(rag_upload_router, prefix="/api/v1")
        except ImportError as e:
            if settings.is_development:
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
            if settings.is_development:
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
            if settings.is_development:
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
            if settings.is_development:
                logger.error("document_kg_admin_not_available", error=str(e))
                raise
            logger.warning("document_kg_admin_not_available", error=str(e))

        # Register unified config API (v5.9.9)
        try:
            from resync.api.unified_config_api import router as unified_config_router

            self.app.include_router(unified_config_router)
            logger.info(
                "unified_config_endpoints_registered", prefix="/api/admin/config"
            )
        except ImportError as e:
            if settings.is_development:
                logger.error("unified_config_api_not_available", error=str(e))
                raise
            logger.warning("unified_config_api_not_available", error=str(e))

        # Register monitoring routers
        try:
            from resync.api.routes.monitoring.routes import monitoring_router
            from resync.core.monitoring_integration import register_dashboard_route

            self.app.include_router(monitoring_router, tags=["Monitoring"])
            register_dashboard_route(self.app)
            logger.info("monitoring_routers_registered")
        except ImportError as e:
            if settings.is_development:
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
            from resync.api.routes.admin.rag_reranker import (
                router as rag_reranker_router,
            )
            from resync.api.routes.admin.semantic_cache import (
                router as semantic_cache_router,
            )
            from resync.api.routes.admin.settings_manager import (
                router as settings_manager_router,
            )
            from resync.api.routes.admin.teams import router as teams_router
            from resync.api.routes.admin.teams_webhook_admin import (
                router as teams_webhook_admin_router,
            )
            from resync.api.routes.admin.teams_notifications_admin import (
                router as teams_notifications_admin_router,
            )
            from resync.api.routes.admin.threshold_tuning import (
                router as threshold_tuning_router,
            )
            from resync.api.routes.admin.tws_instances import (
                router as tws_instances_router,
            )
            from resync.api.routes.admin.users import router as admin_users_router
            from resync.api.routes.admin.v2 import router as admin_v2_router
            from resync.api.routes.admin.skills import router as skills_router

            # API Key Management
            from resync.api.v1.admin import admin_api_keys_router

            # Teams webhook public endpoint
            from resync.api.routes.teams_webhook import router as teams_webhook_router

            # Other routes (migrated from fastapi_app)
            from resync.api.routes.core.status import router as status_router
            from resync.api.routes.monitoring.admin_monitoring import (
                router as admin_monitoring_router,
            )

            # Monitoring routes (migrated from fastapi_app)
            from resync.api.routes.monitoring.ai_monitoring import (
                router as ai_monitoring_router,
            )
            from resync.api.routes.monitoring.metrics_dashboard import (
                router as metrics_dashboard_router,
            )
            from resync.api.routes.monitoring.observability import (
                router as observability_router,
            )

            # learning_router removed in v5.9.3 (drift/eval features unused)
            from resync.api.routes.rag.query import router as rag_query_router
            from resync.api.routes.knowledge.ingest_api import (
                router as knowledge_ingest_router,
            )

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
            from resync.api.routes.core.auth import verify_admin_credentials as _verify_admin

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
            if settings.is_development:
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
        static_dir = settings.base_dir / "static"

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

        # Root redirect
        # Root serves Operator Chat
        @self.app.get("/", include_in_schema=False, response_class=HTMLResponse)
        async def root(request: Request):
            """Serve Operator Chat interface."""
            # Reuse _render_template for CSP nonce injection
            return self._render_template("chat.html", request)

        # Admin panel is now handled by the admin router

        # Revision page
        @self.app.get("/revisao", include_in_schema=False, response_class=HTMLResponse)
        async def revisao_page(request: Request):
            """Serve the revision page."""
            return self._render_template("revisao.html", request)

        # CSP violation report endpoint (with rate limiting to prevent DoS from browser extensions)
        from resync.core.security.rate_limiter_v2 import rate_limit

        @self.app.post("/csp-violation-report", include_in_schema=False)
        @rate_limit("30/minute")
        async def csp_violation_report(request: Request):
            """Handle CSP violation reports with payload size limit."""
            # Limit payload size to prevent DoS
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 4096:
                return JSONResponse(
                    {"status": "ignored", "reason": "payload_too_large"},
                    status_code=413,
                )
            return await self._handle_csp_report(request)

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

        # Hardening: fail-closed in production if nonce missing
        nonce = getattr(request.state, "csp_nonce", None)

        if settings.is_production and not nonce:
            logger.critical("csp_nonce_missing_prod", template=template_name)
            raise HTTPException(status_code=500, detail="Security middleware failed")

        try:
            nonce_value = nonce or ""
            context = {
                "request": request,
                "nonce": nonce_value,
                "csp_nonce": nonce_value,  # backward compatible
                "settings": {
                    "project_name": settings.project_name,
                    "version": settings.project_version,
                    # Security: don't expose environment to prevent fingerprinting
                },
            }
            return self.templates.TemplateResponse(template_name, context)
        except FileNotFoundError:
            logger.error("template_not_found", template=template_name)
            raise HTTPException(
                status_code=404, detail=f"Template {template_name} not found"
            ) from None
        except Exception as e:
            logger.error("template_render_error", template=template_name, error=str(e))
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from None

    async def _handle_csp_report(self, request: Request) -> JSONResponse:
        """
        Handle CSP violation reports with validation.

        Args:
            request: FastAPI request containing CSP report

        Returns:
            JSON response acknowledging receipt
        """
        try:
            from resync.csp_validation import process_csp_report

            result = await process_csp_report(request)

            # Log violation details
            report = result.get("report", {})
            csp_report = (
                report.get("csp-report", report) if isinstance(report, dict) else report
            )

            logger.warning(
                "csp_violation_reported",
                client_host=request.client.host if request.client else "unknown",
                blocked_uri=csp_report.get("blocked-uri", "unknown"),
                violated_directive=csp_report.get("violated-directive", "unknown"),
                effective_directive=csp_report.get("effective-directive", "unknown"),
            )

            return JSONResponse(content={"status": "received"}, status_code=200)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "csp_report_error",
                error_type=type(e).__name__,
                client_host=request.client.host if request.client else "unknown",
            )
            # Always return 200 to prevent information leakage
            return JSONResponse(content={"status": "received"}, status_code=200)


def create_app() -> FastAPI:
    """Entry point para Uvicorn e Pytest.

    Creates a new ApplicationFactory instance per call to avoid
    state contamination between tests.

    Returns:
        Configured FastAPI application
    """
    return ApplicationFactory().create_application()
