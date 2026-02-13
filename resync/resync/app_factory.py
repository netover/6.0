"""
Application factory for creating and configuring the FastAPI application.

This module is imported by resync/main.py and provides the actual application
initialization logic following the factory pattern.

Global State:
    This module creates a module-level ``_factory`` singleton (line ~end) used
    by ``create_app()`` for ASGI server compatibility.  The ``ApplicationFactory``
    class itself holds no mutable state after ``create_application()`` returns;
    all runtime state lives on ``app.state``.
"""

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from resync.core.types.app_state import enterprise_state_from_app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

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
                full_path = Path(self.directory) / path
                
                # Use run_in_executor to avoid blocking the event loop on file I/O
                loop = asyncio.get_running_loop()
                
                if await loop.run_in_executor(None, full_path.exists):
                    stat_result = await loop.run_in_executor(None, os.stat, full_path)
                    
                    # ETag based on size + mtime (standard practice, fast)
                    file_metadata = f"{stat_result.st_size}-{int(stat_result.st_mtime)}"
                    hash_len = getattr(
                        settings, "ETAG_HASH_LENGTH", _DEFAULT_ETAG_HASH_LENGTH
                    )
                    
                    # Calculating hash of a short string is CPU-bound but very fast, 
                    # acceptable in async loop for short strings.
                    digest = hashlib.sha256(file_metadata.encode()).hexdigest()[:hash_len]
                    response.headers["ETag"] = f'"{digest}"'
            except Exception as exc:
                logger.warning("failed_to_generate_etag", error=str(exc))
                # Fallback to simple hash of path if metadata fails
                response.headers["ETag"] = f'"{hash(path)}"'

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

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle with proper startup and shutdown.

        **Startup** (before ``yield``):
            1. Initialises domain singletons on ``app.state`` (TWS, Agent, KG).
            2. Starts TWS monitor and wires DI container.
            3. Connects to Redis with retry.
            4. Launches optional services (monitoring, metrics, cache warming,
               GraphRAG, unified config) — each is isolated so a failure in one
               does not block the others.
            5. Sets ``app.state.startup_complete = True``.

        **Shutdown** (after ``yield``):
            Tears down services in reverse order: monitoring → health → background
            tasks → config system → domain singletons → TWS monitor.

        Side effects:
            - Modifies ``app.state`` (templates, singletons, startup_complete).
            - Creates asyncio background tasks (metrics collector).
            - Initialises global singletons (Redis, GraphRAG, config system).

        Args:
            app: FastAPI application instance.

        Yields:
            None during application runtime.
        """
        logger.info("application_startup_initiated")

        try:
            # Import here to avoid circular dependencies at module load time.
            # These modules depend on resync.settings which in turn may import
            # from modules that reference app_factory — moving them to module
            # scope would create an import cycle.
            from resync.core.exceptions import (
                ConfigurationError,
                RedisAuthError,
                RedisConnectionError,
                RedisInitializationError,
                RedisTimeoutError,
            )
            from resync.core.interfaces import (
                IAgentManager,
                IKnowledgeGraph,
                ITWSClient,
            )
            from resync.core.tws_monitor import get_tws_monitor, shutdown_tws_monitor
            from resync.api_gateway.container import setup_dependencies
            from resync.core.startup import enforce_startup_policy, run_startup_checks
            from resync.core.wiring import (
                init_domain_singletons,
            )

            # -----------------------------------------------------------------
            # Canonical startup validation/health checks
            # -----------------------------------------------------------------
            startup_result = await run_startup_checks()
            if not startup_result.get("overall_health"):
                app_logger.warning(
                    "startup_health_failed",
                    critical_services=startup_result.get("critical_services"),
                    results=startup_result.get("results"),
                    strict=startup_result.get("strict"),
                )
            enforce_startup_policy(startup_result)

            # Core initialisation (failures here are fatal)
            init_domain_singletons(app)
            st = enterprise_state_from_app(app)
            tws_client = st.tws_client
            agent_manager = st.agent_manager
            knowledge_graph = st.knowledge_graph

            await get_tws_monitor(tws_client)
            setup_dependencies(tws_client, agent_manager, knowledge_graph)

            app_logger.info("core_services_initialized")

            # Optional services (failures are non-fatal)
            # Parallelize initialization to reduce startup time
            import asyncio
            await asyncio.gather(
                self._init_proactive_monitoring(app),
                self._init_metrics_collector(),
                self._init_cache_warming(),
                self._init_graphrag(),
                self._init_config_system(),
                return_exceptions=True  # Ensure one failure doesn't stop others
            )

            logger.info("application_startup_completed")
            enterprise_state_from_app(app).startup_complete = True

            # Record startup time for admin dashboard uptime display
            from resync.core.startup_time import set_startup_time
            set_startup_time()

            yield  # Application runs here

        except ConfigurationError as exc:
            app_logger.error(
                "configuration_error",
                error_message=str(exc),
                error_details=exc.details,
                hint=exc.details.get("hint"),
            )
            raise
        except RedisAuthError as exc:
            app_logger.error(
                "redis_authentication_error",
                error_message=str(exc),
                error_details=exc.details,
                hint=exc.details.get("hint"),
                example_redis_url="redis://:yourpassword@localhost:6379",
            )
            raise
        except RedisConnectionError as exc:
            app_logger.error(
                "redis_connection_error",
                error_message=str(exc),
                error_details=exc.details,
                hint=exc.details.get("hint"),
                installation_guide={
                    "macos": "brew install redis",
                    "linux": "apt install redis",
                    "start_command": "redis-server",
                    "test_command": "redis-cli ping (should return 'PONG')",
                },
            )
            raise
        except RedisTimeoutError as exc:
            app_logger.error(
                "redis_timeout_error",
                error_message=str(exc),
                error_details=exc.details,
                hint=exc.details.get("hint"),
            )
            raise
        except RedisInitializationError as exc:
            app_logger.error(
                "redis_initialization_error",
                error_message=str(exc),
                error_details=exc.details,
                hint=exc.details.get("hint"),
            )
            raise
        except Exception as exc:
            logger.critical("application_startup_failed", error=str(exc))
            raise
        finally:
            await self._shutdown_services(app)

    # -----------------------------------------------------------------
    # Lifespan helpers: optional startup services
    # -----------------------------------------------------------------

    @staticmethod
    async def _init_proactive_monitoring(app: FastAPI) -> None:
        """Initialise proactive monitoring (non-fatal on failure)."""
        try:
            from resync.core.monitoring_integration import initialize_proactive_monitoring

            await initialize_proactive_monitoring(app)
            app_logger.info("proactive_monitoring_initialized")
        except ImportError as exc:
            app_logger.info("proactive_monitoring_not_installed", module=str(exc))
        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            app_logger.warning(
                "proactive_monitoring_init_failed",
                error=str(exc),
                hint="Monitoring will be unavailable but app will continue",
                exc_info=True,
            )

    @staticmethod
    async def _init_metrics_collector() -> None:
        """Start the dashboard metrics collector background task."""
        try:
            from resync.api import monitoring_dashboard
            from resync.core.task_tracker import create_tracked_task


            monitoring_dashboard._collector_task = await create_tracked_task(
                monitoring_dashboard.metrics_collector_loop(),
                name="metrics-collector",
            )
            app_logger.info("dashboard_metrics_collector_started")
        except ImportError as exc:
            app_logger.info("metrics_collector_not_installed", module=str(exc))
        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            app_logger.warning(
                "metrics_collector_start_failed",
                error=str(exc),
                hint="Dashboard metrics will be unavailable",
                exc_info=True,
            )

    @staticmethod
    async def _init_cache_warming() -> None:
        """Warm application caches on startup."""
        try:
            from resync.core.cache_utils import warmup_cache_on_startup

            await warmup_cache_on_startup()
            app_logger.info("cache_warming_completed")
        except ImportError as exc:
            app_logger.info("cache_warming_not_installed", module=str(exc))
        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            app_logger.warning(
                "cache_warming_failed",
                error=str(exc),
                hint="Cache will start cold but will warm naturally",
                exc_info=True,
            )

    @staticmethod
    async def _init_graphrag() -> None:
        """Initialise GraphRAG (subgraph retrieval + auto-discovery)."""
        try:
            from resync.core.graphrag_integration import initialize_graphrag
            from resync.core.redis_init import get_redis_client
            from resync.knowledge.retrieval.graph import get_knowledge_graph
            from resync.services.llm_service import get_llm_service
            from resync.services.tws_service import get_tws_client

            if not getattr(settings, "GRAPHRAG_ENABLED", False):
                app_logger.info("graphrag_disabled_by_config")
                return

            llm_service = get_llm_service()
            kg = get_knowledge_graph()
            tws_client = get_tws_client()

            try:
                redis_client = get_redis_client()
            except Exception:
                redis_client = None
                app_logger.warning("graphrag_running_without_redis_cache")

            initialize_graphrag(
                llm_service=llm_service,
                knowledge_graph=kg,
                tws_client=tws_client,
                redis_client=redis_client,
                enabled=True,
            )
            app_logger.info(
                "graphrag_initialized",
                features=["subgraph_retrieval", "auto_discovery"],
            )
        except ImportError as exc:
            app_logger.info("graphrag_not_installed", module=str(exc))
        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            app_logger.warning(
                "graphrag_initialization_failed",
                error=str(exc),
                hint="GraphRAG features will be disabled, but system will continue normally",
            )

    @staticmethod
    async def _init_config_system() -> None:
        """Initialise unified config system with hot reload."""
        try:
            from resync.core.unified_config import initialize_config_system

            await initialize_config_system()
            logger.info(
                "unified_config_initialized",
                hot_reload=True,
                message="All configs loaded with hot reload support",
            )
        except ImportError as exc:
            logger.info("unified_config_not_installed", module=str(exc))
        except Exception as exc:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "unified_config_initialization_failed",
                error=str(exc),
                hint="Config hot reload disabled, but system will use static configs",
            )

    # -----------------------------------------------------------------
    # Lifespan helper: shutdown
    # -----------------------------------------------------------------

    @staticmethod
    async def _shutdown_services(app: FastAPI) -> None:
        """Tear down all services in reverse initialisation order.

        Each shutdown step is isolated: a failure in one does not prevent
        the others from running.  Critical services (domain_singletons)
        log at ``error`` level; optional services log at ``warning``.
        """
        app_logger.info("application_shutdown_initiated")

        try:
            # 1. Proactive monitoring (optional)
            try:
                from resync.core.monitoring_integration import shutdown_proactive_monitoring

                await shutdown_proactive_monitoring(app)
                app_logger.info("proactive_monitoring_shutdown_successful")
            except Exception as exc:
                app_logger.warning("proactive_monitoring_shutdown_error", error=str(exc))

            # 2. Health check service (optional)
            try:
                from resync.core.health import shutdown_health_check_service

                await shutdown_health_check_service()
                app_logger.info("health_service_shutdown_successful")
            except Exception as exc:
                app_logger.warning("health_service_shutdown_error", error=str(exc))

            # 3. Background tasks (important — prevent task leaks)
            try:
                from resync.core.task_tracker import cancel_all_tasks

                cancel_timeout = getattr(
                    settings,
                    "SHUTDOWN_TASK_CANCEL_TIMEOUT",
                    _DEFAULT_TASK_CANCEL_TIMEOUT,
                )
                stats = await cancel_all_tasks(timeout=cancel_timeout)
                app_logger.info(
                    "background_tasks_cancelled",
                    total=stats["total"],
                    cancelled=stats["cancelled"],
                    completed=stats.get("completed", 0),
                )
            except Exception as exc:
                app_logger.warning("background_tasks_cancel_error", error=str(exc))

            # 4. Unified config system (optional)
            try:
                from resync.core.unified_config import shutdown_config_system

                shutdown_config_system()
                app_logger.info("unified_config_shutdown_successful")
            except Exception as exc:
                app_logger.warning("unified_config_shutdown_error", error=str(exc))

            # 5. Domain singletons (critical — resource cleanup)
            try:
                from resync.core.wiring import shutdown_domain_singletons

                await shutdown_domain_singletons(app)
                app_logger.info("domain_singletons_shutdown_successful")
            except Exception as exc:
                app_logger.error("domain_singletons_shutdown_error", error=str(exc))

            # 6. TWS monitor
            try:
                from resync.core.tws_monitor import shutdown_tws_monitor

                await shutdown_tws_monitor()
            except Exception as exc:
                app_logger.warning("tws_monitor_shutdown_error", error=str(exc))

            logger.info("application_shutdown_completed")
        except Exception as exc:
            logger.error("application_shutdown_error", error=str(exc))

    def create_application(self) -> FastAPI:
        """
        Create and configure the FastAPI application.

        Returns:
            Fully configured FastAPI application instance
        """
        # Validate settings first
        self._validate_critical_settings()

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=settings.project_name,
            version=settings.project_version,
            description=settings.description,
            lifespan=self.lifespan,
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
                settings.admin_password.get_secret_value() if settings.admin_password else ""
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
                or secret.startswith("CHANGE_ME_IN_PRODUCTION")
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
            if getattr(settings, "llm_api_key", "") == "dummy_key_for_development":
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
            ) if settings.is_production else 0,
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
        """Configure all middleware in the correct order."""
        from resync.api.middleware.correlation_id import CorrelationIdMiddleware
        from resync.api.middleware.cors_config import CORSConfig
        # CSP is enforced only in production. In non-production environments
        # we avoid adding CSP middleware entirely to prevent interfering with
        # developer tooling (e.g. Swagger UI) and to keep behavior aligned with
        # deployments where CSP is production-only.
        # NOTE: Global exception handling is implemented via FastAPI exception
        # handlers (see ``_configure_exception_handlers``). Avoid adding a
        # catch-all middleware here to prevent double handling and unpredictable
        # precedence.

        # 1. Correlation ID (must be first)
        self.app.add_middleware(CorrelationIdMiddleware)

        # 1.5. Rate Limiting (after Correlation ID, before Exception Handler)
        # Enable whenever rate limiting is configured/enabled. The rate limiter
        # implementation internally decides whether to activate (e.g. disabled
        # in dev/staging by env), so we don't hard-code production-only here.
        try:
            # NOTE: do not create Redis connections during module import (gunicorn --preload).
            # slowapi will use storage_uri from env (REDIS_URL/RATE_LIMIT_REDIS_URL) if provided.
            from resync.core.security.rate_limiter_v2 import setup_rate_limiting

            setup_rate_limiting(self.app)
        except Exception as e:
            # Do not block startup if rate limiting setup fails
            logger.warning(
                "rate_limiting_setup_failed",
                error=str(e),
            )

        # 2. Global exception handling is provided by registered exception
        # handlers (FastAPI/Starlette) rather than a catch-all middleware.

        # 3. CORS Configuration
        from resync.api.middleware.cors_middleware import LoggingCORSMiddleware

        cors_config = CORSConfig()
        cors_policy = cors_config.get_policy(settings.environment.value)

        self.app.add_middleware(
            LoggingCORSMiddleware,
            policy=cors_policy,
            allow_origins=cors_policy.allowed_origins,
            allow_methods=cors_policy.allowed_methods,
            allow_headers=cors_policy.allowed_headers,
            allow_credentials=cors_policy.allow_credentials,
            max_age=cors_policy.max_age,
        )

        # 4. CSP Middleware (production-only)
        if settings.is_production:
            from resync.api.middleware.csp_middleware import CSPMiddleware

            self.app.add_middleware(CSPMiddleware, report_only=False)

        # 5. Additional security headers
        from resync.config.security import add_additional_security_headers

        add_additional_security_headers(self.app)

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
        Register all API routers.

        v5.8.0: Unified API structure - all routes under resync/api/routes/
        """
        # =================================================================
        # CORE ROUTERS (from resync/api/ - backward compatible)
        # =================================================================
        from resync.api.admin import admin_router
        from resync.api.routes.admin.prompts import prompt_router
        from resync.api.agents import agents_router
        from resync.api.audit import router as audit_router
        from resync.api.cache import cache_router
        from resync.api.chat import chat_router
        from resync.api.cors_monitoring import cors_monitor_router
        from resync.api.health import router as health_router
        from resync.api.performance import performance_router

        # Additional routers from main_improved
        try:
            from resync.api.endpoints import router as api_router
            from resync.api.health import config_router
            from resync.api.routes.rag.upload import router as rag_upload_router

            self.app.include_router(api_router, prefix="/api")
            self.app.include_router(config_router, prefix="/api/v1")
            self.app.include_router(rag_upload_router, prefix="/api/v1")
        except ImportError as e:
            logger.warning("optional_routers_not_available", error=str(e))

        # v5.9.9: Enhanced endpoints (orchestrator-based)
        try:
            from resync.api.enhanced_endpoints import enhanced_router

            self.app.include_router(enhanced_router)
            logger.info("enhanced_endpoints_registered", prefix="/api/v2")
        except ImportError as e:
            logger.warning("enhanced_endpoints_not_available", error=str(e))

        # v5.9.9: GraphRAG admin endpoints
        try:
            from resync.api.graphrag_admin import router as graphrag_admin_router

            self.app.include_router(graphrag_admin_router)
            logger.info("graphrag_admin_endpoints_registered", prefix="/api/admin/graphrag")
        except ImportError as e:
            logger.warning("graphrag_admin_not_available", error=str(e))

        # v6.1: Document Knowledge Graph (DKG) admin endpoints
        try:
            from resync.api.document_kg_admin import router as dkg_admin_router

            self.app.include_router(dkg_admin_router)
            logger.info("document_kg_admin_endpoints_registered", prefix="/api/admin/kg")
        except ImportError as e:
            logger.warning("document_kg_admin_not_available", error=str(e))

        # Register unified config API (v5.9.9)
        try:
            from resync.api.unified_config_api import router as unified_config_router

            self.app.include_router(unified_config_router)
            logger.info("unified_config_endpoints_registered", prefix="/api/admin/config")
        except ImportError as e:
            logger.warning("unified_config_api_not_available", error=str(e))

        # Register monitoring routers
        try:
            from resync.api.routes.monitoring.routes import monitoring_router
            from resync.core.monitoring_integration import register_dashboard_route

            self.app.include_router(monitoring_router, tags=["Monitoring"])
            register_dashboard_route(self.app)
            logger.info("monitoring_routers_registered")
        except ImportError as e:
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
            from resync.api.routes.admin.rag_reranker import router as rag_reranker_router
            from resync.api.routes.admin.semantic_cache import router as semantic_cache_router
            from resync.api.routes.admin.settings_manager import router as settings_manager_router
            from resync.api.routes.admin.teams import router as teams_router
            from resync.api.routes.admin.teams_webhook_admin import router as teams_webhook_admin_router
            from resync.api.routes.admin.teams_notifications_admin import router as teams_notifications_admin_router
            from resync.api.routes.admin.threshold_tuning import router as threshold_tuning_router
            from resync.api.routes.admin.tws_instances import router as tws_instances_router
            from resync.api.routes.admin.users import router as admin_users_router
            from resync.api.routes.admin.v2 import router as admin_v2_router

            # Teams webhook public endpoint
            from resync.api.routes.teams_webhook import router as teams_webhook_router

            # Other routes (migrated from fastapi_app)
            from resync.api.routes.core.status import router as status_router
            from resync.api.routes.monitoring.admin_monitoring import (
                router as admin_monitoring_router,
            )

            # Monitoring routes (migrated from fastapi_app)
            from resync.api.routes.monitoring.ai_monitoring import router as ai_monitoring_router
            from resync.api.routes.monitoring.metrics_dashboard import (
                router as metrics_dashboard_router,
            )
            from resync.api.routes.monitoring.observability import router as observability_router

            # learning_router removed in v5.9.3 (drift/eval features unused)
            from resync.api.routes.rag.query import router as rag_query_router
            from resync.api.routes.knowledge.ingest_api import router as knowledge_ingest_router

            # Register unified admin routes
            unified_admin_routers = [
                (backup_router, "/api/v1/admin", ["Admin - Backup"]),
                (admin_config_router, "/api/v1/admin", ["Admin - Config"]),
                (settings_manager_router, "/api/v1/admin", ["Admin - Settings Manager"]),
                (teams_router, "/api/v1/admin", ["Admin - Teams"]),
                (teams_webhook_admin_router, "/api", ["Admin - Teams Webhook Users"]),
                (teams_notifications_admin_router, "/api", ["Admin - Teams Notifications"]),
                (tws_instances_router, "/api/v1/admin", ["Admin - TWS Instances"]),
                (admin_users_router, "/api/v1/admin", ["Admin - Users"]),
                (semantic_cache_router, "/api/v1/admin", ["Admin - Semantic Cache"]),
                (rag_reranker_router, "/api/v1", ["Admin - RAG Reranker"]),
                (threshold_tuning_router, "/api/v1/admin", ["Admin - Threshold Tuning"]),
                (environment_router, "/api/v1/admin", ["Admin - Environment"]),
                (connectors_router, "/api/v1/admin", ["Admin - Connectors"]),
                (feedback_curation_router, "", ["Admin - Feedback Curation"]),
                (admin_v2_router, "/api/v2/admin", ["Admin V2"]),
            ]

            # Register unified monitoring routes (with admin auth)
            from resync.api.auth import verify_admin_credentials as _verify_admin

            unified_monitoring_routers = [
                (ai_monitoring_router, "/api/v1/monitoring", ["Monitoring - AI"]),
                (observability_router, "/api/v1/monitoring", ["Monitoring - Observability"]),
                (metrics_dashboard_router, "/api/v1/monitoring", ["Monitoring - Metrics"]),
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
                    router, prefix=prefix, tags=tags,
                    dependencies=[Depends(_verify_admin)],
                )

            logger.info(
                "unified_routers_registered",
                admin_count=len(unified_admin_routers),
                monitoring_count=len(unified_monitoring_routers),
                other_count=len(unified_other_routers),
            )
        except ImportError as e:
            logger.warning("unified_routers_not_available", error=str(e))

        # =================================================================
        # LEGACY CORE ROUTERS (backward compatible)
        # =================================================================
        routers = [
            (health_router, "/api/v1", ["Health"]),
            (agents_router, "/api/v1/agents", ["Agents"]),
            (chat_router, "/api/v1", ["Chat"]),
            (cache_router, "/api/v1", ["Cache"]),
            (audit_router, "/api/v1", ["Audit"]),
            (cors_monitor_router, "/api/v1", ["CORS"]),
            (performance_router, "/api", ["Performance"]),
            (admin_router, "/api/v1", ["Admin"]),
            (prompt_router, "/api/v1", ["Admin - Prompts"]),
        ]

        for router, prefix, tags in routers:
            self.app.include_router(router, prefix=prefix, tags=tags)

        logger.info("routers_registered", count=len(routers))

    def _mount_static_files(self) -> None:
        """Mount static file directories with caching."""
        static_dir = settings.base_dir / "static"

        if not static_dir.exists():
            logger.warning("static_directory_not_found", path=str(static_dir))
            return

        # Mount main static directory with caching
        self.app.mount("/static", CachedStaticFiles(directory=str(static_dir)), name="static")

        # Mount subdirectories if they exist
        subdirs = ["assets", "css", "js", "img", "fonts"]
        mounted = 1

        for subdir in subdirs:
            subdir_path = static_dir / subdir
            if subdir_path.exists():
                self.app.mount(
                    f"/{subdir}",
                    CachedStaticFiles(directory=str(subdir_path)),
                    name=subdir,
                )
                mounted += 1

        logger.info("static_files_mounted", count=mounted, directory=str(static_dir))

    def _register_special_endpoints(self) -> None:
        """Register special endpoints (frontend, CSP, etc.)."""

        # Root redirect
        @self.app.get("/", include_in_schema=False)
        def root():
            """Redirect root to admin panel."""
            return RedirectResponse(url="/admin", status_code=302)

        # Admin panel is now handled by the admin router

        # Revision page
        @self.app.get("/revisao", include_in_schema=False, response_class=HTMLResponse)
        async def revisao_page(request: Request):
            """Serve the revision page."""
            return self._render_template("revisao.html", request)

        # CSP violation report endpoint
        @self.app.post("/csp-violation-report", include_in_schema=False)
        async def csp_violation_report(request: Request):
            """Handle CSP violation reports."""
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
        """
        if not self.templates:
            raise HTTPException(status_code=500, detail="Template engine not configured")

        try:
            from resync.core.csp_template_response import CSPTemplateResponse

            nonce = getattr(request.state, "csp_nonce", "")
            return CSPTemplateResponse(
                template_name,
                {
                    "request": request,
                    "nonce": nonce,
                    "settings": {
                        "project_name": settings.project_name,
                        "version": settings.project_version,
                        "environment": settings.environment.value,
                    },
                },
                self.templates,
            )
        except FileNotFoundError:
            logger.error("template_not_found", template=template_name)
            raise HTTPException(
                status_code=404, detail=f"Template {template_name} not found"
            ) from None
        except Exception as e:
            logger.error("template_render_error", template=template_name, error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error") from None

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
            csp_report = report.get("csp-report", report) if isinstance(report, dict) else report

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


# Module-level factory instance
_factory = ApplicationFactory()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the main entry point for application creation,
    providing a fully configured FastAPI instance.

    Returns:
        Configured FastAPI application
    """
    return _factory.create_application()