# mypy: ignore-errors
"""
Proactive Monitoring System Integration

This module integrates the proactive monitoring system with the FastAPI application,
registering routes, initializing components, and configuring WebSockets.

Author: Resync Team
Version: 5.2
"""

from typing import Any

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


async def initialize_proactive_monitoring(app: FastAPI) -> None:
    """
    Initialize the proactive monitoring system.

    This function should be called during application startup.

    Args:
        app: FastAPI application instance
    """
    from resync.settings import settings

    # Check if polling is enabled
    if not getattr(settings, "tws_polling_enabled", True):
        logger.info("proactive_monitoring_disabled")
        return

    try:
        logger.info("initializing_proactive_monitoring")

        # 1. Import components
        from resync.core.proactive_monitoring_manager import (
            setup_proactive_monitoring,
        )
        from resync.core.tws_history_rag import init_tws_history_rag

        # 2. Get TWS client
        tws_client = _get_tws_client(app)

        if not tws_client:
            logger.warning("tws_client_not_available_using_mock")
            tws_client = _create_mock_tws_client()

        # 3. Prepare configuration
        monitoring_config = {
            "polling_interval_seconds": getattr(
                settings, "tws_polling_interval_seconds", 30
            ),
            "polling_mode": getattr(settings, "tws_polling_mode", "fixed"),
            "job_stuck_threshold_minutes": getattr(
                settings, "tws_job_stuck_threshold_minutes", 60
            ),
            "job_late_threshold_minutes": getattr(
                settings, "tws_job_late_threshold_minutes", 30
            ),
            "anomaly_failure_rate_threshold": getattr(
                settings, "tws_anomaly_failure_rate_threshold", 0.1
            ),
            "retention_days_full": getattr(settings, "tws_retention_days_full", 7),
            "retention_days_summary": getattr(
                settings, "tws_retention_days_summary", 30
            ),
            "retention_days_patterns": getattr(
                settings, "tws_retention_days_patterns", 90
            ),
            "pattern_detection_enabled": getattr(
                settings, "tws_pattern_detection_enabled", True
            ),
            "pattern_detection_interval_minutes": getattr(
                settings, "tws_pattern_detection_interval_minutes", 60
            ),
            "pattern_min_confidence": getattr(
                settings, "tws_pattern_min_confidence", 0.5
            ),
            "solution_correlation_enabled": getattr(
                settings, "tws_solution_correlation_enabled", True
            ),
            "solution_min_success_rate": getattr(
                settings, "tws_solution_min_success_rate", 0.6
            ),
            "browser_notifications_enabled": getattr(
                settings, "tws_browser_notifications_enabled", True
            ),
            "teams_notifications_enabled": getattr(
                settings, "tws_teams_notifications_enabled", False
            ),
            "teams_webhook_url": getattr(settings, "tws_teams_webhook_url", None),
            "dashboard_theme": getattr(settings, "tws_dashboard_theme", "auto"),
            "dashboard_refresh_seconds": getattr(
                settings, "tws_dashboard_refresh_seconds", 5
            ),
        }

        # 4. Initialize monitoring system
        manager = await setup_proactive_monitoring(
            tws_client=tws_client,
            config=monitoring_config,
            auto_start=True,
        )

        # 5. Initialize history RAG
        if manager and manager.status_store:
            init_tws_history_rag(
                status_store=manager.status_store,
                llm_client=_get_llm_client(app),
            )

        # 6. Store reference in app state
        app.state.monitoring_manager = manager

        logger.info(
            "proactive_monitoring_initialized",
            polling_interval=monitoring_config["polling_interval_seconds"],
            pattern_detection=monitoring_config["pattern_detection_enabled"],
        )

    except Exception as e:
        logger.error("proactive_monitoring_initialization_failed", error=str(e))
        # Do not fail startup, just log the error


async def shutdown_proactive_monitoring(app: FastAPI) -> None:
    """
    Shut down the proactive monitoring system.

    This function should be called during application shutdown.

    Args:
        app: FastAPI application instance
    """
    try:
        from resync.core.proactive_monitoring_manager import (
            shutdown_proactive_monitoring,
        )

        await shutdown_proactive_monitoring()

        if hasattr(app.state, "monitoring_manager"):
            delattr(app.state, "monitoring_manager")

        logger.info("proactive_monitoring_shutdown_complete")

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("proactive_monitoring_shutdown_error", error=str(e))


def register_monitoring_routes(app: FastAPI) -> None:
    """
    Registers the monitoring routes in the application.

    Args:
        app: FastAPI instance
    """
    try:
        from resync.api.routes.monitoring.routes import monitoring_router

        app.include_router(monitoring_router, tags=["Monitoring"])

        logger.info("monitoring_routes_registered")

    except ImportError as e:
        logger.warning("monitoring_routes_not_available", error=str(e))


def register_dashboard_route(app: FastAPI) -> None:
    """
    Registers the route for the real-time monitoring dashboard.

    Args:
        app: FastAPI instance
    """
    from fastapi import Request
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates

    from resync.settings import settings

    templates_dir = settings.base_dir / "templates"

    if not templates_dir.exists():
        logger.warning("templates_directory_not_found")
        return

    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/dashboard/realtime", response_class=HTMLResponse, tags=["Dashboard"])
    def realtime_dashboard(request: Request):
        """TWS real-time monitoring dashboard."""
        from resync.core.monitoring_config import get_monitoring_config

        config = get_monitoring_config()

        return templates.TemplateResponse(
            "realtime_dashboard.html",
            {
                "request": request,
                "config": config.to_frontend_config() if config else {},
            },
        )

    @app.get("/dashboard/tws", response_class=HTMLResponse, tags=["Dashboard"])
    def tws_dashboard(request: Request):
        """Alias for TWS monitoring dashboard."""
        from resync.core.monitoring_config import get_monitoring_config

        config = get_monitoring_config()

        return templates.TemplateResponse(
            "realtime_dashboard.html",
            {
                "request": request,
                "config": config.to_frontend_config() if config else {},
            },
        )

    logger.info("dashboard_routes_registered")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_tws_client(app: "FastAPI") -> Any | None:
    """Get TWS client from the dependency container."""
    try:
        from resync.core.wiring import STATE_TWS_CLIENT

        return getattr(app.state, STATE_TWS_CLIENT)

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("failed_to_get_tws_client", error=str(e))
        return None


def _get_llm_client(app: "FastAPI") -> Any | None:
    """Get LLM client from the dependency container."""
    try:
        from resync.core.wiring import STATE_LLM_SERVICE

        return getattr(app.state, STATE_LLM_SERVICE)

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("failed_to_get_llm_client", error=str(e))
        return None


def _create_mock_tws_client() -> Any:
    """Create a mock TWS client for development."""

    class MockTWSClient:
        """Mock TWS client for development and testing."""

        def query_workstations(self, limit: int = 100) -> dict[str, Any]:
            """Returns mock workstations."""
            import random

            workstations = []
            for i in range(5):
                ws = {
                    "name": f"WS{i + 1:03d}",
                    "status": random.choice(["LINKED", "LINKED", "LINKED", "UNLINKED"]),
                    "agentStatus": "RUNNING",
                    "jobsRunning": random.randint(0, 10),
                    "jobsPending": random.randint(0, 5),
                }
                workstations.append(ws)

            return {"items": workstations}

        def get_plan_jobs(
            self,
            status: list = None,
            limit: int = 500,
        ) -> dict[str, Any]:
            """Returns mock jobs."""
            import random
            from datetime import datetime, timedelta, timezone

            statuses = status or ["EXEC", "READY", "SUCC", "ABEND"]
            jobs = []

            for i in range(random.randint(20, 50)):
                job_status = random.choice(statuses)
                start_time = datetime.now(timezone.utc) - timedelta(
                    minutes=random.randint(5, 120)
                )

                job = {
                    "id": f"job_{i}",
                    "name": f"JOB_{random.choice(['BATCH', 'REPORT', 'BACKUP', 'SYNC'])}_{i:04d}",
                    "jobStream": f"STREAM_{random.randint(1, 5)}",
                    "workstation": f"WS{random.randint(1, 5):03d}",
                    "status": job_status,
                    "returnCode": 0
                    if job_status == "SUCC"
                    else (8 if job_status == "ABEND" else None),
                    "startTime": start_time.isoformat(),
                    "endTime": (
                        start_time + timedelta(minutes=random.randint(1, 30))
                    ).isoformat()
                    if job_status in ["SUCC", "ABEND"]
                    else None,
                    "errorMessage": "Database connection error"
                    if job_status == "ABEND"
                    else None,
                }
                jobs.append(job)

            return {"items": jobs}

    return MockTWSClient()


# =============================================================================
# CONVENIENCE FUNCTION FOR APP FACTORY
# =============================================================================


def setup_monitoring_system(app: FastAPI) -> None:
    """
    Configures the entire monitoring system.

    This is the main function to be called by app_factory.

    Args:
        app: FastAPI instance
    """
    # 1. Register routes
    register_monitoring_routes(app)
    register_dashboard_route(app)

    # 2. Initialize system (will be called in startup)
    # The actual initialization occurs in the lifespan


def get_monitoring_startup_handler(app: FastAPI):
    """
    Returns the startup handler for the monitoring system.

    Args:
        app: FastAPI instance

    Returns:
        Coroutine for initialization
    """

    async def startup():
        await initialize_proactive_monitoring(app)

    return startup


def get_monitoring_shutdown_handler(app: FastAPI):
    """
    Returns the shutdown handler for the monitoring system.

    Args:
        app: FastAPI instance

    Returns:
        Coroutine for finalization
    """

    async def shutdown():
        await shutdown_proactive_monitoring(app)

    return shutdown
