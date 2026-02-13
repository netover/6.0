"""
Unified API Routes for Resync v5.8.0

All routes organized by domain:
- core/: Essential routes (health, auth, chat, status)
- admin/: Administration routes
- monitoring/: Monitoring and observability routes
- agents/: Agent routes
- rag/: RAG routes
- learning/: Learning routes
- enterprise/: Enterprise routes
- system/: System configuration routes
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

__all__ = [
    "get_all_routers",
]


def get_all_routers() -> list[tuple["APIRouter", str, list[str]]]:
    """Get all routers with their prefixes and tags."""
    from .admin.main import admin_router
    from .admin.prompts import prompt_router
    from .agents.agents import router as agents_router
    from .audit import router as audit_router
    from .cache import cache_router
    from .monitoring.dashboard import router as monitoring_dashboard_router
    from .performance import performance_router

    return [
        (health_router, "/api/v1", ["Health"]),
        (auth_router, "/api/v1/auth", ["Auth"]),
        (chat_router, "/api/v1", ["Chat"]),
        (status_router, "/api/v1", ["Status"]),
        (admin_router, "/api/v1", ["Admin"]),
        (admin_router, "", ["Admin"]),
        (prompt_router, "/api/v1", ["Admin - Prompts"]),
        (monitoring_dashboard_router, "/api/v1", ["Monitoring"]),
        (agents_router, "/api/v1/agents", ["Agents"]),
        (cache_router, "/api/v1", ["Cache"]),
        (audit_router, "/api/v1", ["Audit"]),
        (performance_router, "/api", ["Performance"]),
        (cors_monitor_router, "/api/v1", ["CORS"]),
    ]
