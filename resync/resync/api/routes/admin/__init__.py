"""Admin routes consolidated from fastapi_app and api."""
from .feedback_curation import router as feedback_curation_router
from .main import admin_router
from .prompts import prompt_router
from .rag_reranker import router as rag_reranker_router
from .teams_webhook_admin import router as teams_webhook_admin_router

__all__ = [
    "admin_router",
    "prompt_router",
    "feedback_curation_router",
    "rag_reranker_router",
    "teams_webhook_admin_router"
]
