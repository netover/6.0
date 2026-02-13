"""RAG routes for upload and query."""
from .query import router as rag_query_router
from .upload import router as rag_upload_router

__all__ = ["rag_upload_router", "rag_query_router"]
