"""Core routes: health, auth, chat, status."""
from .auth import router as auth_router
from .chat import router as chat_router
from .health import router as health_router
from .status import router as status_router

__all__ = ["health_router", "auth_router", "chat_router", "status_router"]
