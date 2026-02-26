"""
Models Registry - Imports all SQLAlchemy models to ensure they are registered.

This module must be imported before calling init_db() to ensure all
models are registered with SQLAlchemy's Base.metadata.

Usage:
    from resync.core.database.models_registry import register_all_models
    register_all_models()
    await init_db()  # Now all tables will be created
"""

import logging

logger = logging.getLogger(__name__)

_registered = False

def register_all_models():
    """
    Import all SQLAlchemy models to register them with Base.metadata.

    This must be called before init_db() to ensure all tables are created.
    """
    global _registered

    if _registered:
        return

    # Import all model modules to register them with Base
    # NOTE: These imports are for side-effects only (model registration)
    try:
        # Auth models (v5.4.7 - consolidated to core/database)
        from resync.core.database.models.auth import (  # noqa: F401
            AuditLog,
            User,
        )

        logger.debug("Registered auth models")
    except ImportError as e:
        logger.warning("Could not import auth models: %s", e)

    try:
        # Knowledge Graph models (v5.9.3 - simplified)
        from resync.knowledge.models import (  # noqa: F401
            ExtractedTriplet,
            # GraphNode, GraphEdge, GraphSnapshot removed in v5.9.3
            # Graph is now built on-demand from TWS API
        )

        logger.debug("Registered knowledge_graph models")
    except ImportError as e:
        logger.warning("Could not import knowledge_graph models: %s", e)

    _registered = True
    logger.info("All database models registered")

# Convenience: auto-register when module is imported
register_all_models()
