"""ASGI entry point for Resync.

Production path
---------------
In production this project is normally started by an ASGI server importing
the app object directly, e.g.:

    gunicorn --preload -k uvicorn_worker.UvicornWorker resync.main:app

**Important:** all configuration validation and startup health checks are run
from the application's ASGI *lifespan* (see :mod:`resync.core.startup` and
:class:`resync.app_factory.ApplicationFactory`). This guarantees the checks
execute in the real production path (ASGI import + lifespan), and avoids having
two divergent "boot" systems.

Local developer convenience
---------------------------
This module also supports running locally via:

    python -m resync.main

That command simply launches Uvicorn and relies on the same lifespan logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup Environment: Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from resync.app_factory import create_app  # noqa: E402

app = create_app()

def main() -> None:
    """Run a local development server."""
    # Import uvicorn here to avoid loading it in production ASGI path
    # where it's not needed
    import uvicorn
    from dotenv import load_dotenv

    from resync.settings import settings

    load_dotenv(BASE_DIR / ".env")

    host = settings.server_host
    port = settings.server_port
    log_level = settings.log_level.lower()

    uvicorn.run(
        "resync.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes", "on"},
        loop="asyncio",  # P2-05 fix: uvloop is incompatible with Python 3.14 (BaseDefaultEventLoopPolicy removed)
    )

if __name__ == "__main__":
    main()
