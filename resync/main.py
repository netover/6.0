"""ASGI entry point for Resync.

Production path
---------------
In production this project is normally started by an ASGI server importing
the app object directly, e.g.:

    gunicorn --preload -k uvicorn.workers.UvicornWorker resync.main:app

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

from dotenv import load_dotenv

# Setup Environment: Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load .env explicitly before importing settings to avoid reading env vars
# before dotenv has had a chance to populate them
load_dotenv(BASE_DIR / ".env")

from resync.app_factory import ApplicationFactory  # noqa: E402


_factory = ApplicationFactory()
app = _factory.create_application()


def main() -> None:
    """Run a local development server."""

    import uvicorn

    from resync.settings import settings

    host = os.getenv("HOST") or settings.server_host
    port = int(os.getenv("PORT") or settings.server_port)
    log_level = os.getenv("LOG_LEVEL") or settings.log_level.lower()

    uvicorn.run(
        "resync.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes", "on"},
    )


if __name__ == "__main__":
    main()
