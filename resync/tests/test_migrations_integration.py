# pylint
import os
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_alembic_upgrade_head_runs() -> None:
    """Runs `alembic upgrade head` against APP_DATABASE_URL.

    This is intentionally a black-box test: if migrations are broken, the command fails.
    """
    database_url = os.getenv("APP_DATABASE_URL")
    if not database_url:
        pytest.skip("APP_DATABASE_URL not set")

    # Ensure alembic sees the canonical variable regardless of how it was provided.
    env = dict(os.environ)
    env["APP_DATABASE_URL"] = database_url

    # Run alembic in a subprocess to match real usage.
    proc = subprocess.run(  # noqa: S603 - controlled test invocation
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "alembic upgrade head failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
