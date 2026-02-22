# pylint: skip-file
# mypy: ignore-errors
import os
import subprocess
import sys
import pytest


@pytest.mark.integration
def test_alembic_upgrade_head_runs() -> None:
    """Runs `alembic upgrade head` against DATABASE_URL.

    This is intentionally a black-box test: if migrations are broken, the command fails.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    # Ensure alembic sees DATABASE_URL
    env = dict(os.environ)
    env["DATABASE_URL"] = database_url

    # Run alembic in a subprocess to match real usage.
    proc = subprocess.run(
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
