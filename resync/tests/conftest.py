# pylint: disable=all
# mypy: no-rerun
"""
Test Configuration and Fixtures.

v5.6.0: Enhanced testing infrastructure.

Features:
- Async fixtures with proper session management
- Database transaction rollback for test isolation
- Factory-based test data generation
- HTTP client mocking with RESPX
- Property-based testing with Hypothesis

Usage:
    # In tests
    async def test_example(async_client, db_session):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

logger = logging.getLogger(__name__)

# Set test environment before any imports
os.environ["ENVIRONMENT"] = "test"
os.environ["TESTING"] = "true"

# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Application Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def app():
    """Create a test application instance."""
    # v5.8.0: Use unified app from app_factory
    from resync.app_factory import ApplicationFactory

    factory = ApplicationFactory()
    test_app = factory.create_app()

    yield test_app


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Get test database URL."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/resync_test",
    )


@pytest.fixture(scope="session")
async def db_engine(test_database_url):
    """Create a database engine for the test session."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(
        test_database_url,
        poolclass=NullPool,  # Disable pooling for tests
        echo=False,
    )

    # Create tables
    try:
        from resync.models.base import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """
    Create a database session with transaction rollback.

    Each test runs in a transaction that is rolled back after the test,
    ensuring test isolation without database cleanup overhead.
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    async_session_factory = sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with db_engine.connect() as connection:
        # Start a transaction
        transaction = await connection.begin()

        # Create session bound to the connection
        async with async_session_factory(bind=connection) as session:
            yield session

        # Rollback the transaction
        await transaction.rollback()


@pytest.fixture
def mock_db_session():
    """Create a mock database session for unit tests."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# =============================================================================
# Redis Fixtures
# =============================================================================


@pytest.fixture
async def redis_client():
    """Create a Redis client for testing."""
    import redis.asyncio as redis

    redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")
    client = redis.from_url(redis_url)

    # Clear test database
    await client.flushdb()

    yield client

    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for unit tests."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=0)
    mock.expire = AsyncMock(return_value=True)
    mock.ttl = AsyncMock(return_value=-1)
    mock.hget = AsyncMock(return_value=None)
    mock.hset = AsyncMock(return_value=1)
    mock.hgetall = AsyncMock(return_value={})
    mock.pipeline = MagicMock()
    return mock


# =============================================================================
# Authentication Fixtures
# =============================================================================


@pytest.fixture
def test_user_data() -> dict:
    """Sample user data for testing."""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "email": "test@example.com",
        "roles": ["user"],
        "is_active": True,
    }


@pytest.fixture
def admin_user_data() -> dict:
    """Sample admin user data for testing."""
    return {
        "id": "admin-user-456",
        "username": "admin",
        "email": "admin@example.com",
        "roles": ["admin", "user"],
        "is_active": True,
    }


@pytest.fixture
def auth_headers(test_user_data) -> dict:
    """Create authentication headers for testing."""
    from datetime import datetime, timedelta, timezone

    import jwt

    secret_key = os.getenv("SECRET_KEY", "test-secret-key")

    payload = {
        "sub": test_user_data["id"],
        "username": test_user_data["username"],
        "email": test_user_data["email"],
        "roles": test_user_data["roles"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }

    token = jwt.encode(payload, secret_key, algorithm="HS256")

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(admin_user_data) -> dict:
    """Create admin authentication headers for testing."""
    from datetime import datetime, timedelta, timezone

    import jwt

    secret_key = os.getenv("SECRET_KEY", "test-secret-key")

    payload = {
        "sub": admin_user_data["id"],
        "username": admin_user_data["username"],
        "email": admin_user_data["email"],
        "roles": admin_user_data["roles"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }

    token = jwt.encode(payload, secret_key, algorithm="HS256")

    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# HTTP Mocking Fixtures
# =============================================================================


@pytest.fixture
def mock_httpx():
    """
    Mock external HTTP requests using RESPX.

    Usage:
        def test_external_api(mock_httpx):
            mock_httpx.get("https://api.example.com/data").respond(
                json={"key": "value"}
            )
            # Your test code here
    """
    import respx

    with respx.mock(assert_all_called=False) as respx_mock:
        yield respx_mock


# =============================================================================
# Time Fixtures
# =============================================================================


@pytest.fixture
def frozen_time():
    """
    Freeze time for deterministic testing.

    Usage:
        def test_with_frozen_time(frozen_time):
            with frozen_time("2024-01-15 10:00:00"):
                # Time is frozen
                pass
    """
    from datetime import datetime
    from unittest.mock import patch

    class FrozenTime:
        def __call__(self, time_str: str):
            frozen = datetime.fromisoformat(time_str)
            return patch("datetime.datetime", wraps=datetime, now=lambda: frozen)

    return FrozenTime()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Reset any global state if needed


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
