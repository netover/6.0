import pytest
import pytest_asyncio

try:
    import respx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("Optional dependency 'respx' not installed", allow_module_level=True)

import httpx
from httpx import Response
from resync.services.tws_service import OptimizedTWSClient
from resync.core.exceptions import (
    TWSAuthenticationError,
    TWSRateLimitError,
    TWSServerError,
    TWSTimeoutError,
    TWSConnectionError,
)
from resync.settings import settings


@pytest.fixture
def isolated_settings():
    """Provide isolated settings for each test to avoid race conditions."""
    original_retry_total = settings.tws_retry_total
    original_backoff_base = settings.tws_retry_backoff_base
    original_backoff_max = settings.tws_retry_backoff_max

    settings.tws_retry_total = 3
    settings.tws_retry_backoff_base = 0.01
    settings.tws_retry_backoff_max = 0.1

    yield settings

    settings.tws_retry_total = original_retry_total
    settings.tws_retry_backoff_base = original_backoff_base
    settings.tws_retry_backoff_max = original_backoff_max


@pytest_asyncio.fixture
async def tws_client(isolated_settings):
    # Note: password is intentionally a test value for unit tests (not a real credential)
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        username="test_user",
        password="test_password_for_unit_tests_only",  # nosec B106 - test fixture only
        engine_name="test_engine",
        engine_owner="test_owner",
        settings=isolated_settings,
    )
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_authentication_error_401(tws_client):
    """Test that 401 raises TWSAuthenticationError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        mock.get("/test").mock(return_value=Response(401, text="Unauthorized"))

        with pytest.raises(TWSAuthenticationError) as exc:
            await tws_client._get("/test")

        assert exc.value.status_code == 401
        assert "Unauthorized" in str(exc.value)


@pytest.mark.asyncio
async def test_authentication_error_403(tws_client):
    """Test that 403 raises TWSAuthenticationError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        mock.get("/test").mock(return_value=Response(403, text="Forbidden"))

        with pytest.raises(TWSAuthenticationError) as exc:
            await tws_client._get("/test")

        assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_rate_limit_retry_after(tws_client):
    """Test 429 with Retry-After header respects the delay."""
    async with respx.mock(base_url="http://test-tws") as mock:
        # First request returns 429, second returns 200
        # Retry-After: 0 to avoid waiting in valid test
        route = mock.get("/test").mock(
            side_effect=[
                Response(429, headers={"Retry-After": "0"}, text="Slow down"),
                Response(200, json={"status": "ok"}),
            ]
        )

        result = await tws_client._get("/test")

        assert result == {"status": "ok"}
        assert route.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausted(tws_client):
    """Test 429 raises TWSRateLimitError after retries exhausted."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(
            return_value=Response(429, headers={"Retry-After": "0"})
        )

        with pytest.raises(TWSRateLimitError):
            await tws_client._get("/test")

        # Initial + 3 retries (default) = 4 calls (since settings override is flaky)
        assert route.call_count == 4


@pytest.mark.asyncio
async def test_server_error_retry(tws_client):
    """Test 500 error triggers retries."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(
            side_effect=[
                Response(500, text="Server Error"),
                Response(200, json={"status": "recovered"}),
            ]
        )

        result = await tws_client._get("/test")

        assert result == {"status": "recovered"}
        assert route.call_count == 2


@pytest.mark.asyncio
async def test_server_error_exhausted(tws_client):
    """Test 500 error raises TWSServerError after retries."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(return_value=Response(500))

        with pytest.raises(TWSServerError):
            await tws_client._get("/test")

        assert route.call_count == 4


@pytest.mark.asyncio
async def test_connect_timeout(tws_client):
    """Test connection timeout raises TWSTimeoutError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(side_effect=httpx.ConnectTimeout("Timeout"))

        with pytest.raises(TWSTimeoutError):
            await tws_client._get("/test")

        # Should retry on timeouts too
        assert route.call_count == 4


@pytest.mark.asyncio
async def test_read_timeout(tws_client):
    """Test read timeout raises TWSTimeoutError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(side_effect=httpx.ReadTimeout("Timeout"))

        with pytest.raises(TWSTimeoutError):
            await tws_client._get("/test")

        assert route.call_count == 4


@pytest.mark.asyncio
async def test_connection_error(tws_client):
    """Test connection error raises TWSConnectionError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        with pytest.raises(TWSConnectionError):
            await tws_client._get("/test")

        assert route.call_count == 4
