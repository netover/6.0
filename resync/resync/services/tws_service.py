"""
Read‑only client for interacting with the IBM TWS/HWA REST API.

This client implements a subset of the API surface focused on read‑only
operations such as listing objects, retrieving the current plan and its
relationships, and fetching configuration details. Each request is measured
and counted using Prometheus metrics for observability. HTTPX is instrumented
with OpenTelemetry if the instrumentation library is available.

v5.9.3: Added TTL-differentiated caching:
- Job status: 10s (near real-time)
- Logs: 30s (semi-live)
- Static structure: 1h (rarely changes)
- Graph dependencies: 5min

Cache injects _fetched_at for UI transparency.
"""

import asyncio
import random
import time
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from email.utils import parsedate_to_datetime

# Optional OpenTelemetry instrumentation for HTTPX.
# NOTE: instrumenting at import-time is a global side effect; we do it lazily and idempotently.
_HTTPX_OTEL_INSTRUMENTED = False

def _ensure_httpx_instrumented() -> None:
    """Instrument httpx once, if opentelemetry instrumentation is available."""
    global _HTTPX_OTEL_INSTRUMENTED
    if _HTTPX_OTEL_INSTRUMENTED:
        return
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore
    except ImportError:
        _HTTPX_OTEL_INSTRUMENTED = True  # do not retry every instantiation if package is absent
        return
    try:
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(exc, (ImportError, AttributeError, TypeError)):
            raise
        # Best-effort: instrumentation must never break the client.
        logger.debug("suppressed_exception", error=str(exc), exc_info=True)
    finally:
        _HTTPX_OTEL_INSTRUMENTED = True


from resync.core.exceptions import (
    TWSAuthenticationError,
    TWSBadRequestError,
    TWSConnectionError,
    TWSRateLimitError,
    TWSServerError,
    TWSTimeoutError,
    ResourceNotFoundError,
)
from resync.core.metrics_compat import Counter, Histogram
from resync.services.tws_cache import (
    CacheCategory,
    enrich_response_with_cache_meta,
    get_tws_cache,
)

_ID_LIKE_SEGMENT_RE = re.compile(
    r"""(?ix)
    ^(
        \d+                                     # pure digits
        |[0-9a-f]{8,}                             # long hex
        |[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}  # uuid
        |[A-Z]{2,}-\d+(?:-[A-Z0-9]+)*            # e.g. JOB-123-ABC
    )$
    """
)

def _normalize_endpoint_label(path: str) -> str:
    """Collapse ID-like path segments to avoid Prometheus high-cardinality labels."""
    segments = [s for s in path.split("/") if s]
    norm: list[str] = []
    for seg in segments:
        if _ID_LIKE_SEGMENT_RE.match(seg):
            norm.append("{id}")
        else:
            norm.append(seg)
    return "_".join(norm) or "root"

class OptimizedTWSClient:
    """
    A lightweight asynchronous client for the TWS/HWA REST API.

    It wraps an httpx.AsyncClient and exposes convenient methods for common
    read‑only operations. All requests funnel through a single `_get` method
    which records request latency and counts by endpoint and status code.
    """

    # Prometheus metrics shared across all client instances
    _request_latency: Histogram = Histogram(
        "tws_request_latency_seconds",
        "Latency of TWS API requests",
        ["endpoint"],
    )
    _request_count: Counter = Counter(
        "tws_request_total",
        "Total number of TWS API requests",
        ["endpoint", "status"],
    )

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        engine_name: str,
        engine_owner: str,
        trust_env: bool = False,
        settings: Any | None = None,
    ) -> None:
        """
        Construct the TWS client.

        Args:
            base_url: Base URL of the TWS API (e.g., "http://localhost:8080")
            username: Basic auth username
            password: Basic auth password
            engine_name: Default engine name for queries that require it
            engine_owner: Engine owner associated with the engine
            trust_env: If True, use system proxy settings from environment variables.
                       Set to True in corporate environments that require proxy access.
                       Default is False to avoid requiring optional dependencies like socksio.
            settings: Optional override for application settings (for testing/DI).
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.engine_name = engine_name
        self.engine_owner = engine_owner
        
        # Use injected settings or load defaults
        if settings is None:
            from resync.settings import get_settings
            settings = get_settings()

        # httpx client with a base URL and basic authentication. Connection pooling
        # is automatically handled by httpx.AsyncClient.

        timeout_config = httpx.Timeout(
            connect=settings.tws_timeout_connect,
            read=settings.tws_timeout_read,
            write=settings.tws_timeout_write,
            pool=settings.tws_timeout_pool,
        )
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=self.auth,
            trust_env=trust_env,
            timeout=timeout_config,
        )
        # v5.9.3: TTL-differentiated cache
        self._cache = get_tws_cache()

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self.client.aclose()

    # -------------------------------------------------------------------------
    # Cache Management (v5.9.3)
    # -------------------------------------------------------------------------
    def configure_cache(
        self,
        job_status_ttl: int | None = None,
        job_logs_ttl: int | None = None,
        static_ttl: int | None = None,
        graph_ttl: int | None = None,
    ):
        """Configure cache TTLs."""
        self._cache.configure_ttls(
            job_status=job_status_ttl,
            job_logs=job_logs_ttl,
            static_structure=static_ttl,
            graph=graph_ttl,
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """
        Internal helper for GET requests with metrics, retries, and error mapping.

        Args:
            path: The path portion of the URL (should begin with '/')
            params: Optional query parameters
            timeout: Optional custom timeout (uses default if None)

        Returns:
            The parsed JSON response on success.

        Raises:
            TWSAuthenticationError: For 401/403 responses
            TWSRateLimitError: For 429 responses
            ResourceNotFoundError: For 404 responses
            TWSBadRequestError: For 400 responses
            TWSServerError: For 5xx responses
            TWSTimeoutError: For timeout errors
            TWSConnectionError: For network/connection errors
        """
        from resync.settings import get_settings
        
        settings = get_settings()
        endpoint_label = _normalize_endpoint_label(path)
        
        # Determine timeout (custom or default)
        if timeout is None:
            # Use special timeout for joblog endpoints
            if "joblog" in path.lower():
                timeout = settings.tws_joblog_timeout
            else:
                timeout = settings.tws_request_timeout
        
        # Retry configuration
        max_retries = settings.tws_retry_total
        backoff_base = settings.tws_retry_backoff_base
        backoff_max = settings.tws_retry_backoff_max
        
        last_exception = None

        def _compute_backoff(attempt_idx: int) -> float:
            """Compute exponential backoff with *full jitter*.

            Full jitter (random in [0, cap]) helps avoid synchronized retries
            across many clients.
            """

            cap = min(backoff_base * (2 ** attempt_idx), backoff_max)
            return random.uniform(0, cap)

        def _parse_retry_after(value: str) -> int | None:
            """Parse Retry-After, which can be delay-seconds or an HTTP-date."""
            value = value.strip()
            if not value:
                return None
            # delay-seconds
            try:
                return max(0, int(value))
            except ValueError:
                pass
            # HTTP-date
            try:
                dt = parsedate_to_datetime(value)
                if dt is None:
                    return None
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delta = int((dt - now).total_seconds())
                return max(0, delta)
            except (ValueError, TypeError, OverflowError):
                return None
        
        for attempt in range(max_retries + 1):
            start = time.perf_counter()
            status_code = None

            try:
                response = await self.client.get(path, params=params, timeout=timeout)
                elapsed = time.perf_counter() - start
                status_code = response.status_code
                                
                # Check for HTTP errors
                response.raise_for_status()

                # Record success metrics ONCE (only on 2xx/3xx)
                self._request_latency.labels(endpoint=endpoint_label).observe(elapsed)
                self._request_count.labels(
                    endpoint=endpoint_label, status=str(status_code)
                ).inc()

                
                # Parse response body
                content_type = (response.headers.get("content-type") or "").lower()
                if "application/json" in content_type or "+json" in content_type:
                    return response.json()
                if content_type.startswith("text/"):
                    return response.text
                try:
                    return response.json()
                except ValueError:
                    # JSON decode failed
                    return response.text
                    
            except httpx.HTTPStatusError as e:
                elapsed = time.perf_counter() - start
                status_code = e.response.status_code
                
                # Record error metrics ONCE (HTTP 4xx/5xx path)
                self._request_latency.labels(endpoint=endpoint_label).observe(elapsed)
                self._request_count.labels(
                    endpoint=endpoint_label, status=str(status_code)
                ).inc()
                
                # Map HTTP status to custom exceptions
                if status_code == 401 or status_code == 403:
                    raise TWSAuthenticationError(
                        f"TWS authentication failed: {e.response.text[:200]}",
                        details={"status_code": status_code, "response_body": e.response.text}
                    ) from e
                elif status_code == 404:
                    raise ResourceNotFoundError(
                        f"TWS resource not found: {path}",
                        details={"path": path, "params": params}
                    ) from e
                elif status_code == 400:
                    raise TWSBadRequestError(
                        f"TWS bad request: {e.response.text[:200]}",
                        details={"status_code": status_code, "response_body": e.response.text}
                    ) from e
                elif status_code == 429:
                    # Rate limit - check Retry-After header
                    retry_after_header = e.response.headers.get("Retry-After")
                    retry_after = _parse_retry_after(retry_after_header) if retry_after_header else None

                    if attempt < max_retries:
                        # Prefer server-provided Retry-After when present.
                        sleep_s = retry_after if (retry_after is not None) else _compute_backoff(attempt)
                        sleep_s = min(sleep_s, backoff_max)
                        await asyncio.sleep(sleep_s)
                        last_exception = TWSRateLimitError(
                            f"TWS rate limit exceeded (attempt {attempt+1}/{max_retries+1})",
                            details={"retry_after": retry_after, "status_code": status_code},
                        )
                        continue

                    raise TWSRateLimitError(
                        f"TWS rate limit exceeded: {e.response.text[:200]}",
                        details={"retry_after": retry_after, "status_code": status_code, "response_body": e.response.text},
                    ) from e
                elif 500 <= status_code < 600:
                    # Server error - retry with backoff
                    if attempt < max_retries:
                        await asyncio.sleep(_compute_backoff(attempt))
                        last_exception = TWSServerError(
                            f"TWS server error (attempt {attempt+1}/{max_retries+1})",
                            details={"status_code": status_code},
                        )
                        continue
                    else:
                        raise TWSServerError(
                            f"TWS server error after {max_retries} retries: {e.response.text[:200]}",
                            details={"status_code": status_code, "response_body": e.response.text},
                        ) from e
                else:
                    # Other HTTP error - don't retry
                    raise TWSConnectionError(
                        f"TWS HTTP error {status_code}: {e.response.text[:200]}",
                        details={"status_code": status_code, "response_body": e.response.text},
                    ) from e
                    
            except httpx.TimeoutException as e:
                elapsed = time.perf_counter() - start
                
                # Record timeout metrics
                self._request_latency.labels(endpoint=endpoint_label).observe(elapsed)
                self._request_count.labels(endpoint=endpoint_label, status="timeout").inc()
                
                # Retry on timeout
                if attempt < max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                    last_exception = TWSTimeoutError(
                        f"TWS timeout (attempt {attempt+1}/{max_retries+1})",
                        details={"status_code": 504},
                    )
                    continue
                else:
                    raise TWSTimeoutError(
                        f"TWS timeout after {max_retries} retries: {str(e)}",
                        details={"status_code": 504},
                    ) from e
                    
            except httpx.RequestError as e:
                elapsed = time.perf_counter() - start
                
                # Record connection error metrics
                self._request_latency.labels(endpoint=endpoint_label).observe(elapsed)
                self._request_count.labels(endpoint=endpoint_label, status="connection_error").inc()
                
                # Retry on connection errors
                if attempt < max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                    last_exception = TWSConnectionError(
                        f"TWS connection error (attempt {attempt+1}/{max_retries+1}): {str(e)}",
                        details={"status_code": 502},
                    )
                    continue
                else:
                    raise TWSConnectionError(
                        f"TWS connection error after {max_retries} retries: {str(e)}",
                        details={"status_code": 502},
                    ) from e
        
        # Should not reach here, but if we do, raise the last exception
        if last_exception:
            raise last_exception
        raise TWSConnectionError("TWS request failed for unknown reason")

    # ---------------------------------------------------------------------
    # Engine & Configuration
    # ---------------------------------------------------------------------
    async def get_engine_info(self) -> Any:
        """Retrieve high level information about the engine."""
        return await self._get("/twsd/api/v2/engine/info")

    async def get_engine_configuration(self, key: str | None = None) -> Any:
        """Retrieve engine configuration values."""
        params = {"key": key} if key else None
        return await self._get("/twsd/api/v2/engine/configuration", params=params)

    async def list_users(self) -> Any:
        """List users defined in the model."""
        return await self._get("/twsd/api/v2/model/user")

    async def list_groups(self) -> Any:
        """List groups defined in the model."""
        return await self._get("/twsd/api/v2/model/group")

    # ---------------------------------------------------------------------
    # Model queries
    # ---------------------------------------------------------------------
    async def query_jobdefinitions(
        self,
        q: str | None = None,
        folder: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """Search job definitions in the model with optional filters."""
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if folder:
            params["folder"] = folder
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/model/jobdefinition", params=params)

    async def get_jobdefinition(self, jobdef_id: str) -> Any:
        """Retrieve a specific job definition by its ID."""
        return await self._get(f"/twsd/api/v2/model/jobdefinition/{jobdef_id}")

    async def query_jobstreams(
        self,
        q: str | None = None,
        folder: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """Search job streams in the model with optional filters."""
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if folder:
            params["folder"] = folder
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/model/jobstream", params=params)

    async def get_jobstream(self, jobstream_id: str) -> Any:
        """Retrieve a specific job stream by its ID."""
        return await self._get(f"/twsd/api/v2/model/jobstream/{jobstream_id}")

    async def query_workstations(
        self,
        q: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """Search workstations in the model with optional filters."""
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/model/workstation", params=params)

    async def get_workstation(self, workstation_id: str) -> Any:
        """Retrieve a workstation definition by its ID."""
        return await self._get(f"/twsd/api/v2/model/workstation/{workstation_id}")

    # ---------------------------------------------------------------------
    # Current plan queries – Jobs
    # ---------------------------------------------------------------------
    async def query_current_plan_jobs(
        self,
        q: str | None = None,
        folder: str | None = None,
        status: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """
        List or search jobs currently present in the plan.
        Filters include a free form search string, folder path and status.
        """
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if folder:
            params["folder"] = folder
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/plan/job", params=params)

    async def get_current_plan_job(self, job_id: str) -> Any:
        """Retrieve a specific job from the current plan."""
        return await self._get(f"/twsd/api/v2/plan/job/{job_id}")

    async def get_current_plan_job_predecessors(
        self,
        job_id: str,
        depth: int | None = None,
    ) -> Any:
        """Retrieve the predecessors of a job in the current plan."""
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        return await self._get(f"/twsd/api/v2/plan/job/{job_id}/predecessors", params=params)

    async def get_current_plan_job_successors(
        self,
        job_id: str,
        depth: int | None = None,
    ) -> Any:
        """Retrieve the successors of a job in the current plan."""
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        return await self._get(f"/twsd/api/v2/plan/job/{job_id}/successors", params=params)

    async def get_current_plan_job_model(self, job_id: str) -> Any:
        """Retrieve the underlying model of a job in the current plan."""
        return await self._get(f"/twsd/api/v2/plan/job/{job_id}/model")

    async def get_current_plan_job_model_description(self, job_id: str) -> Any:
        """Retrieve the model description of a job in the current plan."""
        return await self._get(f"/twsd/api/v2/plan/job/{job_id}/model/description")

    async def get_current_plan_job_count(self) -> Any:
        """Return the total number of jobs in the current plan."""
        return await self._get("/twsd/api/v2/plan/job/count")

    async def get_current_plan_job_issues(self) -> Any:
        """Return issues detected in the current plan jobs."""
        return await self._get("/twsd/api/v2/plan/job/issues")

    async def get_current_plan_job_joblog(
        self,
        job_id: str | None = None,
        *,
        oql: str | None = None,
        content_only: bool | None = None,
        follow: bool | None = None,
        from_line: int | None = None,
        to_line: int | None = None,
        plan_filter: str | None = None,
        plan_id: str | None = None,
    ) -> Any:
        """Retrieve job log output for a job run in the current plan.

        The WA REST API exposes job log output via ``/plan/job/joblog``. Despite
        the name, this endpoint supports filtering via OQL (``oql`` query
        parameter). Many installations return ``text/plain`` for this endpoint.
        This client therefore returns a string when the response is text.

        Args:
            job_id: Convenience parameter. When provided and ``oql`` is not
                specified, an OQL filter of ``id = '<job_id>'`` is generated.
            oql: Optional OQL filter.
            content_only: If True, omit header/footer from the log.
            follow: If True, stream new output as it becomes available.
            from_line: Start line offset.
            to_line: End line offset.
            plan_filter: Optional plan filter.
            plan_id: Optional plan identifier.
        """

        params: dict[str, Any] = {}
        if oql:
            params["oql"] = oql
        elif job_id:
            # Filter by the run instance id.
            params["oql"] = f"id = '{job_id}'"

        if content_only is not None:
            params["contentOnly"] = "true" if content_only else "false"
        if follow is not None:
            params["follow"] = "true" if follow else "false"
        if from_line is not None:
            params["from_line"] = from_line
        if to_line is not None:
            params["to_line"] = to_line
        if plan_filter is not None:
            params["plan_filter"] = plan_filter
        if plan_id is not None:
            params["plan_id"] = plan_id

        return await self._get("/twsd/api/v2/plan/job/joblog", params=params or None)

    # ---------------------------------------------------------------------
    # Current plan queries – Job Streams
    # ---------------------------------------------------------------------
    async def query_current_plan_jobstreams(
        self,
        q: str | None = None,
        folder: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """List or search job streams in the current plan."""
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if folder:
            params["folder"] = folder
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/plan/jobstream", params=params)

    async def get_current_plan_jobstream(self, jobstream_id: str) -> Any:
        """Retrieve a specific job stream from the current plan."""
        return await self._get(f"/twsd/api/v2/plan/jobstream/{jobstream_id}")

    async def get_current_plan_jobstream_predecessors(
        self,
        jobstream_id: str,
        depth: int | None = None,
    ) -> Any:
        """Retrieve the predecessors of a job stream in the current plan."""
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        return await self._get(
            f"/twsd/api/v2/plan/jobstream/{jobstream_id}/predecessors",
            params=params,
        )

    async def get_current_plan_jobstream_successors(
        self,
        jobstream_id: str,
        depth: int | None = None,
    ) -> Any:
        """Retrieve the successors of a job stream in the current plan."""
        params: dict[str, Any] = {}
        if depth is not None:
            params["depth"] = depth
        return await self._get(
            f"/twsd/api/v2/plan/jobstream/{jobstream_id}/successors",
            params=params,
        )

    async def get_current_plan_jobstream_model_description(self, jobstream_id: str) -> Any:
        """Retrieve the model description of a job stream in the current plan."""
        return await self._get(f"/twsd/api/v2/plan/jobstream/{jobstream_id}/model/description")

    async def get_current_plan_jobstream_count(self) -> Any:
        """Return the total number of job streams in the current plan."""
        return await self._get("/twsd/api/v2/plan/jobstream/count")

    # ---------------------------------------------------------------------
    # Current plan queries – Resources and Folders
    # ---------------------------------------------------------------------
    async def query_current_plan_resources(
        self, q: str | None = None, limit: int | None = 50
    ) -> Any:
        """List or search resources in the current plan."""
        params: dict[str, Any] = {}
        if q:
            params["query"] = q
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/plan/resource", params=params)

    async def get_current_plan_resource(self, resource_id: str) -> Any:
        """Retrieve a specific resource from the current plan."""
        return await self._get(f"/twsd/api/v2/plan/resource/{resource_id}")

    async def get_current_plan_folder_objects_count(self, folder: str | None = None) -> Any:
        """Return the number of plan objects within a folder."""
        params: dict[str, Any] = {}
        if folder:
            params["folder"] = folder
        return await self._get("/twsd/api/v2/plan/folder/objects-count", params=params)

    # ---------------------------------------------------------------------
    # Current plan queries – Consumed Jobs
    # ---------------------------------------------------------------------
    async def get_consumed_jobs_runs(
        self,
        job_name: str | None = None,
        limit: int | None = 50,
    ) -> Any:
        """
        Retrieve runs of consumed jobs in the current plan.

        Args:
            job_name: Optional name filter for the job whose runs are returned.
            limit: Maximum number of runs to return.
        """
        params: dict[str, Any] = {}
        if job_name:
            params["jobName"] = job_name
        if limit is not None:
            params["limit"] = limit
        return await self._get("/twsd/api/v2/plan/consumed-jobs/runs", params=params)

    # =========================================================================
    # CACHED METHODS (v5.9.3 - Near Real-Time Strategy)
    # =========================================================================
    # These methods use TTL-differentiated caching for optimal balance between
    # API protection and data freshness. All responses include _fetched_at
    # timestamp for UI transparency.

    async def get_job_status_cached(
        self,
        job_id: str,
        with_meta: bool = False,
    ) -> Any:
        """
        Get job status with 10-second cache (near real-time).

        Args:
            job_id: Job identifier
            with_meta: If True, returns {data, meta} with cache info

        Returns:
            Job status dict with _fetched_at timestamp
            If with_meta=True: {data: {...}, meta: {cached, age_seconds, freshness}}
        """
        cache_key = f"job_status:{job_id}"

        async def fetch():
            data = await self.get_current_plan_job(job_id)
            data["_fetched_at"] = datetime.now(timezone.utc).isoformat()
            return data

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.JOB_STATUS
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value

    async def get_job_logs_cached(
        self,
        job_id: str,
        with_meta: bool = False,
    ) -> Any:
        """
        Get job logs with 30-second cache.

        Returns:
            Job logs with _fetched_at timestamp
        """
        cache_key = f"job_logs:{job_id}"

        async def fetch():
            # ``/plan/job/joblog`` commonly returns text/plain. Wrap the log in
            # a dict so the cache layer can inject ``_fetched_at`` metadata for
            # UI transparency.
            log_text = await self.get_current_plan_job_joblog(
                job_id,
                content_only=True,
            )
            return {
                "job_id": job_id,
                "log": log_text,
                "_fetched_at": datetime.now(timezone.utc).isoformat(),
            }

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.JOB_LOGS
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value

    async def get_job_dependencies_cached(
        self,
        job_id: str,
        depth: int = 1,
        with_meta: bool = False,
    ) -> dict[str, Any]:
        """
        Get job dependencies with 5-minute cache.

        Returns:
            Dict with predecessors and successors
        """
        cache_key = f"job_deps:{job_id}:d{depth}"

        async def fetch():
            preds = await self.get_current_plan_job_predecessors(job_id, depth)
            succs = await self.get_current_plan_job_successors(job_id, depth)
            return {
                "job_id": job_id,
                "predecessors": preds or [],
                "successors": succs or [],
                "_fetched_at": datetime.now(timezone.utc).isoformat(),
            }

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.GRAPH
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value

    async def get_jobdefinition_cached(
        self,
        jobdef_id: str,
        with_meta: bool = False,
    ) -> Any:
        """
        Get job definition with 1-hour cache (static structure).

        Returns:
            Job definition with _fetched_at timestamp
        """
        cache_key = f"jobdef:{jobdef_id}"

        async def fetch():
            data = await self.get_jobdefinition(jobdef_id)
            if isinstance(data, dict):
                data["_fetched_at"] = datetime.now(timezone.utc).isoformat()
            return data

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.STATIC_STRUCTURE
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value

    async def get_workstation_cached(
        self,
        workstation_id: str,
        with_meta: bool = False,
    ) -> Any:
        """
        Get workstation with 1-hour cache (static structure).

        Returns:
            Workstation data with _fetched_at timestamp
        """
        cache_key = f"ws:{workstation_id}"

        async def fetch():
            data = await self.get_workstation(workstation_id)
            if isinstance(data, dict):
                data["_fetched_at"] = datetime.now(timezone.utc).isoformat()
            return data

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.STATIC_STRUCTURE
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value

    async def query_jobs_cached(
        self,
        q: str | None = None,
        status: str | None = None,
        limit: int = 50,
        with_meta: bool = False,
    ) -> Any:
        """
        Query jobs with 10-second cache.

        Returns:
            List of jobs with _fetched_at timestamp
        """
        cache_key = f"jobs_query:q={(q or '')}:s={(status or '')}:l={limit}"

        async def fetch():
            data = await self.query_current_plan_jobs(q, None, status, limit)
            return {
                "jobs": data if isinstance(data, list) else data.get("items", []),
                "query": q,
                "status_filter": status,
                "_fetched_at": datetime.now(timezone.utc).isoformat(),
            }

        value, is_cached, age = await self._cache.get_or_fetch(
            cache_key, fetch, CacheCategory.JOB_STATUS
        )

        if with_meta:
            return enrich_response_with_cache_meta(value, is_cached, age)
        return value


# =============================================================================
# HELPER FUNCTION
# =============================================================================


def get_tws_client() -> "OptimizedTWSClient":
    """Backward-compatible helper.

    Prefer importing `get_tws_client_singleton` from `resync.core.factories.tws_factory`.
    """
    from resync.core.factories.tws_factory import get_tws_client_singleton
    return get_tws_client_singleton()
