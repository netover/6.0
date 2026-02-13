"""
RAG Service Client for API Gateway.

This module provides a resilient client to communicate with the standalone RAG
microservice.

Hardening goals (production-grade):
- Fail-fast config validation (URL)
- Explicit timeouts for *every* I/O
- Retry with exponential backoff + jitter for transient failures
- Circuit breaker to protect upstream and callers
- Bulkhead (concurrency limit) to avoid saturating the upstream and event loop
- Process-wide singleton to reuse HTTP connection pools and ensure clean shutdown
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from resync.core.exceptions import ConfigurationError, IntegrationError, ServiceUnavailableError
from resync.core.resilience import CircuitBreakerManager, retry_with_backoff_async
from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Response models (typed helpers)
# -----------------------------------------------------------------------------


class RAGJobStatus(BaseModel):
    """Model for RAG job status response."""

    job_id: str
    status: str  # queued, processing, completed, failed
    progress: int | None = None
    message: str | None = None


class RAGUploadResponse(BaseModel):
    """Model for RAG upload response."""

    job_id: str
    filename: str
    status: str


# -----------------------------------------------------------------------------
# Client implementation
# -----------------------------------------------------------------------------


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


class RAGServiceClient:
    """Async HTTP client for the RAG microservice.

    This object owns a single ``httpx.AsyncClient`` (connection pool).
    Prefer using the process-wide singleton via :func:`get_rag_client_singleton`.
    """

    def __init__(self) -> None:
        self.rag_service_url: str = getattr(settings, "RAG_SERVICE_URL", "") or ""
        self.max_retries: int = _safe_int_env("RAG_MAX_RETRIES", 3)
        self.retry_backoff: float = float(os.getenv("RAG_RETRY_BASE_DELAY", "0.5"))
        self._search_timeout_s: float = float(os.getenv("RAG_SEARCH_TIMEOUT", "5.0"))
        self._request_timeout_s: float = float(os.getenv("RAG_REQUEST_TIMEOUT", "30.0"))

        # Bulkhead: cap concurrent in-flight requests to the RAG service.
        self._max_concurrency: int = _safe_int_env("RAG_MAX_CONCURRENCY", 10)
        self._sem = asyncio.Semaphore(self._max_concurrency)

        # Validate URL shape early (fail-fast) — but allow empty URL in dev/test
        # when the feature is not used (calls will raise ConfigurationError).
        if self.rag_service_url:
            parsed = urlparse(self.rag_service_url)
            if parsed.scheme not in {"http", "https"}:
                raise ConfigurationError(
                    "RAG_SERVICE_URL must be an http(s) URL",
                    details={"value": self.rag_service_url},
                )

        # Explicit timeout budget for the underlying client. We still pass a per-request
        # timeout for hot-path calls to keep them bounded.
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=self._request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=max(10, self._max_concurrency), max_keepalive_connections=10),
            follow_redirects=True,
        )

        # Circuit breaker: count only transient classes (network + transient upstream errors).
        self.cbm = CircuitBreakerManager()
        self.cbm.register(
            "rag_service",
            fail_max=_safe_int_env("RAG_CB_FAILURE_THRESHOLD", 5),
            reset_timeout=_safe_int_env("RAG_CB_RECOVERY_TIMEOUT", 60),
            expected_exception=(httpx.RequestError, ServiceUnavailableError),
            exclude=(ConfigurationError, IntegrationError),
        )

        logger.info(
            "rag_client_initialized",
            url=self.rag_service_url or None,
            max_retries=self.max_retries,
            max_concurrency=self._max_concurrency,
        )

    def _ensure_configured(self) -> None:
        if not self.rag_service_url:
            raise ConfigurationError("RAG_SERVICE_URL not configured")

    @staticmethod
    def _extract_body_preview(resp: httpx.Response, limit: int = 2048) -> str:
        try:
            # Try JSON first to avoid huge HTML pages.
            data = resp.json()
            txt = str(data)
        except Exception:
            txt = resp.text
        txt = txt or ""
        if len(txt) > limit:
            return txt[:limit] + "…"
        return txt

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> int | None:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            # Retry-After can be seconds; ignore HTTP date format for simplicity.
            return int(ra)
        except ValueError:
            return None

    def _raise_for_status(self, resp: httpx.Response, *, operation: str) -> None:
        """Translate upstream HTTP failures into app exceptions.

        * 429 / 5xx are treated as transient -> ``ServiceUnavailableError``
        * other 4xx are treated as permanent integration errors -> ``IntegrationError``
        """
        if 200 <= resp.status_code < 300:
            return

        body_preview = self._extract_body_preview(resp)
        retry_after = self._parse_retry_after(resp)

        # Transient failures (safe to retry, count for circuit breaker).
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ServiceUnavailableError(
                f"RAG service transient failure during {operation}",
                retry_after=retry_after,
                details={
                    "operation": operation,
                    "status_code": resp.status_code,
                    "body_preview": body_preview,
                },
            )

        # Permanent failures (do not retry, do not trip breaker).
        raise IntegrationError(
            f"RAG service returned non-retriable error during {operation}",
            details={
                "operation": operation,
                "status_code": resp.status_code,
                "body_preview": body_preview,
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: Any | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        self._ensure_configured()

        url = f"{self.rag_service_url}{path}"

        async def _once() -> httpx.Response:
            async with self._sem:
                return await self.http_client.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    files=files,
                    timeout=timeout_s or self._request_timeout_s,
                )

        async def _call() -> httpx.Response:
            # Circuit breaker wraps the raw I/O; non-transient exceptions won't count.
            resp = await self.cbm.call("rag_service", _once)
            self._raise_for_status(resp, operation=operation)
            return resp

        return await retry_with_backoff_async(
            _call,
            retries=self.max_retries,
            base_delay=self.retry_backoff,
            cap=5.0,
            jitter=True,
            retry_on=(httpx.RequestError, ServiceUnavailableError),
        )

    async def enqueue_file(self, file: Any) -> str:
        """Enqueue a file for RAG processing. Returns the ``job_id``."""

        # Upload endpoints are usually heavier; keep timeout configurable.
        resp = await self._request(
            "POST",
            "/api/v1/upload",
            operation="upload",
            files={"file": (file.filename, file.file, file.content_type)},
            timeout_s=float(os.getenv("RAG_UPLOAD_TIMEOUT", str(getattr(settings, "rag_service_timeout", 300.0) or 300.0))),
        )
        data = resp.json()
        return str(data.get("job_id") or "")

    async def get_job_status(self, job_id: str) -> RAGJobStatus:
        """Get job status by ``job_id``."""

        # Job status should be fast.
        try:
            resp = await self._request(
                "GET",
                f"/api/v1/jobs/{job_id}",
                operation="job_status",
                timeout_s=min(self._request_timeout_s, 10.0),
            )
            return RAGJobStatus(**resp.json())
        except IntegrationError as exc:
            # Preserve legacy behavior for 404 -> not_found pseudo-status.
            status_code = (exc.details or {}).get("status_code")
            if status_code == 404:
                return RAGJobStatus(job_id=job_id, status="not_found", progress=0, message="Job ID not found")
            raise

    async def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Search for relevant documents/snippets in the RAG service.

        The upstream API shape varies across deployments; we first attempt a POST
        JSON contract and fall back to a GET query contract if needed.
        """
        payload = {"query": query, "limit": limit}

        # Preferred: POST /api/v1/search
        try:
            resp = await self._request(
                "POST",
                "/api/v1/search",
                operation="search",
                json=payload,
                timeout_s=min(self._search_timeout_s, self._request_timeout_s),
            )
            return resp.json()
        except IntegrationError as exc:
            status_code = (exc.details or {}).get("status_code")
            if status_code not in {404, 405}:
                raise

        # Fallback: GET /api/v1/search?q=...&limit=...
        resp = await self._request(
            "GET",
            "/api/v1/search",
            operation="search",
            params={"q": query, "query": query, "limit": limit},
            timeout_s=min(self._search_timeout_s, self._request_timeout_s),
        )
        return resp.json()

    async def get_relevant_context(self, query: str) -> str | None:
        """Convenience method used by orchestrators.

        Returns a best-effort concatenation of top hits' text fields.
        """
        try:
            data = await self.search(query=query, limit=5)
        except Exception as exc:
            logger.warning("rag_get_relevant_context_failed", error=str(exc))
            return None

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return None

        snippets: list[str] = []
        for item in results[:5]:
            if not isinstance(item, dict):
                continue
            for k in ("content", "text", "snippet", "chunk", "summary"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    snippets.append(v.strip())
                    break

        if not snippets:
            return None
        return "\n\n".join(snippets)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.http_client.aclose()

    async def __aenter__(self) -> "RAGServiceClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class RAGClient:
    """Compatibility wrapper expected by LangGraph code.

    Many modules do ``from resync.services.rag_client import RAGClient`` and then
    instantiate it (``RAGClient()``). This wrapper delegates to the process-wide
    singleton to avoid leaking HTTP clients per call.
    """

    def __init__(self) -> None:
        self._client = get_rag_client_singleton()

    async def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        return await self._client.search(query=query, limit=limit)

    async def get_relevant_context(self, query: str) -> str | None:
        return await self._client.get_relevant_context(query=query)

    async def enqueue_file(self, file: Any) -> str:
        return await self._client.enqueue_file(file)

    async def get_job_status(self, job_id: str) -> RAGJobStatus:
        return await self._client.get_job_status(job_id)

    async def close(self) -> None:
        # Compatibility: closing the wrapper does nothing; lifecycle closes singleton.
        return None


# -----------------------------------------------------------------------------
# Singleton access (avoids import-time side effects)
# -----------------------------------------------------------------------------


_rag_client_singleton: RAGServiceClient | None = None
_rag_client_lock = __import__("threading").Lock()


def get_rag_client_singleton() -> RAGServiceClient:
    """Get (or create) the process-wide RAG client singleton (thread-safe)."""
    global _rag_client_singleton
    if _rag_client_singleton is None:
        with _rag_client_lock:
            if _rag_client_singleton is None:
                _rag_client_singleton = RAGServiceClient()
    return _rag_client_singleton


async def close_rag_client_singleton() -> None:
    """Close and clear the RAG client singleton (best-effort)."""
    global _rag_client_singleton
    if _rag_client_singleton is not None:
        try:
            await _rag_client_singleton.close()
        finally:
            _rag_client_singleton = None


__all__ = [
    "RAGServiceClient",
    "RAGClient",
    "RAGJobStatus",
    "RAGUploadResponse",
    "get_rag_client_singleton",
    "close_rag_client_singleton",
]
