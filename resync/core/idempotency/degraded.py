"""Degraded idempotency manager for when Redis is unavailable.

This module provides a fail-fast implementation that returns HTTP 503
when Redis is down, instead of silently failing with AttributeError.
"""

from typing import Any

from fastapi import HTTPException, status


class DegradedIdempotencyManager:
    """No-op idempotency manager that fails fast when Redis is unavailable.

    This manager is used in degraded mode when Redis fails during startup
    but strict=False allows the application to continue running.

    All idempotent operations will return HTTP 503 Service Unavailable,
    which is the correct response for a temporarily degraded service.
    """

    async def get_cached_response(
        self, idempotency_key: str, request_data: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Fail fast with 503 when attempting to get cached response.

        Args:
            idempotency_key: The idempotency key
            request_data: Optional request data for validation

        Raises:
            HTTPException: Always raises 503 Service Unavailable
        """
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Idempotency service temporarily unavailable (Redis down)",
            headers={"Retry-After": "60"},  # Suggest retry in 60 seconds
        )

    async def store_response(
        self,
        idempotency_key: str,
        response_data: dict[str, Any],
        request_data: dict[str, Any] | None = None,
    ) -> None:
        """Fail fast with 503 when attempting to store response.

        Args:
            idempotency_key: The idempotency key
            response_data: The response to cache
            request_data: Optional request data for validation

        Raises:
            HTTPException: Always raises 503 Service Unavailable
        """
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Idempotency service temporarily unavailable (Redis down)",
            headers={"Retry-After": "60"},
        )

    async def delete_key(self, idempotency_key: str) -> None:
        """Fail fast with 503 when attempting to delete key.

        Args:
            idempotency_key: The idempotency key to delete

        Raises:
            HTTPException: Always raises 503 Service Unavailable
        """
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Idempotency service temporarily unavailable (Redis down)",
            headers={"Retry-After": "60"},
        )


__all__ = ["DegradedIdempotencyManager"]
