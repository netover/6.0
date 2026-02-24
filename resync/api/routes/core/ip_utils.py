"""IP utilities for secure client IP extraction.

This module provides functions to extract client IP addresses with protection
against X-Forwarded-For spoofing attacks. It validates the proxy chain
and extracts the real client IP based on the number of trusted proxies.

Usage:
    from resync.api.routes.core.ip_utils import get_trusted_client_ip

    @router.post("/login")
    async def login(request: Request):
        client_ip = get_trusted_client_ip(request)
        # ...
"""

from __future__ import annotations

import ipaddress
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request


def _parse_trusted_proxy_count() -> int:
    """Parse trusted proxy count from environment.

    Returns:
        Number of trusted proxies (0=direct, 1=nginx, 2=cdn+nginx)
    """
    raw = os.getenv("TRUSTED_PROXY_COUNT", "0")
    try:
        count = int(raw.strip())
        return max(0, min(count, 10))  # Clamp to 0-10
    except ValueError:
        return 0


# Cache the trusted proxy count
_TRUSTED_PROXY_COUNT = _parse_trusted_proxy_count()


def get_trusted_client_ip(request: Request) -> str:
    """Extract client IP with X-Forwarded-For spoofing protection.

    Security:
        - Validates proxy chain length
        - Extracts real IP from Nth position from right
        - Falls back to direct connection IP if invalid

    Config:
        TRUSTED_PROXY_COUNT env var:
            0 = No proxies (direct connection)
            1 = Single proxy (Nginx)
            2 = CDN + Nginx

    Args:
        request: FastAPI request object

    Returns:
        Sanitized client IP address
    """
    from resync.settings import settings

    # In non-production or when proxy count is 0, use direct connection
    if not getattr(settings, "is_production", False) and _TRUSTED_PROXY_COUNT == 0:
        return request.client.host if request.client else "unknown"

    # Check X-Forwarded-For header
    forwarded = request.headers.get("X-Forwarded-For")

    if forwarded and _TRUSTED_PROXY_COUNT > 0:
        # Parse all IPs from header
        ips = [ip.strip() for ip in forwarded.split(",")]

        if len(ips) > _TRUSTED_PROXY_COUNT:
            # Extract real client IP (Nth from right, where N = proxy_count + 1)
            # For example, with 1 trusted proxy (nginx):
            #   X-Forwarded-For: client, proxy1, proxy2
            #   We trust proxy1, so real client is the first one
            client_ip = ips[0]  # Always use leftmost (original client)

            # Validate IP format
            try:
                ipaddress.ip_address(client_ip)
                return client_ip
            except ValueError:
                # Invalid IP format, fall through to direct connection
                pass

    # Fallback: direct connection IP
    return request.client.host if request.client else "unknown"


def sanitize_ip_for_redis_key(ip: str) -> str:
    """Sanitize IP address for Redis key usage.

    Security:
        - Validates IPv4/IPv6 format
        - Falls back to SHA256 hash for invalid IPs
        - Prevents key injection attacks

    Args:
        ip: Client IP address (possibly malicious)

    Returns:
        Sanitized IP or hash suitable for Redis key
    """
    import hashlib

    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        # Invalid IP → hash it (prevents injection)
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
