"""
GZip Compression Middleware for Resync API.

This module provides compression configuration for reducing response sizes
and improving transfer speeds over the network.
"""

from dataclasses import dataclass
from typing import Literal

from starlette.middleware.gzip import GZipMiddleware


@dataclass
class CompressionConfig:
    """
    Configuration for response compression.

    Attributes:
        enabled: Whether compression is enabled
        minimum_size: Minimum response size in bytes to compress (default 500 bytes)
        compresslevel: GZip compression level (1-9, default 6)
            - 1: Fastest, least compression
            - 6: Balanced (recommended for APIs)
            - 9: Slowest, best compression
    """

    enabled: bool = True
    minimum_size: int = 500
    compresslevel: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = 6


def get_compression_config() -> CompressionConfig:
    """
    Get compression configuration from settings.

    Returns:
        CompressionConfig with values from environment or defaults
    """
    try:
        from resync.settings import settings

        return CompressionConfig(
            enabled=getattr(settings, "compression_enabled", True),
            minimum_size=getattr(settings, "compression_minimum_size", 500),
            compresslevel=getattr(settings, "compression_level", 6),
        )
    except ImportError:
        return CompressionConfig()


def setup_compression(app, config: CompressionConfig | None = None) -> None:
    """
    Add GZip compression middleware to the FastAPI application.

    Args:
        app: The FastAPI application instance
        config: Optional compression configuration. If None, uses defaults.

    Example:
        >>> from fastapi import FastAPI
        >>> from resync.api.middleware.compression import setup_compression
        >>> app = FastAPI()
        >>> setup_compression(app)
    """
    if config is None:
        config = get_compression_config()

    if not config.enabled:
        return

    app.add_middleware(
        GZipMiddleware,
        minimum_size=config.minimum_size,
        compresslevel=config.compresslevel,
    )


__all__ = [
    "CompressionConfig",
    "GZipMiddleware",
    "get_compression_config",
    "setup_compression",
]
