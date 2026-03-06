# pylint
"""Centralized Valkey client factory (canonical)."""

from __future__ import annotations

from fastapi import Depends


def get_valkey_client():
    """Return the canonical Valkey async client."""
    from resync.core.valkey_init import get_valkey_client as _get

    return _get()


def get_valkey_client_sync():
    """Return the canonical Valkey client for sync code paths."""
    return get_valkey_client()


async def get_valkey_client_dep():
    """FastAPI dependency wrapper for Valkey client."""
    return get_valkey_client()


__all__ = [
    "get_valkey_client",
    "get_valkey_client_sync",
    "get_valkey_client_dep",
    "Depends",
]
