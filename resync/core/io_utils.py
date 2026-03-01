from __future__ import annotations

import asyncio
from pathlib import Path

def _read_text_sync(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)

def _write_text_sync(path: Path, data: str, encoding: str) -> None:
    path.write_text(data, encoding=encoding)

async def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text file without blocking the event loop."""
    p = Path(path)
    return await asyncio.to_thread(_read_text_sync, p, encoding)

async def write_text(path: str | Path, data: str, *, encoding: str = "utf-8") -> None:
    """Write a text file without blocking the event loop."""
    p = Path(path)
    await asyncio.to_thread(_write_text_sync, p, data, encoding)
