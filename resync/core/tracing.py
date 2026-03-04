from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class TraceSpan:
    """Minimal trace span used by LangGraph nodes.

    This replaces the previous external-provider-backed tracer with a no-op implementation.
    """

    output: Any | None = None


class NoopTracer:
    @asynccontextmanager
    async def trace(self, name: str, **_: Any) -> AsyncIterator[TraceSpan]:
        span = TraceSpan()
        yield span


_NOOP_TRACER = NoopTracer()


def get_tracer() -> NoopTracer:
    return _NOOP_TRACER
