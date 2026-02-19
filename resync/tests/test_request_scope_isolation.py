from __future__ import annotations

import asyncio
import uuid
from contextvars import ContextVar
from typing import AsyncIterator

import httpx
import pytest
from fastapi import Depends, FastAPI

_request_id: ContextVar[str] = ContextVar("_request_id", default="")

_teardown_calls: list[str] = []


async def request_id_dep() -> AsyncIterator[str]:
    # per-request "resource"
    rid = str(uuid.uuid4())
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)
        _teardown_calls.append(rid)


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/rid")
    async def rid_endpoint(rid: str = Depends(request_id_dep)) -> dict[str, str]:
        # Ensure contextvar matches dependency output in the same request.
        assert _request_id.get() == rid
        return {"rid": rid}

    return app


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_request_scoped_dependency_is_isolated_under_concurrency() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async def call_once() -> str:
            resp = await client.get("/rid")
            resp.raise_for_status()
            return resp.json()["rid"]

        # Run many concurrent requests
        rids = await asyncio.gather(*(call_once() for _ in range(50)))

    # All rids should be unique (no cross-request reuse)
    assert len(set(rids)) == len(rids)

    # Teardown called once per request, matching rids produced
    assert sorted(_teardown_calls) == sorted(rids)
