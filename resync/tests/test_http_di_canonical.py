from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def test_dependency_yield_teardown_runs() -> None:
    closed = {"value": False}

    def dep_with_yield():
        try:
            yield "resource"
        finally:
            closed["value"] = True

    app = FastAPI()

    @app.get("/x")
    def x(res: str = Depends(dep_with_yield)) -> dict[str, str]:
        return {"res": res}

    with TestClient(app) as client:
        r = client.get("/x")
        assert r.status_code == 200
        assert r.json() == {"res": "resource"}

    assert closed["value"] is True
