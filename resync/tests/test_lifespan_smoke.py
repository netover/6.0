from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

from resync.app_factory import create_app
from resync.core.types.app_state import enterprise_state_from_app


if importlib.util.find_spec("sqlalchemy") is None:
    pytest.skip("sqlalchemy not installed in this environment", allow_module_level=True)

REQUIRED_STATE_KEYS = [
    "connection_manager",
    "knowledge_graph",
    "tws_client",
    "agent_manager",
    "hybrid_router",
    "idempotency_manager",
    "llm_service",
]


def test_lifespan_initializes_app_state_singletons() -> None:
    app = create_app()
    assert not getattr(
        getattr(app.state, "enterprise_state", None), "startup_complete", False
    )

    with TestClient(app) as client:
        st = enterprise_state_from_app(client.app)
        assert st.startup_complete is True
        for key in REQUIRED_STATE_KEYS:
            assert getattr(st, key) is not None, f"missing enterprise_state.{key}"

    assert enterprise_state_from_app(app).domain_shutdown_complete is True
