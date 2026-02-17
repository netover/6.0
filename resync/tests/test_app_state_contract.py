from __future__ import annotations

from fastapi import FastAPI
import pytest

from resync.core.wiring import validate_app_state_contract


def test_app_state_contract_fails_fast_when_missing() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError) as exc:
        validate_app_state_contract(app)
    msg = str(exc.value)
    assert "enterprise_state" in msg
