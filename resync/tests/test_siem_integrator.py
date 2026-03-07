from __future__ import annotations

import pytest

from resync.core import siem_integrator
from resync.core.siem_integrator import SIEMEvent


@pytest.mark.asyncio
async def test_get_siem_integrator_returns_real_instance() -> None:
    siem_integrator._siem_integrator_instance = None

    instance = await siem_integrator.get_siem_integrator()

    assert isinstance(instance, siem_integrator.SIEMIntegrator)
    assert instance is siem_integrator.get_siem_integrator_sync()

    await instance.stop()
    siem_integrator._siem_integrator_instance = None


def test_siem_event_uses_distinct_cef_fields_and_utc_timestamp() -> None:
    event = SIEMEvent(
        event_id="evt-1",
        timestamp=0.0,
        source="auth",
        event_type="login",
        severity="low",
        category="authentication",
        message="User login",
        user_id="user-1",
        ip_address="10.0.0.1",
    )

    cef = event.to_cef()
    payload = event.to_json()

    assert "suser=user-1" in cef
    assert "src=10.0.0.1" in cef
    assert cef.count("src=") == 1
    assert '"@timestamp":"1970-01-01T00:00:00+00:00"' in payload
