from __future__ import annotations

import sys
from types import ModuleType

import pytest

from resync.api.routes.admin import v2 as admin_v2


@pytest.mark.asyncio
async def test_check_restart_required_masks_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _get_config_manager():  # type: ignore[no-untyped-def]
        raise RuntimeError("database offline: internal detail")

    fake_module = ModuleType("resync.services.config_manager")
    fake_module.get_config_manager = _get_config_manager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resync.services.config_manager", fake_module)

    result = await admin_v2.check_restart_required()

    assert result == {
        "restart_required": False,
        "error": "restart_status_unavailable",
    }
