from __future__ import annotations

import pytest
from fastapi import HTTPException

from resync.api.routes.admin import users as admin_users


class _FailingAuthService:
    async def create_user(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("duplicate email: internal detail")


@pytest.mark.asyncio
async def test_create_user_masks_value_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admin_users, "get_auth_service", lambda: _FailingAuthService())

    with pytest.raises(HTTPException) as exc_info:
        await admin_users.create_user(
            admin_users.AdminUserCreate(
                username="testuser",
                email="user@example.com",
                password="supersecret",
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid user data"
