from __future__ import annotations

import asyncio

import resync.app_factory as app_factory


def test_app_factory_does_not_monkey_patch_asyncio_iscoroutinefunction() -> None:
    assert app_factory.asyncio.iscoroutinefunction is asyncio.iscoroutinefunction
