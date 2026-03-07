from __future__ import annotations

from uuid import UUID

import pytest

from resync.core.orchestration.events import EventBus, EventType, OrchestrationEvent


@pytest.mark.asyncio
async def test_publish_tolerates_unsubscribe_during_callback() -> None:
    bus = EventBus()
    calls: list[str] = []

    subscription_id: str | None = None

    async def _callback(event: OrchestrationEvent) -> None:
        assert subscription_id is not None
        calls.append(event.trace_id)
        assert bus.unsubscribe(subscription_id) is True

    subscription_id = bus.subscribe(EventType.EXECUTION_STARTED, _callback)

    await bus.publish(
        OrchestrationEvent(
            type=EventType.EXECUTION_STARTED,
            trace_id="trace-1",
            execution_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    )

    assert calls == ["trace-1"]
