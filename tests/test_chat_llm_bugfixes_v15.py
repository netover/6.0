from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_chat_module_import() -> None:
    # Regression test for BUG #1: `os` must be imported in resync.api.chat.
    import importlib

    import resync.api.chat as chat

    importlib.reload(chat)


@pytest.mark.asyncio
async def test_chat_with_metadata_fallback_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for BUG #2: chat_with_metadata() must return the keys
    # expected by the HTTP chat route, even in fallback mode.
    import resync.core.agent_router as agent_router
    from resync.core.agent_manager import UnifiedAgent

    # Avoid heavy router creation (not needed for this test)
    monkeypatch.setattr(agent_router, "create_router", lambda *args, **kwargs: MagicMock())

    agent = UnifiedAgent(agent_manager=MagicMock())

    async def _fake_chat(*args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        return "ok"

    monkeypatch.setattr(agent, "chat", _fake_chat)

    result = await agent.chat_with_metadata(
        message="hello",
        include_history=False,
        tws_instance_id="tws-1",
        conversation_id="cid-1",
    )

    required = {
        "response",
        "intent",
        "confidence",
        "handler",
        "entities",
        "tools_used",
        "processing_time_ms",
        "tws_instance_id",
    }
    assert required.issubset(set(result.keys()))


@pytest.mark.asyncio
async def test_arun_tool_synthesis_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for BUG #4: synthesis call must guard against empty choices.
    # We simulate:
    #   1) First LLM call returns tool_calls.
    #   2) Second LLM call (synthesis) returns choices=[].
    # Expected: no IndexError; returns a fallback string.
    from resync.core.agent_manager import Agent

    # Make agent creation/test deterministic and dependency-light
    monkeypatch.setattr(Agent, "_ensure_litellm_configured", classmethod(lambda cls: None))
    monkeypatch.setattr(Agent, "_convert_tools_to_litellm_format", lambda self: [])

    agent = Agent(tools=[], model="openrouter/free", instructions="", name="Test Agent")

    async def _fake_execute_tool(tool_name: str, arguments: dict) -> str:  # type: ignore[type-arg]
        return "tool_result"

    monkeypatch.setattr(agent, "_execute_tool", _fake_execute_tool)

    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="dummy_tool", arguments="{}"),
    )
    message_obj = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])

    first_choice = SimpleNamespace(message=message_obj)
    first_response = SimpleNamespace(choices=[first_choice])

    synthesis_empty = SimpleNamespace(choices=[])

    calls = {"n": 0}

    async def _fake_call_litellm_with_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return first_response if calls["n"] == 1 else synthesis_empty

    monkeypatch.setattr(agent, "_call_litellm_with_retry", _fake_call_litellm_with_retry)

    # Ensure `import litellm` inside arun doesn't fail in minimal envs
    import sys
    import types

    if "litellm" not in sys.modules:
        sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=False)

    result = await agent.arun("hi")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_auditor_worker_cancel_does_not_corrupt_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for BUG #5: cancelling the auditor worker while it is
    # blocked on `queue.get()` must not call `task_done()` and must not raise ValueError.
    import resync.api.chat as chat

    q: asyncio.Queue[None] = asyncio.Queue()
    monkeypatch.setattr(chat, "_auditor_queue", q, raising=False)

    task = asyncio.create_task(chat._auditor_worker())
    await asyncio.sleep(0)  # let the worker reach the blocking get()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cleanup to avoid cross-test interference
    monkeypatch.setattr(chat, "_auditor_queue", None, raising=False)


@pytest.mark.asyncio
async def test_rate_limiter_exception_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for BUG #6: programming errors must not be silently swallowed.
    import resync.api.chat as chat
    import resync.core.security.rate_limiter_v2 as rl

    async def _boom(_client_ip: str) -> tuple[bool, int]:
        raise TypeError("programming error")

    monkeypatch.setattr(rl, "ws_allow_message", _boom)

    fake_ws = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    with pytest.raises(TypeError):
        await chat._ws_allow_message_checked(fake_ws)  # type: ignore[arg-type]
