"""Helpers to keep LangGraph nodes immutable and delta-based."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable


def build_state_delta(original_state: Mapping[str, Any], updated_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return only keys whose values changed relative to ``original_state``."""
    delta: dict[str, Any] = {}
    for key, value in updated_state.items():
        if key not in original_state or original_state[key] != value:
            delta[key] = value
    return delta


def _looks_like_full_state(result: Mapping[str, Any], original_state: Mapping[str, Any]) -> bool:
    return bool(original_state) and len(result) >= len(original_state) and set(original_state).issubset(result)


def wrap_langgraph_node(node: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a LangGraph node so it never mutates caller state and always returns deltas.

    The wrapper passes a shallow copy of the incoming state into the node. If the node
    returns the mutated working state (legacy pattern), the wrapper converts it to a delta.
    If the node already returns a delta dict, it is preserved as-is.
    Non-mapping results are passed through unchanged.
    """

    is_async = inspect.iscoroutinefunction(node) or inspect.iscoroutinefunction(getattr(node, '__call__', None))

    async def async_wrapper(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        original_state = dict(state)
        working_state = dict(state)
        result = await node(working_state, *args, **kwargs)
        if not isinstance(result, Mapping):
            return result
        if result is working_state or _looks_like_full_state(result, original_state):
            return build_state_delta(original_state, dict(result))
        return dict(result)

    def sync_wrapper(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        original_state = dict(state)
        working_state = dict(state)
        result = node(working_state, *args, **kwargs)
        if not isinstance(result, Mapping):
            return result
        if result is working_state or _looks_like_full_state(result, original_state):
            return build_state_delta(original_state, dict(result))
        return dict(result)

    wrapper = async_wrapper if is_async else sync_wrapper
    setattr(wrapper, '__name__', getattr(node, '__name__', node.__class__.__name__))
    setattr(wrapper, '__doc__', getattr(node, '__doc__', None))
    return wrapper
