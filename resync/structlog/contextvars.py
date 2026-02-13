from __future__ import annotations
from typing import Any

def bind_contextvars(**kwargs: Any) -> None:
    return None

def clear_contextvars() -> None:
    return None

def merge_contextvars(logger, method_name: str, event_dict):
    # pass-through processor signature
    return event_dict
