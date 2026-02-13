from __future__ import annotations
from typing import Any, Dict, Protocol

EventDict = Dict[str, Any]

class WrappedLogger(Protocol):
    def bind(self, **kw: Any) -> Any: ...
