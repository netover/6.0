from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib
import yaml

def load_toml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}

def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}
