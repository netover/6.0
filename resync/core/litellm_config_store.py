from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from resync.core.exceptions import IntegrationError
from resync.core.valkey_init import get_valkey_client, is_valkey_available
import time
import json

CONFIG_PATH = Path(__file__).resolve().parent / "litellm_config.yaml"

HISTORY_LIST_KEY = "resync:llm:config:history:v1"  # LPUSH json entries, LTRIM 0..49

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

async def _push_history_snapshot(previous_text: str) -> None:
    # Keep small history for rollback in Admin UI
    if not is_valkey_available():
        return
    client = get_valkey_client()
    if client is None:
        return
    entry = {
        "ts_ms": int(time.time() * 1000),
        "yaml": previous_text,
    }
    try:
        pipe = client.pipeline()
        pipe.lpush(HISTORY_LIST_KEY, _json_dumps(entry))
        pipe.ltrim(HISTORY_LIST_KEY, 0, 49)
        await pipe.execute()
    except asyncio.CancelledError:
        raise
    except Exception:

        return

async def get_history(limit: int = 20) -> list[dict[str, Any]]:
    if not is_valkey_available():
        return []
    client = get_valkey_client()
    if client is None:
        return []
    raw = await client.lrange(HISTORY_LIST_KEY, 0, max(0, limit - 1))
    out: list[dict[str, Any]] = []
    for r in raw:
        if isinstance(r, (bytes, bytearray)):
            r = r.decode("utf-8", "ignore")
        try:
            obj = json.loads(r)
            if isinstance(obj, dict):
                out.append(obj)
        except asyncio.CancelledError:
            raise
        except Exception:

            continue
    return out



@dataclass(frozen=True)
class LiteLLMConfig:
    raw: dict[str, Any]
    text: str
    path: Path
    mtime_ns: int


def _validate_schema(data: dict[str, Any]) -> None:
    # Minimal validation (fail-fast). Keep permissive for LiteLLM options.
    if "model_list" not in data or not isinstance(data["model_list"], list):
        raise IntegrationError("litellm_config_invalid: model_list must be a list")
    if "model_aliases" in data and not isinstance(data["model_aliases"], dict):
        raise IntegrationError("litellm_config_invalid: model_aliases must be a dict")
    if "router_settings" in data and not isinstance(data["router_settings"], dict):
        raise IntegrationError("litellm_config_invalid: router_settings must be a dict")


def load_litellm_config() -> LiteLLMConfig:
    if not CONFIG_PATH.exists():
        raise IntegrationError(f"litellm_config_missing: {CONFIG_PATH}")

    text = CONFIG_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise IntegrationError("litellm_config_invalid: root must be a mapping")

    _validate_schema(data)
    stat = CONFIG_PATH.stat()
    return LiteLLMConfig(raw=data, text=text, path=CONFIG_PATH, mtime_ns=stat.st_mtime_ns)


async def load_litellm_config_async() -> LiteLLMConfig:
    return await asyncio.to_thread(load_litellm_config)


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


async def save_litellm_config_async(new_text: str) -> LiteLLMConfig:
    def _save() -> LiteLLMConfig:
        data = yaml.safe_load(new_text)
        if not isinstance(data, dict):
            raise IntegrationError("litellm_config_invalid: root must be a mapping")
        _validate_schema(data)
        previous = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
        # store previous in history (best-effort)
        try:
            import asyncio as _asyncio
            _loop = _asyncio.get_running_loop()
            _loop.create_task(_push_history_snapshot(previous))
        except asyncio.CancelledError:
            raise
        except Exception:

            import logging
            logging.getLogger(__name__).debug("Ignored exception in /mnt/data/proj_v5/resync/core/litellm_config_store.py", exc_info=True)
        _atomic_write(CONFIG_PATH, new_text)
        stat = CONFIG_PATH.stat()
        return LiteLLMConfig(raw=data, text=new_text, path=CONFIG_PATH, mtime_ns=stat.st_mtime_ns)

    return await asyncio.to_thread(_save)
