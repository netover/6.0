from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PromptType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"


@dataclass(frozen=True)
class PromptConfig:
    variables: list[str]


@dataclass
class Prompt:
    id: str
    name: str
    type: PromptType
    version: str
    content: str
    description: str = ""
    config: PromptConfig = PromptConfig(variables=[])
    default_values: dict[str, str] | None = None
    model_hint: str | None = None
    temperature_hint: float | None = None
    max_tokens_hint: int | None = None
    is_active: bool = True
    is_default: bool = False

    def compile(self, **variables: Any) -> str:
        # Simple template replacement: {var}
        vals = {}
        if self.default_values:
            vals.update(self.default_values)
        vals.update({k: str(v) for k, v in variables.items()})
        out = self.content
        for key, val in vals.items():
            out = out.replace("{" + key + "}", val)
        return out


class PromptManager:
    """YAML-backed prompt manager (external prompt provider removed).

    Prompts are loaded from resync/prompts/agent_prompts.yaml.
    """

    def __init__(self, prompt_file: Path) -> None:
        self._prompt_file = prompt_file
        self._loaded: dict[str, Prompt] | None = None

    def _load(self) -> dict[str, Prompt]:
        if self._loaded is not None:
            return self._loaded
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load prompt definitions") from e

        data = yaml.safe_load(self._prompt_file.read_text(encoding="utf-8")) or {}
        prompts: dict[str, Prompt] = {}
        for item in data.get("prompts", []):
            pid = str(item.get("id") or item.get("name") or "")
            if not pid:
                continue
            ptype = PromptType(str(item.get("type", "system")))
            prompts[pid] = Prompt(
                id=pid,
                name=str(item.get("name", pid)),
                type=ptype,
                version=str(item.get("version", "1.0.0")),
                content=str(item.get("content", "")),
                description=str(item.get("description", "")),
                config=PromptConfig(variables=list(item.get("variables", []) or [])),
                default_values=dict(item.get("default_values", {}) or {}),
                model_hint=item.get("model_hint"),
                temperature_hint=item.get("temperature_hint"),
                max_tokens_hint=item.get("max_tokens_hint"),
                is_active=bool(item.get("is_active", True)),
                is_default=bool(item.get("is_default", False)),
            )
        self._loaded = prompts
        return prompts

    async def get_prompt(self, prompt_id: str) -> Prompt | None:
        return self._load().get(prompt_id)

    async def list_prompts(self) -> list[Prompt]:
        return list(self._load().values())


_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        prompt_file = Path(__file__).resolve().parent.parent / "prompts" / "agent_prompts.yaml"
        _prompt_manager = PromptManager(prompt_file=prompt_file)
    return _prompt_manager
