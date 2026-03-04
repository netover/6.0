"""Agent registry and startup validations.

The codebase has multiple "agent" concepts:
- Native agents configured via :class:`resync.core.agent_manager.AgentManager`.
- Specialist agents (job_analyst/dependency/resource/knowledge).
- LangGraph-powered graphs and their prompt dependencies.

This module provides a single inventory with deterministic validations that can
run at startup and can be queried from the admin API.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from resync.core.agent_manager import AgentManager
from resync.core.prompts import get_prompt_manager
from resync.core.specialists.models import SpecialistType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentEntry:
    """Single inventory entry."""

    kind: str  # native | specialist | langgraph
    id: str
    display_name: str
    model: str | None
    tools: list[str]
    validations: dict[str, ValidationResult]


@dataclass(frozen=True, slots=True)
class AgentRegistry:
    """Full registry output."""

    entries: list[AgentEntry]
    summary: dict[str, Any]


def _collect_prompt_ids_from_code(package_root: Path) -> set[str]:
    """Collect prompt_id string literals from the codebase.

    We look for keyword arguments named "prompt_id" with a constant string
    value. This is intentionally conservative (won't try to evaluate dynamic
    expressions).
    """

    prompt_ids: set[str] = set()
    for py in package_root.rglob("*.py"):
        # Avoid scanning tests/venv etc. (none should exist in resync package)
        try:
            src = py.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(src, filename=str(py))
        except SyntaxError:
            # Don't fail startup due to one file; surface in validations.
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "prompt_id":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, str
                ):
                    prompt_ids.add(kw.value.value)

    return prompt_ids


def _validate_tools(agent_manager: AgentManager, tools: Iterable[str]) -> ValidationResult:
    missing = [t for t in tools if t not in agent_manager.tools]
    return ValidationResult(
        ok=len(missing) == 0,
        details={"missing": missing, "available_count": len(agent_manager.tools)},
    )


def _validate_prompts(prompt_ids: Iterable[str]) -> ValidationResult:
    prompt_manager = get_prompt_manager()
    missing: list[str] = []
    for pid in prompt_ids:
        try:
            prompt_manager.get_prompt(pid)
        except KeyError:
            missing.append(pid)
    return ValidationResult(ok=len(missing) == 0, details={"missing": missing})


def build_agent_registry(agent_manager: AgentManager) -> AgentRegistry:
    """Build registry inventory + validations."""

    entries: list[AgentEntry] = []

    # 1) Native agents (AgentManager configs)
    for cfg in agent_manager.agent_configs:
        tools = list(getattr(cfg, "tools", []) or [])
        validations = {
            "tools": _validate_tools(agent_manager, tools),
        }
        entries.append(
            AgentEntry(
                kind="native",
                id=cfg.id,
                display_name=cfg.name,
                model=getattr(cfg, "model_name", None),
                tools=tools,
                validations=validations,
            )
        )

    # 2) Specialist agents
    for st in SpecialistType:
        entries.append(
            AgentEntry(
                kind="specialist",
                id=st.value,
                display_name=f"Specialist: {st.value}",
                model=None,
                tools=[],
                validations={},
            )
        )

    # 3) LangGraph prompt dependencies (best-effort static scan)
    # Scan only langgraph package (keeps runtime small and focused)
    langgraph_root = Path(__file__).resolve().parents[2] / "langgraph"
    prompt_ids = _collect_prompt_ids_from_code(langgraph_root) if langgraph_root.exists() else set()
    prompt_validation = _validate_prompts(prompt_ids)
    entries.append(
        AgentEntry(
            kind="langgraph",
            id="langgraph",
            display_name="LangGraph graphs",
            model=None,
            tools=[],
            validations={
                "prompts": prompt_validation,
                "prompt_ids": ValidationResult(ok=True, details={"found": sorted(prompt_ids)}),
            },
        )
    )

    summary = {
        "total": len(entries),
        "native": len([e for e in entries if e.kind == "native"]),
        "specialist": len([e for e in entries if e.kind == "specialist"]),
        "langgraph": len([e for e in entries if e.kind == "langgraph"]),
        "prompts_missing": prompt_validation.details.get("missing", []),
    }
    return AgentRegistry(entries=entries, summary=summary)


def validate_registry_or_raise(registry: AgentRegistry, *, strict: bool) -> None:
    """Fail-fast validation helper for startup."""

    issues: list[str] = []
    for entry in registry.entries:
        for name, res in entry.validations.items():
            if not res.ok:
                issues.append(f"{entry.kind}:{entry.id}:{name}:{res.details}")

    if issues:
        msg = "agent_registry_validation_failed"
        logger.error(msg, issues_count=len(issues), issues=issues[:50])
        if strict:
            raise RuntimeError(f"{msg}: {issues[:10]}")
