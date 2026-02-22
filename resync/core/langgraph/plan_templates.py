"""Deterministic planner templates for Resync v6.1 (Golden Path).

Why templates?
- TWS/AIOps troubleshooting follows predictable sequences.
- Templates reduce latency and prevent the LLM from "inventing" steps.

Design constraints:
- Must be JSON-serializable when stored in AgentState (LangGraph persistence).
- The executor in :mod:`resync.core.langgraph.agent_graph` interprets these steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PlanStep:
    """One executable step in an execution plan."""

    id: str
    action: str
    description: str
    requires: list[str] = field(default_factory=list)
    on_failure: str = "skip"  # "skip" | "abort"


def create_plan(template_key: str) -> dict:
    """Create a JSON-serializable plan from a template key."""
    steps = [
        {
            "id": s.id,
            "action": s.action,
            "description": s.description,
            "requires": list(s.requires),
            "on_failure": s.on_failure,
            "completed": False,
            "error": None,
        }
        for s in PLAN_TEMPLATES.get(template_key, [])
    ]

    return {
        "template": template_key,
        "steps": steps,
        "total_steps": len(steps),
    }


PLAN_TEMPLATES: dict[str, list[PlanStep]] = {
    # Troubleshooting: collect -> analyze -> synthesize
    "troubleshoot": [
        PlanStep(
            id="collect",
            action="orchestrator_collect",
            description="Coletar status, logs, dependências e histórico em paralelo",
            on_failure="abort",
        ),
        PlanStep(
            id="analyze",
            action="analyze_evidence",
            description="Extrair sinais (RC/ABEND, erros nos logs, dependências afetadas)",
            requires=["collect"],
        ),
        PlanStep(
            id="synthesize",
            action="synthesize_diagnosis",
            description="Sintetizar diagnóstico e próximos passos",
            requires=["collect"],
        ),
    ],
    # Status: collect light -> format
    "status": [
        PlanStep(
            id="collect",
            action="orchestrator_collect_light",
            description="Coletar status e dependências (sem logs)",
        ),
        PlanStep(
            id="format",
            action="format_status",
            description="Formatar resposta de status",
            requires=["collect"],
        ),
    ],
    # Action: validate -> approve -> execute -> verify
    "action": [
        PlanStep(
            id="validate",
            action="validate_action",
            description="Validar se a ação é permitida no estado atual",
            on_failure="abort",
        ),
        PlanStep(
            id="approve",
            action="request_approval",
            description="Solicitar aprovação HITL",
            requires=["validate"],
            on_failure="abort",
        ),
        PlanStep(
            id="execute",
            action="execute_action",
            description="Executar a ação no TWS",
            requires=["approve"],
            on_failure="abort",
        ),
        PlanStep(
            id="verify",
            action="verify_action",
            description="Verificar se a ação teve efeito real no TWS",
            requires=["execute"],
        ),
    ],
}
