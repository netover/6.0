"""ROMA graph nodes built with LLMFactory and resilient parsing."""

from __future__ import annotations

import json
from typing import Any

from resync.core.langgraph.roma_models import RomaState, SubTask
from resync.core.structured_logger import get_logger
from resync.core.utils.llm_factories import LLMFactory

logger = get_logger(__name__)


async def atomizer_node(state: RomaState) -> dict[str, Any]:
    """Detect whether the user query is atomic or needs decomposition."""
    query = state.get("user_query", "").strip()
    if not query:
        return {"is_atomic": True, "execution_logs": ["empty_query_defaulted_to_atomic"]}

    prompt = (
        "Classify if this task is atomic or composite. "
        "Return ONLY JSON with {\"is_atomic\": boolean}.\n\n"
        f"Query: {query}"
    )
    try:
        response = await LLMFactory.call_llm(prompt=prompt, model="tws-intent", temperature=0.1)
        payload = json.loads(response)
        is_atomic = bool(payload.get("is_atomic", False))
    except Exception as e:
        logger.warning("roma_atomizer_fallback", error=str(e))
        is_atomic = False

    return {
        "is_atomic": is_atomic,
        "execution_logs": [f"atomizer_result={'atomic' if is_atomic else 'composite'}"],
    }


async def planner_node(state: RomaState) -> dict[str, Any]:
    """Create a deterministic plan for composite tasks."""
    query = state.get("user_query", "")
    prompt = (
        "Break the request into up to 5 practical tasks. "
        "Return ONLY JSON array with objects: id,title,description.\n\n"
        f"Request: {query}"
    )

    try:
        response = await LLMFactory.call_llm(prompt=prompt, model="tws-reasoning", temperature=0.1)
        parsed = json.loads(response)
        tasks = [SubTask.model_validate(item) for item in parsed if isinstance(item, dict)]
    except Exception as e:
        logger.warning("roma_planner_fallback", error=str(e))
        tasks = [
            SubTask(
                id="task-1",
                title="Handle user request",
                description=query or "No query provided",
            )
        ]

    return {
        "plan": tasks,
        "execution_logs": [f"planner_created_{len(tasks)}_tasks"],
    }


async def executor_node(state: RomaState) -> dict[str, Any]:
    """Execute each subtask through a concise reasoning prompt."""
    plan = state.get("plan", [])
    results: list[dict[str, Any]] = []

    for task in plan:
        prompt = (
            "You are executing one orchestration subtask. "
            "Provide a concise action/result summary for this task.\n\n"
            f"Task: {task.title}\nDescription: {task.description}"
        )
        try:
            output = await LLMFactory.call_llm(prompt=prompt, model="tws-reasoning", temperature=0.2)
            results.append({"task_id": task.id, "status": "done", "output": output})
        except Exception as e:
            logger.warning("roma_executor_task_failed", task_id=task.id, error=str(e))
            results.append({"task_id": task.id, "status": "failed", "output": str(e)})

    return {
        "execution_results": results,
        "execution_logs": [f"executor_completed_{len(results)}_results"],
    }


async def aggregator_node(state: RomaState) -> dict[str, Any]:
    """Aggregate outputs into a user-facing summary."""
    if state.get("is_atomic") and not state.get("plan"):
        summary = "Solicitação tratada como tarefa atômica; nenhuma decomposição adicional necessária."
        return {"final_response": summary, "execution_logs": ["aggregator_atomic_short_circuit"]}

    results = state.get("execution_results", [])
    if not results:
        return {
            "final_response": "Não foi possível gerar resultados para a solicitação.",
            "execution_logs": ["aggregator_no_results"],
        }

    lines = [f"- {item['task_id']}: {item['status']}" for item in results]
    return {
        "final_response": "Execução ROMA concluída:\n" + "\n".join(lines),
        "execution_logs": ["aggregator_built_final_response"],
    }


async def verifier_node(state: RomaState) -> dict[str, Any]:
    """Provide lightweight verification notes for observability."""
    results = state.get("execution_results", [])
    failed = [item["task_id"] for item in results if item.get("status") == "failed"]

    if failed:
        notes = [f"Falhas detectadas nas tarefas: {', '.join(failed)}"]
    else:
        notes = ["Todas as tarefas ROMA foram concluídas sem falhas registradas."]

    return {
        "verification_notes": notes,
        "execution_logs": ["verifier_completed"],
    }
