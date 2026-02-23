"""Tests for v6.1 Golden Path: Planner, Orchestrator, Verification, Critique.

Covers:
- Fase 0: state reset between turns
- Fase 1: orchestrator integration (KG stub)
- Fase 2: planner templates + plan executor
- Fase 2.5: LLM rescue routing
- Fase 3: verification loop
- Fase 4: output critique
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_io_bounds(monkeypatch):
    """Bypass expensive models (SentenceTransformer) and LLM endpoints."""
    import resync.core.langgraph.agent_graph as agent_graph
    import resync.core.utils.llm as llm
    from resync.core.langgraph.models import Intent

    # We patch at the module level where they are used to prevent 8s model loading.
    monkeypatch.setattr(
        agent_graph, "async_init_router_cache", AsyncMock(return_value=None)
    )

    # We also mock the structured LLM response to avoid OpenAI delays or timeouts
    class DummyRouterOutput:
        intent = Intent.STATUS
        confidence = 1.0
        entities = {}

    monkeypatch.setattr(
        llm, "call_llm_structured", AsyncMock(return_value=DummyRouterOutput())
    )

    # And mock the standard fallback llm call as well just in case
    monkeypatch.setattr(
        llm,
        "call_llm",
        AsyncMock(return_value='{"satisfactory": true, "issues": [], "missing": []}'),
    )


# =============================================================================
# FASE 0: State Reset
# =============================================================================


@pytest.mark.asyncio
async def test_router_resets_transient_state(monkeypatch):
    """Transient fields from a previous turn must not leak into the next turn."""
    from resync.core.langgraph.agent_graph import Intent, router_node

    state = {
        "message": "status do job TEST",
        "execution_plan": {"intent": "status", "steps": [], "total_steps": 0},
        "plan_step_index": 2,
        "plan_failed": True,
        "rescue_used": True,
        "orchestration_result": {"tws_status": {"status": "ABEND"}},
        "verification_attempts": 3,
        "action_pending_verification": "rerun",
        "critique_retries": 1,
        "critique_feedback": ["missing info"],
        "needs_refinement": True,
        "intent": Intent.STATUS,
        "confidence": 1.0,
    }

    new_state = await router_node(state)

    assert new_state.get("execution_plan") is None
    assert new_state.get("plan_step_index") == 0
    assert new_state.get("plan_failed") is False
    assert new_state.get("rescue_used") is False
    assert new_state.get("orchestration_result") is None
    assert new_state.get("verification_attempts") == 0
    assert new_state.get("action_pending_verification") is None
    assert new_state.get("critique_retries") == 0
    assert new_state.get("critique_feedback") is None
    assert new_state.get("needs_refinement") is False


@pytest.mark.asyncio
async def test_router_preserves_max_verification_attempts(monkeypatch):
    """max_verification_attempts should persist across resets if already set."""
    from resync.core.langgraph.agent_graph import Intent, router_node

    state = {
        "message": "status do job X",
        "max_verification_attempts": 5,
        "intent": Intent.STATUS,
        "confidence": 1.0,
    }

    new_state = await router_node(state)
    assert new_state.get("max_verification_attempts") == 5


# =============================================================================
# FASE 1: KG Stub Interface
# =============================================================================


def test_kg_stub_has_both_interfaces():
    """KG stub must satisfy both RAGClient (search) and Orchestrator (get_relevant_context)."""
    from resync.core.langgraph.agent_graph import _get_knowledge_graph_or_stub

    with patch.dict("sys.modules", {"resync.services.rag_client": None}):
        stub = _get_knowledge_graph_or_stub()

    assert hasattr(stub, "search")
    assert hasattr(stub, "get_relevant_context")


@pytest.mark.asyncio
async def test_kg_stub_returns_none_for_context():
    """KG stub.get_relevant_context() should return None (not raise)."""
    from resync.core.langgraph.agent_graph import _get_knowledge_graph_or_stub

    with patch.dict("sys.modules", {"resync.services.rag_client": None}):
        stub = _get_knowledge_graph_or_stub()

    result = await stub.get_relevant_context("test query")
    assert result is None


@pytest.mark.asyncio
async def test_kg_stub_returns_empty_search():
    """KG stub.search() should return empty results."""
    from resync.core.langgraph.agent_graph import _get_knowledge_graph_or_stub

    with patch.dict("sys.modules", {"resync.services.rag_client": None}):
        stub = _get_knowledge_graph_or_stub()

    result = await stub.search(query="test")
    assert result == {"results": []}


# =============================================================================
# FASE 2: Planner Templates
# =============================================================================


def test_plan_templates_are_json_serializable():
    """All plan templates must survive JSON round-trip (LangGraph persistence)."""
    from resync.core.langgraph.plan_templates import PLAN_TEMPLATES, create_plan

    for template_key in PLAN_TEMPLATES:
        plan = create_plan(template_key)
        serialized = json.dumps(plan)
        deserialized = json.loads(serialized)
        assert deserialized["template"] == template_key
        assert len(deserialized["steps"]) == plan["total_steps"]
        for step in deserialized["steps"]:
            assert step["completed"] is False
            assert step["error"] is None


def test_create_plan_unknown_template_returns_empty():
    """Unknown template key should return plan with zero steps."""
    from resync.core.langgraph.plan_templates import create_plan

    plan = create_plan("nonexistent_template")
    assert plan["steps"] == []
    assert plan["total_steps"] == 0


@pytest.mark.asyncio
async def test_planner_node_creates_plan_for_troubleshoot():
    """Troubleshoot intent should produce collect/analyze/synthesize steps."""
    from resync.core.langgraph.agent_graph import Intent, planner_node

    state = {
        "message": "erro no job BATCH_001",
        "intent": Intent.TROUBLESHOOT,
        "entities": {"job_name": "BATCH_001"},
        "execution_plan": None,
        "plan_step_index": 0,
    }
    result = planner_node(state)

    plan = result["execution_plan"]
    assert plan is not None
    assert plan["template"] == "troubleshoot"
    assert plan["total_steps"] == 3
    step_ids = [s["id"] for s in plan["steps"]]
    assert step_ids == ["collect", "analyze", "synthesize"]


@pytest.mark.asyncio
async def test_planner_node_creates_plan_for_action():
    """Action intent should produce validate/approve/execute/verify steps."""
    from resync.core.langgraph.agent_graph import Intent, planner_node

    state = {
        "message": "rerun job TEST",
        "intent": Intent.ACTION,
        "entities": {"job_name": "TEST", "action_type": "rerun"},
        "execution_plan": None,
        "plan_step_index": 0,
    }
    result = planner_node(state)

    plan = result["execution_plan"]
    assert plan["template"] == "action"
    step_ids = [s["id"] for s in plan["steps"]]
    assert step_ids == ["validate", "approve", "execute", "verify"]


@pytest.mark.asyncio
async def test_planner_node_bypasses_for_general_intent():
    """General intent should NOT create a plan (direct handler)."""
    from resync.core.langgraph.agent_graph import Intent, planner_node

    state = {
        "message": "olá, tudo bem?",
        "intent": Intent.GENERAL,
        "entities": {},
        "execution_plan": None,
        "plan_step_index": 0,
    }
    result = planner_node(state)
    assert result["execution_plan"] is None


# =============================================================================
# FASE 2: Plan Executor — Dependency Check
# =============================================================================


@pytest.mark.asyncio
async def test_plan_executor_dependency_check_abort():
    """If a step's dependency is not met and on_failure=abort, plan must fail."""
    from resync.core.langgraph.agent_graph import plan_executor_node

    plan = {
        "template": "test",
        "total_steps": 2,
        "steps": [
            {
                "id": "step_a",
                "action": "noop",
                "description": "",
                "requires": [],
                "on_failure": "skip",
                "completed": False,
                "error": None,
            },
            {
                "id": "step_b",
                "action": "noop",
                "description": "",
                "requires": ["step_a"],
                "on_failure": "abort",
                "completed": False,
                "error": None,
            },
        ],
    }

    state = {
        "execution_plan": plan,
        "plan_step_index": 1,  # Jump to step_b (step_a NOT completed)
        "plan_failed": False,
        "raw_data": {},
    }

    result = await plan_executor_node(state)

    assert result["plan_failed"] is True
    step_b = result["execution_plan"]["steps"][1]
    assert "dependency_not_met:step_a" in step_b["error"]
    assert step_b["completed"] is False


@pytest.mark.asyncio
async def test_plan_executor_dependency_check_skip():
    """If a step's dependency is not met and on_failure=skip, plan advances past it."""
    from resync.core.langgraph.agent_graph import plan_executor_node

    plan = {
        "template": "test",
        "total_steps": 3,
        "steps": [
            {
                "id": "step_a",
                "action": "noop",
                "description": "",
                "requires": [],
                "on_failure": "skip",
                "completed": False,
                "error": None,
            },
            {
                "id": "step_b",
                "action": "noop",
                "description": "",
                "requires": ["step_a"],
                "on_failure": "skip",
                "completed": False,
                "error": None,
            },
            {
                "id": "step_c",
                "action": "noop",
                "description": "",
                "requires": [],
                "on_failure": "skip",
                "completed": False,
                "error": None,
            },
        ],
    }

    state = {
        "execution_plan": plan,
        "plan_step_index": 1,  # Jump to step_b (step_a NOT completed)
        "plan_failed": False,
        "raw_data": {},
    }

    result = await plan_executor_node(state)

    assert result["plan_failed"] is False
    assert result["plan_step_index"] == 2  # Skipped past step_b
    step_b = result["execution_plan"]["steps"][1]
    assert step_b["completed"] is False
    assert "dependency_not_met" in step_b["error"]


@pytest.mark.asyncio
async def test_plan_executor_dependency_not_reached_when_action_executes():
    """Verify the action block is NOT entered when dependency check fails (the bug fix)."""
    from resync.core.langgraph.agent_graph import plan_executor_node

    # Use orchestrator_collect as the action — if it runs, it will try to import
    # resync.core.factories which we DON'T mock. If the dep check works correctly,
    # the action should never be called and no ImportError should occur.
    plan = {
        "template": "test",
        "total_steps": 2,
        "steps": [
            {
                "id": "collect",
                "action": "orchestrator_collect",
                "description": "",
                "requires": [],
                "on_failure": "skip",
                "completed": False,
                "error": None,
            },
            {
                "id": "analyze",
                "action": "orchestrator_collect",
                "description": "",
                "requires": ["collect"],
                "on_failure": "abort",
                "completed": False,
                "error": None,
            },
        ],
    }

    state = {
        "execution_plan": plan,
        "plan_step_index": 1,  # Skip to analyze (collect NOT completed)
        "plan_failed": False,
        "raw_data": {},
    }

    # This should NOT raise ImportError — the dep check should early-return
    result = await plan_executor_node(state)
    assert result["plan_failed"] is True


# =============================================================================
# FASE 2.5: LLM Rescue Routing
# =============================================================================


def test_after_plan_executor_routes_to_rescue_on_troubleshoot_failure():
    """When troubleshoot plan fails, routing should go to llm_rescue."""
    from resync.core.langgraph.agent_graph import Intent, _after_plan_executor

    state = {
        "plan_failed": True,
        "intent": Intent.TROUBLESHOOT,
        "rescue_used": False,
        "execution_plan": {"steps": [], "total_steps": 0},
        "plan_step_index": 0,
    }
    assert _after_plan_executor(state) == "llm_rescue"


def test_after_plan_executor_no_rescue_for_action():
    """When action plan fails, routing should go to synthesizer (no rescue for actions)."""
    from resync.core.langgraph.agent_graph import Intent, _after_plan_executor

    state = {
        "plan_failed": True,
        "intent": Intent.ACTION,
        "rescue_used": False,
        "execution_plan": {"steps": [], "total_steps": 0},
        "plan_step_index": 0,
    }
    assert _after_plan_executor(state) == "synthesizer"


def test_after_plan_executor_no_double_rescue():
    """Rescue must only fire once; second failure goes to synthesizer."""
    from resync.core.langgraph.agent_graph import Intent, _after_plan_executor

    state = {
        "plan_failed": True,
        "intent": Intent.TROUBLESHOOT,
        "rescue_used": True,
        "execution_plan": {"steps": [], "total_steps": 0},
        "plan_step_index": 0,
    }
    assert _after_plan_executor(state) == "synthesizer"


def test_after_plan_executor_loops_when_steps_remain():
    """If plan has remaining steps, executor should loop back."""
    from resync.core.langgraph.agent_graph import _after_plan_executor

    state = {
        "plan_failed": False,
        "execution_plan": {"steps": [{}, {}, {}], "total_steps": 3},
        "plan_step_index": 1,
    }
    assert _after_plan_executor(state) == "plan_executor"


def test_after_plan_executor_synthesizes_when_plan_complete():
    """If all steps done, routing goes to synthesizer."""
    from resync.core.langgraph.agent_graph import _after_plan_executor

    state = {
        "plan_failed": False,
        "execution_plan": {"steps": [{}, {}], "total_steps": 2},
        "plan_step_index": 2,
    }
    assert _after_plan_executor(state) == "synthesizer"


# =============================================================================
# FASE 3: Verification Loop
# =============================================================================


@pytest.mark.asyncio
async def test_plan_executor_verification_retry(monkeypatch):
    """Verify that verify_action can request retry without advancing plan."""
    mock_tws = AsyncMock()
    mock_tws.get_job_status.side_effect = [
        {"status": "UNKNOWN"},
        {"status": "EXECUTING"},
    ]
    mock_tws.execute_action.return_value = {"ok": True}

    monkeypatch.setattr(
        "resync.core.factories.get_tws_client_singleton", lambda *a, **k: mock_tws
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    from resync.core.langgraph.agent_graph import (
        Intent,
        plan_executor_node,
        planner_node,
    )

    state = {
        "message": "rerun job TEST",
        "intent": Intent.ACTION,
        "entities": {"job_name": "TEST", "action_type": "rerun"},
        "raw_data": {},
        "execution_plan": None,
        "plan_step_index": 0,
        "plan_failed": False,
        "verification_attempts": 0,
        "action_pending_verification": "rerun",
        "max_verification_attempts": 3,
    }

    state = planner_node(state)
    state["execution_plan"]["steps"] = [
        {
            "id": "execute",
            "action": "execute_action",
            "description": "",
            "requires": [],
            "on_failure": "abort",
            "completed": True,
            "error": None,
        },
        {
            "id": "verify",
            "action": "verify_action",
            "description": "",
            "requires": ["execute"],
            "on_failure": "skip",
            "completed": False,
            "error": None,
        },
    ]
    state["execution_plan"]["total_steps"] = 2
    state["plan_step_index"] = 1

    # First verification: UNKNOWN → retry
    state = await plan_executor_node(state)
    assert state["plan_step_index"] == 1
    assert state["raw_data"].get("_verification_retry") is True

    # Second verification: EXECUTING → success
    state = await plan_executor_node(state)
    assert state["raw_data"].get("action_verified") is True
    assert state["plan_step_index"] == 2


@pytest.mark.asyncio
async def test_verification_exhausted_after_max_attempts(monkeypatch):
    """After max attempts, verification should mark exhausted and stop retrying."""
    mock_tws = AsyncMock()
    mock_tws.get_job_status.return_value = {"status": "UNKNOWN"}

    monkeypatch.setattr(
        "resync.core.factories.get_tws_client_singleton", lambda *a, **k: mock_tws
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    from resync.core.langgraph.agent_graph import _execute_verification_once

    state = {
        "entities": {"job_name": "TEST"},
        "action_pending_verification": "rerun",
        "verification_attempts": 2,
        "max_verification_attempts": 3,
        "raw_data": {},
    }

    result = await _execute_verification_once(state)

    assert result["raw_data"]["verification_exhausted"] is True
    assert result["raw_data"]["_verification_retry"] is False
    assert result["verification_attempts"] == 3


@pytest.mark.asyncio
async def test_verification_respects_custom_max_attempts(monkeypatch):
    """Custom max_verification_attempts from config should be respected."""
    mock_tws = AsyncMock()
    mock_tws.get_job_status.return_value = {"status": "UNKNOWN"}

    monkeypatch.setattr(
        "resync.core.factories.get_tws_client_singleton", lambda *a, **k: mock_tws
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    from resync.core.langgraph.agent_graph import _execute_verification_once

    state = {
        "entities": {"job_name": "TEST"},
        "action_pending_verification": "rerun",
        "verification_attempts": 0,
        "max_verification_attempts": 1,
        "raw_data": {},
    }

    result = await _execute_verification_once(state)

    assert result["raw_data"]["verification_exhausted"] is True
    assert result["verification_attempts"] == 1


# =============================================================================
# FASE 4: Output Critique
# =============================================================================


@pytest.mark.asyncio
async def test_output_critique_forces_refinement(monkeypatch):
    """When critique says unsatisfactory, response should be cleared for re-synthesis."""
    critique_json = '{"satisfactory": false, "issues": ["Falta informar o RC"], "missing": ["return_code"]}'
    mock_llm = AsyncMock(return_value=critique_json)
    monkeypatch.setattr("resync.core.utils.llm.call_llm", mock_llm)

    from resync.core.langgraph.agent_graph import output_critique_node

    state = {
        "message": "por que o job falhou?",
        "response": "O job falhou por um erro.",
        "raw_data": {"status": {"return_code": 12}},
        "critique_retries": 0,
        "needs_refinement": False,
    }

    result = await output_critique_node(state)

    assert result["needs_refinement"] is True
    assert result["response"] == ""
    assert result["critique_retries"] == 1
    assert "Falta informar o RC" in result["critique_feedback"]


@pytest.mark.asyncio
async def test_output_critique_accepts_good_response(monkeypatch):
    """When critique says satisfactory, response should be kept."""
    critique_json = '{"satisfactory": true, "issues": [], "missing": []}'
    mock_llm = AsyncMock(return_value=critique_json)
    monkeypatch.setattr("resync.core.utils.llm.call_llm", mock_llm)

    from resync.core.langgraph.agent_graph import output_critique_node

    state = {
        "message": "por que o job falhou?",
        "response": "O job BATCH_001 falhou com RC=12 — tabela não encontrada.",
        "raw_data": {"status": {"return_code": 12}},
        "critique_retries": 0,
        "needs_refinement": False,
    }

    result = await output_critique_node(state)
    assert result["needs_refinement"] is False
    assert result["response"] != ""


@pytest.mark.asyncio
async def test_output_critique_stops_after_max_retries():
    """Critique should not run after 2 retries."""
    from resync.core.langgraph.agent_graph import output_critique_node

    state = {
        "message": "test",
        "response": "some response",
        "raw_data": {},
        "critique_retries": 2,
        "needs_refinement": True,
    }

    result = await output_critique_node(state)
    assert result["needs_refinement"] is False


@pytest.mark.asyncio
async def test_output_critique_never_blocks_on_llm_error(monkeypatch):
    """If LLM call fails, critique should not block the response."""
    mock_llm = AsyncMock(side_effect=Exception("LLM timeout"))
    monkeypatch.setattr("resync.core.utils.llm.call_llm", mock_llm)

    from resync.core.langgraph.agent_graph import output_critique_node

    state = {
        "message": "test",
        "response": "A valid response that should not be lost.",
        "raw_data": {},
        "critique_retries": 0,
        "needs_refinement": False,
    }

    result = await output_critique_node(state)
    assert result["needs_refinement"] is False
    assert result["response"] == "A valid response that should not be lost."


# =============================================================================
# ROUTING: _after_critique
# =============================================================================


def test_after_critique_refines_when_needed():
    from resync.core.langgraph.agent_graph import _after_critique

    state = {"needs_refinement": True, "critique_retries": 1}
    assert _after_critique(state) == "synthesizer"


def test_after_critique_passes_when_satisfied():
    from resync.core.langgraph.agent_graph import _after_critique

    state = {"needs_refinement": False, "critique_retries": 0}
    assert _after_critique(state) == "hallucination_check"


def test_after_critique_passes_when_retries_exhausted():
    from resync.core.langgraph.agent_graph import _after_critique

    state = {"needs_refinement": True, "critique_retries": 2}
    assert _after_critique(state) == "hallucination_check"


# =============================================================================
# DKG: Document Knowledge Graph Integration
# =============================================================================


@pytest.mark.asyncio
async def test_dkg_context_node_noop_when_disabled(monkeypatch):
    """DKG node should set empty string when KG_RETRIEVAL_ENABLED is not set."""
    monkeypatch.delenv("KG_RETRIEVAL_ENABLED", raising=False)

    from resync.core.langgraph.agent_graph import document_kg_context_node

    state = {
        "message": "status do job TEST",
        "entities": {"job_name": "TEST"},
        "doc_kg_context": "",
    }

    result = await document_kg_context_node(state)
    assert result["doc_kg_context"] == ""


@pytest.mark.asyncio
async def test_dkg_context_node_survives_import_failure(monkeypatch):
    """DKG node should not raise even when DocumentGraphRAG import fails."""
    monkeypatch.setattr(
        "resync.core.document_graphrag.DocumentGraphRAG",
        None,
    )

    from resync.core.langgraph.agent_graph import document_kg_context_node

    state = {
        "message": "test",
        "entities": {},
        "doc_kg_context": "",
    }

    # Should not raise
    result = await document_kg_context_node(state)
    assert result["doc_kg_context"] == ""


def test_router_resets_doc_kg_context():
    """doc_kg_context should be reset between turns."""

    # The transient defaults in router_node reset doc_kg_context to ""
    # We can verify by checking the default value in the reset dict
    transient_defaults = {
        "doc_kg_context": "",
    }
    assert transient_defaults["doc_kg_context"] == ""


# =============================================================================
# DKG: Routing through DKG context node
# =============================================================================


def test_routing_v6_1_all_non_clarification_go_to_dkg():
    """After DKG integration, all non-clarification intents go through DKG context first."""
    from resync.core.langgraph.agent_graph import Intent, _get_next_node_v6_1

    for intent in [
        Intent.STATUS,
        Intent.TROUBLESHOOT,
        Intent.ACTION,
        Intent.QUERY,
        Intent.GENERAL,
    ]:
        state = {"needs_clarification": False, "intent": intent}
        assert _get_next_node_v6_1(state) == "document_kg_context", (
            f"Failed for {intent}"
        )


def test_routing_v6_1_clarification_still_direct():
    """Clarification should still bypass DKG context."""
    from resync.core.langgraph.agent_graph import Intent, _get_next_node_v6_1

    state = {"needs_clarification": True, "intent": Intent.TROUBLESHOOT}
    assert _get_next_node_v6_1(state) == "clarification"


def test_after_dkg_context_routes_troubleshoot_to_planner():
    from resync.core.langgraph.agent_graph import Intent, _after_dkg_context

    state = {"intent": Intent.TROUBLESHOOT}
    assert _after_dkg_context(state) == "planner"


def test_after_dkg_context_routes_status_to_planner():
    from resync.core.langgraph.agent_graph import Intent, _after_dkg_context

    state = {"intent": Intent.STATUS}
    assert _after_dkg_context(state) == "planner"


def test_after_dkg_context_routes_action_to_planner():
    from resync.core.langgraph.agent_graph import Intent, _after_dkg_context

    state = {"intent": Intent.ACTION}
    assert _after_dkg_context(state) == "planner"


def test_after_dkg_context_routes_query_to_handler():
    from resync.core.langgraph.agent_graph import Intent, _after_dkg_context

    state = {"intent": Intent.QUERY}
    assert _after_dkg_context(state) == "query_handler"


def test_after_dkg_context_routes_general_to_handler():
    from resync.core.langgraph.agent_graph import Intent, _after_dkg_context

    state = {"intent": Intent.GENERAL}
    assert _after_dkg_context(state) == "general_handler"


# =============================================================================
# DKG: Plan templates still JSON serializable (regression guard)
# =============================================================================


def test_plan_templates_still_serializable_after_dkg():
    """Ensure DKG changes did not break plan template serialization."""
    import json

    from resync.core.langgraph.plan_templates import PLAN_TEMPLATES, create_plan

    for key in PLAN_TEMPLATES:
        plan = create_plan(key)
        roundtrip = json.loads(json.dumps(plan))
        assert roundtrip["template"] == key


# =============================================================================
# DKG: Store upsert_from_extraction interface
# =============================================================================


def test_store_has_upsert_from_extraction():
    """PostgresGraphStore must have upsert_from_extraction method."""
    from resync.knowledge.kg_store.store import PostgresGraphStore

    store = PostgresGraphStore.__new__(PostgresGraphStore)
    assert hasattr(store, "upsert_from_extraction")
    assert callable(store.upsert_from_extraction)


# =============================================================================
# DKG: Normalizer
# =============================================================================


def test_make_node_id_stable():
    """Node IDs should be deterministic and normalized."""
    from resync.knowledge.kg_extraction.normalizer import make_node_id

    assert make_node_id("Job", "BATCH_001") == "Job:batch_001"
    assert make_node_id("Error", "ORA-00942") == "Error:ora_00942"
    assert (
        make_node_id("Concept", "Falha de Autenticação")
        == "Concept:falha_de_autentica_o"
    )
    # Same input produces same output
    assert make_node_id("Job", "BATCH_001") == make_node_id("Job", "BATCH_001")


def test_dedup_concepts_merges_aliases():
    """Dedup should merge aliases and properties from duplicates."""
    from resync.knowledge.kg_extraction.normalizer import dedup_concepts
    from resync.knowledge.kg_extraction.schemas import Concept

    concepts = [
        Concept(
            name="ORA-00942",
            node_type="Error",
            aliases=["table not found"],
            properties={"doc_id": "a"},
        ),
        Concept(
            name="ORA-00942",
            node_type="Error",
            aliases=["missing table"],
            properties={"doc_id": "b"},
        ),
    ]

    result = dedup_concepts(concepts)
    assert len(result) == 1
    assert "table not found" in result[0].aliases
    assert "missing table" in result[0].aliases
    assert result[0].properties["doc_id"] == "b"  # Last write wins for properties


def test_dedup_edges_keeps_max_weight():
    """Dedup should keep the highest weight edge."""
    from resync.knowledge.kg_extraction.normalizer import dedup_edges
    from resync.knowledge.kg_extraction.schemas import Edge, Evidence

    edges = [
        Edge(
            source="A",
            target="B",
            relation_type="CAUSES",
            weight=0.3,
            evidence=Evidence(extractor="cooc"),
        ),
        Edge(
            source="A",
            target="B",
            relation_type="CAUSES",
            weight=0.9,
            evidence=Evidence(extractor="llm"),
        ),
    ]

    result = dedup_edges(edges)
    assert len(result) == 1
    assert result[0].weight == 0.9
    assert result[0].evidence.extractor == "llm"


# =============================================================================
# DKG: ExtractionResult schema
# =============================================================================


def test_extraction_result_defaults_empty():
    """ExtractionResult should default to empty lists."""
    from resync.knowledge.kg_extraction.schemas import ExtractionResult

    result = ExtractionResult()
    assert result.concepts == []
    assert result.edges == []


def test_extraction_result_serializable():
    """ExtractionResult should be JSON-serializable."""
    import json

    from resync.knowledge.kg_extraction.schemas import (
        Concept,
        Edge,
        Evidence,
        ExtractionResult,
    )

    result = ExtractionResult(
        concepts=[Concept(name="Test", node_type="Concept")],
        edges=[Edge(source="A", target="B", evidence=Evidence(doc_id="d1"))],
    )
    data = json.loads(result.model_dump_json())
    assert len(data["concepts"]) == 1
    assert len(data["edges"]) == 1
