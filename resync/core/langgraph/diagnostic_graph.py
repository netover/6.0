"""
Autonomous Diagnostic Resolution Graph for Resync v5.4.0

This module implements a cyclic decision graph for autonomous problem
resolution in TWS/HWA environments.

The AI navigates through states:
1. DIAGNOSE - Analyze the problem
2. RESEARCH - Consult documentation and history
3. VERIFY - Check current system state
4. PROPOSE - Suggest solution
5. (Optional) EXECUTE - Apply fix with approval
6. VALIDATE - Confirm resolution

The graph can cycle through these states until confident of solution.

Graph Architecture:

    START
      │
      ▼
   [DIAGNOSE] ◄───────────────────┐
      │                           │
      ▼                           │
   [RESEARCH] (RAG + History)     │
      │                           │
      ▼                           │
   [VERIFY] (TWS State)           │
      │                           │
      ├──► (uncertain) ───────────┘
      │
      ▼ (confident)
   [PROPOSE]
      │
      ├──► (needs_action)──► [APPROVE]──► [EXECUTE]──► [VALIDATE]
      │                                                    │
      └──► (info_only)──────────────────────────────────► [END]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# Check LangGraph availability
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"


# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class DiagnosticPhase(str, Enum):
    """Current phase in diagnostic resolution."""

    DIAGNOSE = "diagnose"
    RESEARCH = "research"
    VERIFY = "verify"
    PROPOSE = "propose"
    APPROVE = "approve"
    EXECUTE = "execute"
    VALIDATE = "validate"
    COMPLETE = "complete"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence in diagnosis/solution."""

    LOW = "low"  # < 40% - needs more info
    MEDIUM = "medium"  # 40-70% - tentative
    HIGH = "high"  # 70-90% - confident
    CERTAIN = "certain"  # > 90% - very confident


class DiagnosticState(TypedDict, total=False):
    """
    State for diagnostic resolution graph.

    Flows through the graph as diagnosis progresses.
    """

    # Input
    problem_description: str
    user_id: str | None
    session_id: str | None
    tws_instance_id: str | None
    skill_context: str  # GAP C: Skill injection for diagnostic

    # Current phase
    phase: DiagnosticPhase
    iteration: int
    max_iterations: int

    # Diagnosis
    symptoms: list[str]
    possible_causes: list[dict[str, Any]]
    root_cause: str | None
    root_cause_confidence: float

    # Research results
    documentation_context: list[dict[str, Any]]
    historical_incidents: list[dict[str, Any]]
    similar_resolutions: list[dict[str, Any]]

    # TWS State
    affected_jobs: list[str]
    affected_workstations: list[str]
    current_job_states: dict[str, Any]
    error_logs: list[str]

    # Solution
    proposed_solution: str | None
    solution_steps: list[str]
    requires_action: bool
    risk_level: str  # low, medium, high

    # Execution
    approval_status: str | None  # pending, approved, rejected
    execution_result: str | None
    validation_result: str | None

    # Output
    response: str
    recommendations: list[str]

    # Errors
    errors: list[str]


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic resolution."""

    # Iteration limits
    max_iterations: int = 5
    min_confidence_for_proposal: float = 0.7

    # Approval settings
    require_approval_for_actions: bool = True
    auto_approve_low_risk: bool = False

    # Research settings
    max_rag_results: int = 5
    include_historical: bool = True

    # LLM settings
    model: str = "meta/llama-3.1-70b-instruct"
    temperature: float = 0.3  # Lower for diagnostic accuracy


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================


class DiagnoseNode:
    """
    Analyze problem and extract symptoms.

    Uses LLM to understand the problem description and identify
    possible causes based on TWS domain knowledge.
    """

    def __init__(self, config: DiagnosticConfig):
        self.config = config

    async def __call__(self, state: DiagnosticState) -> DiagnosticState:
        logger.info("DiagnoseNode: iteration %s", state.get('iteration', 0))

        problem = state.get("problem_description", "")

        # Extract symptoms from problem description
        symptoms = await self._extract_symptoms(problem)

        # Generate possible causes
        possible_causes = await self._generate_hypotheses(problem, symptoms)

        # Update state
        state["symptoms"] = symptoms
        state["possible_causes"] = possible_causes
        state["phase"] = DiagnosticPhase.RESEARCH

        logger.info("Diagnosis: %s symptoms, %s possible causes", len(symptoms), len(possible_causes))

        return state

    async def _extract_symptoms(self, problem: str) -> list[str]:
        """Extract symptoms from problem description."""
        try:
            from resync.services.llm_service import get_llm_completion

            prompt = f"""Analyze this TWS/HWA problem and extract specific symptoms.

Problem: {problem}

List each symptom on a new line. Be specific about:
- Job names mentioned
- Error messages
- Status codes
- Time patterns
- Affected systems

Symptoms:"""

            response = await get_llm_completion(
                prompt,
                model=self.config.model,
                temperature=0.2,
                max_tokens=500,
            )

            # Parse symptoms from response
            symptoms = [
                s.strip().lstrip("- ").lstrip("•")
                for s in response.strip().split("\n")
                if s.strip() and not s.startswith("Symptoms")
            ]

            return symptoms[:10]  # Limit to 10 symptoms

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Symptom extraction failed: %s", e)
            return [problem]  # Use original problem as single symptom

    async def _generate_hypotheses(
        self,
        problem: str,
        symptoms: list[str],
    ) -> list[dict[str, Any]]:
        """Generate possible causes for the symptoms."""
        try:
            from resync.services.llm_service import get_llm_completion

            symptoms_text = "\n".join(f"- {s}" for s in symptoms)

            prompt = f"""Based on these TWS/HWA symptoms, generate possible root causes.

Problem: {problem}

Symptoms:
{symptoms_text}

For each possible cause, provide:
1. Cause description
2. Likelihood (low/medium/high)
3. What to check to verify

Format as JSON array:
[{{"cause": "...", "likelihood": "...", "verification": "..."}}]

Causes:"""

            response = await get_llm_completion(
                prompt,
                model=self.config.model,
                temperature=0.3,
                max_tokens=800,
            )

            # Parse JSON from response
            import json
            import re

            # Find JSON array in response
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                return json.loads(json_match.group())

            return [{"cause": problem, "likelihood": "medium", "verification": "Check job status"}]

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Hypothesis generation failed: %s", e)
            return []


class ResearchNode:
    """
    Research documentation and history for relevant context.

    Uses:
    - Hybrid RAG (BM25 + Vector) for documentation
    - Historical incident database
    - Similar resolution patterns
    """

    def __init__(self, config: DiagnosticConfig):
        self.config = config

    async def __call__(self, state: DiagnosticState) -> DiagnosticState:
        logger.info("ResearchNode: gathering context")

        problem = state.get("problem_description", "")
        symptoms = state.get("symptoms", [])
        causes = state.get("possible_causes", [])

        # Build search query from symptoms and causes
        search_query = self._build_search_query(problem, symptoms, causes)

        # Search documentation using hybrid retrieval
        docs = await self._search_documentation(search_query)
        state["documentation_context"] = docs

        # Search historical incidents
        if self.config.include_historical:
            history = self._search_history(search_query)
            state["historical_incidents"] = history
            state["similar_resolutions"] = [h for h in history if h.get("resolution")]

        state["phase"] = DiagnosticPhase.VERIFY

        logger.info(
            f"Research: {len(docs)} docs, {len(state.get('historical_incidents', []))} incidents"
        )

        return state

    def _build_search_query(
        self,
        problem: str,
        symptoms: list[str],
        causes: list[dict],
    ) -> str:
        """Build optimized search query."""
        # Combine problem + top symptoms + likely causes
        parts = [problem]
        parts.extend(symptoms[:3])
        parts.extend(
            [c.get("cause", "") for c in causes[:2] if c.get("likelihood") in ["high", "medium"]]
        )

        return " ".join(parts)[:500]  # Limit query length

    async def _search_documentation(self, query: str) -> list[dict]:
        """Search TWS documentation using hybrid retrieval."""
        try:
            from resync.knowledge.ingestion.embedding_service import EmbeddingService
            from resync.knowledge.retrieval.hybrid_retriever import HybridRetriever
            from resync.knowledge.store.pgvector_store import PgVectorStore

            embedder = EmbeddingService()
            store = PgVectorStore()
            retriever = HybridRetriever(embedder, store)

            return await retriever.retrieve(
                query,
                top_k=self.config.max_rag_results,
            )

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("Documentation search failed: %s", e)
            return []

    def _search_history(self, query: str) -> list[dict]:
        """Search historical incident database."""
        # DEBT: Implement historical incident search when ITSM integration is ready
        # For now, return empty list
        return []


class VerifyNode:
    """
    Verify current TWS state to confirm diagnosis.

    Checks:
    - Current job states
    - Workstation status
    - Recent error logs
    """

    def __init__(self, config: DiagnosticConfig):
        self.config = config

    async def __call__(self, state: DiagnosticState) -> DiagnosticState:
        logger.info("VerifyNode: checking TWS state")

        # Get affected entities from symptoms
        jobs, workstations = self._extract_entities(state)
        state["affected_jobs"] = jobs
        state["affected_workstations"] = workstations

        # Check current states
        if jobs or workstations:
            current_states = await self._check_tws_state(
                jobs,
                workstations,
                state.get("tws_instance_id"),
            )
            state["current_job_states"] = current_states

        # Get error logs
        state["error_logs"] = self._get_error_logs(jobs)

        # Calculate confidence based on verification
        confidence = self._calculate_confidence(state)
        state["root_cause_confidence"] = confidence

        # Determine root cause if confident enough
        if confidence >= self.config.min_confidence_for_proposal:
            state["root_cause"] = self._determine_root_cause(state)
            state["phase"] = DiagnosticPhase.PROPOSE
        else:
            # Need more iterations
            iteration = state.get("iteration", 0) + 1
            if iteration >= state.get("max_iterations", self.config.max_iterations):
                # Max iterations reached, propose anyway
                state["root_cause"] = self._determine_root_cause(state)
                state["phase"] = DiagnosticPhase.PROPOSE
            else:
                # Go back to diagnose with new info
                state["iteration"] = iteration
                state["phase"] = DiagnosticPhase.DIAGNOSE

        logger.info("Verification: confidence={confidence:.2f}, phase={state['phase']}")

        return state

    def _extract_entities(self, state: DiagnosticState) -> tuple[list[str], list[str]]:
        """Extract job and workstation names from symptoms."""
        import re

        jobs = []
        workstations = []

        # Job pattern (uppercase alphanumeric)
        job_pattern = r"\b[A-Z][A-Z0-9_\-]{2,39}\b"

        # Check symptoms
        for symptom in state.get("symptoms", []):
            matches = re.findall(job_pattern, symptom.upper())
            jobs.extend(matches)

        # Check problem description
        problem = state.get("problem_description", "")
        jobs.extend(re.findall(job_pattern, problem.upper()))

        # Deduplicate
        jobs = list(set(jobs))[:10]

        return jobs, workstations

    async def _check_tws_state(
        self,
        jobs: list[str],
        workstations: list[str],
        tws_instance_id: str | None,
    ) -> dict[str, Any]:
        """Check current TWS job/workstation states."""
        try:
            from resync.services.tws_service import get_tws_client

            client = await get_tws_client()

            states = {}
            for job in jobs[:5]:  # Limit to 5 jobs
                try:
                    status = await client.get_job_status(job)
                    states[job] = status
                except Exception as e:
                    states[job] = {"error": str(e)}

            return states

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("TWS state check failed: %s", e)
            return {}

    def _get_error_logs(self, jobs: list[str]) -> list[str]:
        """Get recent error logs for affected jobs."""
        # DEBT: Implement error log retrieval when centralized logging is integrated
        return []

    def _calculate_confidence(self, state: DiagnosticState) -> float:
        """Calculate confidence in diagnosis."""
        confidence = 0.3  # Base confidence

        # Boost if documentation found
        if state.get("documentation_context"):
            confidence += 0.2

        # Boost if similar incidents found
        if state.get("similar_resolutions"):
            confidence += 0.2

        # Boost if TWS state verified
        if state.get("current_job_states"):
            confidence += 0.2

        # Boost if root cause is clear from possible causes
        causes = state.get("possible_causes", [])
        high_likelihood = [c for c in causes if c.get("likelihood") == "high"]
        if len(high_likelihood) == 1:
            confidence += 0.1

        return min(confidence, 1.0)

    def _determine_root_cause(self, state: DiagnosticState) -> str:
        """Determine most likely root cause."""
        causes = state.get("possible_causes", [])

        # Prefer high likelihood causes
        high = [c for c in causes if c.get("likelihood") == "high"]
        if high:
            return high[0].get("cause", "Unknown cause")

        # Fall back to medium
        medium = [c for c in causes if c.get("likelihood") == "medium"]
        if medium:
            return medium[0].get("cause", "Unknown cause")

        # Fall back to any cause
        if causes:
            return causes[0].get("cause", "Unknown cause")

        return "Unable to determine root cause"


class ProposeNode:
    """
    Propose solution based on diagnosis.

    Generates:
    - Solution description
    - Step-by-step resolution guide
    - Risk assessment
    - Whether action is needed
    """

    def __init__(self, config: DiagnosticConfig):
        self.config = config

    async def __call__(self, state: DiagnosticState) -> DiagnosticState:
        logger.info("ProposeNode: generating solution")

        # Generate solution using LLM with all context
        solution = await self._generate_solution(state)

        state["proposed_solution"] = solution.get("description", "")
        state["solution_steps"] = solution.get("steps", [])
        state["requires_action"] = solution.get("requires_action", False)
        state["risk_level"] = solution.get("risk_level", "medium")
        state["recommendations"] = solution.get("recommendations", [])

        # Determine next phase
        if state["requires_action"]:
            state["phase"] = DiagnosticPhase.APPROVE
        else:
            # Information-only response
            state["response"] = self._format_response(state)
            state["phase"] = DiagnosticPhase.COMPLETE

        return state

    async def _generate_solution(self, state: DiagnosticState) -> dict[str, Any]:
        """Generate solution using LLM."""
        try:
            from resync.services.llm_service import get_llm_completion

            # Build context
            context_parts = []

            # Skill Knowledge (GAP C: Skill Injection)
            if state.get("skill_context"):
                context_parts.append(f"Skill Knowledge:\n{state['skill_context'][:2000]}")

            # Root cause
            if state.get("root_cause"):
                context_parts.append(f"Root Cause: {state['root_cause']}")

            # Documentation context
            if state.get("documentation_context"):
                docs = state["documentation_context"][:3]
                doc_text = "\n".join([f"- {d.get('content', '')[:200]}" for d in docs])
                context_parts.append(f"Documentation:\n{doc_text}")

            # Similar resolutions
            if state.get("similar_resolutions"):
                resolutions = state["similar_resolutions"][:2]
                res_text = "\n".join([f"- {r.get('resolution', '')[:200]}" for r in resolutions])
                context_parts.append(f"Similar Resolutions:\n{res_text}")

            context = "\n\n".join(context_parts)

            prompt = f"""Based on this TWS/HWA problem diagnosis, propose a solution.

Problem: {state.get("problem_description", "")}

Symptoms: {", ".join(state.get("symptoms", [])[:5])}

{context}

Provide:
1. Solution description
2. Step-by-step resolution (numbered list)
3. Risk level (low/medium/high)
4. Whether manual action is required
5. Recommendations

Format as JSON:
{{
    "description": "...",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "risk_level": "low|medium|high",
    "requires_action": true|false,
    "recommendations": ["...", "..."]
}}

Solution:"""

            response = await get_llm_completion(
                prompt,
                model=self.config.model,
                temperature=0.3,
                max_tokens=1000,
            )

            # Parse JSON
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())

            return {
                "description": response,
                "steps": [],
                "risk_level": "medium",
                "requires_action": False,
                "recommendations": [],
            }

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Solution generation failed: %s", e)
            return {
                "description": f"Unable to generate solution: {e}",
                "steps": [],
                "risk_level": "unknown",
                "requires_action": False,
                "recommendations": ["Contact support for assistance"],
            }

    def _format_response(self, state: DiagnosticState) -> str:
        """Format final response for user."""
        parts = []

        # Root cause
        if state.get("root_cause"):
            parts.append(f"**Root Cause:** {state['root_cause']}")

        # Solution
        if state.get("proposed_solution"):
            parts.append(f"\n**Solution:** {state['proposed_solution']}")

        # Steps
        if state.get("solution_steps"):
            parts.append("\n**Resolution Steps:**")
            for i, step in enumerate(state["solution_steps"], 1):
                parts.append(f"{i}. {step}")

        # Recommendations
        if state.get("recommendations"):
            parts.append("\n**Recommendations:**")
            for rec in state["recommendations"]:
                parts.append(f"- {rec}")

        return "\n".join(parts)


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def create_diagnostic_graph(
    config: DiagnosticConfig | None = None,
) -> StateGraph | None:
    """
    Create the diagnostic resolution graph.

    Returns None if LangGraph is not available.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, diagnostic graph disabled")
        return None

    config = config or DiagnosticConfig()

    # Create nodes
    diagnose = DiagnoseNode(config)
    research = ResearchNode(config)
    verify = VerifyNode(config)
    propose = ProposeNode(config)

    # Build graph
    graph = StateGraph(DiagnosticState)

    # Add nodes
    graph.add_node("diagnose", diagnose)
    graph.add_node("research", research)
    graph.add_node("verify", verify)
    graph.add_node("propose", propose)

    # Add edges
    graph.add_edge("diagnose", "research")
    graph.add_edge("research", "verify")

    # Conditional edge from verify
    def route_after_verify(state: DiagnosticState) -> str:
        phase = state.get("phase")
        if phase == DiagnosticPhase.PROPOSE:
            return "propose"
        if phase == DiagnosticPhase.DIAGNOSE:
            return "diagnose"  # Loop back
        return END

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "propose": "propose",
            "diagnose": "diagnose",
            END: END,
        },
    )

    # Edge from propose to end
    graph.add_edge("propose", END)

    # Set entry point
    graph.set_entry_point("diagnose")

    return graph.compile()


# =============================================================================
# HIGH-LEVEL API
# =============================================================================


async def diagnose_problem(
    problem_description: str,
    tws_instance_id: str | None = None,
    skill_context: str = "",
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """
    Run autonomous diagnostic resolution.

    Args:
        problem_description: Description of the problem
        tws_instance_id: Optional TWS instance ID
        config: Optional configuration

    Returns:
        Dictionary with diagnosis results and recommendations
    """
    config = config or DiagnosticConfig()

    # Create initial state
    initial_state: DiagnosticState = {
        "problem_description": problem_description,
        "tws_instance_id": tws_instance_id,
        "skill_context": skill_context,
        "phase": DiagnosticPhase.DIAGNOSE,
        "iteration": 0,
        "max_iterations": config.max_iterations,
        "symptoms": [],
        "possible_causes": [],
        "documentation_context": [],
        "historical_incidents": [],
        "similar_resolutions": [],
        "affected_jobs": [],
        "affected_workstations": [],
        "current_job_states": {},
        "error_logs": [],
        "recommendations": [],
        "errors": [],
    }

    # Check if LangGraph is available
    graph = create_diagnostic_graph(config)

    if graph:
        # Run through LangGraph
        try:
            final_state = await graph.ainvoke(initial_state)
            return {
                "success": True,
                "root_cause": final_state.get("root_cause"),
                "confidence": final_state.get("root_cause_confidence", 0),
                "solution": final_state.get("proposed_solution"),
                "steps": final_state.get("solution_steps", []),
                "recommendations": final_state.get("recommendations", []),
                "response": final_state.get("response", ""),
                "requires_action": final_state.get("requires_action", False),
                "risk_level": final_state.get("risk_level", "unknown"),
            }
        except Exception as e:
            logger.error("Diagnostic graph failed: %s", e)

    # Fallback: run nodes manually without graph
    try:
        diagnose_node = DiagnoseNode(config)
        research_node = ResearchNode(config)
        verify_node = VerifyNode(config)
        propose_node = ProposeNode(config)

        state = initial_state
        state = await diagnose_node(state)
        state = await research_node(state)
        state = await verify_node(state)
        state = await propose_node(state)

        return {
            "success": True,
            "root_cause": state.get("root_cause"),
            "confidence": state.get("root_cause_confidence", 0),
            "solution": state.get("proposed_solution"),
            "steps": state.get("solution_steps", []),
            "recommendations": state.get("recommendations", []),
            "response": state.get("response", ""),
            "requires_action": state.get("requires_action", False),
            "risk_level": state.get("risk_level", "unknown"),
        }

    except Exception as e:
        logger.error("Manual diagnostic failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "root_cause": None,
            "confidence": 0,
            "solution": None,
            "steps": [],
            "recommendations": ["Contact support for assistance"],
            "response": f"Diagnostic failed: {e}",
        }


__all__ = [
    "DiagnosticPhase",
    "ConfidenceLevel",
    "DiagnosticState",
    "DiagnosticConfig",
    "DiagnoseNode",
    "ResearchNode",
    "VerifyNode",
    "ProposeNode",
    "create_diagnostic_graph",
    "diagnose_problem",
]
