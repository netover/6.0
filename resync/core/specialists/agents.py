"""
TWS Specialist Agents Implementation (v5.2.3.24 - Agno-Free).

Provides 4 specialized AI agents coordinated by an orchestrator:
1. Job Analyst - Return codes, ABENDs, execution analysis
2. Dependency Specialist - Predecessors, successors, impact analysis
3. Resource Specialist - Workstations, CPU/memory, conflicts
4. Knowledge Specialist - Documentation RAG, troubleshooting

v5.2.3.24: Removed Agno dependency, using native LiteLLM + prompts.

Author: Resync Team
Version: 5.2.3.24
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import structlog
import yaml

from resync.core.specialists.models import (
    DEFAULT_SPECIALIST_CONFIGS,
    DEFAULT_TEAM_CONFIG,
    QueryClassification,
    SpecialistConfig,
    SpecialistResponse,
    SpecialistType,
    TeamConfig,
    TeamResponse,
)
from resync.core.specialists.tools import (
    CalendarTool,
    DependencyGraphTool,
    ErrorCodeTool,
    JobLogTool,
    WorkstationTool,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# PROMPT LOADER
# =============================================================================

def _load_specialist_prompts() -> dict[str, Any]:
    """Load specialist prompts from YAML file."""
    prompts_path = Path(__file__).parent.parent.parent / "prompts" / "specialist_prompts.yaml"

    if prompts_path.exists():
        with open(prompts_path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    logger.warning("specialist_prompts_not_found", path=str(prompts_path))
    return {}


SPECIALIST_PROMPTS = _load_specialist_prompts()


# =============================================================================
# LITELLM INTEGRATION (REPLACES AGNO)
# =============================================================================

async def _call_llm(
    prompt: str,
    system_prompt: str,
    model: str = "ollama/qwen2.5:3b",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Call LLM using LiteLLM (unified interface).

    v5.2.3.24: Replaces Agno agent.arun() calls.
    """
    try:
        import litellm
        litellm.suppress_debug_info = True

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("llm_call_failed", error=str(e))
        return f"[LLM Error] {str(e)}"


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================


class QueryClassifier:
    """
    Classifies queries to determine which specialists should handle them.

    Uses pattern matching for fast classification without LLM calls.
    """

    # Intent patterns (Portuguese + English)
    PATTERNS = {
        SpecialistType.JOB_ANALYST: [
            r"\bRC[=:]\s*\d+",
            r"\bABEND\s+\w+",
            r"\b(failed|falhou|erro|error)\b",
            r"\b(log|logs|sysout)\b",
            r"\breturn\s*code\b",
            r"\b(S0C\d|U\d{4})\b",
        ],
        SpecialistType.DEPENDENCY: [
            r"\b(depende|depends?|dependency|dependência)\b",
            r"\b(predecessor|predecessores|antecessor)\b",
            r"\b(successor|sucessor|downstream|upstream)\b",
            r"\b(impact|impacto|afeta|affects?)\b",
            r"\b(chain|cadeia|flow|fluxo)\b",
        ],
        SpecialistType.RESOURCE: [
            r"\b(workstation|WS\d+|servidor|server)\b",
            r"\b(resource|recurso|capacity|capacidade)\b",
            r"\b(available|disponível|status)\b",
            r"\b(schedule|agenda|calendar|calendário)\b",
            r"\b(CPU|memory|memória|slot)\b",
        ],
        SpecialistType.KNOWLEDGE: [
            r"\b(how\s+to|como)\b",
            r"\b(documentation|documentação|manual)\b",
            r"\b(procedure|procedimento|processo)\b",
            r"\b(best\s+practice|boas?\s+práticas?)\b",
            r"\b(configure|configurar|setup)\b",
            r"\b(explain|explicar?|what\s+is|o\s+que\s+é)\b",
        ],
    }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify query to determine routing.

        Returns QueryClassification with recommended specialists.
        """
        query_lower = query.lower()

        # Match patterns
        specialist_scores: dict[SpecialistType, int] = {}

        for spec_type, patterns in self.PATTERNS.items():
            score = sum(
                1 for pattern in patterns
                if re.search(pattern, query_lower, re.IGNORECASE)
            )
            if score > 0:
                specialist_scores[spec_type] = score

        # Sort by score
        recommended = sorted(
            specialist_scores.keys(),
            key=lambda x: specialist_scores[x],
            reverse=True,
        )

        # Extract entities
        entities = self._extract_entities(query)

        # Determine query type
        if SpecialistType.JOB_ANALYST in recommended[:2]:
            query_type = "troubleshooting"
        elif SpecialistType.DEPENDENCY in recommended[:2]:
            query_type = "dependency_analysis"
        elif SpecialistType.RESOURCE in recommended[:2]:
            query_type = "resource_status"
        elif SpecialistType.KNOWLEDGE in recommended[:2]:
            query_type = "documentation"
        else:
            query_type = "general"

        return QueryClassification(
            query_type=query_type,
            recommended_specialists=recommended or [SpecialistType.KNOWLEDGE],
            entities=entities,
            confidence=min(0.95, 0.5 + 0.15 * len(recommended)),
        )

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        """Extract TWS entities from query."""
        entities = {
            "jobs": [],
            "workstations": [],
            "error_codes": [],
        }

        # Job names (uppercase with underscores/numbers)
        job_pattern = r"\b([A-Z][A-Z0-9_]{2,})\b"
        entities["jobs"] = re.findall(job_pattern, query)

        # Workstation names
        ws_pattern = r"\b(WS\d+|TWS_[A-Z0-9_]+)\b"
        entities["workstations"] = re.findall(ws_pattern, query, re.IGNORECASE)

        # Error/ABEND codes
        abend_pattern = r"\b([SU]\d{3}[A-Z]?)\b"
        rc_pattern = r"RC[=:]\s*(\d+)"
        entities["error_codes"] = (
            re.findall(abend_pattern, query, re.IGNORECASE) +
            [f"RC={rc}" for rc in re.findall(rc_pattern, query, re.IGNORECASE)]
        )

        return entities


# =============================================================================
# SPECIALIST AGENT CLASSES (AGNO-FREE)
# =============================================================================


class BaseSpecialist:
    """
    Base class for specialist agents.

    v5.2.3.24: Uses direct LiteLLM calls instead of Agno.
    """

    specialist_type: SpecialistType

    def __init__(
        self,
        config: SpecialistConfig | None = None,
        model_name: str | None = None,
    ):
        self.config = config or DEFAULT_SPECIALIST_CONFIGS.get(
            self.specialist_type, SpecialistConfig(specialist_type=self.specialist_type)
        )
        self.model_name = model_name or self.config.model_name
        self._system_prompt = self._load_system_prompt()

    @property
    def name(self) -> str:
        """Agent display name."""
        return f"TWS {self.specialist_type.value.replace('_', ' ').title()}"

    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML."""
        prompt_key = self.specialist_type.value.lower()

        if prompt_key in SPECIALIST_PROMPTS:
            return SPECIALIST_PROMPTS[prompt_key].get("system_prompt", "")

        return f"You are a TWS {self.specialist_type.value} specialist."

    def _get_tools(self) -> list:
        """Get available tools. Override in subclasses."""
        return []

    def _execute_tools(self, query: str, context: dict | None) -> str:
        """Execute relevant tools and format results."""
        # Base implementation - override in subclasses
        return ""

    async def process(self, query: str, context: dict | None = None) -> SpecialistResponse:
        """
        Process a query and return a response.

        v5.2.3.24: Uses LiteLLM directly instead of Agno.
        """
        start_time = time.time()
        tools_used = []

        try:
            # Execute tools to get data
            tool_results = await self._execute_tools(query, context)
            tools_used = self._get_tool_names()

            # Format prompt with context and tool results
            formatted_prompt = self._format_prompt(query, context, tool_results)

            # Call LLM
            response_text = await _call_llm(
                prompt=formatted_prompt,
                system_prompt=self._system_prompt,
                model=self.model_name,
            )

            processing_time = int((time.time() - start_time) * 1000)

            return SpecialistResponse(
                specialist_type=self.specialist_type,
                response=response_text,
                confidence=0.85,
                tools_used=tools_used,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(
                "specialist_processing_error",
                specialist=self.specialist_type.value,
                error=str(e),
            )
            return SpecialistResponse(
                specialist_type=self.specialist_type,
                response="",
                confidence=0.0,
                processing_time_ms=processing_time,
                error=str(e),
            )

    def _format_prompt(self, query: str, context: dict | None, tool_results: str) -> str:
        """Format the prompt with context and tool results."""
        parts = [query]

        if context and (entities := context.get("entities")):
            if jobs := entities.get("jobs"):
                parts.append(f"\nJobs mentioned: {', '.join(jobs)}")
            if ws := entities.get("workstations"):
                parts.append(f"\nWorkstations: {', '.join(ws)}")
            if errors := entities.get("error_codes"):
                parts.append(f"\nError codes: {', '.join(errors)}")

        if tool_results:
            parts.append(f"\n\n--- Tool Results ---\n{tool_results}")

        return "\n".join(parts)

    def _get_tool_names(self) -> list[str]:
        """Get names of tools used."""
        return [t.__name__ if callable(t) else str(t) for t in self._get_tools()]


class JobAnalystAgent(BaseSpecialist):
    """
    Specialist for job execution analysis.

    v5.2.3.24: Agno-free implementation.
    """

    specialist_type = SpecialistType.JOB_ANALYST

    def __init__(self, config: SpecialistConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.job_log_tool = JobLogTool()
        self.error_code_tool = ErrorCodeTool()

    def _get_tools(self) -> list:
        return [
            self.job_log_tool.get_job_log,
            self.job_log_tool.analyze_return_code,
            self.error_code_tool.lookup_error,
        ]

    async def _execute_tools(self, query: str, context: dict | None) -> str:
        """Execute job analysis tools."""
        results = []
        entities = (context or {}).get("entities", {})

        # Get job logs for mentioned jobs
        for job in entities.get("jobs", [])[:2]:  # Limit to 2 jobs
            try:
                log = await self.job_log_tool.get_job_log(job)
                if log:
                    results.append(f"Log for {job}:\n{log[:500]}...")
            except Exception as e:
                logger.debug("Could not get log for %s: %s", job, e)

        # Lookup error codes
        for code in entities.get("error_codes", [])[:3]:
            try:
                info = await self.error_code_tool.lookup_error(code)
                if info:
                    results.append(f"Error {code}: {info}")
            except Exception as e:
                logger.debug("Could not lookup %s: %s", code, e)

        return "\n\n".join(results)


class DependencySpecialist(BaseSpecialist):
    """
    Specialist for dependency and workflow analysis.

    v5.2.3.24: Agno-free implementation.
    """

    specialist_type = SpecialistType.DEPENDENCY

    def __init__(self, config: SpecialistConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.dependency_tool = DependencyGraphTool()

    def _get_tools(self) -> list:
        return [
            self.dependency_tool.get_predecessors,
            self.dependency_tool.get_successors,
            self.dependency_tool.analyze_impact,
        ]

    async def _execute_tools(self, query: str, context: dict | None) -> str:
        """Execute dependency analysis tools."""
        results = []
        entities = (context or {}).get("entities", {})

        for job in entities.get("jobs", [])[:2]:
            try:
                preds = await self.dependency_tool.get_predecessors(job)
                succs = await self.dependency_tool.get_successors(job)

                if preds:
                    results.append(f"Predecessors of {job}: {', '.join(preds)}")
                if succs:
                    results.append(f"Successors of {job}: {', '.join(succs)}")

            except Exception as e:
                logger.debug("Could not get dependencies for %s: %s", job, e)

        return "\n".join(results)


class ResourceSpecialist(BaseSpecialist):
    """
    Specialist for resource and capacity analysis.

    v5.2.3.24: Agno-free implementation.
    """

    specialist_type = SpecialistType.RESOURCE

    def __init__(self, config: SpecialistConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.workstation_tool = WorkstationTool()
        self.calendar_tool = CalendarTool()

    def _get_tools(self) -> list:
        return [
            self.workstation_tool.get_workstation_status,
            self.calendar_tool.get_calendar_schedule,
        ]

    async def _execute_tools(self, query: str, context: dict | None) -> str:
        """Execute resource analysis tools."""
        results = []
        entities = (context or {}).get("entities", {})

        for ws in entities.get("workstations", [])[:3]:
            try:
                status = await self.workstation_tool.get_workstation_status(ws)
                if status:
                    results.append(f"Status of {ws}: {status}")
            except Exception as e:
                logger.debug("Could not get status for %s: %s", ws, e)

        return "\n".join(results)


class KnowledgeSpecialist(BaseSpecialist):
    """
    Specialist for documentation and troubleshooting knowledge.

    v5.2.3.24: Agno-free implementation using RAG.
    """

    specialist_type = SpecialistType.KNOWLEDGE

    def __init__(
        self,
        config: SpecialistConfig | None = None,
        knowledge_base: Any | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.knowledge_base = knowledge_base

    def _get_tools(self) -> list:
        return ["rag_search"]

    async def _execute_tools(self, query: str, context: dict | None) -> str:
        """Search knowledge base for relevant documents."""
        if not self.knowledge_base:
            return ""

        try:
            # Use RAG to search
            results = await self.knowledge_base.search(query, top_k=3)

            if results:
                docs = [f"- {r.get('content', '')[:200]}..." for r in results]
                return "Relevant documentation:\n" + "\n".join(docs)

        except Exception as e:
            logger.debug("Knowledge search failed: %s", e)

        return ""


# =============================================================================
# TEAM ORCHESTRATOR (AGNO-FREE)
# =============================================================================


class TWSSpecialistTeam:
    """
    Orchestrates the 4 specialist agents as a coordinated team.

    v5.2.3.24: Removed Agno Team dependency, uses direct orchestration.
    """

    def __init__(
        self,
        config: TeamConfig | None = None,
        knowledge_base: Any | None = None,
    ):
        self.config = config or DEFAULT_TEAM_CONFIG
        self.knowledge_base = knowledge_base

        self.classifier = QueryClassifier()

        # Initialize specialists
        self.specialists: dict[SpecialistType, BaseSpecialist] = {}
        self._init_specialists()

        logger.info(
            "specialist_team_initialized",
            specialists=list(self.specialists.keys()),
            execution_mode=self.config.execution_mode,
            agno_free=True,
        )

    def _init_specialists(self):
        """Initialize all specialist agents."""
        specialist_classes = {
            SpecialistType.JOB_ANALYST: JobAnalystAgent,
            SpecialistType.DEPENDENCY: DependencySpecialist,
            SpecialistType.RESOURCE: ResourceSpecialist,
            SpecialistType.KNOWLEDGE: KnowledgeSpecialist,
        }

        for spec_type, spec_class in specialist_classes.items():
            spec_config = self.config.specialists.get(
                spec_type, DEFAULT_SPECIALIST_CONFIGS.get(spec_type)
            )

            if spec_config and spec_config.enabled:
                kwargs = {}
                if spec_type == SpecialistType.KNOWLEDGE:
                    kwargs["knowledge_base"] = self.knowledge_base

                self.specialists[spec_type] = spec_class(config=spec_config, **kwargs)

    async def process(
        self,
        query: str,
        context: dict | None = None,
        use_all_specialists: bool = False,
    ) -> TeamResponse:
        """
        Process a query using the specialist team.

        v5.2.3.24: Agno-free implementation.
        """
        start_time = time.time()

        # Classify query
        classification = self.classifier.classify(query)

        # Determine which specialists to use
        if use_all_specialists:
            specialists_to_use = list(self.specialists.values())
        else:
            specialists_to_use = [
                self.specialists[spec_type]
                for spec_type in classification.recommended_specialists
                if spec_type in self.specialists
            ]

        # If no specialists matched, use Knowledge as fallback
        if not specialists_to_use:
            if SpecialistType.KNOWLEDGE in self.specialists:
                specialists_to_use = [self.specialists[SpecialistType.KNOWLEDGE]]
            else:
                specialists_to_use = list(self.specialists.values())[:1]

        logger.info(
            "processing_query",
            query_type=classification.query_type,
            specialists=[s.specialist_type.value for s in specialists_to_use],
            parallel=self.config.parallel_execution,
        )

        # Add entities to context
        context = context or {}
        context["entities"] = classification.entities

        # Execute specialists
        if self.config.parallel_execution:
            specialist_responses = await self._execute_parallel(specialists_to_use, query, context)
        else:
            specialist_responses = await self._execute_sequential(specialists_to_use, query, context)

        # Synthesize responses
        synthesized = await self._synthesize_responses(query, specialist_responses, classification)

        total_time = int((time.time() - start_time) * 1000)

        return TeamResponse(
            query=query,
            synthesized_response=synthesized,
            specialist_responses=specialist_responses,
            execution_mode=self.config.execution_mode,
            total_processing_time_ms=total_time,
            specialists_used=[r.specialist_type for r in specialist_responses if r.is_successful],
            query_classification=classification.query_type,
            confidence=self._calculate_confidence(specialist_responses),
        )

    async def _execute_parallel(
        self,
        specialists: list[BaseSpecialist],
        query: str,
        context: dict,
    ) -> list[SpecialistResponse]:
        """Execute specialists in parallel."""
        tasks = [
            spec.process(query, context)
            for spec in specialists[: self.config.max_parallel_specialists]
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        result = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                result.append(
                    SpecialistResponse(
                        specialist_type=specialists[i].specialist_type,
                        response="",
                        confidence=0.0,
                        error=str(resp),
                    )
                )
            else:
                result.append(resp)

        return result

    async def _execute_sequential(
        self,
        specialists: list[BaseSpecialist],
        query: str,
        context: dict,
    ) -> list[SpecialistResponse]:
        """Execute specialists sequentially."""
        responses = []
        for spec in specialists:
            try:
                resp = await spec.process(query, context)
                responses.append(resp)
            except Exception as e:
                responses.append(
                    SpecialistResponse(
                        specialist_type=spec.specialist_type,
                        response="",
                        confidence=0.0,
                        error=str(e),
                    )
                )
        return responses

    async def _synthesize_responses(
        self,
        query: str,
        responses: list[SpecialistResponse],
        classification: QueryClassification,
    ) -> str:
        """Synthesize specialist responses into a coherent answer."""
        successful_responses = [r for r in responses if r.is_successful]

        if not successful_responses:
            return "Desculpe, não foi possível processar sua consulta. Por favor, tente novamente."

        # If single response, return directly
        if len(successful_responses) == 1:
            return successful_responses[0].response

        # Multiple responses - synthesize with LLM
        responses_text = "\n\n".join([
            f"### {r.specialist_type.value.title()}\n{r.response}"
            for r in successful_responses
        ])

        synthesis_prompt = SPECIALIST_PROMPTS.get("synthesis", {}).get(
            "combine_responses",
            "Combine these specialist responses into a unified answer:\n\n{responses}"
        )

        return await _call_llm(
            prompt=synthesis_prompt.format(responses=responses_text),
            system_prompt="You are a helpful assistant that synthesizes technical information.",
            model=self.config.orchestrator_model,
        )


    def _calculate_confidence(self, responses: list[SpecialistResponse]) -> float:
        """Calculate overall confidence from specialist responses."""
        successful = [r for r in responses if r.is_successful]

        if not successful:
            return 0.0

        return sum(r.confidence for r in successful) / len(successful)


# =============================================================================
# SINGLETON FUNCTIONS
# =============================================================================

_team_instance: TWSSpecialistTeam | None = None


def get_specialist_team() -> TWSSpecialistTeam:
    """Get the singleton TWSSpecialistTeam instance."""
    global _team_instance
    if _team_instance is None:
        _team_instance = TWSSpecialistTeam()
    return _team_instance


def create_specialist_team(
    config: "SpecialistConfig | None" = None,
    llm_client: Any = None,
) -> TWSSpecialistTeam:
    """Create a new TWSSpecialistTeam instance.

    Args:
        config: Optional specialist configuration
        llm_client: Optional LLM client

    Returns:
        New TWSSpecialistTeam instance
    """
    from resync.core.specialists.models import SpecialistConfig

    if config is None:
        config = SpecialistConfig()

    return TWSSpecialistTeam(config=config, llm_client=llm_client)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "QueryClassifier",
    "BaseSpecialist",
    "JobAnalystAgent",
    "DependencySpecialist",
    "ResourceSpecialist",
    "KnowledgeSpecialist",
    "TWSSpecialistTeam",
    "SPECIALIST_PROMPTS",
    "get_specialist_team",
    "create_specialist_team",
]
