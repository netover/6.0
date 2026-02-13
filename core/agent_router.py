"""
Hybrid Agent Router for Resync v5.4.1.

This module provides intelligent routing between 3 execution paths:
1. RAG-only: Quick, low-risk queries using knowledge base
2. Agentic (Agno Team): Multi-step tasks requiring tool use
3. Diagnostic (LangGraph): Sensitive troubleshooting with HITL

Architecture:
    User Message → Intent Classifier → Hybrid Router → Appropriate Path → Response

Features:
    - Automatic intent detection and routing
    - Minimum autonomy principle (use least autonomy needed)
    - Seamless fallback between paths
    - Security through separation of concerns

v5.4.1 Enhancements (PR-3):
    - Formalized 3-path routing
    - RAG content treated as untrusted data
    - Cost/loop controls per path
    - Explicit HITL checkpoints

Author: Resync Team
Version: 5.4.1
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# ROUTING MODES
# =============================================================================


class RoutingMode(str, Enum):
    """
    Routing modes for hybrid agent system.

    Choose the minimum autonomy necessary:
    - RAG_ONLY: Cheapest, fastest, no tool execution
    - AGENTIC: Multi-step with tools, controlled loops
    - DIAGNOSTIC: Full autonomy with HITL checkpoints
    """

    RAG_ONLY = "rag_only"  # Knowledge base search only
    AGENTIC = "agentic"  # Agno Team with tools
    DIAGNOSTIC = "diagnostic"  # LangGraph with HITL


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================


class Intent(str, Enum):
    """
    Classified intents for user messages.
    Each intent maps to a routing mode.
    """

    # TWS Operations
    STATUS = "status"  # Check system/job/workstation status
    TROUBLESHOOTING = "troubleshooting"  # Diagnose and fix issues
    JOB_MANAGEMENT = "job_management"  # Run, stop, rerun jobs
    MONITORING = "monitoring"  # Real-time monitoring queries

    # Analysis & Reporting
    ANALYSIS = "analysis"  # Deep analysis of patterns/trends
    REPORTING = "reporting"  # Generate reports

    # General
    GENERAL = "general"  # General questions, help
    GREETING = "greeting"  # Hello, hi, etc.

    # System
    UNKNOWN = "unknown"  # Cannot classify


# Intent to routing mode mapping
INTENT_TO_ROUTING: dict[Intent, RoutingMode] = {
    Intent.GREETING: RoutingMode.RAG_ONLY,
    Intent.GENERAL: RoutingMode.RAG_ONLY,
    Intent.REPORTING: RoutingMode.RAG_ONLY,
    Intent.STATUS: RoutingMode.AGENTIC,
    Intent.MONITORING: RoutingMode.AGENTIC,
    Intent.ANALYSIS: RoutingMode.AGENTIC,
    Intent.JOB_MANAGEMENT: RoutingMode.AGENTIC,
    Intent.TROUBLESHOOTING: RoutingMode.DIAGNOSTIC,
    Intent.UNKNOWN: RoutingMode.RAG_ONLY,
}


@dataclass
class IntentClassification:
    """Result of intent classification."""

    primary_intent: Intent
    confidence: float  # 0.0 to 1.0
    secondary_intents: list[Intent] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    requires_tools: bool = False
    suggested_routing: RoutingMode = RoutingMode.RAG_ONLY
    matched_skills: list[str] = field(default_factory=list)  # Maps Intent to Skill names

    @property
    def is_high_confidence(self) -> bool:
        """Check if classification confidence is high enough for routing."""
        return self.confidence >= 0.7

    @property
    def needs_clarification(self) -> bool:
        """Check if we should ask for clarification."""
        return self.confidence < 0.4


class IntentClassifier:
    """
    Classifies user messages into intents using a combination of
    rule-based keyword matching and optional LLM enhancement.
    """

    # Keyword patterns for each intent (Portuguese and English)
    INTENT_PATTERNS: dict[Intent, list[str]] = {
        Intent.STATUS: [
            r"\bstatus\b",
            r"\bsituação\b",
            r"\bestado\b",
            r"\bcomo\s+est[áa]\b",
            r"\bqual\s+o\s+status\b",
            r"\bworkstation[s]?\b",
            r"\bjob[s]?\b.*\b(rodando|executando|running)\b",
            r"\bonline\b",
            r"\boffline\b",
            r"\blinked\b",
            r"\bunlinked\b",
            r"\bativo[s]?\b",
            r"\binativo[s]?\b",
        ],
        Intent.TROUBLESHOOTING: [
            r"\babend\b",
            r"\berro[s]?\b",
            r"\bfalha[s]?\b",
            r"\bfailed\b",
            r"\bproblema[s]?\b",
            r"\bissue[s]?\b",
            r"\bbug[s]?\b",
            r"\bpor\s*que\b.*\b(falhou|erro|abend)\b",
            r"\bcausa\b",
            r"\broot\s*cause\b",
            r"\bdiagn[oó]stic\b",
            r"\bresolver\b",
            r"\bfix\b",
            r"\bcorrigir\b",
            r"\bsolução\b",
            r"\binvestigar\b",
            r"\banalisar\s+(erro|falha|problema)\b",
            r"\brc\s*=\s*\d+\b",
            r"\breturn\s*code\b",
            r"\bs0c\d\b",
        ],
        Intent.JOB_MANAGEMENT: [
            r"\bexecutar\b",
            r"\brodar\b",
            r"\brun\b",
            r"\bstart\b",
            r"\bparar\b",
            r"\bstop\b",
            r"\bcancel\b",
            r"\bkill\b",
            r"\brerun\b",
            r"\breexecutar\b",
            r"\breiniciar\b",
            r"\bsubmit\b",
            r"\brelease\b",
            r"\bhold\b",
            r"\bagendar\b",
            r"\bschedule\b",
        ],
        Intent.MONITORING: [
            r"\bmonitor\w*\b",
            r"\bacompanhar\b",
            r"\bwatch\b",
            r"\balert[as]?\b",
            r"\bnotifica\w*\b",
            r"\btempo\s+real\b",
            r"\breal[\s-]*time\b",
            r"\bdashboard\b",
            r"\bmétricas?\b",
            r"\bmetrics?\b",
        ],
        Intent.ANALYSIS: [
            r"\banalis[ae]\w*\b",
            r"\banalyz[e]?\w*\b",
            r"\btendênci\w*\b",
            r"\btrend[s]?\b",
            r"\bpadr[ãa]o\b",
            r"\bpattern[s]?\b",
            r"\bhistóric\w*\b",
            r"\bhistor\w*\b",
            r"\bcompar\w*\b",
            r"\bcorrelação\b",
            r"\bperformance\b",
            r"\bdesempenho\b",
        ],
        Intent.REPORTING: [
            r"\brelatório\w*\b",
            r"\breport\w*\b",
            r"\bgerar\b.*\b(relatório|report)\b",
            r"\bexport\w*\b",
            r"\bextrair\b",
            r"\bressum\w*\b",
            r"\bsummar\w*\b",
            r"\bdocument\w*\b",
        ],
        Intent.GREETING: [
            r"^(ol[áa]|oi|hey|hi|hello|bom\s*dia|boa\s*(tarde|noite))[\s!?.,]*$",
            r"^(tudo\s*bem|como\s*vai)[\s!?.,]*$",
        ],
        Intent.GENERAL: [
            r"\bajud[ae]\b",
            r"\bhelp\b",
            r"\bo\s*que\s*(é|são|significa)\b",
            r"\bwhat\s*is\b",
            r"\bcomo\s+(funciona|usar|faz)\b",
            r"\bhow\s+(to|do|does)\b",
            r"\bexplica\w*\b",
            r"\bexplain\b",
        ],
    }

    # Entity extraction patterns
    ENTITY_PATTERNS: dict[str, str] = {
        "job_name": r"\bjob\s+([A-Z0-9_]+)\b|\b([A-Z][A-Z0-9_]{2,})\b",
        "workstation": r"\bworkstation\s+([A-Z0-9_]+)\b|\b(TWS_[A-Z0-9_]+)\b",
        "error_code": r"\brc\s*=\s*(\d+)\b|\breturn\s*code\s*(\d+)\b",
        "abend_code": r"\b([SU]0?C?\d+[A-Z]?)\b",
        "time_reference": r"\b(hoje|ontem|última\s+hora|last\s+hour|yesterday|today)\b",
    }

    def __init__(self, llm_classifier: Callable | None = None):
        """
        Initialize the classifier.

        Args:
            llm_classifier: Optional async function for LLM-based classification
        """
        self.llm_classifier = llm_classifier
        self._compiled_patterns: dict[Intent, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns
            ]

    def classify(self, message: str) -> IntentClassification:
        """
        Classify a user message into an intent.

        Args:
            message: The user's input message

        Returns:
            IntentClassification with primary intent and routing suggestion
        """
        message = message.strip()

        if not message:
            return IntentClassification(
                primary_intent=Intent.UNKNOWN,
                confidence=0.0,
                suggested_routing=RoutingMode.RAG_ONLY,
            )

        # Score each intent based on pattern matches
        scores: dict[Intent, float] = {}

        for intent, patterns in self._compiled_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern.search(message))
            if match_count > 0:
                # Score based on matches relative to total patterns
                scores[intent] = min(1.0, match_count / max(len(patterns) * 0.3, 1))

        # Determine primary intent
        if scores:
            sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            primary_intent = sorted_intents[0][0]
            confidence = sorted_intents[0][1]
            secondary_intents = [i for i, _ in sorted_intents[1:3] if _ >= 0.3]
        else:
            primary_intent = Intent.UNKNOWN
            confidence = 0.2
            secondary_intents = []

        # Extract entities
        entities = self._extract_entities(message)

        # Determine if tools are required
        requires_tools = primary_intent in (
            Intent.STATUS,
            Intent.TROUBLESHOOTING,
            Intent.JOB_MANAGEMENT,
            Intent.MONITORING,
            Intent.ANALYSIS,
        )

        # Determine routing mode
        suggested_routing = INTENT_TO_ROUTING.get(primary_intent, RoutingMode.RAG_ONLY)

        # Upgrade routing if tools are needed and confidence is high
        if requires_tools and confidence >= 0.6:
            if primary_intent == Intent.TROUBLESHOOTING:
                suggested_routing = RoutingMode.DIAGNOSTIC
            else:
                suggested_routing = RoutingMode.AGENTIC

        # Map Intent to Skills (gerúndio)
        matched_skills: list[str] = []
        if primary_intent == Intent.TROUBLESHOOTING:
            matched_skills.append("analyzing-job-logs")
        elif primary_intent == Intent.MONITORING:
            matched_skills.append("monitoring-workstations")
        elif primary_intent == Intent.JOB_MANAGEMENT:
            matched_skills.append("managing-jobs")
        elif primary_intent == Intent.STATUS:
            matched_skills.append("checking-status")
        elif primary_intent == Intent.ANALYSIS:
            matched_skills.append("analyzing-performance")

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            entities=entities,
            requires_tools=requires_tools,
            suggested_routing=suggested_routing,
            matched_skills=matched_skills,
        )

    def _extract_entities(self, message: str) -> dict[str, list[str]]:
        """Extract entities from the message."""
        entities: dict[str, list[str]] = {}

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            # Flatten tuple results from groups
            flat_matches = []
            for m in matches:
                if isinstance(m, tuple):
                    flat_matches.extend([x for x in m if x])
                else:
                    flat_matches.append(m)
            if flat_matches:
                entities[entity_type] = list(set(flat_matches))

        return entities


# =============================================================================
# ROUTING RESULT
# =============================================================================


@dataclass
class RoutingResult:
    """Result of hybrid routing."""

    response: str
    routing_mode: RoutingMode
    intent: str
    confidence: float
    handler: str
    tools_used: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    trace_id: str | None = None

    # HITL status
    requires_approval: bool = False
    approval_id: str | None = None


# =============================================================================
# HANDLER PROTOCOL
# =============================================================================


class Handler(Protocol):
    """Protocol for message handlers."""

    def handle(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle a message and return response."""
        ...


# =============================================================================
# BASE HANDLER
# =============================================================================


class BaseHandler(ABC):
    """Base class for all handlers."""

    def __init__(self, agent_manager: Any = None):
        self.agent_manager = agent_manager
        self.last_tools_used: list[str] = []

    @abstractmethod
    def handle(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle the message and return a response."""

    async def _call_tool(self, tool_name: str, **kwargs) -> str | None:
        """Call a tool and track usage."""
        if self.agent_manager and hasattr(self.agent_manager, "call_tool"):
            result = await self.agent_manager.call_tool(tool_name, **kwargs)
            if result:
                self.last_tools_used.append(tool_name)
            return result
        return None

    async def _get_agent_response(self, agent_id: str, message: str, skill_context: str = "") -> str:
        """Get response from a specific agent injecting skill context."""
        if self.agent_manager:
            agent = await self.agent_manager.get_agent(agent_id)
            if agent and hasattr(agent, "arun"):
                return await agent.arun(message, skill_context=skill_context)
        return ""


# =============================================================================
# RAG-ONLY HANDLER (Path 1)
# =============================================================================


class RAGOnlyHandler(BaseHandler):
    """
    Handler for RAG-only queries.

    Uses knowledge base search without tool execution.
    Fastest and cheapest path.
    """

    async def handle(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        self.last_tools_used = []

        try:
            # Search knowledge base
            from resync.core.specialists.tools import RAGTool

            rag = RAGTool()
            results = rag.search_knowledge_base(
                query=message,
                top_k=5,
                use_hybrid=True,
            )

            if results.get("results"):
                self.last_tools_used.append("search_knowledge_base")

                # Format response from RAG results
                response_parts = ["Com base na documentação disponível:\n"]
                for i, result in enumerate(results["results"][:3], 1):
                    content = result.get("content", result.get("text", ""))
                    if content:
                        response_parts.append(f"{i}. {content[:500]}...")

                return "\n".join(response_parts)

            # Fallback to general response
            return self._general_response(message, classification)

        except Exception as e:
            logger.warning("RAG search failed: %s", e)
            return self._general_response(message, classification)

    def _general_response(
        self,
        message: str,
        classification: IntentClassification,
    ) -> str:
        """Generate a general response without RAG."""
        if classification.primary_intent == Intent.GREETING:
            return (
                "Olá! 👋 Sou o assistente do Resync para TWS. "
                "Posso ajudar com status de jobs, troubleshooting, e muito mais. "
                "Como posso ajudar?"
            )

        return (
            "Entendi sua pergunta. Posso ajudar com:\n\n"
            "• **Status:** Verificar situação de jobs e workstations\n"
            "• **Troubleshooting:** Diagnosticar e resolver problemas\n"
            "• **Monitoramento:** Acompanhar execuções em tempo real\n"
            "• **Análise:** Identificar padrões e tendências\n\n"
            "Como posso ajudar?"
        )


# =============================================================================
# AGENTIC HANDLER (Path 2) - Enhanced with Parallel Execution (PR-8)
# =============================================================================


class AgenticHandler(BaseHandler):
    """
    Handler for agentic queries using Agno Team.

    v5.4.2 Enhancements (PR-8):
    - Parallel execution for read-only operations
    - Sub-agent dispatch for complex searches

    Executes multi-step tasks with tool use.
    Controlled by tool_call_limit and max_steps.
    """

    def __init__(self, agent_manager: Any = None):
        super().__init__(agent_manager)
        self.tool_call_limit = 10
        self.max_steps = 15
        self._parallel_executor = None

    @property
    def parallel_executor(self):
        """Lazy load parallel executor."""
        if self._parallel_executor is None:
            from resync.core.specialists.parallel_executor import ParallelToolExecutor

            self._parallel_executor = ParallelToolExecutor()
        return self._parallel_executor

    async def handle(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        self.last_tools_used = []

        try:
            # Route to appropriate specialist based on intent
            handler_map = {
                Intent.STATUS: self._handle_status,
                Intent.MONITORING: self._handle_monitoring,
                Intent.ANALYSIS: self._handle_analysis,
                Intent.JOB_MANAGEMENT: self._handle_job_management,
            }

            handler = handler_map.get(classification.primary_intent)
            if handler:
                return await handler(message, context, classification)

            # Default: use general agent
            return await self._get_agent_response("tws-general", message)

        except Exception as e:
            logger.error("Agentic handler error: %s", e)
            return f"Erro ao processar a solicitação: {e}"

    async def _handle_status(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle status queries with parallel execution."""
        from resync.core.specialists.parallel_executor import (
            ExecutionStrategy,
            ToolRequest,
        )

        # Check for specific workstations/jobs
        workstations = classification.entities.get("workstation", [])
        job_names = classification.entities.get("job_name", [])

        requests = []

        # Add workstation status requests (parallel-safe)
        # v5.7.1 FIX: Correct parameter name (ws_name → workstation_name)
        if workstations:
            for ws in workstations[:5]:  # Limit to 5
                requests.append(
                    ToolRequest(
                        tool_name="get_workstation_status",
                        parameters={"workstation_name": ws},
                    )
                )
        else:
            requests.append(
                ToolRequest(
                    tool_name="get_workstation_status",
                    parameters={},
                )
            )

        # Add job status requests if mentioned
        # v5.7.1 FIX: Correct parameter name (lines → max_lines)
        if job_names:
            for job in job_names[:5]:  # Limit to 5
                requests.append(
                    ToolRequest(
                        tool_name="get_job_log",
                        parameters={"job_name": job, "max_lines": 10},
                    )
                )

        # Execute in parallel (all read-only)
        responses = await self.parallel_executor.execute(
            requests, strategy=ExecutionStrategy.CONCURRENT
        )

        self.last_tools_used = [r.tool_name for r in responses if r.success]

        # Format response
        response_parts = ["**📊 Status do Sistema**\n"]

        for r in responses:
            if r.success and r.result:
                if r.tool_name == "get_workstation_status":
                    result = r.result
                    if isinstance(result, dict):
                        if "workstations" in result:
                            for ws in result["workstations"]:
                                status_icon = "🟢" if ws.get("status") == "ONLINE" else "🔴"
                                response_parts.append(
                                    f"{status_icon} **{ws.get('name')}**: "
                                    f"{ws.get('status')} - {ws.get('jobs_running', 0)} jobs"
                                )
                        else:
                            ws = result
                            status_icon = "🟢" if ws.get("status") == "ONLINE" else "🔴"
                            response_parts.append(
                                f"{status_icon} **{ws.get('workstation', 'WS')}**: "
                                f"{ws.get('status')} - {ws.get('jobs_running', 0)} jobs"
                            )

                elif r.tool_name == "get_job_log":
                    result = r.result
                    if isinstance(result, dict):
                        job = result.get("job_name", "Job")
                        status = result.get("status", "UNKNOWN")
                        status_icon = "✅" if status in ["COMPLETED", "SUCCESS"] else "❌"
                        response_parts.append(f"{status_icon} **{job}**: {status}")

        if len(response_parts) == 1:
            response_parts.append("Não foi possível obter informações de status.")

        response_parts.append(f"\n_Consultados {len(responses)} recursos em paralelo_")

        return "\n".join(response_parts)

    async def _handle_monitoring(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle monitoring queries with parallel data collection."""
        from resync.core.specialists.parallel_executor import (
            ExecutionStrategy,
            ToolRequest,
        )

        # Collect metrics and status in parallel
        requests = [
            ToolRequest(tool_name="get_system_metrics", parameters={}),
            ToolRequest(tool_name="get_workstation_status", parameters={}),
        ]

        responses = await self.parallel_executor.execute(
            requests, strategy=ExecutionStrategy.CONCURRENT
        )

        self.last_tools_used = [r.tool_name for r in responses if r.success]

        response = "**📊 Monitoramento TWS**\n\n"

        for r in responses:
            if r.success and r.result and r.tool_name == "get_system_metrics":
                m = r.result.get("metrics", r.result)
                response += f"Taxa de sucesso: {m.get('job_success_rate', 0) * 100:.1f}%\n"
                response += f"Workstations ativas: {m.get('active_workstations', 0)}\n"
                response += f"Jobs na fila: {m.get('queue_depth', 0)}\n"
                response += f"Taxa de erros: {m.get('error_rate', 0) * 100:.1f}%\n"

        response += "\n💡 Use o dashboard para monitoramento em tempo real."

        return response

    async def _handle_analysis(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle analysis queries with parallel sub-agents for multiple jobs."""
        job_names = classification.entities.get("job_name", [])

        if len(job_names) > 1:
            # Multiple jobs: use parallel sub-agents (PR-10)
            try:
                from resync.core.specialists.sub_agent import SubAgent

                agents = [
                    SubAgent(prompt=f"Analyze job {job} performance and history")
                    for job in job_names[:5]
                ]

                results = await SubAgent.execute_parallel(agents, max_concurrent=5)

                self.last_tools_used = ["dispatch_parallel_sub_agents"]

                response = "**📈 Análise Paralela de Jobs**\n\n"
                for _i, (job, result) in enumerate(zip(job_names, results, strict=False)):
                    status_icon = "✅" if result.status.value == "completed" else "⚠️"
                    response += f"{status_icon} **{job}**: {result.summary}\n"

                response += f"\n_Analisados {len(job_names)} jobs em paralelo_"
                return response

            except Exception as e:
                logger.warning("Parallel sub-agent failed: %s", e)

        # Single job or fallback
        if job_names:
            from resync.core.specialists.tools import JobLogTool

            job_tool = JobLogTool()
            job_name = job_names[0]

            history = job_tool.get_job_history(job_name, days=7)
            self.last_tools_used.append("get_job_history")

            return (
                f"**Análise do Job {job_name}**\n\n"
                f"Período: últimos {history.get('period_days', 7)} dias\n"
                f"Execuções: {history.get('total_executions', 0)}\n"
                f"Taxa de sucesso: {history.get('success_rate', 0) * 100:.1f}%\n"
                f"Duração média: {history.get('avg_duration_seconds', 0) / 60:.1f} min\n"
                f"Falhas: {history.get('failure_count', 0)}\n"
                f"Tendência: {history.get('trend', 'estável')}"
            )

        # NOVO: Buscar o conteúdo das skills (Deep Loading)
        from resync.core.skill_manager import get_skill_manager
        
        skill_mgr = get_skill_manager()
        skill_text = ""
        for skill_name in classification.matched_skills:
            content = skill_mgr.get_skill_content(skill_name)
            if content:
                skill_text += f"\n\n=== {skill_name.upper()} ===\n{content}"

        # Quando for chamar o agente, passe a skill
        return await self._get_agent_response("job-analyst", message, skill_context=skill_text)

    async def _handle_job_management(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        """Handle job management queries with risk assessment."""
        from resync.core.specialists.tools import (
            RiskLevel,
            get_tool_catalog,
        )

        job_names = classification.entities.get("job_name", [])

        if not job_names:
            return (
                "Para executar operações em jobs, preciso que você especifique "
                "o nome do job. Por exemplo: 'Rerun do job ETL_DAILY_BACKUP'"
            )

        # Check for destructive operations
        destructive_keywords = ["stop", "cancel", "kill", "parar", "cancelar"]
        is_destructive = any(kw in message.lower() for kw in destructive_keywords)

        # PR-12: Assess risk level
        catalog = get_tool_catalog()
        risk_level = catalog.assess_risk(
            "execute_tws_command",
            {"job_name": job_names[0], "action": message},
        )

        risk_icons = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴",
        }

        if is_destructive or risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            return (
                f"{risk_icons.get(risk_level, '⚠️')} **Confirmação Necessária** "
                f"(Risco: {risk_level.value.upper()})\n\n"
                f"Você solicitou uma operação no job `{job_names[0]}`.\n\n"
                f"Esta operação requer aprovação humana. "
                f"Por favor, confirme no painel de aprovações ou digite: "
                f"'Confirmo {message.lower()}'"
            )

        return await self._get_agent_response("tws-general", message)


# =============================================================================
# DIAGNOSTIC HANDLER (Path 3)
# =============================================================================


class DiagnosticHandler(BaseHandler):
    """
    Handler for diagnostic queries using LangGraph.

    Sensitive troubleshooting with HITL checkpoints.
    Uses cyclic graph for iterative diagnosis.
    """

    async def handle(
        self,
        message: str,
        context: dict[str, Any],
        classification: IntentClassification,
    ) -> str:
        self.last_tools_used = []

        try:
            # Try to use LangGraph diagnostic
            from resync.core.langgraph.diagnostic_graph import diagnose_problem

            result = await diagnose_problem(
                problem_description=message,
                tws_instance_id=context.get("tws_instance_id"),
            )

            if result.get("success"):
                self.last_tools_used.append("diagnostic_graph")

                response_parts = []

                if result.get("root_cause"):
                    response_parts.append(f"**🔍 Causa Raiz:** {result['root_cause']}")

                if result.get("solution"):
                    response_parts.append(f"\n**💡 Solução:** {result['solution']}")

                if result.get("steps"):
                    response_parts.append("\n**📋 Passos para Resolução:**")
                    for i, step in enumerate(result["steps"], 1):
                        response_parts.append(f"{i}. {step}")

                if result.get("recommendations"):
                    response_parts.append("\n**📌 Recomendações:**")
                    for rec in result["recommendations"]:
                        response_parts.append(f"• {rec}")

                if result.get("requires_action"):
                    response_parts.append(
                        f"\n\n⚠️ **Atenção:** Esta solução requer ações que "
                        f"precisam de aprovação humana. Nível de risco: {result.get('risk_level', 'médio')}"
                    )

                return "\n".join(response_parts)

            # Fallback to manual analysis
            return self._manual_troubleshooting(message, classification)

        except ImportError:
            logger.warning("LangGraph not available, using fallback")
            return self._manual_troubleshooting(message, classification)
        except Exception as e:
            logger.error("Diagnostic handler error: %s", e)
            return self._manual_troubleshooting(message, classification)

    def _manual_troubleshooting(
        self,
        message: str,
        classification: IntentClassification,
    ) -> str:
        """Manual troubleshooting when LangGraph is not available."""
        from resync.core.specialists.tools import (
            JobLogTool,
            SearchHistoryTool,
        )

        response_parts = ["**🔧 Análise de Troubleshooting**\n"]

        # Check for specific error codes
        error_codes = classification.entities.get("error_code", [])
        abend_codes = classification.entities.get("abend_code", [])
        job_names = classification.entities.get("job_name", [])

        job_tool = JobLogTool()

        # Analyze ABEND codes
        if abend_codes:
            for code in abend_codes[:2]:
                analysis = job_tool.analyze_abend_code(code)
                self.last_tools_used.append("analyze_abend_code")

                response_parts.append(
                    f"\n**ABEND {code}:** {analysis.get('description', 'Desconhecido')}"
                )
                response_parts.append("\nCausas comuns:")
                for cause in analysis.get("common_causes", [])[:3]:
                    response_parts.append(f"• {cause}")
                response_parts.append("\nAções recomendadas:")
                for action in analysis.get("recommended_actions", [])[:3]:
                    response_parts.append(f"• {action}")

        # Analyze return codes
        if error_codes:
            for code in error_codes[:2]:
                try:
                    rc = int(code.replace("RC=", "").strip())
                    analysis = job_tool.analyze_return_code(rc)
                    self.last_tools_used.append("analyze_return_code")

                    response_parts.append(
                        f"\n**Return Code {rc}:** {analysis.get('description', '')}"
                    )
                    response_parts.append(f"Severidade: {analysis.get('severity', 'N/A')}")
                except ValueError as exc:
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        # Get job log if job name provided
        if job_names:
            log = job_tool.get_job_log(job_names[0])
            self.last_tools_used.append("get_job_log")

            response_parts.append(f"\n**Log do Job {job_names[0]}:**")
            response_parts.append(f"Status: {log.get('status', 'N/A')}")
            if log.get("error_details"):
                response_parts.append(f"Detalhes: {log['error_details']}")

        # Search history for similar incidents
        try:
            history_tool = SearchHistoryTool()
            history = history_tool.search_history(message, limit=3)
            self.last_tools_used.append("search_history")

            if history.get("recommendations"):
                response_parts.append("\n**Baseado em incidentes similares:**")
                for rec in history["recommendations"][:2]:
                    response_parts.append(f"• {rec}")
        except Exception as exc:
            logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        if len(response_parts) == 1:
            response_parts.append(
                "\nNão encontrei informações específicas sobre este problema. "
                "Por favor, forneça mais detalhes como:\n"
                "• Nome do job afetado\n"
                "• Código de erro (ABEND ou RC)\n"
                "• Mensagem de erro completa"
            )

        return "\n".join(response_parts)


# =============================================================================
# HYBRID ROUTER
# =============================================================================


class HybridRouter:
    """
    Main router that orchestrates the 3 execution paths.

    Follows the principle of minimum autonomy:
    use the least autonomous path that can handle the request.
    """

    def __init__(
        self,
        agent_manager: Any = None,
        classifier: IntentClassifier | None = None,
    ):
        self.agent_manager = agent_manager
        self.classifier = classifier or IntentClassifier()

        # Initialize handlers
        self._handlers: dict[RoutingMode, BaseHandler] = {
            RoutingMode.RAG_ONLY: RAGOnlyHandler(agent_manager),
            RoutingMode.AGENTIC: AgenticHandler(agent_manager),
            RoutingMode.DIAGNOSTIC: DiagnosticHandler(agent_manager),
        }

        logger.info(
            "hybrid_router_initialized",
            modes=list(self._handlers.keys()),
        )

    async def route(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        force_mode: RoutingMode | None = None,
    ) -> RoutingResult:
        """
        Route a message through the appropriate path.

        Args:
            message: User message
            context: Additional context
            force_mode: Force a specific routing mode

        Returns:
            RoutingResult with response and metadata
        """
        start_time = time.time()
        context = context or {}

        # Classify intent
        classification = self.classifier.classify(message)

        # Determine routing mode
        routing_mode = force_mode or classification.suggested_routing

        logger.info(
            "routing_decision",
            intent=classification.primary_intent.value,
            confidence=classification.confidence,
            routing_mode=routing_mode.value,
        )

        # Get handler
        handler = self._handlers.get(routing_mode)
        if not handler:
            handler = self._handlers[RoutingMode.RAG_ONLY]
            routing_mode = RoutingMode.RAG_ONLY

        # Execute handler
        try:
            response = await handler.handle(message, context, classification)
            tools_used = handler.last_tools_used if hasattr(handler, "last_tools_used") else []
        except Exception as e:
            logger.error("Handler error: %s", e)
            response = f"Erro ao processar: {e}"
            tools_used = []

        processing_time = int((time.time() - start_time) * 1000)

        return RoutingResult(
            response=response,
            routing_mode=routing_mode,
            intent=classification.primary_intent.value,
            confidence=classification.confidence,
            handler=handler.__class__.__name__,
            tools_used=tools_used,
            entities=classification.entities,
            processing_time_ms=processing_time,
        )


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


# Alias for backward compatibility
class AgentRouter(HybridRouter):
    """Backward-compatible alias for HybridRouter."""


def create_router(agent_manager: Any) -> HybridRouter:
    """
    Factory function to create a configured HybridRouter.

    Args:
        agent_manager: The AgentManager instance

    Returns:
        Configured HybridRouter ready for use
    """
    classifier = IntentClassifier()
    router = HybridRouter(agent_manager, classifier)

    logger.info(
        "agent_router_created",
        routing_modes=list(router._handlers.keys()),
    )

    return router


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RoutingMode",
    "Intent",
    "INTENT_TO_ROUTING",
    # Classes
    "IntentClassification",
    "IntentClassifier",
    "RoutingResult",
    "BaseHandler",
    "RAGOnlyHandler",
    "AgenticHandler",
    "DiagnosticHandler",
    "HybridRouter",
    "AgentRouter",  # Backward compatibility
    # Factory
    "create_router",
]
