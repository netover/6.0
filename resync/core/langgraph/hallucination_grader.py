"""
Hallucination Grader - Validates LLM responses are grounded in retrieved documents.

This module implements a hallucination detection system based on best practices from:
- LangGraph Adaptive RAG patterns
- Self-RAG (Self-Reflective RAG) paper concepts
- Corrective RAG patterns

The grader acts as a fact-checker, ensuring LLM responses don't fabricate information
that isn't supported by the retrieved context/documents.

Key Features:
- Binary grading (grounded/not grounded)
- Confidence scoring
- Detailed reasoning for debugging
- Support for multiple LLM backends
- Async-first design
- Metrics tracking for monitoring

Usage:
    from resync.core.langgraph.hallucination_grader import (
        HallucinationGrader,
        grade_hallucination,
        hallucination_check_node,
    )

    # Simple usage
    result = await grade_hallucination(
        documents=["The job BATCH_001 failed with RC=12"],
        generation="The job BATCH_001 failed with return code 12.",
        question="What happened to BATCH_001?"
    )

    if result.is_grounded:
        print("Response is factually grounded")
    else:
        print(f"Hallucination detected: {result.reasoning}")

Author: Resync Team
Version: 1.0.0 (v5.2.3.27)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from resync.core.structured_logger import get_logger
from resync.settings import settings

logger = get_logger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination assessment.

    This model is used with LLM structured output to get consistent,
    parseable responses from the grading LLM.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts: 'yes' if grounded, 'no' if hallucinated"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the grading (0.0 to 1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the grading decision"
    )


class GradeAnswer(BaseModel):
    """
    Binary score for answer relevance assessment.

    Checks if the generation actually addresses the user's question.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question: 'yes' if it does, 'no' if it doesn't"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the answer does/doesn't address the question"
    )


# =============================================================================
# GRADING RESULT DATACLASS
# =============================================================================


class GradeDecision(str, Enum):
    """Decision outcomes from the grading process."""

    USEFUL = "useful"              # Grounded AND answers question
    NOT_GROUNDED = "not_grounded"  # Hallucination detected
    NOT_USEFUL = "not_useful"      # Grounded but doesn't answer question
    ERROR = "error"                # Grading failed


@dataclass
class HallucinationGradeResult:
    """
    Complete result from hallucination grading.

    Contains both the binary decision and detailed metadata
    for debugging, monitoring, and continuous improvement.
    """

    # Core results
    is_grounded: bool
    decision: GradeDecision

    # Detailed scores
    hallucination_score: GradeHallucinations | None = None
    answer_score: GradeAnswer | None = None

    # Metadata
    latency_ms: float = 0.0
    model_used: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Debug info
    documents_count: int = 0
    generation_length: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "is_grounded": self.is_grounded,
            "decision": self.decision.value,
            "hallucination_score": self.hallucination_score.model_dump() if self.hallucination_score else None,
            "answer_score": self.answer_score.model_dump() if self.answer_score else None,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "documents_count": self.documents_count,
            "generation_length": self.generation_length,
            "error": self.error,
        }


# =============================================================================
# PROMPTS
# =============================================================================


HALLUCINATION_GRADER_SYSTEM_PROMPT = """Você é um avaliador especializado em verificar se uma resposta de IA está fundamentada em fatos recuperados.

Sua tarefa é determinar se a resposta gerada pelo LLM está baseada/suportada pelo conjunto de fatos fornecidos.

REGRAS DE AVALIAÇÃO:
1. A resposta deve ser baseada APENAS nos fatos fornecidos
2. Se a resposta contém informações que NÃO estão nos fatos, é uma alucinação
3. Inferências razoáveis dos fatos são aceitáveis
4. Generalizações sem base nos fatos são alucinações
5. Números, datas e códigos devem corresponder exatamente aos fatos

CONTEXTO TWS (Tivoli Workload Scheduler):
- Termos como job, workstation, return code, ABEND são técnicos e devem ser precisos
- Códigos de erro (RC=X, AWSB####) devem corresponder exatamente
- Status de jobs (SUCC, ABEND, EXEC, HOLD) devem ser precisos

Responda com:
- binary_score: "yes" se a resposta está fundamentada nos fatos, "no" se há alucinação
- confidence: sua confiança de 0.0 a 1.0
- reasoning: breve explicação da sua decisão"""


HALLUCINATION_GRADER_USER_TEMPLATE = """Conjunto de fatos recuperados:
{documents}

Resposta gerada pelo LLM:
{generation}

A resposta está fundamentada nos fatos? Avalie cuidadosamente."""


ANSWER_GRADER_SYSTEM_PROMPT = """Você é um avaliador que verifica se uma resposta de IA realmente responde à pergunta do usuário.

REGRAS:
1. A resposta deve abordar diretamente a pergunta feita
2. Informações parciais contam como "não responde" se faltam elementos críticos
3. Respostas tangenciais não contam como respostas válidas

Responda com:
- binary_score: "yes" se a resposta resolve a pergunta, "no" se não resolve
- reasoning: breve explicação"""


ANSWER_GRADER_USER_TEMPLATE = """Pergunta do usuário:
{question}

Resposta gerada:
{generation}

A resposta resolve adequadamente a pergunta do usuário?"""


# =============================================================================
# HALLUCINATION GRADER CLASS
# =============================================================================


class HallucinationGrader:
    """
    Hallucination Grader for RAG responses.

    Validates that LLM-generated responses are grounded in the retrieved
    documents and actually address the user's question.

    Implements a two-stage grading process:
    1. Hallucination check: Is the response grounded in facts?
    2. Answer check: Does the response address the question?

    Example:
        grader = HallucinationGrader()
        result = await grader.grade(
            documents=retrieved_docs,
            generation=llm_response,
            question=user_query
        )

        if result.decision == GradeDecision.USEFUL:
            return llm_response
        else:
            # Handle hallucination or irrelevant response
            ...
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        check_answer_relevance: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize the Hallucination Grader.

        Args:
            model: LLM model to use for grading (defaults to settings.llm_model)
            temperature: Temperature for grading LLM (0.0 for deterministic)
            check_answer_relevance: Whether to also check if answer addresses question
            max_retries: Number of retries on grading failure
        """
        self.model = model or settings.llm_model or "gpt-4o-mini"
        self.temperature = temperature
        self.check_answer_relevance = check_answer_relevance
        self.max_retries = max_retries

        # Metrics
        self._total_grades = 0
        self._hallucinations_detected = 0
        self._avg_latency_ms = 0.0

        logger.info(
            "hallucination_grader_initialized",
            model=self.model,
            check_answer_relevance=self.check_answer_relevance,
        )

    async def grade(
        self,
        documents: list[str] | str,
        generation: str,
        question: str | None = None,
    ) -> HallucinationGradeResult:
        """
        Grade an LLM generation for hallucination.

        Args:
            documents: Retrieved documents/facts (list of strings or single string)
            generation: The LLM-generated response to check
            question: Original user question (optional, for answer relevance check)

        Returns:
            HallucinationGradeResult with grading decision and metadata
        """
        start_time = time.time()

        # Normalize documents to string
        if isinstance(documents, list):
            docs_text = "\n\n---\n\n".join(documents)
            docs_count = len(documents)
        else:
            docs_text = documents
            docs_count = 1

        try:
            # Stage 1: Check for hallucination
            hallucination_result = await self._grade_hallucination(
                documents=docs_text,
                generation=generation,
            )

            is_grounded = hallucination_result.binary_score == "yes"

            # Stage 2: Check answer relevance (if enabled and grounded)
            answer_result = None
            if self.check_answer_relevance and is_grounded and question:
                answer_result = await self._grade_answer(
                    question=question,
                    generation=generation,
                )

            # Determine final decision
            if not is_grounded:
                decision = GradeDecision.NOT_GROUNDED
            elif answer_result and answer_result.binary_score == "no":
                decision = GradeDecision.NOT_USEFUL
            else:
                decision = GradeDecision.USEFUL

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._total_grades += 1
            if not is_grounded:
                self._hallucinations_detected += 1
            self._avg_latency_ms = (
                (self._avg_latency_ms * (self._total_grades - 1) + latency_ms)
                / self._total_grades
            )

            result = HallucinationGradeResult(
                is_grounded=is_grounded,
                decision=decision,
                hallucination_score=hallucination_result,
                answer_score=answer_result,
                latency_ms=latency_ms,
                model_used=self.model,
                documents_count=docs_count,
                generation_length=len(generation),
            )

            logger.info(
                "hallucination_grade_complete",
                is_grounded=is_grounded,
                decision=decision.value,
                confidence=hallucination_result.confidence,
                latency_ms=round(latency_ms, 2),
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("hallucination_grade_error", error=str(e))

            return HallucinationGradeResult(
                is_grounded=True,  # Default to grounded on error (fail open)
                decision=GradeDecision.ERROR,
                latency_ms=latency_ms,
                model_used=self.model,
                documents_count=docs_count,
                generation_length=len(generation),
                error=str(e),
            )

    async def _grade_hallucination(
        self,
        documents: str,
        generation: str,
    ) -> GradeHallucinations:
        """
        Internal method to grade hallucination using LLM.

        Uses structured output for reliable parsing.
        """
        from resync.core.utils.llm import call_llm

        # Build prompt
        user_message = HALLUCINATION_GRADER_USER_TEMPLATE.format(
            documents=documents,
            generation=generation,
        )

        full_prompt = f"""SYSTEM: {HALLUCINATION_GRADER_SYSTEM_PROMPT}

USER: {user_message}

Responda em JSON com o formato:
{{"binary_score": "yes" ou "no", "confidence": 0.0-1.0, "reasoning": "explicação"}}"""

        for attempt in range(self.max_retries + 1):
            try:
                response = await call_llm(
                    prompt=full_prompt,
                    model=self.model,
                    max_tokens=200,
                    temperature=self.temperature,
                )

                # Parse JSON response
                return self._parse_hallucination_response(response)

            except Exception as e:
                if attempt == self.max_retries:
                    raise
                logger.warning(
                    "hallucination_grade_retry",
                    attempt=attempt + 1,
                    error=str(e),
                )

        # Should not reach here
        return GradeHallucinations(
            binary_score="yes",
            confidence=0.0,
            reasoning="Grading failed, defaulting to grounded",
        )

    async def _grade_answer(
        self,
        question: str,
        generation: str,
    ) -> GradeAnswer:
        """
        Internal method to grade if answer addresses the question.
        """
        from resync.core.utils.llm import call_llm

        user_message = ANSWER_GRADER_USER_TEMPLATE.format(
            question=question,
            generation=generation,
        )

        full_prompt = f"""SYSTEM: {ANSWER_GRADER_SYSTEM_PROMPT}

USER: {user_message}

Responda em JSON com o formato:
{{"binary_score": "yes" ou "no", "reasoning": "explicação"}}"""

        try:
            response = await call_llm(
                prompt=full_prompt,
                model=self.model,
                max_tokens=150,
                temperature=self.temperature,
            )

            return self._parse_answer_response(response)

        except Exception as e:
            logger.error("answer_grade_error", error=str(e))
            return GradeAnswer(
                binary_score="yes",
                reasoning=f"Grading failed: {str(e)}",
            )

    def _parse_hallucination_response(self, response: str) -> GradeHallucinations:
        """Parse LLM response into GradeHallucinations model."""
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Normalize binary_score
            score = data.get("binary_score", "yes").lower().strip()
            if score not in ("yes", "no"):
                score = "yes" if "yes" in score or "grounded" in score.lower() else "no"

            return GradeHallucinations(
                binary_score=score,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.warning("hallucination_parse_fallback", error=str(e))

            # Fallback: look for keywords
            response_lower = response.lower()
            if "no" in response_lower and "hallucin" in response_lower:
                return GradeHallucinations(
                    binary_score="no",
                    confidence=0.6,
                    reasoning="Parsed from text: hallucination detected",
                )
            if "yes" in response_lower or "grounded" in response_lower:
                return GradeHallucinations(
                    binary_score="yes",
                    confidence=0.6,
                    reasoning="Parsed from text: appears grounded",
                )

            # Default to grounded (fail open)
            return GradeHallucinations(
                binary_score="yes",
                confidence=0.5,
                reasoning="Could not parse response, defaulting to grounded",
            )

    def _parse_answer_response(self, response: str) -> GradeAnswer:
        """Parse LLM response into GradeAnswer model."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            score = data.get("binary_score", "yes").lower().strip()
            if score not in ("yes", "no"):
                score = "yes" if "yes" in score else "no"

            return GradeAnswer(
                binary_score=score,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            # Fallback
            response_lower = response.lower()
            if "no" in response_lower and ("address" in response_lower or "answer" in response_lower):
                return GradeAnswer(
                    binary_score="no",
                    reasoning="Parsed from text: doesn't address question",
                )
            return GradeAnswer(
                binary_score="yes",
                reasoning="Could not parse response, defaulting to addresses question",
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content."""
        # Try to find JSON block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        return text

    def get_metrics(self) -> dict[str, Any]:
        """Get grading metrics for monitoring."""
        return {
            "total_grades": self._total_grades,
            "hallucinations_detected": self._hallucinations_detected,
            "hallucination_rate": (
                self._hallucinations_detected / self._total_grades
                if self._total_grades > 0
                else 0.0
            ),
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "model": self.model,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


# Module-level grader instance (lazy initialization)
_default_grader: HallucinationGrader | None = None


def get_hallucination_grader() -> HallucinationGrader:
    """Get or create the default hallucination grader instance."""
    global _default_grader
    if _default_grader is None:
        _default_grader = HallucinationGrader()
    return _default_grader


async def grade_hallucination(
    documents: list[str] | str,
    generation: str,
    question: str | None = None,
) -> HallucinationGradeResult:
    """
    Convenience function to grade hallucination.

    Args:
        documents: Retrieved documents/facts
        generation: LLM-generated response to check
        question: Original user question (optional)

    Returns:
        HallucinationGradeResult

    Example:
        result = await grade_hallucination(
            documents=["Job BATCH_001 failed with RC=12"],
            generation="The job BATCH_001 failed with return code 12.",
            question="What happened to BATCH_001?"
        )
        print(f"Grounded: {result.is_grounded}")
    """
    grader = get_hallucination_grader()
    return await grader.grade(
        documents=documents,
        generation=generation,
        question=question,
    )


async def is_response_grounded(
    documents: list[str] | str,
    generation: str,
) -> bool:
    """
    Simple boolean check if response is grounded.

    Args:
        documents: Retrieved documents/facts
        generation: LLM-generated response

    Returns:
        True if grounded, False if hallucination detected
    """
    result = await grade_hallucination(documents, generation)
    return result.is_grounded


# =============================================================================
# LANGGRAPH NODE
# =============================================================================


async def hallucination_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node for hallucination checking.

    Integrates with the agent graph to validate responses before
    returning them to the user.

    Expected state keys:
    - response: The LLM-generated response to check
    - tool_output: Retrieved documents/context (optional)
    - message: Original user message/question

    Adds to state:
    - hallucination_check: GradeResult dictionary
    - is_grounded: Boolean flag
    - hallucination_decision: GradeDecision value
    """
    logger.debug("hallucination_check_node_start")

    response = state.get("response", "")
    documents = state.get("tool_output") or state.get("raw_data", {})
    question = state.get("message", "")

    # If no response or no documents, skip check
    if not response or not documents:
        logger.debug("hallucination_check_skip", reason="no response or documents")
        state["hallucination_check"] = None
        state["is_grounded"] = True
        state["hallucination_decision"] = GradeDecision.USEFUL.value
        return state

    # Convert documents to string if needed
    if isinstance(documents, dict):
        docs_str = json.dumps(documents, ensure_ascii=False)
    elif isinstance(documents, list):
        docs_str = "\n\n".join(str(d) for d in documents)
    else:
        docs_str = str(documents)

    # Grade the response
    result = await grade_hallucination(
        documents=docs_str,
        generation=response,
        question=question,
    )

    # Update state
    state["hallucination_check"] = result.to_dict()
    state["is_grounded"] = result.is_grounded
    state["hallucination_decision"] = result.decision.value

    # If hallucination detected, add warning to metadata
    if not result.is_grounded:
        metadata = state.get("metadata", {})
        metadata["hallucination_warning"] = True
        metadata["hallucination_reasoning"] = (
            result.hallucination_score.reasoning
            if result.hallucination_score
            else "Unknown"
        )
        state["metadata"] = metadata

        logger.warning(
            "hallucination_detected",
            decision=result.decision.value,
            confidence=result.hallucination_score.confidence if result.hallucination_score else 0,
        )

    return state


def get_hallucination_route(state: dict[str, Any]) -> str:
    """
    Routing function for conditional edge after hallucination check.

    Returns:
        - "regenerate" if hallucination detected (for retry)
        - "output" if grounded (proceed to response)
    """
    decision = state.get("hallucination_decision", GradeDecision.USEFUL.value)

    if decision == GradeDecision.NOT_GROUNDED.value:
        retry_count = state.get("hallucination_retry_count", 0)
        max_retries = state.get("max_hallucination_retries", 2)

        if retry_count < max_retries:
            return "regenerate"

    return "output"


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Classes
    "HallucinationGrader",
    "GradeHallucinations",
    "GradeAnswer",
    "GradeDecision",
    "HallucinationGradeResult",
    # Functions
    "grade_hallucination",
    "is_response_grounded",
    "get_hallucination_grader",
    # LangGraph
    "hallucination_check_node",
    "get_hallucination_route",
]
