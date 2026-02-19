# resync/core/ia_auditor.py
import asyncio
from typing import Any

import httpx

from resync.core.audit_lock import DistributedAuditLock
from resync.core.audit_queue import AsyncAuditQueue
from resync.core.constants import (
    AUDIT_DELETION_CONFIDENCE_THRESHOLD,
    AUDIT_FLAGGING_CONFIDENCE_THRESHOLD,
    AUDIT_HIGH_RATING_THRESHOLD,
    RECENT_MEMORIES_FETCH_LIMIT,
)
from resync.core.exceptions import (
    AuditError,
    DatabaseError,
    KnowledgeGraphError,
    LLMError,
    ParsingError,
)
from resync.core.structured_logger import get_logger
from resync.core.utils.json_parser import parse_llm_json_response
from resync.core.utils.llm import call_llm
from resync.settings import settings

logger = get_logger(__name__)

# Lazy initialization of knowledge graph (Context Store with SQLite)
_knowledge_graph = None


def _get_knowledge_graph():
    global _knowledge_graph
    if _knowledge_graph is None:
        from resync.core.context_store import ContextStore

        _knowledge_graph = ContextStore()
    return _knowledge_graph


audit_lock = DistributedAuditLock()
audit_queue = AsyncAuditQueue()

# Concurrency control for LLM analysis to prevent rate limiting
# Default to 5 concurrent analyses
_analysis_semaphore: asyncio.Semaphore | None = None


def _get_analysis_semaphore() -> asyncio.Semaphore:
    """Get or create the analysis semaphore."""
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(5)
    return _analysis_semaphore


async def _validate_memory_for_analysis(mem: dict[str, Any]) -> bool:
    """Checks if a memory is valid for analysis."""
    memory_id = str(mem.get("id", ""))
    if await _get_knowledge_graph().is_memory_already_processed(memory_id):
        logger.debug("memory_already_processed", memory_id=memory_id)
        return False

    rating = mem.get("rating")
    if (
        rating is not None
        and isinstance(rating, (int, float))
        and rating >= AUDIT_HIGH_RATING_THRESHOLD
    ):
        logger.debug("memory_has_high_rating", memory_id=memory_id, rating=rating)
        return False

    if not mem.get("user_query") or not mem.get("agent_response"):
        logger.debug("memory_missing_required_fields", memory_id=memory_id)
        return False

    # Skip if memory is already approved by human
    if await _get_knowledge_graph().is_memory_approved(memory_id):
        logger.debug("memory_already_approved_by_human", memory_id=memory_id)
        return False

    return True


async def _get_llm_analysis(user_query: str, agent_response: str) -> dict[str, Any] | None:
    """Gets the analysis of a memory from the LLM."""
    prompt = f"""
    You are an expert TWS (IBM MQ/Workload Scheduler) auditor.
    Evaluate if the agent's response is correct for the user's query.

    Query: "{user_query}"
    Response: "{agent_response}"

    Consider:
    - Technical errors? (e.g., suggesting /tmp cleanup for a permission error)
    - Irrelevant response?
    - Contradictory information?

    Return ONLY a JSON object in the format:
    {{ "is_incorrect": true/false, "confidence": 0.0-1.0, "reason": "string" }}
    """
    try:
        # Increased max_tokens from 200 to 500 to allow for more detailed analysis
        # and comprehensive reasoning in the auditor's evaluation of agent responses
        result = await call_llm(prompt, model=settings.AUDITOR_MODEL_NAME, max_tokens=500)
        if not result:
            return None

        return parse_llm_json_response(
            result,
            required_keys=["is_incorrect", "confidence", "reason"],
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error("llm_network_error", error=str(e), exc_info=True)
        raise LLMError("Network error during memory audit analysis") from e
    except ParsingError as e:
        logger.error("llm_json_parsing_failed", error=str(e), exc_info=True)
        # Return None to indicate a non-critical failure for this memory
        return None
    except Exception as e:
        logger.critical("unexpected_error_in_llm_analysis", error=str(e), exc_info=True)
        # Encapsulate unexpected errors in a domain-specific one
        raise LLMError("Failed to get LLM analysis for memory audit") from e


async def _perform_action_on_memory(
    mem: dict[str, Any], analysis: dict[str, Any]
) -> tuple[str, str | dict[str, Any]] | None:
    """Performs the appropriate action on a memory based on the LLM analysis."""
    memory_id = str(mem.get("id", ""))
    confidence = float(analysis.get("confidence", 0))

    if analysis.get("is_incorrect") and confidence > AUDIT_DELETION_CONFIDENCE_THRESHOLD:
        logger.info(
            "deleting_memory",
            memory_id=memory_id,
            confidence=confidence,
            reason=analysis.get("reason", "N/A"),
        )

        # === CONTINUAL LEARNING INTEGRATION ===
        # Convert audit finding to knowledge graph entries
        try:
            from resync.core.continual_learning import process_audit_finding

            await process_audit_finding(
                memory_id=memory_id,
                user_query=str(mem.get("user_query", "")),
                agent_response=str(mem.get("agent_response", "")),
                is_incorrect=True,
                confidence=confidence,
                reason=str(analysis.get("reason", "N/A")),
            )
        except ImportError:
            logger.debug("continual_learning_module_not_available")
        except Exception as e:
            logger.warning("continual_learning_pipeline_error", error=str(e))
        # === END CONTINUAL LEARNING INTEGRATION ===

        success = await _get_knowledge_graph().atomic_check_and_delete(memory_id)
        return ("delete", memory_id) if success else None

    if analysis.get("is_incorrect") and confidence > AUDIT_FLAGGING_CONFIDENCE_THRESHOLD:
        reason = str(analysis.get("reason", "N/A"))
        logger.warning("flagging_memory", memory_id=memory_id, confidence=confidence, reason=reason)

        # === CONTINUAL LEARNING INTEGRATION ===
        # Convert audit finding to knowledge graph entries (for flagged items too)
        try:
            from resync.core.continual_learning import process_audit_finding

            await process_audit_finding(
                memory_id=memory_id,
                user_query=str(mem.get("user_query", "")),
                agent_response=str(mem.get("agent_response", "")),
                is_incorrect=True,
                confidence=confidence,
                reason=reason,
            )
        except ImportError:
            logger.debug("continual_learning_module_not_available")
        except Exception as e:
            logger.warning("continual_learning_pipeline_error", error=str(e))
        # === END CONTINUAL LEARNING INTEGRATION ===

        success = await _get_knowledge_graph().atomic_check_and_flag(memory_id, reason, confidence)
        if success:
            mem["ia_audit_reason"] = reason
            mem["ia_audit_confidence"] = confidence
            return "flag", mem

    return None


async def _fetch_recent_memories() -> list[dict[str, Any]]:
    """Fetches recent memories from knowledge graph."""
    try:
        memories = await _get_knowledge_graph().get_memories(limit=RECENT_MEMORIES_FETCH_LIMIT)
        return memories or []
    except KnowledgeGraphError as e:
        logger.error("failed_to_fetch_memories", error=str(e), exc_info=True)
        raise AuditError("Failed to fetch memories for analysis") from e


async def _process_memory_batch(
    memories: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Processes a batch of memories through concurrent analysis.

    Returns:
        Tuple of (deleted_memory_ids, flagged_memories)
    """
    tasks = [analyze_memory(mem) for mem in memories]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log exceptions but don't fail the batch
    for result in results:
        if isinstance(result, Exception):
            logger.warning("memory_analysis_failed", error=str(result))

    # Extract valid results and categorize by action
    valid_results = [
        (action, data)
        for r in results
        if r and not isinstance(r, Exception)
        for action, data in [r]
    ]

    deleted = [data for action, data in valid_results if action == "delete"]
    flagged = [data for action, data in valid_results if action == "flag"]

    return deleted, flagged


async def _store_flagged_memories(flagged: list[dict[str, Any]]) -> None:
    """Stores flagged memories in audit queue for human review."""
    for mem in flagged:
        try:
            # Use generic enqueue since enqueue_for_review is not available in AsyncAuditQueue
            await audit_queue.enqueue(
                action="review_memory",
                payload={
                    "memory_id": str(mem["id"]),
                    "reason": mem.get("ia_audit_reason", "Flagged by IA"),
                    "confidence": float(mem.get("ia_audit_confidence", 0.0)),
                    "memory_content": mem,
                },
                priority=1,  # High priority for flagged items
            )
        except (DatabaseError, AttributeError) as e:
            logger.error(
                "failed_to_enqueue_flagged_memory",
                memory_id=str(mem.get("id")),
                error=str(e),
                exc_info=True,
            )


async def analyze_memory(
    mem: dict[str, Any],
) -> tuple[str, str | dict[str, Any]] | None:
    """Analyzes a single memory and returns an action if necessary."""
    memory_id = str(mem.get("id", ""))
    
    # Use semaphore to limit concurrent analysis and prevent rate limiting
    async with _get_analysis_semaphore():
        try:
            async with await audit_lock.acquire(memory_id, timeout=30):
                if not await _validate_memory_for_analysis(mem):
                    return None

                # Check if memory is already flagged or approved before LLM analysis
                if await _get_knowledge_graph().is_memory_flagged(memory_id):
                    logger.debug("memory_already_flagged_by_ia", memory_id=memory_id)
                    return None

                if await _get_knowledge_graph().is_memory_approved(memory_id):
                    logger.debug("memory_already_approved_by_human", memory_id=memory_id)
                    return None

                analysis = await _get_llm_analysis(
                    str(mem.get("user_query", "")), str(mem.get("agent_response", ""))
                )
                if not analysis:
                    return None

                return await _perform_action_on_memory(mem, analysis)

        except LLMError:
            # Error already logged in _get_llm_analysis, just bubble it up if needed
            logger.warning("skipping_memory_due_to_llm_failure", memory_id=memory_id)
            return None
        except AuditError as e:
            # Lock acquisition failure
            logger.warning("could_not_acquire_lock_for_memory", memory_id=memory_id, error=str(e))
            return None
        except (KnowledgeGraphError, DatabaseError) as e:
            # Catch specific data access errors
            logger.error(
                "database_or_knowledge_graph_error_analyzing_memory",
                memory_id=memory_id,
                error=str(e),
                exc_info=True,
            )
            return None
        except Exception as e:
            # Catch unexpected errors to prevent crashing the batch
            logger.error(
                "unexpected_error_analyzing_memory",
                memory_id=memory_id,
                error=str(e),
                exc_info=True,
            )
            return None


async def _cleanup_locks() -> None:
    """Safely cleans up expired audit locks."""
    try:
        await audit_lock.cleanup_expired_locks(max_age=60)
    except Exception as _e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(_e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.warning("error_cleaning_up_expired_locks", exc_info=True)


# Old functions removed - replaced by refactored versions above


async def analyze_and_flag_memories() -> dict[str, int | str]:
    """
    Orchestrates the memory analysis workflow.

    Fetches recent memories, processes them concurrently, and handles
    the results by deleting incorrect memories and queuing flagged ones
    for human review.

    Returns:
        Dictionary with counts of deleted and flagged memories
    """
    await _cleanup_locks()

    memories = await _fetch_recent_memories()
    if not memories:
        logger.info("no_memories_to_analyze")
        return {"deleted": 0, "flagged": 0}

    deleted, flagged = await _process_memory_batch(memories)

    if flagged:
        await _store_flagged_memories(flagged)

    logger.info(
        "memory_analysis_completed",
        deleted_count=len(deleted),
        flagged_count=len(flagged),
    )

    return {"deleted": len(deleted), "flagged": len(flagged)}
