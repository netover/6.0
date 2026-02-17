"""
Memory Integration for LangGraph Agent v5.2.3.26

Integrates Long-term Memory with the LangGraph agent workflow.

This module provides:
1. Context assembly with user memories
2. Session-end memory extraction hooks
3. Memory-aware prompt construction

Usage in agent_graph.py:
    from resync.core.memory.integration import (
        assemble_memory_context,
        extract_session_memories,
    )

    # Before processing query
    memory_context = await assemble_memory_context(user_id, query)
    full_prompt = f"{memory_context}\n\n{base_prompt}"

    # After session ends
    await extract_session_memories(user_id, conversation, session_id)

Author: Resync Team
Version: 5.2.3.26
"""

from __future__ import annotations

import logging
from typing import Any

from resync.core.memory.conversation_memory import (
    get_conversation_memory,
)
from resync.core.memory.long_term_memory import (
    get_long_term_memory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT ASSEMBLY
# =============================================================================


async def assemble_memory_context(
    user_id: str,
    query: str | None = None,
    session_id: str | None = None,
    include_conversation: bool = True,
    include_long_term: bool = True,
    max_conversation_turns: int = 5,
    max_long_term_memories: int = 10,
) -> str:
    """
    Assemble complete memory context for a query.

    Combines:
    1. Conversation history (short-term)
    2. Long-term memories (facts + patterns)

    Args:
        user_id: User identifier
        query: Current query for reactive memory retrieval
        session_id: Current session ID
        include_conversation: Include short-term conversation history
        include_long_term: Include long-term memories
        max_conversation_turns: Max conversation turns to include
        max_long_term_memories: Max long-term memories to include

    Returns:
        Formatted context string for LLM prompt injection
    """
    sections = []

    # 1. Long-term memories (user facts and patterns)
    if include_long_term:
        ltm = get_long_term_memory()
        ltm_context = await ltm.get_memory_context(
            user_id=user_id,
            query=query,
            max_memories=max_long_term_memories,
        )
        if ltm_context:
            sections.append(ltm_context)

    # 2. Conversation history (recent turns)
    if include_conversation and session_id:
        conv_memory = get_conversation_memory()
        session = await conv_memory.get_or_create_session(session_id)
        conv_context = session.get_context_for_prompt(max_conversation_turns)
        if conv_context:
            sections.append(conv_context)

    if not sections:
        return ""

    return "\n\n".join(sections)


async def assemble_full_prompt(
    user_id: str,
    query: str,
    base_system_prompt: str,
    session_id: str | None = None,
) -> str:
    """
    Assemble a complete prompt with memory context.

    Args:
        user_id: User identifier
        query: Current user query
        base_system_prompt: Base system prompt
        session_id: Optional session ID for conversation context

    Returns:
        Complete prompt with memory injected
    """
    memory_context = await assemble_memory_context(
        user_id=user_id,
        query=query,
        session_id=session_id,
    )

    if memory_context:
        return f"{base_system_prompt}\n\n{memory_context}\n\nUSER: {query}"
    return f"{base_system_prompt}\n\nUSER: {query}"


# =============================================================================
# SESSION LIFECYCLE HOOKS
# =============================================================================


async def extract_session_memories(
    user_id: str,
    conversation: list[dict[str, str]],
    session_id: str,
    min_turns: int = 3,
) -> int:
    """
    Extract long-term memories from a completed session.

    Should be called when a session ends or after significant interactions.

    Args:
        user_id: User identifier
        conversation: List of messages [{"role": "user"|"assistant", "content": "..."}]
        session_id: Session ID for provenance
        min_turns: Minimum turns required for extraction (skip trivial sessions)

    Returns:
        Number of memories extracted
    """
    # Skip trivial sessions
    if len(conversation) < min_turns * 2:  # user + assistant = 2 messages per turn
        logger.debug("Skipping memory extraction for short session %s", session_id)
        return 0

    ltm = get_long_term_memory()

    try:
        memories = await ltm.extract_from_session(
            user_id=user_id,
            conversation=conversation,
            session_id=session_id,
        )
        return len(memories)

    except Exception as e:
        logger.error("Memory extraction failed for session %s: %s", session_id, e)
        return 0


async def on_session_end(
    user_id: str,
    session_id: str,
) -> int:
    """
    Hook to call when a session ends.

    Retrieves conversation from ConversationMemory and extracts long-term memories.

    Args:
        user_id: User identifier
        session_id: Session ID

    Returns:
        Number of memories extracted
    """
    conv_memory = get_conversation_memory()
    session = await conv_memory.get_or_create_session(session_id)

    # Convert to list format
    conversation = [
        {"role": msg.role, "content": msg.content}
        for msg in session.messages
    ]

    return await extract_session_memories(
        user_id=user_id,
        conversation=conversation,
        session_id=session_id,
    )


# =============================================================================
# AGENT STATE INTEGRATION
# =============================================================================


async def enrich_agent_state(
    state: dict[str, Any],
    include_long_term: bool = True,
) -> dict[str, Any]:
    """
    Enrich agent state with memory context.

    Use this in the router node to add memory context to the state.

    Args:
        state: Current agent state
        include_long_term: Whether to include long-term memories

    Returns:
        Enriched state with memory_context field
    """
    user_id = state.get("user_id")
    query = state.get("message", "")
    session_id = state.get("session_id")

    if not user_id:
        return state

    memory_context = await assemble_memory_context(
        user_id=user_id,
        query=query,
        session_id=session_id,
        include_long_term=include_long_term,
    )

    state["memory_context"] = memory_context

    # Also update LLM messages if present
    if memory_context and "llm_messages" in state:
        # Inject memory as system context
        system_msg = {
            "role": "system",
            "content": f"Context about this user:\n{memory_context}",
        }
        state["llm_messages"].insert(0, system_msg)

    return state


async def save_turn_to_memory(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """
    Save a conversation turn to both short-term and potentially long-term memory.

    Args:
        user_id: User identifier
        session_id: Session ID
        user_message: User's message
        assistant_response: Assistant's response
    """
    # 1. Save to conversation memory (short-term)
    conv_memory = get_conversation_memory()
    await conv_memory.add_turn(
        session_id=session_id,
        user_message=user_message,
        assistant_response=assistant_response,
    )

    # 2. Periodic long-term extraction (every 5 turns)
    session = await conv_memory.get_or_create_session(session_id)
    if session.turn_count % 5 == 0 and session.turn_count >= 5:
        # Background extraction (don't block response)
        try:
            conversation = [
                {"role": msg.role, "content": msg.content}
                for msg in session.messages
            ]
            await extract_session_memories(user_id, conversation, session_id)
        except Exception as e:
            logger.warning("Background memory extraction failed: %s", e)


# =============================================================================
# USER MEMORY MANAGEMENT
# =============================================================================


async def get_user_memory_summary(user_id: str) -> dict[str, Any]:
    """
    Get a summary of what we know about a user.

    Useful for debugging and user transparency.

    Args:
        user_id: User identifier

    Returns:
        Summary dict with memory statistics and samples
    """
    ltm = get_long_term_memory()
    stats = await ltm.get_statistics(user_id)

    # Get sample memories
    store = await ltm._ensure_store()
    all_memories = await store.get_user_memories(user_id)

    samples = []
    for mem in all_memories[:5]:
        samples.append({
            "id": mem.id,
            "type": "declarative" if hasattr(mem, "content") else "procedural",
            "preview": getattr(mem, "content", None) or getattr(mem, "pattern", ""),
            "confidence": mem.effective_confidence,
        })

    return {
        **stats,
        "samples": samples,
    }


async def delete_user_memories(user_id: str) -> dict[str, Any]:
    """
    Delete all long-term memories for a user (GDPR compliance).

    Args:
        user_id: User identifier

    Returns:
        Deletion result
    """
    ltm = get_long_term_memory()
    count = await ltm.delete_user_memories(user_id)

    return {
        "user_id": user_id,
        "memories_deleted": count,
        "status": "success",
    }


async def confirm_memory(user_id: str, memory_id: str) -> bool:
    """
    User confirms a memory is correct.

    Args:
        user_id: User identifier
        memory_id: Memory ID to confirm

    Returns:
        Success status
    """
    ltm = get_long_term_memory()
    return await ltm.confirm_memory(memory_id)


async def reject_memory(user_id: str, memory_id: str) -> bool:
    """
    User says a memory is incorrect.

    Args:
        user_id: User identifier
        memory_id: Memory ID to reject

    Returns:
        Success status
    """
    ltm = get_long_term_memory()
    return await ltm.contradict_memory(memory_id)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Context assembly
    "assemble_memory_context",
    "assemble_full_prompt",
    # Session hooks
    "extract_session_memories",
    "on_session_end",
    # Agent integration
    "enrich_agent_state",
    "save_turn_to_memory",
    # User management
    "get_user_memory_summary",
    "delete_user_memories",
    "confirm_memory",
    "reject_memory",
]
