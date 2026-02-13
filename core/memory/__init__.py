"""
Memory System for Resync v5.2.3.26

Provides multi-turn conversation support with context persistence.

Two types of memory:
1. Conversation Memory (short-term) - Session-based, expires with session
2. Long-term Memory (persistent) - User facts and behavior patterns

Based on Google's Context Engineering principles.
"""

from .conversation_memory import (
    ConversationContext,
    ConversationMemory,
    InMemoryStore,
    MemoryStore,
    Message,
    RedisMemoryStore,
    get_conversation_memory,
)
from .integration import (
    assemble_full_prompt,
    # Context assembly
    assemble_memory_context,
    confirm_memory,
    delete_user_memories,
    # Agent integration
    enrich_agent_state,
    # Session hooks
    extract_session_memories,
    # User management
    get_user_memory_summary,
    on_session_end,
    reject_memory,
    save_turn_to_memory,
)
from .long_term_memory import (
    DeclarativeCategory,
    DeclarativeMemory,
    InMemoryLongTermStore,
    LongTermMemoryManager,
    # Stores
    LongTermMemoryStore,
    # Core classes
    MemoryExtractor,
    # Data classes
    MemoryProvenance,
    # Enums
    MemoryType,
    ProceduralCategory,
    ProceduralMemory,
    RedisLongTermStore,
    RetrievalMode,
    # Singleton
    get_long_term_memory,
)

__all__ = [
    # Conversation Memory (short-term)
    "Message",
    "ConversationContext",
    "MemoryStore",
    "RedisMemoryStore",
    "InMemoryStore",
    "ConversationMemory",
    "get_conversation_memory",
    # Long-term Memory (v5.2.3.26)
    "MemoryType",
    "DeclarativeCategory",
    "ProceduralCategory",
    "RetrievalMode",
    "MemoryProvenance",
    "DeclarativeMemory",
    "ProceduralMemory",
    "LongTermMemoryStore",
    "InMemoryLongTermStore",
    "RedisLongTermStore",
    "MemoryExtractor",
    "LongTermMemoryManager",
    "get_long_term_memory",
    # Integration functions
    "assemble_memory_context",
    "assemble_full_prompt",
    "extract_session_memories",
    "on_session_end",
    "enrich_agent_state",
    "save_turn_to_memory",
    "get_user_memory_summary",
    "delete_user_memories",
    "confirm_memory",
    "reject_memory",
]
