"""
Chat routes for FastAPI with RAG integration and Hybrid Agent routing.

v5.4.1 Enhancements (PR-4):
- HybridRouter with RAG-only, Agentic, and Diagnostic modes
- Tool execution with guardrails
- HITL support for write operations
- Improved intent classification with routing suggestions

v5.4.0 Enhancements:
- Conversational memory for multi-turn dialogues
- Hybrid retrieval (BM25 + Vector) for better TWS job search
- Anaphora resolution ("restart it" -> "restart job AWSBH001")

This module provides the chat API endpoints using the HybridRouter system
which automatically routes messages to the appropriate handler based on
intent classification and complexity analysis.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
import secrets

# v5.4.1: Import HybridRouter (fallback to UnifiedAgent for compatibility)
try:
    from resync.core.agent_router import HybridRouter, RoutingMode

    # v5.7.1 FIX: Import provider to get router with AgentManager
    from resync.core.wiring import get_hybrid_router as get_hybrid_router_provider

    _use_hybrid_router = True
except ImportError:
    from resync.core.agent_manager import unified_agent

    _use_hybrid_router = False

# Import RAG components
# v5.4.0: Import memory system
from resync.api.dependencies_v2 import get_logger, get_current_user
from resync.api.models.requests import ChatHistoryQuery, ChatMessageRequest
from resync.api.models.responses_v2 import ChatMessageResponse
from resync.core.memory import ConversationContext, get_conversation_memory
from resync.knowledge.ingestion.embedding_service import EmbeddingService
from resync.knowledge.ingestion.ingest import IngestService
from resync.knowledge.retrieval.retriever import RagRetriever
from resync.knowledge.store.pgvector_store import get_vector_store

router = APIRouter()
logger = None  # Will be injected by dependency

# v5.4.1: HybridRouter instance (singleton)
_hybrid_router: HybridRouter | None = None


class RagComponentsManager:
    """Singleton manager for RAG components to avoid global state issues."""
    
    _instance: "RagComponentsManager | None" = None
    _initialized: bool = False
    _embedding_service = None
    _vector_store = None
    _retriever = None
    _ingest_service = None
    _hybrid_retriever = None
    
    def __new__(cls) -> "RagComponentsManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_components(self):
        """Lazy initialization of RAG components within async context"""
        if self._initialized:
            return self._embedding_service, self._vector_store, self._retriever, self._ingest_service
        
        # v5.7.1 FIX: Thread-safe initialization
        if not hasattr(self, "_lock"):
            import asyncio
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._initialized:
                return self._embedding_service, self._vector_store, self._retriever, self._ingest_service

            try:
                self._embedding_service = EmbeddingService()
                self._vector_store = get_vector_store()
                self._retriever = RagRetriever(self._embedding_service, self._vector_store)
                self._ingest_service = IngestService(self._embedding_service, self._vector_store)

                try:
                    from resync.knowledge.retrieval.hybrid_retriever import HybridRetriever

                    self._hybrid_retriever = HybridRetriever(self._embedding_service, self._vector_store)
                    if logger:
                        logger.info("Hybrid retriever initialized (BM25 + Vector)")
                except Exception as e:
                    if logger:
                        logger.warning("Hybrid retriever not available, using standard: %s", e)

                self._initialized = True
                if logger:
                    logger.info("RAG components initialized successfully (lazy)")
            except Exception as e:
                if logger:
                    logger.error("Failed to initialize RAG components: %s", e)
                self._embedding_service = None
                self._vector_store = None
                self._retriever = None
                self._ingest_service = None

        return self._embedding_service, self._vector_store, self._retriever, self._ingest_service


# Singleton instance
_rag_manager = RagComponentsManager()


async def _get_rag_components():
    """Lazy initialization of RAG components within async context"""
    return await _rag_manager.get_components()


async def _get_or_create_session(session_id: str | None) -> ConversationContext:
    """Get or create conversation session for memory."""
    memory = get_conversation_memory()
    return await memory.get_or_create_session(session_id)


async def _save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    metadata: dict | None = None,
) -> None:
    """Save conversation turn to memory."""
    try:
        memory = get_conversation_memory()
        await memory.add_turn(session_id, user_message, assistant_response, metadata)
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        if logger:
            logger.warning("Failed to save conversation turn: %s", e)


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_message(
    request: ChatMessageRequest,
    background_tasks: BackgroundTasks,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    x_routing_mode: str | None = Header(None, alias="X-Routing-Mode"),
    # Temporarily disabled authentication for testing
    current_user: dict | None = Depends(get_current_user),
    hybrid_router=Depends(get_hybrid_router_provider),
    logger_instance=Depends(get_logger),
):
    """
    Send chat message to Resync AI Assistant.

    v5.4.1: Uses HybridRouter for intelligent routing:
    - RAG-only: Quick knowledge base queries (fastest, cheapest)
    - Agentic: Multi-step tasks requiring tools
    - Diagnostic: Complex troubleshooting with HITL

    Pass X-Routing-Mode header to force a specific mode (rag_only, agentic, diagnostic).

    v5.4.0: Now supports multi-turn conversations with memory.
    Pass X-Session-ID header to maintain conversation context.
    """
    from resync.settings import settings
    from resync.settings import settings
    
    # v6.2.1: Implemented secure Limited Access pattern
    # Use 'operator_api_key' if provided in X-Operator-Key header
    x_operator_key = request.headers.get("X-Operator-Key")
    is_operator = False
    
    if x_operator_key and settings.operator_api_key:
        if secrets.compare_digest(x_operator_key, settings.operator_api_key.get_secret_value()):
            is_operator = True

    if settings.is_production and not current_user and not is_operator:
        logger_instance.warning("unauthorized_chat_access_attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


    # v5.7.1 FIX: Removed global logger injection to prevent context contamination
    # logger = logger_instance 

    try:
        # v5.4.0: Get or create conversation session
        session_id = x_session_id or (
            request.metadata.get("session_id") if request.metadata else None
        )
        context = await _get_or_create_session(session_id)

        # v5.4.0: Resolve anaphoric references ("it", "that job")
        memory = get_conversation_memory()
        resolved_message = memory.resolve_reference(context, request.message)

        # v5.4.0: Get conversation history for context
        conversation_context = context.get_context_for_prompt(max_messages=5)

        # v5.4.1: Parse optional routing mode override
        force_mode = None
        if x_routing_mode:
            try:
                force_mode = RoutingMode(x_routing_mode)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid X-Routing-Mode: {x_routing_mode}",
                )

        # v5.4.1: Use HybridRouter if available
        if _use_hybrid_router:
            # Route the message
            result = await hybrid_router.route(
                message=resolved_message,
                context={
                    "tws_instance_id": request.tws_instance_id,
                    "session_id": context.session_id,
                    "conversation_history": conversation_context,
                },
                force_mode=force_mode,
            )

            response_message = result.response

            logger_instance.info(
                "chat_message_processed",
                user_id="test_user",
                session_id=context.session_id,
                routing_mode=result.routing_mode.value,
                intent=result.intent,
                confidence=result.confidence,
                handler=result.handler,
                tools_used=result.tools_used,
                tws_instance_id=request.tws_instance_id,
                message_length=len(request.message),
                response_length=len(response_message),
                processing_time_ms=result.processing_time_ms,
                turn_count=context.turn_count + 1,
            )

            # v5.4.0: Save conversation turn in background
            background_tasks.add_task(
                _save_conversation_turn,
                context.session_id,
                request.message,
                response_message,
                {
                    "routing_mode": result.routing_mode.value,
                    "intent": result.intent,
                    "handler": result.handler,
                    "tools_used": result.tools_used,
                },
            )

            return ChatMessageResponse(
                message=response_message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_id=result.handler,
                is_final=True,
                metadata={
                    "routing_mode": result.routing_mode.value,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "tools_used": result.tools_used,
                    "entities": result.entities,
                    "tws_instance_id": request.tws_instance_id,
                    "session_id": context.session_id,
                    "turn_count": context.turn_count + 1,
                    "requires_approval": result.requires_approval,
                    "approval_id": result.approval_id,
                },
            )

        # Fallback to UnifiedAgent for compatibility
        result = await unified_agent.chat_with_metadata(
            message=resolved_message,
            include_history=True,
            tws_instance_id=request.tws_instance_id,
            extra_context=conversation_context if conversation_context else None,
        )

        response_message = result["response"]

        logger_instance.info(
            "chat_message_processed",
            user_id="test_user",
            session_id=context.session_id,
            intent=result["intent"],
            confidence=result["confidence"],
            handler=result["handler"],
            tools_used=result["tools_used"],
            tws_instance_id=request.tws_instance_id,
            message_length=len(request.message),
            response_length=len(response_message),
            processing_time_ms=result["processing_time_ms"],
            turn_count=context.turn_count + 1,
        )

        # v5.4.0: Save conversation turn in background
        background_tasks.add_task(
            _save_conversation_turn,
            context.session_id,
            request.message,
            response_message,
            {"intent": result["intent"], "handler": result["handler"]},
        )

        return ChatMessageResponse(
            message=response_message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=result["handler"],
            is_final=True,
            metadata={
                "intent": result["intent"],
                "confidence": result["confidence"],
                "tools_used": result["tools_used"],
                "entities": result["entities"],
                "tws_instance_id": request.tws_instance_id,
                "session_id": context.session_id,
                "turn_count": context.turn_count + 1,
            },
        )

    except Exception as e:
        logger_instance.error("chat_message_error", error=str(e), user_id="test_user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message",
        ) from e


@router.post("/chat/analyze", response_model=dict)
async def analyze_message(request: ChatMessageRequest, logger_instance=Depends(get_logger)):
    """
    Analyze a message without processing it.

    Returns the intent classification, confidence score, suggested routing mode,
    and which handler would process the message. Useful for debugging and
    understanding how the router interprets different queries.
    """
    # v5.7.1 FIX: Removed global logger injection
    # logger = logger_instance

    try:
        from resync.core.agent_router import IntentClassifier

        classifier = IntentClassifier()
        classification = classifier.classify(request.message)

        return {
            "message": request.message,
            "primary_intent": classification.primary_intent.value,
            "confidence": classification.confidence,
            "secondary_intents": [i.value for i in classification.secondary_intents],
            "entities": classification.entities,
            "requires_tools": classification.requires_tools,
            "is_high_confidence": classification.is_high_confidence,
            "needs_clarification": classification.needs_clarification,
            # v5.4.1: Add routing suggestion
            "suggested_routing": getattr(classification, "suggested_routing", None),
        }

    except Exception as e:
        logger_instance.error("analyze_message_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to analyze message"
        ) from e


@router.get("/chat/history")
async def chat_history(
    query_params: ChatHistoryQuery = Depends(),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    # Temporarily disabled authentication for testing
    # current_user: dict = Depends(get_current_user)
    logger_instance=Depends(get_logger),
):
    """Get chat history for the current session."""
    # v5.4.1: Try to get from memory first
    if x_session_id:
        try:
            memory = get_conversation_memory()
            context = await memory.get_session(x_session_id)
            if context:
                return {
                    "history": context.messages[-query_params.limit :],
                    "session_id": x_session_id,
                    "total_messages": len(context.messages),
                }
        except Exception as exc:
            logger_instance.debug("suppressed_exception", error=str(exc), exc_info=True)

    # Fallback
    if not _use_hybrid_router:
        history = unified_agent.get_history()
        return {
            "history": history,
            "agent_id": "unified",
            "total_messages": len(history),
        }

    return {
        "history": [],
        "session_id": x_session_id,
        "total_messages": 0,
    }


@router.delete("/chat/history")
async def clear_chat_history(
    query_params: ChatHistoryQuery = Depends(),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    # Temporarily disabled authentication for testing
    current_user: dict | None = Depends(get_current_user),
    logger_instance=Depends(get_logger),
):
    """Clear chat history for the current session."""
    # v5.4.1: Clear from memory if session provided
    if x_session_id:
        try:
            memory = get_conversation_memory()
            await memory.clear_session(x_session_id)
            logger_instance.info("chat_history_cleared", session_id=x_session_id)
            return {"message": "Chat history cleared successfully", "session_id": x_session_id}
        except Exception as exc:
            logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

    # Fallback
    if not _use_hybrid_router:
        unified_agent.clear_history()

    logger_instance.info("chat_history_cleared", user_id="test_user")
    return {"message": "Chat history cleared successfully"}


@router.get("/chat/intents")
async def list_supported_intents():
    """
    List all supported intents and their descriptions.

    v5.4.1: Also shows which routing mode each intent uses.
    """
    from resync.core.agent_router import Intent

    intent_info = {
        Intent.STATUS.value: {
            "description": "Check system, job, or workstation status",
            "routing": "agentic",
        },
        Intent.TROUBLESHOOTING.value: {
            "description": "Diagnose and resolve issues, analyze errors",
            "routing": "diagnostic",
        },
        Intent.JOB_MANAGEMENT.value: {
            "description": "Run, stop, rerun, or schedule jobs",
            "routing": "agentic",
        },
        Intent.MONITORING.value: {
            "description": "Real-time monitoring and alerts",
            "routing": "agentic",
        },
        Intent.ANALYSIS.value: {
            "description": "Deep analysis of patterns and trends",
            "routing": "agentic",
        },
        Intent.REPORTING.value: {
            "description": "Generate reports and summaries",
            "routing": "rag_only",
        },
        Intent.GREETING.value: {
            "description": "Greetings and introductions",
            "routing": "rag_only",
        },
        Intent.GENERAL.value: {
            "description": "General questions and help",
            "routing": "rag_only",
        },
    }

    routing_modes = {
        "rag_only": "Quick knowledge base queries (fastest, cheapest)",
        "agentic": "Multi-step tasks with tool execution",
        "diagnostic": "Complex troubleshooting with HITL checkpoints",
    }

    return {
        "intents": intent_info,
        "routing_modes": routing_modes,
        "total_intents": len(intent_info),
    }
