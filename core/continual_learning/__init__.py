"""
Continual Learning Module for Resync.

This module provides continuous improvement capabilities:

1. **Feedback Store** - Tracks user feedback on RAG responses
2. **Feedback-Aware Retriever** - Reranks results based on feedback history
3. **Audit-to-KG Pipeline** - Converts audit errors into knowledge graph entries
4. **Context Enrichment** - Enhances queries with learned context
5. **Active Learning** - Identifies uncertain responses for human review
6. **Knowledge Incorporator** - Transforms approved feedback into searchable knowledge (v5.2.3.20)

Usage:
    from resync.core.continual_learning import (
        get_feedback_store,
        create_feedback_aware_retriever,
        get_audit_to_kg_pipeline,
        get_context_enricher,
        get_active_learning_manager,
        get_knowledge_incorporator,  # v5.2.3.20
    )

Architecture:

    Query → Context Enrichment → RAG (with Feedback Reranking) → Response
                                        ↓
                              Active Learning Check
                                        ↓
                              [If uncertain → Human Review Queue]
                                        ↓
                              Feedback Recording → Feedback Store
                                        ↓
                              [If approved → Knowledge Incorporator → Vector DB]
                                        ↓
                              [If error detected → Audit-to-KG Pipeline]
                                        ↓
                              Knowledge Graph Updates
"""

from resync.core.continual_learning.active_learning import (
    ActiveLearningDecision,
    ActiveLearningManager,
    ReviewItem,
    ReviewReason,
    ReviewStatus,
    check_for_review,
    get_active_learning_manager,
)
from resync.core.continual_learning.audit_to_kg_pipeline import (
    AuditResult,
    AuditToKGPipeline,
    ErrorRelationType,
    ErrorTriplet,
    get_audit_to_kg_pipeline,
    process_audit_finding,
)
from resync.core.continual_learning.context_enrichment import (
    ContextEnricher,
    EnrichmentResult,
    EntityContext,
    enrich_query,
    get_context_enricher,
)
from resync.core.continual_learning.feedback_retriever import (
    FeedbackAwareRetriever,
    create_feedback_aware_retriever,
)
from resync.core.continual_learning.feedback_store import (
    FeedbackRating,
    FeedbackStore,
    get_feedback_store,
)
from resync.core.continual_learning.knowledge_incorporator import (
    KnowledgeIncorporator,
    get_knowledge_incorporator,
    incorporate_feedback,
)
from resync.core.continual_learning.orchestrator import (
    ContinualLearningOrchestrator,
    ContinualLearningResult,
    get_continual_learning_orchestrator,
    process_with_continual_learning,
)
from resync.core.continual_learning.threshold_tuning import (
    AuditLogEntry,
    AutoTuningMode,
    ThresholdBounds,
    ThresholdConfig,
    ThresholdMetrics,
    ThresholdRecommendation,
    ThresholdTuningManager,
    get_threshold_tuning_manager,
)

__all__ = [
    # Feedback Store
    "FeedbackRating",
    "FeedbackStore",
    "get_feedback_store",
    # Knowledge Incorporator (v5.2.3.20)
    "KnowledgeIncorporator",
    "get_knowledge_incorporator",
    "incorporate_feedback",
    # Feedback-Aware Retriever
    "FeedbackAwareRetriever",
    "create_feedback_aware_retriever",
    # Audit-to-KG Pipeline
    "AuditToKGPipeline",
    "AuditResult",
    "ErrorTriplet",
    "ErrorRelationType",
    "get_audit_to_kg_pipeline",
    "process_audit_finding",
    # Context Enrichment
    "ContextEnricher",
    "EnrichmentResult",
    "EntityContext",
    "get_context_enricher",
    "enrich_query",
    # Active Learning
    "ActiveLearningManager",
    "ActiveLearningDecision",
    "ReviewItem",
    "ReviewReason",
    "ReviewStatus",
    "get_active_learning_manager",
    "check_for_review",
    # Orchestrator
    "ContinualLearningOrchestrator",
    "ContinualLearningResult",
    "get_continual_learning_orchestrator",
    "process_with_continual_learning",
    # Threshold Tuning
    "ThresholdTuningManager",
    "AutoTuningMode",
    "ThresholdConfig",
    "ThresholdBounds",
    "ThresholdMetrics",
    "ThresholdRecommendation",
    "AuditLogEntry",
    "get_threshold_tuning_manager",
]
