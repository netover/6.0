"""
Knowledge Incorporator - Transforma feedback aprovado em conhecimento.

v5.2.3.20: Implementa o fluxo "Golden Record" para active learning:

Fluxo:
1. Admin aprova feedback com correção
2. Sistema cria documento Q&A formatado
3. Documento é inserido no pgvector com metadados especiais
4. Feedback é marcado como "incorporated"

O documento criado tem prioridade alta no retrieval por ter:
- source_tier: "verified" (máxima credibilidade)
- authority_tier: 1 (máxima autoridade)
- doc_type: "golden_record" (conhecimento validado)

Author: Resync Team
Version: 5.2.3.20
"""

from datetime import datetime, timezone
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class KnowledgeIncorporator:
    """
    Incorpora feedback humano ao knowledge base.

    Transforma feedback aprovado por especialistas em documentos
    de conhecimento que são indexados no vector store com
    máxima prioridade de retrieval.
    """

    def __init__(
        self,
        ingest_service=None,
        embedder=None,
        store=None,
        collection: str = "golden_records",
    ):
        """
        Initialize KnowledgeIncorporator.

        Args:
            ingest_service: IngestService instance (optional, will create if needed)
            embedder: Embedder instance (optional, will create if needed)
            store: VectorStore instance (optional, will create if needed)
            collection: Collection name for golden records
        """
        self._ingest_service = ingest_service
        self._embedder = embedder
        self._store = store
        self._collection = collection
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of dependencies."""
        if self._initialized:
            return

        if self._ingest_service is None:
            from resync.knowledge.ingestion.embedding_service import (
                create_embedding_service,
            )
            from resync.knowledge.ingestion.ingest import IngestService
            from resync.knowledge.store import get_vector_store

            self._embedder = self._embedder or create_embedding_service()
            self._store = self._store or get_vector_store()
            self._ingest_service = IngestService(
                embedder=self._embedder,
                store=self._store,
            )

        self._initialized = True
        logger.info("knowledge_incorporator_initialized", collection=self._collection)

    async def incorporate_feedback(
        self,
        feedback_id: int,
        original_question: str,
        user_correction: str,
        metadata: dict[str, Any] | None = None,
        reviewer_id: str | None = None,
    ) -> str:
        """
        Transforma feedback aprovado em documento de conhecimento.

        Args:
            feedback_id: ID do feedback no banco
            original_question: Pergunta original do usuário
            user_correction: Resposta correta (do especialista)
            metadata: Metadados adicionais
            reviewer_id: ID do revisor que aprovou

        Returns:
            doc_id: ID do documento criado no vector store
        """
        self._ensure_initialized()

        # Gerar ID único para o documento
        doc_id = f"golden_record_{feedback_id}"

        # Formato do documento "Golden Record"
        # Estruturado para maximizar relevância no retrieval
        doc_content = self._format_golden_record(
            original_question=original_question,
            user_correction=user_correction,
            feedback_id=feedback_id,
            reviewer_id=reviewer_id,
        )

        # Timestamp ISO
        ts_iso = datetime.now(timezone.utc).isoformat()

        # Ingerir documento com metadados de alta prioridade
        chunks_added = await self._ingest_service.ingest_document_advanced(
            tenant="default",
            doc_id=doc_id,
            source=f"expert_feedback_{feedback_id}",
            text=doc_content,
            ts_iso=ts_iso,
            document_title=f"Q&A Verificada: {original_question[:50]}...",
            tags=["golden_record", "expert_verified", "qa"],
            # Alta autoridade para priorizar no retrieval
            doc_type="golden_record",
            source_tier="verified",
            authority_tier=1,  # Máxima autoridade
            is_deprecated=False,
            # Aplicável a todos os contextos
            platform="all",
            environment="all",
            # Configurações de chunking
            chunking_strategy="fixed_size",  # Q&A geralmente são curtas
            max_tokens=500,
            overlap_tokens=0,  # Sem overlap para Q&A
        )

        logger.info(
            "knowledge_incorporated",
            feedback_id=feedback_id,
            doc_id=doc_id,
            chunks_added=chunks_added,
            reviewer_id=reviewer_id,
        )

        return doc_id

    def _format_golden_record(
        self,
        original_question: str,
        user_correction: str,
        feedback_id: int,
        reviewer_id: str | None = None,
    ) -> str:
        """
        Formata o documento Golden Record para máxima efetividade no retrieval.

        O formato é otimizado para:
        1. Matching semântico com a pergunta original
        2. Contexto claro da resposta
        3. Metadados de verificação inline
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Formato estruturado para retrieval
        return f"""# Pergunta Verificada

**Pergunta:** {original_question}

## Resposta Oficial

{user_correction}

---

**Metadados de Verificação:**
- Fonte: Feedback de especialista #{feedback_id}
- Verificado por: {reviewer_id or "admin"}
- Data de verificação: {timestamp}
- Status: ✅ Conhecimento verificado e aprovado

**Palavras-chave relacionadas:**
{self._extract_keywords(original_question, user_correction)}
"""

    def _extract_keywords(
        self,
        question: str,
        answer: str,
    ) -> str:
        """
        Extrai palavras-chave relevantes para melhorar o retrieval.

        Foca em termos técnicos do TWS/HWA.
        """
        import re

        # Combinar texto
        combined = f"{question} {answer}".upper()

        keywords = []

        # Padrões TWS comuns
        patterns = [
            r"JOB[_\s]?\w+",  # Job names
            r"AWSB\w+\d+",    # Error codes AWSB
            r"EQQQ\w+\d+",    # Error codes EQQQ
            r"ABEND\s*\w*",   # ABEND codes
            r"RC\s*=?\s*\d+", # Return codes
            r"STATUS\s*\w+",  # Status keywords
        ]

        for pattern in patterns:
            matches = re.findall(pattern, combined)
            keywords.extend(matches)

        # Remover duplicatas e formatar
        unique_keywords = list(set(keywords))[:10]

        if unique_keywords:
            return ", ".join(unique_keywords)
        return "TWS, IBM Workload Scheduler, automação"

    async def incorporate_batch(
        self,
        feedbacks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Incorpora múltiplos feedbacks em batch.

        Args:
            feedbacks: Lista de dicts com feedback_id, original_question, user_correction

        Returns:
            Dict com estatísticas de incorporação
        """
        results = {
            "total": len(feedbacks),
            "success": 0,
            "failed": 0,
            "doc_ids": [],
            "errors": [],
        }

        for fb in feedbacks:
            try:
                doc_id = await self.incorporate_feedback(
                    feedback_id=fb["feedback_id"],
                    original_question=fb["original_question"],
                    user_correction=fb["user_correction"],
                    metadata=fb.get("metadata"),
                    reviewer_id=fb.get("reviewer_id"),
                )
                results["doc_ids"].append(doc_id)
                results["success"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "feedback_id": fb.get("feedback_id"),
                    "error": str(e),
                })
                logger.error(
                    "batch_incorporate_error",
                    feedback_id=fb.get("feedback_id"),
                    error=str(e),
                )

        logger.info(
            "batch_incorporation_complete",
            total=results["total"],
            success=results["success"],
            failed=results["failed"],
        )

        return results

    async def remove_incorporated(self, feedback_id: int) -> bool:
        """
        Remove um documento incorporado (rollback).

        Args:
            feedback_id: ID do feedback cujo documento deve ser removido

        Returns:
            True se removido com sucesso
        """
        self._ensure_initialized()

        doc_id = f"golden_record_{feedback_id}"

        try:
            from resync.knowledge.config import CFG

            await self._store.delete_by_doc_id(doc_id, collection=CFG.collection_write)
            logger.info("incorporated_document_removed", doc_id=doc_id)
            return True
        except Exception as e:
            logger.error("remove_incorporated_error", doc_id=doc_id, error=str(e))
            return False


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_instance: KnowledgeIncorporator | None = None


def get_knowledge_incorporator() -> KnowledgeIncorporator:
    """Get or create the singleton KnowledgeIncorporator instance."""
    global _instance
    if _instance is None:
        _instance = KnowledgeIncorporator()
    return _instance


async def incorporate_feedback(
    feedback_id: int,
    original_question: str,
    user_correction: str,
    **kwargs,
) -> str:
    """Convenience function to incorporate feedback."""
    incorporator = get_knowledge_incorporator()
    return await incorporator.incorporate_feedback(
        feedback_id=feedback_id,
        original_question=original_question,
        user_correction=user_correction,
        **kwargs,
    )
