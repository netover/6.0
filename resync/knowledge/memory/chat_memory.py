"""
Chat Memory Module - Armazenamento de Histórico de Conversas com Conformidade GDPR.

Este módulo fornece:
- Armazenamento de turns de chat em Postgres (pgvector)
- Redação automática de PII antes do armazenamento
- TTL (Time-To-Live) automático para conformidade GDPR
- Busca semântica em conversas anteriores

Autor: Resync Team
Versão: 1.0.0
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from resync.core.gdpr_compliance import DataAnonymizer, GDPRComplianceConfig
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


CHAT_MEMORY_TTL_DAYS = int(os.environ.get("CHAT_MEMORY_TTL_DAYS", "30"))
CHAT_MEMORY_COLLECTION = "chat_history"


@dataclass
class ChatTurn:
    """
    Representa um único turn de conversa.

    Attributes:
        session_id: ID único da sessão (para agrupar conversas)
        user_id: ID do usuário (para personalização)
        role: 'user' | 'assistant'
        content: Texto da mensagem
        expires_at: Data de expiração (TTL GDPR)
        metadata: Dados adicionais opcionais
        created_at: Timestamp de criação
    """

    session_id: str
    user_id: str
    role: str
    content: str
    expires_at: datetime = field(
        default_factory=lambda: (
            datetime.now(timezone.utc) + timedelta(days=CHAT_MEMORY_TTL_DAYS)
        )
    )
    metadata: Optional[dict] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMemoryStore:
    """
    Armazém de memória de chat com conformidade GDPR.

    Features:
    - Persistência em Postgres (usando PgVector existente)
    - Redação automática de PII
    - TTL automático (30 dias por padrão)
    - Busca semântica em histórico
    """

    def __init__(
        self,
        vector_store: Any = None,
        embedder: Any = None,
        ttl_days: int = CHAT_MEMORY_TTL_DAYS,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._ttl_days = ttl_days
        self._anonymizer = DataAnonymizer(GDPRComplianceConfig())
        self._session_cache: dict[str, list[ChatTurn]] = {}
        self._cache_max_size = 100
        self._init_lock: asyncio.Lock | None = None

    async def _get_vector_store(self):
        """Lazy load do vector store."""
        if self._vector_store is None:
            if self._init_lock is None:
                self._init_lock = asyncio.Lock()
            async with self._init_lock:
                if self._vector_store is None:
                    from resync.knowledge.store.pgvector_store import PgVectorStore
        
                    self._vector_store = PgVectorStore()
        return self._vector_store

    async def _get_embedder(self):
        """Lazy load do embedder."""
        if self._embedder is None:
            if self._init_lock is None:
                self._init_lock = asyncio.Lock()
            async with self._init_lock:
                if self._embedder is None:
                    from resync.knowledge.ingestion.embedding_service import get_embedder
        
                    self._embedder = get_embedder()
        return self._embedder

    async def add_turn(self, turn: ChatTurn) -> bool:
        """
        Armazena um turn de chat com REDAÇÃO OBRIGATÓRIA de PII.
        """
        try:
            safe_data = self._anonymizer.anonymize_personal_data(
                {
                    "content": turn.content,
                    "session_id": turn.session_id,
                    "user_id": turn.user_id,
                }
            )

            safe_content = safe_data.get("content", turn.content)

            if safe_content != turn.content:
                logger.info(
                    "pii_redacted",
                    original_length=len(turn.content),
                    redacted_length=len(safe_content),
                    session_id=turn.session_id[:8],
                )

            turn.expires_at = datetime.now(timezone.utc) + timedelta(
                days=self._ttl_days
            )
            turn_id = self._generate_turn_id(turn)

            embedder = await self._get_embedder()
            vector = await embedder.embed(safe_content, timeout=10.0)

            vector_store = await self._get_vector_store()

            await vector_store.upsert_batch(
                ids=[turn_id],
                vectors=[vector],
                payloads=[
                    {
                        "content": safe_content,
                        "session_id": turn.session_id,
                        "user_id": turn.user_id,
                        "role": turn.role,
                        "expires_at": turn.expires_at.isoformat(),
                        "metadata": turn.metadata or {},
                    }
                ],
                collection=CHAT_MEMORY_COLLECTION,
                timeout=15.0,
            )

            self._update_cache(turn)

            logger.debug(
                "chat_turn_stored",
                turn_id=turn_id[:8],
                session_id=turn.session_id[:8],
                ttl_days=self._ttl_days,
            )

            return True

        except Exception as e:
            logger.error("chat_turn_store_failed", error=str(e))
            return False

    async def search(
        self, query: str, user_id: str, session_id: Optional[str] = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Busca em conversas anteriores por similaridade semântica.
        """
        try:
            embedder = await self._get_embedder()
            vector = await embedder.embed(query, timeout=5.0)

            vector_store = await self._get_vector_store()

            filters = {"user_id": user_id}
            if session_id:
                filters["session_id"] = session_id

            results = await vector_store.query(
                vector=vector,
                top_k=top_k,
                collection=CHAT_MEMORY_COLLECTION,
                filters=filters,
                with_vectors=False,
            )

            now = datetime.now(timezone.utc)
            valid_results = []

            for result in results:
                expires_at = result.get("payload", {}).get("expires_at")
                if expires_at:
                    exp_date = datetime.fromisoformat(expires_at)
                    if exp_date > now:
                        valid_results.append(result)
                else:
                    valid_results.append(result)

            logger.debug(
                "chat_history_search",
                query=query[:50],
                user_id=user_id[:8],
                results_found=len(valid_results),
            )

            return valid_results

        except Exception as e:
            logger.error("chat_history_search_failed", error=str(e))
            return []

    async def cleanup_expired(self) -> int:
        """
        Deleta registros expirados do banco de dados.
        """
        try:
            logger.info("chat_memory_cleanup_triggered", ttl_days=self._ttl_days)
            vector_store = await self._get_vector_store()
            pool = await vector_store._get_pool()
            
            async with pool.acquire() as conn:
                now_iso = datetime.now(timezone.utc).isoformat()
                result = await conn.execute(
                    """
                    DELETE FROM document_embeddings 
                    WHERE collection_name = $1 
                    AND metadata ? 'expires_at'
                    AND metadata->>'expires_at' < $2
                    """,
                    CHAT_MEMORY_COLLECTION,
                    now_iso
                )
                deleted = int(result.split()[-1])
                logger.info("chat_memory_cleanup_finished", deleted_count=deleted)
                return deleted

        except Exception as e:
            logger.error("chat_memory_cleanup_failed", error=str(e))
            return 0

    def _generate_turn_id(self, turn: ChatTurn) -> str:
        """Gera ID único para o turn."""
        content_hash = hashlib.sha256(
            f"{turn.session_id}:{turn.user_id}:{turn.content}:{turn.created_at.isoformat()}".encode()
        ).hexdigest()[:16]
        return f"chat_{content_hash}"

    def _update_cache(self, turn: ChatTurn) -> None:
        """Atualiza cache local de sessions."""
        if turn.session_id not in self._session_cache:
            self._session_cache[turn.session_id] = []

        self._session_cache[turn.session_id].append(turn)

        if len(self._session_cache) > self._cache_max_size:
            oldest_session = min(
                self._session_cache.keys(),
                key=lambda s: self._session_cache[s][0].created_at,
            )
            del self._session_cache[oldest_session]


_chat_memory_store: Optional[ChatMemoryStore] = None


def get_chat_memory_store() -> ChatMemoryStore:
    """Get or cria singleton do ChatMemoryStore."""
    global _chat_memory_store
    if _chat_memory_store is None:
        _chat_memory_store = ChatMemoryStore()
    return _chat_memory_store


__all__ = ["ChatTurn", "ChatMemoryStore", "get_chat_memory_store"]
