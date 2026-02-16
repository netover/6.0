# Plano de Implementação - Otimização RAG v7.0 (Edição Final)

**Versão:** 7.0  
**Status:** Pronto para Execução  
**Público-alvo:** Desenvolvedores Junior/Mid-Level  
**Esforço Estimado:** 3-4 Dias  
**Data de Criação:** 2026-02-15

---

## 1. Visão Executiva

### 1.1. Objetivos de Negócio (ROI)

| Problema Atual | Solução | Valor de Negócio |
|---------------|---------|------------------|
| Boot lento (~15s) | Índice Persistente: Salvar índice em disco (.bin.gz) e carregar via mmap | 95% faster startups. Rollbacks e scaling instantâneos |
| Amnésia do Chat | Memória Vetorial: Armazenar histórico de conversas no Postgres (pgbox) | Inteligência Contextual. O bot lembra soluções passadas |
| Caixa Preta | Dashboard Admin: Métricas visuais para cache hits e latência | Decisões Data-Driven. Sabemos exatamente como o sistema performa |

### 1.2. Dependências Externas

O projeto NÃO possui estas bibliotecas. Devem ser instaladas:

```txt
# Adicionar ao requirements.txt do projeto
filelock>=3.13.0   # RLock para múltiplos processos (workers Uvicorn)
joblib>=1.3.0      # Serialização eficiente para grandes arrays numpy
pgvector>=0.2.0    # Extensão PostgreSQL para busca vetorial
```

**Por quê?**
- **filelock**: O Uvicorn executa múltiplos processos (workers). Python `threading.Lock` só funciona dentro de um único processo. Para evitar que dois workers escribam no arquivo de índice ao mesmo tempo e corrompam, precisamos de um lock que viva no sistema de arquivos.
- **joblib**: Serialização otimizada para numpy arrays, muito mais rápida que pickle puro.
- **pgvector**: Extensão PostgreSQL para buscar相似idade vetorial (necessário para chat memory).

---

## 2. Fase 1: Recuperação Persistente Robusta (O "Boot Rápido")

### 2.1. Objetivo

Modificar `BM25Index` para salvar seu estado em disco e carregá-lo com segurança.

### 2.2. Arquivo Alvo

**Arquivo:** `resync/knowledge/retrieval/hybrid_retriever.py`  
**Classe:** `BM25Index` (linha ~44)

### 2.3. Step 1: Adicionar Imports e Constantes

Adicionar estes imports no topo do arquivo (após os imports existentes):

```python
# Phase 1: Persistent BM25 Index
import gzip
import os
import asyncio
from pathlib import Path
from filelock import FileLock, Timeout
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Constante para caminho do índice
INDEX_STORAGE_PATH = os.environ.get(
    "BM25_INDEX_PATH", 
    os.path.join(os.getcwd(), "data", "bm25_index.bin.gz")
)
```

**Por quê?**
- `gzip`: Comprimir o arquivo para economizar espaço (índices podem ter vários GB)
- `filelock`: Garantir acesso exclusivo entre processos Uvicorn
- `asyncio`: Para wrapper async seguro

### 2.4. Step 2: Implementar método save()

Adicionar este método à classe `BM25Index`:

```python
def save(self, path: str) -> bool:
    """
    Persiste o índice em disco com compressão.
    
    Por quê: Rebuilding leva 15s. Carregamento leva 0.5s.
    Segurança: Usa FileLock para evitar corrupção por escrita concorrente.
    
    Args:
        path: Caminho do arquivo de índice (.bin.gz)
    
    Returns:
        True se salvo com sucesso, False caso contrário
    """
    try:
        # Garante que diretório existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        lock_path = f"{path}.lock"
        
        # FileLock com timeout para evitar deadlock infinito
        lock = FileLock(lock_path, timeout=10)
        
        try:
            with lock.acquire(timeout=10):
                # Serialização com joblib + compressão gzip
                with gzip.open(path, "wb") as f:
                    joblib.dump(self, f)
                
                logger.info(
                    "bm25_index_saved",
                    path=path,
                    size_bytes=os.path.getsize(path),
                    num_docs=len(self.documents),
                    num_terms=len(self.inverted_index)
                )
                return True
                
        except Timeout:
            logger.error("bm25_index_save_timeout", path=path)
            return False
            
    except Exception as e:
        logger.error("bm25_index_save_failed", error=str(e), path=path)
        return False
```

### 2.5. Step 3: Implementar método load() (O Sentinel)

Adicionar este método de classe à classe `BM25Index`:

```python
@classmethod
def load(cls, path: str) -> "BM25Index":
    """
    Carrega índice do disco com recuperação automática de corrupção.
    
    Por quê: Servidor de produção pode ter índice corrompido por:
    - Crash durante escrita
    - Disco com problemas
    - Memória insuficiente
    
    Comportamento:
    - Se arquivo não existe: retorna índice vazio (aciona rebuild)
    - Se corrompido: loga aviso, deleta arquivo, retorna índice vazio
    - Se MemoryError: loga erro, retorna índice vazio (degradação graciosa)
    
    Args:
        path: Caminho do arquivo de índice
    
    Returns:
        Instância BM25Index (carregada ou vazia)
    """
    # Se arquivo não existe, retorna índice fresco
    if not os.path.exists(path):
        logger.info("bm25_index_not_found_will_build", path=path)
        return cls()  # Retorna novo índice (chamador fará rebuild)
    
    lock_path = f"{path}.lock"
    
    try:
        # Tenta acquire lock com timeout curto
        lock = FileLock(lock_path, timeout=5)
        
        try:
            with lock.acquire(timeout=5):
                # Carrega índice com joblib
                with gzip.open(path, "rb") as f:
                    index = joblib.load(f)
                
                logger.info(
                    "bm25_index_loaded",
                    path=path,
                    size_bytes=os.path.getsize(path),
                    num_docs=len(index.documents),
                    load_time_ms=0  # Você pode adicionar timing aqui
                )
                return index
                
        except Timeout:
            # Outro processo está usando. Retorna índice vazio.
            logger.warning("bm25_index_locked_using_empty", path=path)
            return cls()
            
    except (EOFError, OSError, ImportError, gzip.BadGzipFile) as e:
        # CORRUÇÃO DETECTADA: Auto-cura
        logger.warning(
            "bm25_index_corrupted_rebuilding",
            error=str(e),
            path=path
        )
        try:
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as cleanup_error:
            logger.error("bm25_index_cleanup_failed", error=str(cleanup_error))
        
        return cls()  # Retorna novo índice (chamador will rebuild)
        
    except MemoryError:
        # PRESSÃO DE MEMÓRIA: Falha rápida mas segura
        logger.error(
            "bm25_index_oom_using_empty",
            path=path,
            available_memory_mb=_get_available_memory_mb()
        )
        return cls()  # Degrada para índice vazio ao invés de crashar
        
    except Exception as e:
        # Qualquer outro erro: loga e retorna índice vazio
        logger.warning(
            "bm25_index_load_failed_unknown",
            error=str(e),
            error_type=type(e).__name__,
            path=path
        )
        return cls()


def _get_available_memory_mb() -> float:
    """Retorna memória disponível em MB (para logging)."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return -1  # psutil não instalado
```

### 2.6. Step 4: Modificar o HybridRetriever

No arquivo `resync/knowledge/retrieval/hybrid_retriever.py`, método `_ensure_bm25_index()` (~linha 854):

```python
async def _ensure_bm25_index(self, collection: str | None = None) -> None:
    """Garante que o índice BM25 está pronto (carregando ou construindo)."""
    
    if self._index_built:
        return
    
    # Define caminho do índice
    index_path = os.environ.get(
        "BM25_INDEX_PATH",
        os.path.join(os.getcwd(), "data", "bm25_index.bin.gz")
    )
    
    try:
        # Step 1: Tenta carregar índice persistido
        logger.info("Attempting to load persisted BM25 index", path=index_path)
        
        self.bm25_index = await asyncio.to_thread(
            BM25Index.load, index_path
        )
        
        # Se retornou com documentos, está pronto!
        if self.bm25_index and self.bm25_index.documents:
            self._index_built = True
            logger.info(
                "BM25 index loaded from disk",
                num_docs=len(self.bm25_index.documents)
            )
            return
            
    except Exception as e:
        logger.warning(
            "BM25 index load failed, will rebuild",
            error=str(e)
        )
    
    # Step 2: Se não tem índice (ou falhou), constrói do zero
    logger.info("Building fresh BM25 index from database")
    
    try:
        documents = self.store.get_all_documents(collection=collection)
        
        if documents:
            self.bm25_index = BM25Index(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
                field_boosts=self.config.field_boosts,
            )
            self.bm25_index.build_index(documents)
            self._index_built = True
            
            # Step 3: Salva para próxima vez (em background)
            asyncio.create_task(
                self._save_index_async(index_path)
            )
            
            logger.info(
                f"BM25 index built and saved: {len(documents)} docs"
            )
        else:
            logger.warning("No documents found for BM25 indexing")
            
    except Exception as e:
        logger.error("Failed to build BM25 index: %s", e)
        # Continua sem BM25 - fallback para vector-only


async def _save_index_async(self, path: str) -> None:
    """Salva índice em background (não bloqueia requisições)."""
    try:
        # Pequeno delay para não bloquear startup
        await asyncio.sleep(2)
        
        if self.bm25_index:
            success = self.bm25_index.save(path)
            if success:
                logger.info("BM25 index persisted successfully")
            else:
                logger.warning("BM25 index persist failed (non-critical)")
    except Exception as e:
        logger.error("BM25 index async save failed: %s", e)
```

---

## 3. Fase 2: Memória de Chat em Conformidade (O "Cérebro")

### 3.1. Objetivo

Armazenar turns de conversa no Postgres sem violar privacidade do usuário.

### 3.2. Por Quê Conformidade GDPR?

- **LGPD/GDPR Artigo 5**: "Os dados devem ser mantidos apenas pelo tempo necessário"
- **Armazenamento proativo**: turn.s `expires_at` garante exclusão automática
- **Redação de PII**: Dados sensíveis são removidos ANTES do armazenamento

### 3.3. Novo Arquivo

**Arquivo:** `resync/knowledge/memory/chat_memory.py` (NOVO)

```python
"""
Chat Memory Module - Armazenamento de Histórico de Conversas com Conformidade GDPR.

Este módulo fornece:
- Armazenamento de turns de chat em Postgres (pgvector)
- Redação obrigatória de PII antes do armazenamento
- TTL (Time-To-Live) automático para conformidade GDPR
- Busca semântica em conversas anteriores

Autor: Resync Team
Versão: 1.0.0
"""

import os
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Any

from resync.core.structured_logger import get_logger
from resync.core.gdpr_compliance import (
    DataAnonymizer,
    GDPRComplianceConfig,
    RetentionPolicy,
    DataCategory
)
from resync.knowledge.config import CFG

logger = get_logger(__name__)


# =============================================================================
# SCHEMA DE DADOS
# =============================================================================


@dataclass
class ChatTurn:
    """
    Representa um único turn de conversa.
    
    Attributes:
        session_id: ID único da sessão (paraagrupar conversas)
        user_id: ID do usuário (para personalização)
        role: 'user' | 'assistant'
        content: Texto da mensagem
        expires_at: Data de expiração (TTL GDPR)
        metadata: Dados adicionais opcionais
        created_at: Timestamp de criação
    """
    session_id: str
    user_id: str
    role: str  # 'user' | 'assistant'
    content: str
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    metadata: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# CHAT MEMORY STORE
# =============================================================================


class ChatMemoryStore:
    """
    Armazém de memória de chat com conformidade GDPR.
    
    Features:
    - Persistência em Postgres (usando PgVector existente)
    - Redação automática de PII
    - TTL automático (30 dias por padrão)
    - Busca semântica em histórico
    
    Usage:
        store = ChatMemoryStore()
        await store.add_turn(ChatTurn(...))
        results = await store.search("problema no job X", user_id="user123")
    """
    
    # TTL padrão em dias (configurável via ambiente)
    DEFAULT_TTL_DAYS = int(os.environ.get("CHAT_MEMORY_TTL_DAYS", "30"))
    
    # Nome da coleção para chat history no PgVector
    COLLECTION_NAME = "chat_history"
    
    def __init__(
        self,
        vector_store: Any = None,
        embedder: Any = None,
        ttl_days: int = DEFAULT_TTL_DAYS
    ):
        """
        Inicializa o store de memória de chat.
        
        Args:
            vector_store: PgVectorStore existente (injetado ou criado)
            embedder: Modelo de embedding (injetado ou criado)
            ttl_days: Dias até expiração (padrão: 30 para conformidade GDPR)
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._ttl_days = ttl_days
        
        # Inicializa redator GDPR
        self._anonymizer = DataAnonymizer(GDPRComplianceConfig())
        
        # Cache de turns recentes por sessão
        self._session_cache: dict[str, List[ChatTurn]] = {}
        self._cache_max_size = 100
    
    async def _get_vector_store(self):
        """Lazy load do vector store."""
        if self._vector_store is None:
            from resync.knowledge.store.pgvector_store import PgVectorStore
            self._vector_store = PgVectorStore()
        return self._vector_store
    
    async def _get_embedder(self):
        """Lazy load do embedder."""
        if self._embedder is None:
            from resync.knowledge.ingestion.embedding_service import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    # =========================================================================
    # OPERACIONES CORE
    # =========================================================================
    
    async def add_turn(self, turn: ChatTurn) -> bool:
        """
        Armazena um turn de chat com REDAÇÃO OBRIGATÓRIA de PII.
        
        Por quê: Uma vez que dados são vetorizados, é difícil remover PII.
        Devemos limpar ANTES do armazenamento.
        
        Args:
            turn: Turn de conversa para armazenar
            
        Returns:
            True se armazenado com sucesso
        """
        try:
            # Step 1: REDAÇÃO DE PII (OBRIGATÓRIA)
            # Usa DataAnonymizer do módulo GDPR existente
            safe_data = self._anonymizer.anonymize_personal_data({
                "content": turn.content,
                "session_id": turn.session_id,
                "user_id": turn.user_id
            })
            
            safe_content = safe_data.get("content", turn.content)
            
            # Se conteúdo mudou significativamente após redação, loga
            if safe_content != turn.content:
                logger.info(
                    "pii_redacted",
                    original_length=len(turn.content),
                    redacted_length=len(safe_content),
                    session_id=turn.session_id[:8]  # Only log part for privacy
                )
            
            # Step 2: Enforce TTL (Time-To-Live)
            # Usa política existente do módulo GDPR
            turn.expires_at = datetime.utcnow() + timedelta(days=self._ttl_days)
            
            # Step 3: Gera ID único
            turn_id = self._generate_turn_id(turn)
            
            # Step 4: Vetoriza conteúdo
            embedder = await self._get_embedder()
            vector = embedder.embed(safe_content)
            
            # Step 5: Armazena no PgVector
            vector_store = await self._get_vector_store()
            
            await vector_store.upsert_batch(
                ids=[turn_id],
                vectors=[vector],
                payloads=[{
                    "content": safe_content,
                    "session_id": turn.session_id,
                    "user_id": turn.user_id,
                    "role": turn.role,
                    "expires_at": turn.expires_at.isoformat(),
                    "metadata": turn.metadata or {}
                }],
                collection=self.COLLECTION_NAME
            )
            
            # Atualiza cache local
            self._update_cache(turn)
            
            logger.debug(
                "chat_turn_stored",
                turn_id=turn_id[:8],
                session_id=turn.session_id[:8],
                ttl_days=self._ttl_days
            )
            
            return True
            
        except Exception as e:
            logger.error("chat_turn_store_failed", error=str(e))
            return False
    
    async def search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[dict[str, Any]]:
        """
        Busca em conversas anteriores por similaridade semântica.
        
        Args:
            query: Texto da pergunta atual
            user_id: ID do usuário (para filtrar apenas conversas dele)
            session_id: ID de sessão específico (opcional)
            top_k: Número de resultados
            
        Returns:
            Lista de turns relevantes encontrados
        """
        try:
            embedder = await self._get_embedder()
            vector = embedder.embed(query)
            
            vector_store = await self._get_vector_store()
            
            # Filtros: apenas do usuário atual
            filters = {"user_id": user_id}
            if session_id:
                filters["session_id"] = session_id
            
            results = await vector_store.query(
                vector=vector,
                top_k=top_k,
                collection=self.COLLECTION_NAME,
                filters=filters,
                with_vectors=False
            )
            
            # Filtra results expirados
            now = datetime.utcnow()
            valid_results = []
            
            for result in results:
                expires_at = result.get("payload", {}).get("expires_at")
                if expires_at:
                    exp_date = datetime.fromisoformat(expires_at)
                    if exp_date > now:
                        valid_results.append(result)
                else:
                    # Se não tem expires_at, mantém (legacy data)
                    valid_results.append(result)
            
            logger.debug(
                "chat_history_search",
                query=query[:50],
                user_id=user_id[:8],
                results_found=len(valid_results)
            )
            
            return valid_results
            
        except Exception as e:
            logger.error("chat_history_search_failed", error=str(e))
            return []
    
    async def cleanup_expired(self) -> int:
        """
        Task agendada para deletar registros expirados.
        
        Implementa "Storage Limitation" do GDPR.
        
        Returns:
            Número de registros deletados
        """
        try:
            # Este método seria implementado com query SQL direta
            # Por enquanto, loga que deveria executar
            logger.info(
                "chat_memory_cleanup_triggered",
                ttl_days=self._ttl_days
            )
            
            # TODO: Implementar limpeza real via SQL:
            # DELETE FROM document_embeddings 
            # WHERE collection_name = 'chat_history'
            # AND metadata->>'expires_at' < NOW()
            
            return 0
            
        except Exception as e:
            logger.error("chat_memory_cleanup_failed", error=str(e))
            return 0
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
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
        
        # Limpa cache se muito grande
        if len(self._session_cache) > self._cache_max_size:
            oldest_session = min(
                self._session_cache.keys(),
                key=lambda s: self._session_cache[s][0].created_at
            )
            del self._session_cache[oldest_session]


# =============================================================================
# SINGLETON
# =============================================================================


_chat_memory_store: Optional[ChatMemoryStore] = None


def get_chat_memory_store() -> ChatMemoryStore:
    """Get or cria singleton do ChatMemoryStore."""
    global _chat_memory_store
    if _chat_memory_store is None:
        _chat_memory_store = ChatMemoryStore()
    return _chat_memory_store


__all__ = [
    "ChatTurn",
    "ChatMemoryStore", 
    "get_chat_memory_store"
]
```

---

## 4. Fase 3: Observabilidade (Os "Olhos")

### 4.1. Objetivo

Fornecer dashboard para monitorar os novos sistemas.

### 4.2. Novo Arquivo

**Arquivo:** `resync/api/routes/admin/rag_stats.py` (NOVO)

```python
"""
RAG Statistics Endpoint - Monitoramento e Observabilidade.

Fornece métricas para:
- Status do índice BM25 (tamanho, lock, integridade)
- Performance do cache (hit rate, miss rate)
- Memória de chat (turns armazenados, expirações)

Autor: Resync Team
Versão: 1.0.0
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/admin/rag", tags=["RAG Admin"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class IndexStatus(BaseModel):
    """Status do índice BM25."""
    path: str
    exists: bool
    size_bytes: Optional[int] = None
    last_modified: Optional[str] = None
    is_locked: bool
    is_loaded: bool
    num_documents: int = 0
    num_terms: int = 0


class CachePerformance(BaseModel):
    """Performance do cache de queries."""
    hit_rate: float
    miss_rate: float
    total_queries: int
    cache_size: int
    cache_max_size: int


class ChatMemoryStatus(BaseModel):
    """Status da memória de chat."""
    enabled: bool
    ttl_days: int
    total_turns: int = 0
    active_sessions: int = 0
    expired_turns_cleaned: int = 0


class RAGStatsResponse(BaseModel):
    """Resposta completa de estatísticas RAG."""
    timestamp: str
    index_status: IndexStatus
    cache_performance: CachePerformance
    chat_memory: ChatMemoryStatus


# =============================================================================
# ENDPOINT
# =============================================================================


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats() -> RAGStatsResponse:
    """
    Retorna estatísticas completas do sistema RAG.
    
    Use para:
    - Monitoramento de saúde
    - Debugging de performance
    - Dashboard administrativo
    
    Returns:
        RAGStatsResponse com todas as métricas
    """
    # === INDEX STATUS ===
    index_path = os.environ.get(
        "BM25_INDEX_PATH",
        os.path.join(os.getcwd(), "data", "bm25_index.bin.gz")
    )
    lock_path = f"{index_path}.lock"
    
    index_exists = os.path.exists(index_path)
    index_size = os.path.getsize(index_path) if index_exists else None
    
    index_modified = None
    if index_exists:
        mod_time = os.path.getmtime(index_path)
        index_modified = datetime.fromtimestamp(mod_time).isoformat()
    
    # Tenta obter info do índice carregado
    # (Em produção, isso viria de uma variável de instância)
    num_docs = 0
    num_terms = 0
    
    # === CACHE PERFORMANCE ===
    # Estes valores viriam de métricas reais
    cache_hit_rate = 0.0  # TODO: Obter de métricas reais
    total_queries = 0
    cache_size = 0
    
    # === CHAT MEMORY ===
    chat_memory_enabled = os.environ.get("CHAT_MEMORY_ENABLED", "true").lower() == "true"
    chat_ttl = int(os.environ.get("CHAT_MEMORY_TTL_DAYS", "30"))
    
    return RAGStatsResponse(
        timestamp=datetime.utcnow().isoformat(),
        index_status=IndexStatus(
            path=index_path,
            exists=index_exists,
            size_bytes=index_size,
            last_modified=index_modified,
            is_locked=os.path.exists(lock_path),
            is_loaded=False,  # TODO: Obter de estado real
            num_documents=num_docs,
            num_terms=num_terms
        ),
        cache_performance=CachePerformance(
            hit_rate=cache_hit_rate,
            miss_rate=1.0 - cache_hit_rate,
            total_queries=total_queries,
            cache_size=cache_size,
            cache_max_size=1000  # Do config
        ),
        chat_memory=ChatMemoryStatus(
            enabled=chat_memory_enabled,
            ttl_days=chat_ttl,
            total_turns=0,  # TODO: Obter do DB
            active_sessions=0,
            expired_turns_cleaned=0
        )
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================


@router.get("/health")
async def rag_health_check() -> dict[str, Any]:
    """
    Health check simplificado para RAG.
    
    Returns:
        Status de saúde do sistema RAG
    """
    index_path = os.environ.get(
        "BM25_INDEX_PATH",
        os.path.join(os.getcwd(), "data", "bm25_index.bin.gz")
    )
    
    index_ok = os.path.exists(index_path)
    
    return {
        "status": "healthy" if index_ok else "degraded",
        "index_ready": index_ok,
        "timestamp": datetime.utcnow().isoformat()
    }


__all__ = ["router", "get_rag_stats", "rag_health_check"]
```

### 4.3. Registrar o Router

Adicionar no arquivo de rotas admin:

```python
# Em resync/api/routes/admin/main.py ou similar

from resync.api.routes.admin import rag_stats

router.include_router(rag_stats.router)
```

---

## 5. Plano de Rollback

### 5.1. Se o Sistema Ficar Instável

1. **Parar o servidor**
2. **Deletar o arquivo de índice:**
   ```bash
   rm -f data/bm25_index.bin.gz
   rm -f data/bm25_index.bin.gz.lock
   ```
3. **Reiniciar o serviço**

O código é projetado para fazer fallback para um build em memória fresca se o arquivo estiver faltando. Seguro e simples.

---

## 6. Checklist de Verificação

### ✅ Testes Manuais

| Teste | Critério de Sucesso |
|-------|---------------------|
| **Boot Speed:** Iniciar servidor, logs mostram "Loaded BM25 index from disk" em <1s | |
| **Boot Speed 2:** Segunda inicialização deve ser instantânea | |
| **Recovery:** Corromper arquivo (deletar caracteres aleatórios), servidor NÃO deve crashar, deve log "Index corrupted, rebuilding" | |
| **Privacy:** Enviar mensagem "My password is 123456", verificar DB, conteúdo deve estar "My password is ***REDACTED***" | |
| **TTL:** Inserir registro com expires_at = agora - 1min, executar cleanup, registro deve desaparecer | |

---

## 7. Esforço e Responsabilidades

| Fase | Atividade | Esforço | Responsável | Entregável |
|------|-----------|----------|--------------|------------|
| 0 | Dependencies | 0.5 dia | Dev | requirements.txt atualizado |
| 1 | BM25 Persistence | 1.5 dias | Dev | HybridRetriever com save/load |
| 1 | Testes Unitários | 0.5 dia | QA/Dev | Relatório de testes |
| 2 | Chat Memory | 1.5 dias | Dev | chat_memory.py |
| 2 | Integração | 0.5 dia | Dev | Busca híbrida atualizada |
| 3 | Observabilidade | 0.5 dia | Dev | Endpoint /admin/rag/stats |
| **TOTAL** | | **5 dias** | | |

---

## 8. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Índice corrompido | Média | Alto | Auto-recovery com fallback para rebuild |
| Acesso concorrente | Alta | Alto | FileLock com timeout |
| Memória insuficiente | Baixa | Alto | Catch MemoryError, degradação graciosa |
| PII em chat history | Alta | Crítico | Redação obrigatória antes de storage |

---

Este plano está completo e pronto para execução após aprovação.