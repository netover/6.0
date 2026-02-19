# Plano de Correções - PR #35 Issues Pendentes

## Visão Geral

Este documento detalha todas as issues identificadas nas revisões da PR #35 que ainda precisam ser corrigidas. As correções críticas já foram aplicadas, mas há dezenas de issues adicionais que precisam ser tratadas.

---

## 1. Issues de Segurança (Security)

### 1.1 BM25 Insecure Deserialization (joblib.load)
**Severidade:** CRITICAL
**Arquivo:** `resync/knowledge/retrieval/hybrid_retriever.py`
**Linha:** 241-285
**Problema:** O método `BM25Index.load` usa `joblib.load()` para deserializar arquivos locais, o que pode levar a execução de código arbitrário se um atacante conseguir influenciar o `INDEX_STORAGE_PATH`.

**Solução Proposta:**
```python
# Substituir joblib.load por uma alternativa segura
# Opção 1: Usar pickle com verificação de hash
# Opção 2: Usar um formato seguro como JSON + validar estrutura
# Opção 3: Adicionar verificação de assinatura do arquivo

import hashlib
import json

def load(cls, path: str) -> 'BM25Index':
    # Verificar integridade do arquivo
    expected_hash = cls._get_expected_hash(path)
    actual_hash = cls._compute_file_hash(path)
    
    if expected_hash != actual_hash:
        raise SecurityError("File integrity check failed")
    
    # Carregar dados de forma segura
    with open(path, 'r') as f:
        data = json.load(f)
    
    return cls._from_dict(data)
```

### 1.2 Semantic Cache - Cross-User Data Leakage
**Severidade:** CRITICAL
**Arquivo:** `resync/core/cache/semantic_cache.py`
**Linha:** 165
**Problema:** O user_id é嵌入ado no texto da query, o que não fornece isolamento confiável entre usuários. Embeddings são semanticamente similares entre usuários diferentes.

**Solução Proposta:**
```python
# Adicionar campo user_id como tag no schema e usar FilterExpression
from qdrant_client import Filter, FieldCondition

async def _search_redisvl(
    self, 
    query: str, 
    user_id: str,
    limit: int = 5
) -> list[dict]:
    # Usar filtro de user_id para isolamento
    search_filter = Filter(
        must=[
            FieldCondition(key="user_id", match={"value": user_id})
        ]
    )
    
    results = await self.collection.search(
        query_vector=embedding,
        query_filter=search_filter,
        limit=limit
    )
```

### 1.3 Sensitive Logging in SQL Injection Detection
**Severidade:** WARNING
**Arquivo:** `resync/api/middleware/database_security_middleware.py`
**Linha:** 70-86
**Problema:** O middleware loga valores fornecidos pelo usuário (mesmo truncados a 100 chars), o que pode vazar secrets ou PII.

**Solução Proposta:**
```python
# Hash de inputs antes de logar
import hashlib

def _sanitize_for_logging(self, user_input: str) -> str:
    """Hash input to avoid logging sensitive data."""
    return hashlib.sha256(user_input.encode()).hexdigest()[:8]

def _log_violation(self, pattern: str, user_input: str):
    sanitized = self._sanitize_for_logging(user_input)
    logger.warning(
        "SQL injection pattern detected",
        pattern=pattern,
        input_hash=sanitized,  # Em vez do input truncado
    )
```

---

## 2. Issues de Race Condition e Concorrência

### 2.1 Agent Manager Race Condition
**Severidade:** CRITICAL
**Arquivo:** `resync/core/agent_manager.py`
**Linha:** 462
**Problema:** `_create_agent(agent_id)` é chamado FORA do lock, então requisições concorrentes para o mesmo `agent_id` criam múltiplos agentes redundantes.

**Solução Proposta:**
```python
# Mover _create_agent para dentro do lock
async def get_agent(self, agent_id: str) -> Agent:
    async with self._agent_creation_lock:
        if agent_id in self._agents:
            return self._agents[agent_id]
        
        # AGORA dentro do lock
        agent = await self._create_agent(agent_id)
        self._agents[agent_id] = agent
        return agent
```

### 2.2 TWSStore Double-Check Locking Bug
**Severidade:** CRITICAL
**Arquivo:** `resync/api/dependencies_v2.py`
**Linha:** 118
**Problema:** `TWSStore()` creation e `await store.initialize()` estão FORA do lock, então chamadas concorrentes criam múltiplas instâncias.

**Solução Proposta:**
```python
# Mover criação e inicialização para dentro do lock
async def get_tws_store() -> TWSStore:
    global _tws_store_instance
    if _tws_store_instance is None:
        async with _tws_store_lock:
            if _tws_store_instance is None:
                # Criar E inicializar dentro do lock
                store = TWSStore()
                await store.initialize()
                _tws_store_instance = store
    return _tws_store_instance
```

### 2.3 Runtime Metrics Histogram/Gauge Bug
**Severidade:** CRITICAL
**Arquivo:** `resync/core/metrics/runtime_metrics.py`
**Linha:** 175
**Problema:** `record_histogram` stores histograms in `self._dynamic_gauges` em vez de `_dynamic_histograms` (copy-paste error).

**Solução Proposta:**
```python
# Adicionar dict para histograms e corrigir o método
class RuntimeMetrics:
    def __init__(self):
        self._dynamic_gauges: dict[str, Gauge] = {}
        self._dynamic_histograms: dict[str, Histogram] = {}  # ADICIONAR
    
    def record_histogram(self, name: str, value: float, tags: dict = None):
        if name not in self._dynamic_histograms:  # CORRIGIR
            self._dynamic_histograms[name] = Histogram(name)
        self._dynamic_histograms[name].observe(value)
```

### 2.4 WebSocket Client Concurrency
**Severidade:** WARNING
**Arquivo:** `resync/api/websocket/handlers.py`
**Linha:** 567-643
**Problema:** Lista global `connected_clients` é mutada de múltiplos handlers async sem sincronização.

**Solução Proposta:**
```python
import asyncio

class WebSocketHandler:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        async with self._clients_lock:
            self._clients.add(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        async with self._clients_lock:
            self._clients.discard(websocket)
```

### 2.5 EventBus Subscription Leak
**Severidade:** CRITICAL
**Arquivo:** `resync/api/routes/orchestration.py`
**Linha:** 189
**Problema:** O endpoint WebSocket subscreve um handler mas nunca desinscreve, causando leak de memória.

**Solução Proposta:**
```python
# Adicionar método unsubscribe ao EventBus
class EventBus:
    def __init__(self):
        self._handlers: list[Callable] = []
    
    def subscribe_all(self, handler: Callable):
        self._handlers.append(handler)
        return len(self._handlers) - 1  # Retorna subscription_id
    
    def unsubscribe(self, subscription_id: int):
        if 0 <= subscription_id < len(self._handlers):
            self._handlers[subscription_id] = None

# No endpoint WebSocket
async def ws_execute(config_id: str, websocket: WebSocket):
    sub_id = event_bus.subscribe_all(handler)
    try:
        # Processar
        pass
    finally:
        event_bus.unsubscribe(sub_id)  # LIMPAR
```

### 2.6 Orchestration Runner Task GC
**Severidade:** WARNING
**Arquivo:** `resync/core/orchestration/runner.py`
**Linha:** 75
**Problema:** O resultado de `asyncio.create_task()` não é salvo, então a task pode ser coletada pelo GC.

**Solução Proposta:**
```python
class OrchestrationRunner:
    def __init__(self):
        self._running_tasks: set[asyncio.Task] = set()
    
    async def run_async(self, workflow: Workflow):
        task = asyncio.create_task(self._run_workflow(workflow))
        self._running_tasks.add(task)  # SALVAR reference
        task.add_done_callback(self._running_tasks.discard)
        return task
```

---

## 3. Issues de Deprecated APIs

### 3.1 datetime.utcnow Deprecation
**Severidade:** WARNING
**Arquivos:** Múltiplos (48 occurrences)
**Problema:** `datetime.utcnow` é deprecated desde Python 3.12.

**Solução Proposta:**
```python
# Substituir:
from datetime import datetime, timezone

# ANTES (deprecated)
created_at = datetime.utcnow()

# DEPOIS (correct)
created_at = datetime.now(timezone.utc)

# Para default_factory em models:
from datetime import datetime, timezone

def _utc_now():
    return datetime.now(timezone.utc)

class MyModel:
    created_at: datetime = Field(default_factory=_utc_now)
```

**Arquivos afetados:**
- resync/models/error_models.py
- resync/knowledge/models.py
- resync/knowledge/ontology/provenance.py
- resync/knowledge/retrieval/tws_relations.py
- resync/core/database/models/auth.py
- resync/core/database/models/orchestration.py
- resync/core/continual_learning/threshold_tuning.py
- resync/core/database/models/metrics.py
- resync/core/continual_learning/orchestrator.py
- resync/core/database/models/teams.py
- resync/core/database/models/teams_notifications.py
- resync/core/database/repositories/tws_repository.py
- resync/api/validation/common.py
- resync/api/validation/chat.py
- resync/core/database/repositories/orchestration_execution_repo.py
- resync/core/monitoring/evidently_monitor.py
- resync/core/orchestration/events.py
- resync/core/metrics/lightweight_store.py
- resync/api/v1/admin/admin_api_keys.py
- resync/core/orchestration/runner.py
- resync/core/specialists/tools.py
- resync/core/specialists/models.py
- resync/core/langfuse/prompt_manager.py
- resync/core/langfuse/observability.py
- resync/core/tws_multi/instance.py
- resync/core/tws_multi/session.py

---

## 4. Issues de Lógica de Negócio

### 4.1 Regex Validation After Compile
**Severidade:** WARNING
**Arquivo:** `resync/api/middleware/database_security_middleware.py`
**Linha:** 94-110
**Problema:** O código compila cada padrão (verificando sintaxe) e depois rejeita regex usage em produção. Compilar antes de verificar ambiente é desperdiçar CPU.

**Solução Proposta:**
```python
def validate_regex_pattern(self, pattern: str, environment: str) -> bool:
    # Verificar ambiente PRIMEIRO (fail fast)
    if environment == "production":
        raise ValidationError("Regex patterns not allowed in production")
    
    # Só compilar se não for produção
    try:
        re.compile(pattern)
        return True
    except re.error as e:
        raise ValidationError(f"Invalid regex pattern: {e}")
```

### 4.2 Pagination Division by Zero
**Severidade:** WARNING
**Arquivo:** `resync/api/utils/helpers.py`
**Linha:** 67-71
**Problema:** O helper de paginação calcula `offset // limit + 1` sem proteger contra `limit == 0`.

**Solução Proposta:**
```python
def get_pagination(
    total: int,
    page: int = 1,
    limit: int = 10
) -> dict:
    if limit <= 0:
        limit = 10  # Default fallback
    
    total_pages = (total + limit - 1) // limit if limit > 0 else 0
    
    # Proteger contra division by zero
    current_page = (offset // limit + 1) if limit > 0 else 1
    
    return {
        "total": total,
        "page": current_page,
        "limit": limit,
        "total_pages": total_pages
    }
```

### 4.3 RAG Service Missing Await
**Severidade:** WARNING
**Arquivo:** `resync/api/services/rag_service.py`
**Linha:** 116-126
**Problema:** Várias chamadas de método do RAG service são invocadas sem `await`.

**Solução Proposta:**
```python
# Verificar se os métodos são async e adicionar await
documents = await self.rag_service.list_documents(collection)
document = await self.rag_service.get_document(doc_id)
await self.rag_service.delete_document(doc_id)
stats = await self.rag_service.get_stats()
```

### 4.4 ConnectionManager Race Condition
**Severidade:** WARNING
**Arquivo:** `resync/core/connection_manager.py`
**Linha:** 73-109
**Problema:** O `disconnect` síncrono faz fallback para mutação sem lock.

**Solução Proposta:**
```python
async def disconnect(self, client_id: str):
    async with self._lock:
        if client_id in self._connections:
            websocket = self._connections.pop(client_id)
            await websocket.close()
```

---

## 5. Issues de Documentação e Logging

### 5.1 Authentication Log Regression
**Severidade:** SUGGESTION
**Arquivo:** `resync/api/auth/service.py`
**Linha:** 131
**Problema:** Todos os caminhos de falha de autenticação emitem a mesma mensagem de log.

**Solução Proposta:**
```python
# Restaurar mensagens distintas
logger.warning("Authentication failed: user not found", user_id=user_id)
logger.warning("Authentication failed: invalid password", user_id=user_id)
logger.warning("Authentication failed: inactive user", user_id=user_id)
```

### 5.2 Docstring/Code Mismatch
**Severidade:** WARNING
**Arquivo:** `resync/api/responses.py`
**Linha:** 160
**Problema:** Docstring lista `JSONResponse` como prioridade #2, mas o código retorna `MsgSpecJSONResponse` primeiro.

**Solução Proposta:**
```python
async def negotiate_response(
    accept_header: str | None = None
) -> type[JSONResponse]:
    """
    Returns the best response class based on installed libraries.
    
    Priority:
    1. MsgSpecJSONResponse (if msgspec installed)
    2. JSONResponse (stdlib fallback)
    3. ORJSONResponse (if orjson installed)
    """
    # Reordenar conforme a documentação
    ...
```

---

## 6. Issues de Scripts e Utilitários

### 6.1 Hardcoded Paths
**Severidade:** WARNING
**Arquivos:** 
- `apply_fixes_ast.py` linha 113
- `apply_fixes_from_audit.py` linha 230
- `fix_llm_service2.py` linha 41

**Problema:** Caminhos absolutos hardcoded tornam o script não-portátil.

**Solução Proposta:**
```python
# Usar caminhos relativos ou argumentos de linha de comando
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_PROJECT_PATH = SCRIPT_DIR.parent / "resync"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PROJECT_PATH)
    args = parser.parse_args()
```

### 6.2 AST Unparse Strips Comments
**Severidade:** WARNING
**Arquivo:** `apply_fixes_ast.py`
**Linha:** 96
**Problema:** `ast.unparse()` remove todos os comentários e formatação original.

**Solução Proposta:**
```python
# Usar libcst em vez de ast para preservar comentários
import libcst as cst

class CommentPreservingTransformer(cst.CSTTransformer):
    def leave_Import(self, original, updated):
        # Preservar comentários
        return updated

# Ou usar abordagem line-based
def modify_file_in_place(filepath: str, modifications: list[tuple]):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Aplicar modificações preservando estrutura
    ...
```

---

## 7. Plano de Execução

### Fase 1: Issues Críticas (Immediately)
1. ✅ BM25 Insecure Deserialization - Requer análise de segurança
2. ✅ Semantic Cache Cross-User Leak - Requer schema change
3. ✅ Agent Manager Race Condition - Requer lock fix
4. ✅ EventBus Subscription Leak - Requer unsubscribe implementation

### Fase 2: High Priority (This Sprint)
1. datetime.utcnow Deprecation (48 occurrences) - Automatizável
2. TWSStore Double-Check Locking
3. Runtime Metrics Histogram/Gauge Bug
4. WebSocket Client Concurrency

### Fase 3: Medium Priority (Next Sprint)
1. Pagination Division by Zero
2. RAG Service Missing Await
3. ConnectionManager Race Condition
4. Logging Issues

### Fase 4: Low Priority (Backlog)
1. Documentation Fixes
2. Hardcoded Paths
3. Script Improvements
4. Regex Validation Optimization

---

## 8. Métricas de Progresso

| Fase | Issues | Status |
|------|--------|--------|
| Críticas (PR #35) | 2 | ✅ Corrigido |
| Fase 1 | 4 | ⏳ Pendente |
| Fase 2 | 4 | ⏳ Pendente |
| Fase 3 | 4 | ⏳ Pendente |
| Fase 4 | 4 | ⏳ Pendente |
| **Total** | **18** | **2✅ / 16⏳** |

---

## 9. Notas Adicionais

### 9.1 Dependencies Required
- libcst (para preservação de comentários)
- hashlib (built-in)
- asyncio (built-in)

### 9.2 Testes Recomendados
- Testar concurrency com múltiplos clients simultâneos
- Testar edge cases de pagination (limit=0, negative)
- Testar isolation de semantic cache entre usuários
- Testar memory leak de EventBus após múltiplas conexões

### 9.3 Riscos
- Mudanças em datetime podem afetar serialization de dados existentes
- Lock changes podem causar deadlocks se mal implementados
- Schema changes em cache requerem migration

---

*Documento gerado em: 2026-02-19*
*PR Reference: https://github.com/netover/6.0/pull/35*
