# Plano de Refatoração: asyncio.TaskGroup como Motor Central de Concorrência

## Visão Geral

Este plano detalha a implementação completa das 4 fases de refatoração para consolidar o `asyncio.TaskGroup` como motor central de concorrência, garantindo estabilidade do event loop, consistência de dados em shutdowns e performance sob alta concorrência.

---

## Fase 1: Desbloqueio de Inicialização (Severidade P0)

**Objetivo:** Eliminar o TypeError imediato no boot e restabelecer o fail-fast de compliance.

### 1.1 Refatorar `EncryptedAuditTrail.start()` (resync/core/encrypted_audit.py)

**Problema Identificado:**
- O método `start()` na linha 427 é síncrono (`def start(self)`) e não aceita parâmetros
- O `EnterpriseManager` já tenta chamar `self._encrypted_audit.start(tg=tg)` na linha 426 de manager.py
- Isso causa TypeError: `start() got an unexpected keyword argument 'tg'`

**Ações:**
1. Modificar assinatura do método `start()` para aceitar `tg: asyncio.TaskGroup | None = None`
2. Tornar o método async
3. Substituir o uso de `track_task()` por lógica condicional:
   - Se `tg` for fornecido: usar `tg.create_task()`
   - Se não for fornecido (fallback para testes): usar `asyncio.create_task()`

**Código Atual (linhas 427-450):**
```python
def start(self) -> None:
    """
    Start background workers.

    Must be called from an async context with a running event loop.
    """
    if self._running:
        return

    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError(
            "EncryptedAuditTrail.start() must be called from an async context "
            "with a running event loop"
        ) from exc

    self._running = True
    self._tasks = [
        track_task(self._flush_worker(), name="audit_flush_worker"),
        track_task(self._archival_worker(), name="audit_archival_worker"),
        track_task(self._verification_worker(), name="audit_verification_worker"),
    ]
    logger.info("Encrypted audit trail system started")
```

**Código Alvo:**
```python
async def start(self, tg: asyncio.TaskGroup | None = None) -> None:
    """
    Start background workers.

    Args:
        tg: Optional TaskGroup for structured concurrency.
             If provided, tasks will be managed by the TaskGroup for proper shutdown.
    """
    if self._running:
        return

    # Validar contexto async (apenas se tg não for fornecido, pois TaskGroup já requer contexto async)
    if not tg:
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "EncryptedAuditTrail.start() must be called from an async context "
                "with a running event loop"
            ) from exc

    self._running = True
    
    if tg:
        self._tasks = [
            tg.create_task(self._flush_worker(), name="audit_flush_worker"),
            tg.create_task(self._archival_worker(), name="audit_archival_worker"),
            tg.create_task(self._verification_worker(), name="audit_verification_worker"),
        ]
    else:
        # Fallback seguro apenas para testes isolados
        self._tasks = [
            asyncio.create_task(self._flush_worker(), name="audit_flush_worker"),
            asyncio.create_task(self._archival_worker(), name="audit_archival_worker"),
            asyncio.create_task(self._verification_worker(), name="audit_verification_worker"),
        ]
    logger.info("Encrypted audit trail system started")
```

---

## Fase 2: Unificação de Concorrência Estruturada (Severidade P1)

**Objetivo:** Eliminar tarefas "órfãs" e garantir Graceful Shutdown sem corromper transações.

### 2.1 Remover import e uso de `track_task` no EncryptedAuditTrail

**Arquivo:** resync/core/encrypted_audit.py

**Ações:**
1. Remover linha 26: `from resync.core.task_tracker import track_task`
2. A linha 26 deve ser completamente removida após a Fase 1, pois não será mais necessária

### 2.2 Adequar o OrchestrationRunner (resync/core/orchestration/runner.py)

**Problema Identificado:**
- Linha 83: `asyncio.create_task(self._run_loop(execution.id))` cria tarefa "solta" que não será aguardada no shutdown
- O método `start_execution` na linha 41-85 não aceita TaskGroup

**Ações:**
1. Adicionar parâmetro `tg: asyncio.TaskGroup | None = None` ao método `start_execution`
2. Armazenar `self.tg = tg` no construtor para uso posterior
3. Substituir `asyncio.create_task(...)` por lógica condicional

**Código Atual (linhas 31-43):**
```python
class OrchestrationRunner:
    """
    Executes orchestration workflows based on configuration and state strings.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.agent_adapter = AgentAdapter()
        self.event_bus = event_bus
```

**Código Alvo (construtor):**
```python
class OrchestrationRunner:
    """
    Executes orchestration workflows based on configuration and state strings.
    """

    def __init__(
        self, 
        session_factory: async_sessionmaker[AsyncSession],
        tg: asyncio.TaskGroup | None = None
    ):
        self.session_factory = session_factory
        self.agent_adapter = AgentAdapter()
        self.event_bus = event_bus
        self.tg = tg  # TaskGroup para execução de workflows
```

**Código Atual (linhas 79-85):**
```python
            # Start background processing
            # In a real system, this might be a Celery task or similar.
            # Here we run it as an asyncio task (fire and forget handled by caller or here?)
            # Ideally, we return trace_id immediately and run async.
            asyncio.create_task(self._run_loop(execution.id))

            return trace_id
```

**Código Alvo (método start_execution):**
```python
    async def start_execution(
        self, 
        config_id: UUID, 
        input_data: dict, 
        user_id: str | None = None,
        tg: asyncio.TaskGroup | None = None
    ) -> str:
        """
        Starts a new execution for a given config.
        Returns the trace_id.
        
        Args:
            config_id: The configuration ID to execute
            input_data: Input data for the workflow
            user_id: Optional user ID for audit
            tg: Optional TaskGroup to run the execution in
        """
        # ... código existente até a parte de criação de task ...
        
        # Start background processing
        if tg:
            tg.create_task(self._run_loop(execution.id), name=f"workflow_{trace_id}")
        else:
            # Fallback para backward compatibility
            asyncio.create_task(self._run_loop(execution.id))

        return trace_id
```

### 2.3 Revisar Lifespan (resync/core/startup.py)

**Verificação:** O startup.py já está corretamente implementado:
- Linha 706: `async with asyncio.TaskGroup() as bg_tasks:`
- Linha 710: `await get_tws_monitor(st.tws_client, tg=bg_tasks)`
- Linha 722: Nested TaskGroup `init_tg` para inicialização

**Ação necessária:** Verificar se EnterpriseManager.initialize() é chamado passando o TaskGroup do lifespan.

Procurar no código onde `enterprise_manager.initialize()` é chamado:
```python
# Deve ser algo como:
await enterprise_manager.initialize(tg=bg_tasks)
```

---

## Fase 3: Otimização de Memória e Estruturas (Severidade P2)

**Objetivo:** Prevenir Out of Memory (OOM), picos de Garbage Collection e lentidão em I/O sob stress.

### 3.1 Substituir Buffer Linear por Deque Atômico

**Arquivo:** resync/core/encrypted_audit.py
**Linhas:** 17-18 (imports), 376 (declaração)

**Ações:**
1. Adicionar import na linha 18: `import collections`
2. Modificar linha 376 de `self.pending_entries: list[AuditEntry] = []` para usar deque

**Código Atual (linha 17-18):**
```python
from dataclasses import dataclass
from pathlib import Path
```

**Código Alvo (imports):**
```python
from dataclasses import dataclass
from pathlib import Path
import collections
```

**Código Atual (linha 376):**
```python
self.pending_entries: list[AuditEntry] = []
```

**Código Alvo:**
```python
# Buffer principal usando deque com limite fixo (capacidade ajustável via config)
# maxlen garante que eventos antigos são descartados automaticamente em vez de crescer infinitamente
self._buffer: collections.deque[AuditEntry] = collections.deque(
    maxlen=self.config.max_memory_entries
)

# Mantemos pending_entries como property que retorna list(self._buffer) para backward compatibility
# OU podemos migrar todos os acessos para usar _buffer diretamente
```

### 3.2 Atualizar Métodos que Acessam o Buffer

**Método:** `log_event` (linhas 475-535)

Precisa ser atualizado para usar `_buffer` em vez de `pending_entries`:
```python
async def log_event(...) -> str:
    # ... código de criação de entry ...
    
    async with self._mem_lock:
        if self.config.enable_hash_chaining:
            entry.previous_hash = self.chain_hash
            entry.chain_hash = self._calculate_chain_hash(entry)
            self.chain_hash = entry.chain_hash

        if self.config.enable_signatures:
            entry.signature = self._calculate_signature(entry)

        entry.encryption_key_id = self.key_manager.get_active_key().key_id

        current_len = len(self._buffer)
        if current_len >= self.config.max_memory_entries:
            # O deque com maxlen já descarta automaticamente, mas podemos notificar
            now = time.time()
            if now - self._last_buffer_warning_time > self._buffer_warning_throttle_seconds:
                logger.warning(
                    "audit_buffer_full",
                    maxlen=self.config.max_memory_entries,
                    current_len=current_len + 1,
                )
                self._last_buffer_warning_time = now

        self._buffer.append(entry)  # Usa append ao invés de append
        self.total_entries += 1
```

### 3.3 Implementar Padrão de Swap Atômico no Flush

**Arquivo:** resync/core/encrypted_audit.py
**Método:** `_flush_pending_entries()` (linhas 956-1012)

**Problema:** O flush atual pode bloquear a estrutura durante I/O

**Solução - swap atômico:**
```python
async def _flush_pending_entries(self) -> None:
    """
    Flush pending entries to disk using atomic swap pattern.
    
    The swap pattern allows new entries to be appended to the buffer
    while flush is in progress, improving throughput under load.
    """
    # Fase 1: Swap atômico - remove todos os itens de uma vez
    async with self._mem_lock:
        if not self._buffer:
            return
        current_batch = list(self._buffer)
        self._buffer.clear()
    
    # Fase 2: I/O operations fora do lock
    # ... (lógica de escrita existente, usando current_batch) ...
    
    # Fase 3: Se flush falhou, readicionar ao buffer
    if flush_failed:
        async with self._mem_lock:
            self._buffer.extendleft(reversed(current_batch))
```

---

## Fase 4: Validação Final e Rollout

### 4.1 Teste de Stress de Shutdown

**Procedimento:**
1. Subir o servidor localmente: `uvicorn resync.main:app`
2. Disparar uma carga de requests:
   ```bash
   # Script de teste de carga
   for i in {1..100}; do 
       curl -X POST http://localhost:8000/api/v1/... 
   done &
   ```
3. Enviar sinal SIGINT (Ctrl+C)
4. Verificar nos logs:
   - O sistema deve registrar a espera pela conclusão/cancelamento das tarefas
   - "Database connections closed" deve aparecer APÓS as tarefas terminarem

**Logs esperados:**
```
INFO: Shutdown initiated, waiting for tasks...
INFO: audit_flush_worker completed
INFO: audit_archival_worker completed  
INFO: Database connections closed
```

### 4.2 Validação de Tipagem

**Comando:**
```bash
mypy --strict \
    resync/core/encrypted_audit.py \
    resync/core/orchestration/runner.py \
    resync/core/enterprise/manager.py
```

**Verificações necessárias:**
- [ ] Todas as funções devem ter retornos corretamente tipados
- [ ] Parâmetros TaskGroup devem ser opcionais (None por default)
- [ ] Nenhum Any implícito em caminhos críticos
- [ ] Compatibilidade com Python 3.11+ (ExceptionGroup)

### 4.3 Testes de Integração

**Executar testes existentes:**
```bash
pytest resync/tests/ -v -k "test_lifespan or test_startup" --tb=short
```

---

## Dependências e Ordem de Execução

```mermaid
graph TD
    A[Fase 1: P0] --> B[Fase 2: P1]
    B --> C[Fase 3: P2]
    C --> D[Fase 4: Validação]
    
    A1[Modificar start() aceitar tg] --> A2[Testar boot]
    B1[Remover track_task] --> B2[Adequar runner.py]
    C1[Substituir list por deque] --> C2[Implementar swap]
    
    A2 --> B1
    B2 --> C1
```

---

## Riscos e Mitigações

| Risco | Severidade | Mitigação |
|-------|------------|-----------|
| Quebra de backward compatibility | Alta | Manter fallback para `asyncio.create_task()` quando `tg` não for fornecido |
| Perda de eventos durante swap | Alta | Garantir que swap é atômico sob `_mem_lock`; readicionar em caso de falha |
| Tasks órfãs em testes | Média | Usar fixtures que garantem cleanup de TaskGroups |
| TypeError em chamadas existentes | Alta | Manter compatibilidade - tg é opcional |

---

## Resumo de Alterações por Arquivo

### resync/core/encrypted_audit.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 26 | Remover import de `track_task` | 2 |
| 18 | Adicionar `import collections` | 3 |
| 376 | Trocar `list` por `deque(maxlen=...)` | 3 |
| 427-450 | Modificar `start()` para async + aceitar tg | 1 |
| 508-531 | Usar `_buffer` em vez de `pending_entries` | 3 |
| 956-1012 | Implementar swap atômico no flush | 3 |

### resync/core/orchestration/runner.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 36-40 | Adicionar parâmetro `tg` ao `__init__` | 2 |
| 41-85 | Adicionar parâmetro `tg` ao `start_execution` | 2 |
| 83 | Trocar `asyncio.create_task` por condicional | 2 |

### resync/core/enterprise/manager.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 426 | Já está correto - chamar `start(tg=tg)` | 1 |

### resync/core/startup.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 706-750 | Já está correto | - |

---

## Checklist de Implementação

### Fase 1 (P0) - Desbloqueio de Inicialização
- [ ] Modificar assinatura de `EncryptedAuditTrail.start()` para `async def start(self, tg: asyncio.TaskGroup | None = None)`
- [ ] Substituir `track_task()` por lógica condicional com `tg.create_task()`
- [ ] Testar que o boot funciona sem TypeError

### Fase 2 (P1) - Unificação de Concorrência
- [ ] Adequar `OrchestrationRunner.__init__` para aceitar TaskGroup
- [ ] Adequar `OrchestrationRunner.start_execution` para aceitar TaskGroup
- [ ] Substituir `asyncio.create_task` por lógica condicional no runner
- [ ] Verificar integração do lifespan com EnterpriseManager

### Fase 3 (P2) - Otimização de Memória
- [ ] Importar `collections` em encrypted_audit.py
- [ ] Substituir `list` por `collections.deque(maxlen=...)`
- [ ] Atualizar `log_event` para usar `_buffer`
- [ ] Implementar padrão de swap atômico no flush

### Fase 4 - Validação
- [ ] Executar testes de integração
- [ ] Rodar mypy --strict
- [ ] Testar shutdown com carga
- [ ] Verificar logs de graceful shutdown

## Visão Geral

Este plano detalha a implementação completa das 4 fases de refatoração para consolidar o `asyncio.TaskGroup` como motor central de concorrência, garantindo estabilidade do event loop, consistência de dados em shutdowns e performance sob alta concorrência.

---

## Fase 1: Desbloqueio de Inicialização (Severidade P0)

**Objetivo:** Eliminar o TypeError imediato no boot e restabelecer o fail-fast de compliance.

### 1.1 Refatorar `EncryptedAuditTrail.start()` (resync/core/encrypted_audit.py)

**Problema Identificado:**
- O método `start()` na linha 427 é síncrono (`def start(self)`) e não aceita parâmetros
- O `EnterpriseManager` já tenta chamar `self._encrypted_audit.start(tg=tg)` na linha 426 de manager.py
- Isso causa TypeError: `start() got an unexpected keyword argument 'tg'`

**Ações:**
1. Modificar assinatura do método `start()` para aceitar `tg: asyncio.TaskGroup | None = None`
2. Tornar o método async
3. Substituir o uso de `track_task()` por lógica condicional:
   - Se `tg` for fornecido: usar `tg.create_task()`
   - Se não for fornecido (fallback para testes): usar `asyncio.create_task()`

**Código Atual (linhas 427-450):**
```python
def start(self) -> None:
    """
    Start background workers.

    Must be called from an async context with a running event loop.
    """
    if self._running:
        return

    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError(
            "EncryptedAuditTrail.start() must be called from an async context "
            "with a running event loop"
        ) from exc

    self._running = True
    self._tasks = [
        track_task(self._flush_worker(), name="audit_flush_worker"),
        track_task(self._archival_worker(), name="audit_archival_worker"),
        track_task(self._verification_worker(), name="audit_verification_worker"),
    ]
    logger.info("Encrypted audit trail system started")
```

**Código Alvo:**
```python
async def start(self, tg: asyncio.TaskGroup | None = None) -> None:
    """
    Start background workers.

    Args:
        tg: Optional TaskGroup for structured concurrency.
             If provided, tasks will be managed by the TaskGroup for proper shutdown.
    """
    if self._running:
        return

    # Validar contexto async (apenas se tg não for fornecido, pois TaskGroup já requer contexto async)
    if not tg:
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "EncryptedAuditTrail.start() must be called from an async context "
                "with a running event loop"
            ) from exc

    self._running = True
    
    if tg:
        self._tasks = [
            tg.create_task(self._flush_worker(), name="audit_flush_worker"),
            tg.create_task(self._archival_worker(), name="audit_archival_worker"),
            tg.create_task(self._verification_worker(), name="audit_verification_worker"),
        ]
    else:
        # Fallback seguro apenas para testes isolados
        self._tasks = [
            asyncio.create_task(self._flush_worker(), name="audit_flush_worker"),
            asyncio.create_task(self._archival_worker(), name="audit_archival_worker"),
            asyncio.create_task(self._verification_worker(), name="audit_verification_worker"),
        ]
    logger.info("Encrypted audit trail system started")
```

---

## Fase 2: Unificação de Concorrência Estruturada (Severidade P1)

**Objetivo:** Eliminar tarefas "órfãs" e garantir Graceful Shutdown sem corromper transações.

### 2.1 Remover import e uso de `track_task` no EncryptedAuditTrail

**Arquivo:** resync/core/encrypted_audit.py

**Ações:**
1. Remover linha 26: `from resync.core.task_tracker import track_task`
2. A linha 26 deve ser completamente removida após a Fase 1, pois não será mais necessária

### 2.2 Adequar o OrchestrationRunner (resync/core/orchestration/runner.py)

**Problema Identificado:**
- Linha 83: `asyncio.create_task(self._run_loop(execution.id))` cria tarefa "solta" que não será aguardada no shutdown
- O método `start_execution` na linha 41-85 não aceita TaskGroup

**Ações:**
1. Adicionar parâmetro `tg: asyncio.TaskGroup | None = None` ao método `start_execution`
2. Armazenar `self.tg = tg` no construtor para uso posterior
3. Substituir `asyncio.create_task(...)` por lógica condicional

**Código Atual (linhas 31-43):**
```python
class OrchestrationRunner:
    """
    Executes orchestration workflows based on configuration and state strings.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.agent_adapter = AgentAdapter()
        self.event_bus = event_bus
```

**Código Alvo (construtor):**
```python
class OrchestrationRunner:
    """
    Executes orchestration workflows based on configuration and state strings.
    """

    def __init__(
        self, 
        session_factory: async_sessionmaker[AsyncSession],
        tg: asyncio.TaskGroup | None = None
    ):
        self.session_factory = session_factory
        self.agent_adapter = AgentAdapter()
        self.event_bus = event_bus
        self.tg = tg  # TaskGroup para execução de workflows
```

**Código Atual (linhas 79-85):**
```python
            # Start background processing
            # In a real system, this might be a Celery task or similar.
            # Here we run it as an asyncio task (fire and forget handled by caller or here?)
            # Ideally, we return trace_id immediately and run async.
            asyncio.create_task(self._run_loop(execution.id))

            return trace_id
```

**Código Alvo (método start_execution):**
```python
    async def start_execution(
        self, 
        config_id: UUID, 
        input_data: dict, 
        user_id: str | None = None,
        tg: asyncio.TaskGroup | None = None
    ) -> str:
        """
        Starts a new execution for a given config.
        Returns the trace_id.
        
        Args:
            config_id: The configuration ID to execute
            input_data: Input data for the workflow
            user_id: Optional user ID for audit
            tg: Optional TaskGroup to run the execution in
        """
        # ... código existente até a parte de criação de task ...
        
        # Start background processing
        if tg:
            tg.create_task(self._run_loop(execution.id), name=f"workflow_{trace_id}")
        else:
            # Fallback para backward compatibility
            asyncio.create_task(self._run_loop(execution.id))

        return trace_id
```

### 2.3 Revisar Lifespan (resync/core/startup.py)

**Verificação:** O startup.py já está corretamente implementado:
- Linha 706: `async with asyncio.TaskGroup() as bg_tasks:`
- Linha 710: `await get_tws_monitor(st.tws_client, tg=bg_tasks)`
- Linha 722: Nested TaskGroup `init_tg` para inicialização

**Ação necessária:** Verificar se EnterpriseManager.initialize() é chamado passando o TaskGroup do lifespan.

Procurar no código onde `enterprise_manager.initialize()` é chamado:
```python
# Deve ser algo como:
await enterprise_manager.initialize(tg=bg_tasks)
```

---

## Fase 3: Otimização de Memória e Estruturas (Severidade P2)

**Objetivo:** Prevenir Out of Memory (OOM), picos de Garbage Collection e lentidão em I/O sob stress.

### 3.1 Substituir Buffer Linear por Deque Atômico

**Arquivo:** resync/core/encrypted_audit.py
**Linhas:** 17-18 (imports), 376 (declaração)

**Ações:**
1. Adicionar import na linha 18: `import collections`
2. Modificar linha 376 de `self.pending_entries: list[AuditEntry] = []` para usar deque

**Código Atual (linha 17-18):**
```python
from dataclasses import dataclass
from pathlib import Path
```

**Código Alvo (imports):**
```python
from dataclasses import dataclass
from pathlib import Path
import collections
```

**Código Atual (linha 376):**
```python
self.pending_entries: list[AuditEntry] = []
```

**Código Alvo:**
```python
# Buffer principal usando deque com limite fixo (capacidade ajustável via config)
# maxlen garante que eventos antigos são descartados automaticamente em vez de crescer infinitamente
self._buffer: collections.deque[AuditEntry] = collections.deque(
    maxlen=self.config.max_memory_entries
)

# Mantemos pending_entries como property que retorna list(self._buffer) para backward compatibility
# OU podemos migrar todos os acessos para usar _buffer diretamente
```

### 3.2 Atualizar Métodos que Acessam o Buffer

**Método:** `log_event` (linhas 475-535)

Precisa ser atualizado para usar `_buffer` em vez de `pending_entries`:
```python
async def log_event(...) -> str:
    # ... código de criação de entry ...
    
    async with self._mem_lock:
        if self.config.enable_hash_chaining:
            entry.previous_hash = self.chain_hash
            entry.chain_hash = self._calculate_chain_hash(entry)
            self.chain_hash = entry.chain_hash

        if self.config.enable_signatures:
            entry.signature = self._calculate_signature(entry)

        entry.encryption_key_id = self.key_manager.get_active_key().key_id

        current_len = len(self._buffer)
        if current_len >= self.config.max_memory_entries:
            # O deque com maxlen já descarta automaticamente, mas podemos notificar
            now = time.time()
            if now - self._last_buffer_warning_time > self._buffer_warning_throttle_seconds:
                logger.warning(
                    "audit_buffer_full",
                    maxlen=self.config.max_memory_entries,
                    current_len=current_len + 1,
                )
                self._last_buffer_warning_time = now

        self._buffer.append(entry)  # Usa append ao invés de append
        self.total_entries += 1
```

### 3.3 Implementar Padrão de Swap Atômico no Flush

**Arquivo:** resync/core/encrypted_audit.py
**Método:** `_flush_pending_entries()` (linhas 956-1012)

**Problema:** O flush atual pode bloquear a estrutura durante I/O

**Solução - swap atômico:**
```python
async def _flush_pending_entries(self) -> None:
    """
    Flush pending entries to disk using atomic swap pattern.
    
    The swap pattern allows new entries to be appended to the buffer
    while flush is in progress, improving throughput under load.
    """
    # Fase 1: Swap atômico - remove todos os itens de uma vez
    async with self._mem_lock:
        if not self._buffer:
            return
        current_batch = list(self._buffer)
        self._buffer.clear()
    
    # Fase 2: I/O operations fora do lock
    # ... (lógica de escrita existente, usando current_batch) ...
    
    # Fase 3: Se flush falhou, readicionar ao buffer
    if flush_failed:
        async with self._mem_lock:
            self._buffer.extendleft(reversed(current_batch))
```

---

## Fase 4: Validação Final e Rollout

### 4.1 Teste de Stress de Shutdown

**Procedimento:**
1. Subir o servidor localmente: `uvicorn resync.main:app`
2. Disparar uma carga de requests:
   ```bash
   # Script de teste de carga
   for i in {1..100}; do 
       curl -X POST http://localhost:8000/api/v1/... 
   done &
   ```
3. Enviar sinal SIGINT (Ctrl+C)
4. Verificar nos logs:
   - O sistema deve registrar a espera pela conclusão/cancelamento das tarefas
   - "Database connections closed" deve aparecer APÓS as tarefas terminarem

**Logs esperados:**
```
INFO: Shutdown initiated, waiting for tasks...
INFO: audit_flush_worker completed
INFO: audit_archival_worker completed  
INFO: Database connections closed
```

### 4.2 Validação de Tipagem

**Comando:**
```bash
mypy --strict \
    resync/core/encrypted_audit.py \
    resync/core/orchestration/runner.py \
    resync/core/enterprise/manager.py
```

**Verificações necessárias:**
- [ ] Todas as funções devem ter retornos corretamente tipados
- [ ] Parâmetros TaskGroup devem ser opcionais (None por default)
- [ ] Nenhum Any implícito em caminhos críticos
- [ ] Compatibilidade com Python 3.11+ (ExceptionGroup)

### 4.3 Testes de Integração

**Executar testes existentes:**
```bash
pytest resync/tests/ -v -k "test_lifespan or test_startup" --tb=short
```

---

## Dependências e Ordem de Execução

```mermaid
graph TD
    A[Fase 1: P0] --> B[Fase 2: P1]
    B --> C[Fase 3: P2]
    C --> D[Fase 4: Validação]
    
    A1[Modificar start() aceitar tg] --> A2[Testar boot]
    B1[Remover track_task] --> B2[Adequar runner.py]
    C1[Substituir list por deque] --> C2[Implementar swap]
    
    A2 --> B1
    B2 --> C1
```

---

## Riscos e Mitigações

| Risco | Severidade | Mitigação |
|-------|------------|-----------|
| Quebra de backward compatibility | Alta | Manter fallback para `asyncio.create_task()` quando `tg` não for fornecido |
| Perda de eventos durante swap | Alta | Garantir que swap é atômico sob `_mem_lock`; readicionar em caso de falha |
| Tasks órfãs em testes | Média | Usar fixtures que garantem cleanup de TaskGroups |
| TypeError em chamadas existentes | Alta | Manter compatibilidade - tg é opcional |

---

## Resumo de Alterações por Arquivo

### resync/core/encrypted_audit.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 26 | Remover import de `track_task` | 2 |
| 18 | Adicionar `import collections` | 3 |
| 376 | Trocar `list` por `deque(maxlen=...)` | 3 |
| 427-450 | Modificar `start()` para async + aceitar tg | 1 |
| 508-531 | Usar `_buffer` em vez de `pending_entries` | 3 |
| 956-1012 | Implementar swap atômico no flush | 3 |

### resync/core/orchestration/runner.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 36-40 | Adicionar parâmetro `tg` ao `__init__` | 2 |
| 41-85 | Adicionar parâmetro `tg` ao `start_execution` | 2 |
| 83 | Trocar `asyncio.create_task` por condicional | 2 |

### resync/core/enterprise/manager.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 426 | Já está correto - chamar `start(tg=tg)` | 1 |

### resync/core/startup.py

| Linha(s) | Alteração | Fase |
|----------|-----------|------|
| 706-750 | Já está correto | - |

---

## Checklist de Implementação

### Fase 1 (P0) - Desbloqueio de Inicialização
- [ ] Modificar assinatura de `EncryptedAuditTrail.start()` para `async def start(self, tg: asyncio.TaskGroup | None = None)`
- [ ] Substituir `track_task()` por lógica condicional com `tg.create_task()`
- [ ] Testar que o boot funciona sem TypeError

### Fase 2 (P1) - Unificação de Concorrência
- [ ] Adequar `OrchestrationRunner.__init__` para aceitar TaskGroup
- [ ] Adequar `OrchestrationRunner.start_execution` para aceitar TaskGroup
- [ ] Substituir `asyncio.create_task` por lógica condicional no runner
- [ ] Verificar integração do lifespan com EnterpriseManager

### Fase 3 (P2) - Otimização de Memória
- [ ] Importar `collections` em encrypted_audit.py
- [ ] Substituir `list` por `collections.deque(maxlen=...)`
- [ ] Atualizar `log_event` para usar `_buffer`
- [ ] Implementar padrão de swap atômico no flush

### Fase 4 - Validação
- [ ] Executar testes de integração
- [ ] Rodar mypy --strict
- [ ] Testar shutdown com carga
- [ ] Verificar logs de graceful shutdown

