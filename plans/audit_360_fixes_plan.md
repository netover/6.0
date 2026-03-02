# Plano Detalhado de Correções - 360° Audit netover/6.0

## Visão Geral

Este documento detalha o plano de correção para os problemas identificados no audit 360° da branch `new1`. As correções são organizadas por severidade e dependências.

---

## 🎯 Fase 1: P0 Critical (Correções Imediatas)

### P0-01: Duplicate `except asyncio.TimeoutError` em `send_personal_message`

**Arquivo:** [`resync/core/websocket_pool_manager.py`](resync/core/websocket_pool_manager.py:456)

**Problema:** Dois blocos `except` consecutivos para `TimeoutError`, sendo o segundo código morto.

**Correção:**

```python
# Substituir linhas 456-472 por:
except asyncio.TimeoutError:
    logger.warning(
        "ws_send_timeout client=%s timeout=%.1fs — marking error and removing",
        client_id,
        self._send_timeout_seconds,
    )
    conn_info.mark_error()
    await self._remove_connection(client_id)
    return False
```

**Ação Required:**
- [ ] Remover bloco duplicado nas linhas 464-472
- [ ] Adicionar logging de warning com timeout info
- [ ] Garantir que `conn_info.mark_error()` seja chamado

---

### P0-02: `broadcast_json` sem `asyncio.wait_for` (DoS Vulnerability)

**Arquivo:** [`resync/core/websocket_pool_manager.py`](resync/core/websocket_pool_manager.py:652)

**Problema:** `await conn_info.websocket.send_json(data)` sem timeout permite clientes lentos bloquearem o event loop.

**Correção:**

```python
# Na função _send_json_with_error_handling, substituir linha 652:
# DE:
await conn_info.websocket.send_json(data)
# PARA:
await asyncio.wait_for(
    conn_info.websocket.send_json(data),
    timeout=self._send_timeout_seconds,
)
```

**Ação Required:**
- [ ] Envolver `send_json` com `asyncio.wait_for`
- [ ] O `except asyncio.TimeoutError` existente na linha 659 agora funcionará corretamente

---

### P0-04: `yield` dentro de `asyncio.TaskGroup` sem tratamento de exceções

**Arquivo:** [`resync/core/startup.py`](resync/core/startup.py:803)

**Problema:** O `yield` na linha 883 está dentro do TaskGroup. Se uma background task falhar, a app morre sem graceful shutdown.

**Correção:**

```python
# Modificar a estrutura do TaskGroup (linhas 803-885):
async with asyncio.TaskGroup() as bg_tasks:
    app.state.bg_tasks = bg_tasks
    # ... setup code ...
    
    try:
        yield  # app runs here
    except* Exception as eg:
        # Agora captura exceções de todas as background tasks
        for exc in eg.exceptions:
            logger.critical(
                "bg_task_crashed",
                error=str(exc),
                type=type(exc).__name__,
            )
        # Não faz raise - app já vai para shutdown
```

**Ação Required:**
- [ ] Adicionar `try-except*` ao redor do `yield`
- [ ] Logar exceções de background tasks com nível crítico
- [ ] Garantir graceful shutdown

---

## 🎯 Fase 2: P1 High (Correções de Curto Prazo)

### P1-01: Atributo inexistente `STARTUP_TCP_CHECK_TIMEOUT`

**Arquivo:** [`resync/core/startup.py:208`](resync/core/startup.py:208)

**Problema:** Usa `getattr(settings, "STARTUP_TCP_CHECK_TIMEOUT", 3.0)` que sempre retorna default.

**Correção:**

```python
# Linha 208 - corrigir para:
raw_timeout = getattr(settings, "tws_timeout_connect", 3.0)
```

**Ação Required:**
- [ ] Corrigir nome do atributo de `STARTUP_TCP_CHECK_TIMEOUT` para `tws_timeout_connect`

---

### P1-02: Atributo inexistente `RAG_SERVICE_TIMEOUT`

**Arquivo:** [`resync/core/startup.py:318`](resync/core/startup.py:318)

**Problema:** Usa `getattr(settings, "RAG_SERVICE_TIMEOUT", 5.0)` que sempre retorna default.

**Correção:**

```python
# Linha 318 - corrigir para:
timeout = float(getattr(settings, "rag_service_timeout", 5.0))
```

**Ação Required:**
- [ ] Corrigir nome do atributo de `RAG_SERVICE_TIMEOUT` para `rag_service_timeout`

---

### P1-03: Race Condition em `get_redis_initializer()`

**Arquivo:** [`resync/core/redis_init.py:462`](resync/core/redis_init.py:462)

**Problema:** Dois coroutines concorrentes podem criar múltiplas instâncias de `RedisInitializer`.

**Correção:**

```python
# Adicionar lock global no topo do arquivo:
import threading

_redis_initializer_create_lock = threading.Lock()

# Modificar a função:
def get_redis_initializer() -> RedisInitializer:
    global _redis_initializer
    if _redis_initializer is None:
        with _redis_initializer_create_lock:
            if _redis_initializer is None:
                _redis_initializer = RedisInitializer()
    return _redis_initializer
```

**Ação Required:**
- [ ] Adicionar import de `threading`
- [ ] Criar lock global `_redis_initializer_create_lock`
- [ ] Implementar double-checked locking

---

### P1-05: Interface `IContainer.dispose()` async vs sync mismatch

**Arquivo:** [`resync/api_gateway/container.py`](resync/api_gateway/container.py:39)

**Problema:** Interface declara `def dispose(self) -> None` (sync), mas implementação é `async def dispose(self)`.

**Correção:**

```python
# Na interface IContainer (linha 39):
# DE:
@abstractmethod
def dispose(self) -> None:
# PARA:
@abstractmethod
async def dispose(self) -> None:
```

**Ação Required:**
- [ ] Mudar assinatura de `dispose()` na interface para `async`
- [ ] Verificar callers de `container.dispose()` para garantir que usam `await`

---

### P1-07: `optional_timeout` conflita com timeouts internos

**Arquivo:** [`resync/core/startup.py:815`](resync/core/startup.py:815)

**Problema:** `max(0.5, min(3.0, float(startup_timeout) / 2.0))` sempre retorna max 3s, mas funções internas têm timeout de 10-15s.

**Correção:**

```python
# Linha 815 - corrigir para:
optional_timeout = max(5.0, float(startup_timeout) * 0.8)
# ou simplesmente remover o min():
optional_timeout = float(startup_timeout) * 0.8
```

**Ação Required:**
- [ ] Remover o `min(3.0, ...)` que limita a 3 segundos
- [ ] Ajustar para usar percentage do startup_timeout

---

### P1-08: `Semaphore(100)` hardcoded em `broadcast_json`

**Arquivo:** [`resync/core/websocket_pool_manager.py:615`](resync/core/websocket_pool_manager.py:615)

**Problema:** `broadcast()` usa `self._broadcast_concurrency` mas `broadcast_json()` hardcoda 100.

**Correção:**

```python
# Linha 615 - corrigir para:
sem = asyncio.Semaphore(self._broadcast_concurrency)  # era hardcoded 100
```

**Ação Required:**
- [ ] Usar `self._broadcast_concurrency` em vez de 100

---

## 🎯 Fase 3: P2 Medium (Correções de Médio Prazo)

### P2-07: Duplicate `TimeoutError` em exception tuple

**Arquivo:** [`resync/core/startup.py:131`](resync/core/startup.py:131)

**Problema:** `TimeoutError` já é capturado na linha 127, mas também está na tuple da linha 131.

**Correção:**

```python
# Linha 131 - remover TimeoutError da tuple:
except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, ConnectionError) as e:
```

**Mesma correção para linha 160** (`_http_healthy` function).

**Ação Required:**
- [ ] Remover `TimeoutError` da tuple na linha 131
- [ ] Remover `TimeoutError` da tuple na linha 160

---

## 🎯 Fase 4: P3 Low (Correções de Estilo)

### P3-04: `except Exception:` para import de orjson

**Arquivo:** [`resync/core/websocket_pool_manager.py:7`](resync/core/websocket_pool_manager.py:7)

**Problema:** Usa `except Exception:` em vez de `except ImportError:`.

**Correção:**

```python
# Linha 7 - corrigir para:
try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore
```

**Ação Required:**
- [ ] Mudar `except Exception:` para `except ImportError:`

---

## 📋 Ordem de Execução Recomendada

```
FASE 1 (P0 - Imediato)
├── P0-01: Remover duplicate except TimeoutError
├── P0-02: Adicionar asyncio.wait_for em broadcast_json
└── P0-04: Adicionar exception handling no TaskGroup yield

FASE 2 (P1 - Curto Prazo)
├── P1-01: Corrigir tws_timeout_connect attribute
├── P1-02: Corrigir rag_service_timeout attribute
├── P1-03: Adicionar lock em get_redis_initializer
├── P1-05: Corrigir dispose() async signature
├── P1-07: Corrigir optional_timeout calculation
└── P1-08: Usar broadcast_concurrency em broadcast_json

FASE 3 (P2 - Médio Prazo)
└── P2-07: Remover duplicate TimeoutError

FASE 4 (P3 - Estilo)
└── P3-04: Corrigir except ImportError
```

---

## 🔍 Dependências e Riscos

| Correção | Depende de | Risco | Mitigação |
|----------|-----------|-------|-----------|
| P0-02 | P0-01 (precisa do timeout var) | Baixo | Isolado ao método |
| P0-04 | Nenhuma | Médio | Requer teste de shutdown |
| P1-03 | Nenhuma | Médio | Requer teste de concorrência |
| P1-05 | Verificar callers | Alto | Buscar todos os usos de dispose() |
| P1-07 | Nenhuma | Baixo | Mudança simples |

---

## ✅ Checklist de Validação Pré-Deploy

- [ ] `pytest -x resync/tests/ -k websocket` — verificar sem TimeoutError duplicado
- [ ] `pytest -x resync/tests/ -k redis` — verificar health loop
- [ ] `python -m compileall resync/` — sem erros de sintaxe
- [ ] `ruff check resync/core/websocket_pool_manager.py --select B015,F704`
- [ ] `ruff check resync/core/startup.py --select B014`
- [ ] Smoke test: subir com `RESYNC_DISABLE_REDIS=1`
- [ ] Teste de carga WS: 200 conexões simultâneas
- [ ] Verificar que `APP_SECRET_KEY` levanta erro claro se ausente

---

## 📝 Notas Adicionais

1. **P0-03 já está corrigido** - O validator existe em [`settings_validators.py:448`](resync/settings_validators.py:448)
2. **P2-01 (dead imports)** - Verificado que `threading`, `ClassVar`, `Iterator` são usados; não é necessária correção
3. **P1-04 e P1-06** - Não validados completamente; podem necessitar de investigação adicional
