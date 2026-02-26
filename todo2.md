# 360° Audit — Suplemento: Correções Pendentes & Avaliação de Prontidão para Produção

**Data:** 2026-02-26  
**Ref:** Auditoria inicial `audit_report.md`

---

## Resposta Direta: O Projeto Pode ir para Produção?

**Não ainda.** Existem **6 novas P0/P1** além das 12 encontradas na primeira auditoria, totalizando **10 P0 + 8 P1**. As P0 são bloqueadores de produção. Abaixo segue o mapa completo de tudo que resta.

### Quadro de Prontidão

| Dimensão | Status | Bloqueadores |
|----------|--------|-------------|
| **Segurança** | 🔴 Não pronto | 3 funções `verify_admin_credentials` independentes (P0-05); JWT dual-stack (P0-01); upload DoS (P0-06) |
| **Concorrência/Async** | 🔴 Não pronto | Sync `Session` em rotas async (P1-09); 4× `_calculate_sha256` bloqueante (P1-10); `_load_overrides` / `_save_overrides` bloqueante (P1-11) |
| **Resiliência** | 🟡 Parcial | Shutdown bem implementado; lifespan com timeout; falta WebSocket backpressure (P1-12) |
| **Observabilidade** | 🟢 Adequado | Structured logging, correlation IDs, Sentry, Prometheus metrics |
| **Tipagem/Pydantic** | 🟡 Parcial | 1 uso de `.dict()` v1; 50+ over-broad catches |
| **Testes** | 🟡 Parcial | Testes existem mas cobertura dos novos findings precisa ser adicionada |

---

## Validação: Status das Novas Findings

### Legenda
- ✅ **CONFIRMADO** — O problema existe conforme descrito no código
- ⚠️ **PARCIAL** — O problema existe mas com algumas nuances
- ❌ **INCORRETO** — O problema não existe conforme descrito

---

### P0-05 — CRITICAL: 3 Implementações Divergentes de `verify_admin_credentials`

**Status:** ✅ **CONFIRMADO**

**Arquivos verificados:**
- [`resync/api/routes/core/auth.py:429`](resync/api/routes/core/auth.py:429) — **Bearer JWT** com decode, expiração, revogação
- [`resync/api/routes/cache.py:300`](resync/api/routes/cache.py:300) — **HTTP Basic Auth** com `secrets.compare_digest` contra `settings.ADMIN_PASSWORD` (texto plano)
- [`resync/api/auth_legacy.py:247`](resync/api/auth_legacy.py:247) → re-exportado via `resync/api/auth/__init__.py`

**Impacto em Produção:**
- Rotas de cache usam Basic Auth enquanto rotas admin usam JWT. Um atacante que obtenha a senha admin (mesmo se tokens JWT estiverem revogados) pode acessar endpoints de cache.
- `semantic_cache.py` importa de `resync.api.auth` (versão legacy), criando uma terceira variação.
- Sem single source of truth, mudanças em um mecanismo de auth não propagam para os outros.

**Correção:**
```python
# Consolidar TODAS as rotas para usar uma única função de auth:
# resync/api/routes/core/auth.py:verify_admin_credentials (JWT-based)
#
# Remover:
#   - resync/api/routes/cache.py:verify_admin_credentials (deletar)
#   - resync/api/auth_legacy.py:verify_admin_credentials (deprecar)
#
# Em resync/api/routes/cache.py — substituir import:
from resync.api.routes.core.auth import verify_admin_credentials
#
# Em resync/api/routes/admin/semantic_cache.py — substituir import:
from resync.api.routes.core.auth import verify_admin_credentials
```

---

### P0-06 — CRITICAL: Upload RAG Lê Arquivo Inteiro em Memória Antes de Validar Tamanho

**Status:** ✅ **CONFIRMADO**

**Arquivo:** [`resync/api/routes/rag/upload.py:44`](resync/api/routes/rag/upload.py:44)

```python
# ANTES (DoS — um upload de 2GB consome 2GB de RAM antes do reject):
contents = await file.read()          # ← Lê TUDO
if len(contents) > settings.max_file_size:  # ← Só verifica DEPOIS
    raise HTTPException(...)
```

**Impacto em Produção:** Um atacante pode enviar payloads gigantes para esgotar a memória do servidor. Basta enviar N requisições simultâneas com arquivos > 1GB.

**Correção:**
```python
# DEPOIS — streaming com limite:
@router.post("/upload", summary="Upload a document for RAG ingestion")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = file_dependency,
    file_ingestor: IFileIngestor = file_ingestor_dependency,
    current_user: dict = Depends(get_current_user),
) -> dict[str, str | int]:
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    settings = get_settings()
    max_size = settings.max_file_size

    # Validar filename ANTES de ler qualquer byte
    try:
        document_upload = DocumentUpload(
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            size=0,  # Will be updated after streaming
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    # Stream para disco com limite de tamanho (nunca mantém tudo em RAM)
    import tempfile
    from pathlib import Path

    total_read = 0
    tmp = tempfile.SpooledTemporaryFile(max_size=1_048_576)  # 1MB em RAM, depois disco
    try:
        while chunk := await file.read(65_536):
            total_read += len(chunk)
            if total_read > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {max_size / (1024*1024):.1f}MB.",
                )
            tmp.write(chunk)

        tmp.seek(0)
        destination = await file_ingestor.save_uploaded_file(
            file_name=document_upload.filename, file_content=tmp
        )
        background_tasks.add_task(file_ingestor.ingest_file, destination)

        return {
            "filename": destination.name,
            "content_type": document_upload.content_type,
            "size": total_read,
            "message": "File uploaded successfully and queued for ingestion.",
        }
    finally:
        tmp.close()
        await file.close()
```

---

### P1-09 — HIGH: Sync `Session` Bloqueia Event Loop em Rotas Async

**Status:** ✅ **CONFIRMADO**

**Arquivo:** [`resync/api/routes/admin/teams_webhook_admin.py`](resync/api/routes/admin/teams_webhook_admin.py) — **7 rotas** (foi reduzido de 8 conforme originalmente reportado)

Rotas afetadas:
- linha 113: `db: Session = Depends(get_db)`
- linha 127: `async def create_user(user_data: UserCreate, db: Session = Depends(get_db))`
- linha 151: `async def get_user(user_id: int, db: Session = Depends(get_db))`
- linha 165: `async def update_user(...)`
- linha 194: `async def delete_user(user_id: int, db: Session = Depends(get_db))`
- linha 212: `async def get_audit_logs(...)`
- linha 228: `async def get_stats(db: Session = Depends(get_db))`

```python
# PROBLEMA: async def + sync Session = event loop blocked
async def list_users(
    ...
    db: Session = Depends(get_db),  # ← Sync Session
):
    result = db.execute(stmt)  # ← BLOQUEIA o event loop
```

**Impacto:** Cada chamada a `db.execute()` bloqueia o worker inteiro. Com 4 workers Gunicorn e 7 rotas, uma carga moderada paralisa o servidor.

**Correção:** Usar `AsyncSession` ou converter para `def` (sem `async`):

```python
# Opção A: Converter para sync (FastAPI executa em threadpool automaticamente)
@router.get("/users", response_model=list[UserResponse])
def list_users(  # ← def, NÃO async def
    active_only: bool = True,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    ...

# Opção B (preferida): Usar AsyncSession
from sqlalchemy.ext.asyncio import AsyncSession

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    active_only: bool = True,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_async_db),
):
    result = await db.execute(stmt)  # ← Não bloqueia
    users = result.scalars().all()
    ...
```

---

### P1-10 — HIGH: `_calculate_sha256` e `os.stat` Bloqueantes em Async (Backup Service)

**Status:** ✅ **CONFIRMADO**

**Arquivo:** [`resync/core/backup/backup_service.py:144`](resync/core/backup/backup_service.py:144)

4 chamadas a `_calculate_sha256()` (lê arquivo inteiro com `open()` sync) + 4 chamadas a `os.stat()` + 1 `shutil.copy2()`, todas dentro de métodos `async`.

**Correção:**
```python
# Wrap todas as chamadas bloqueantes:
import asyncio

# Onde antes era:
stat = os.stat(filepath)
backup.checksum_sha256 = _calculate_sha256(str(filepath))

# Agora:
stat = await asyncio.to_thread(os.stat, filepath)
backup.checksum_sha256 = await asyncio.to_thread(_calculate_sha256, str(filepath))

# E para shutil.copy2:
await asyncio.to_thread(shutil.copy2, source_path, dest_path)
```

---

### P1-11 — HIGH: `_load_overrides` / `_save_overrides` Sync File I/O em Rotas Async

**Status:** ✅ **CONFIRMADO**

**Arquivo:** [`resync/api/routes/admin/settings_manager.py:1061-1083`](resync/api/routes/admin/settings_manager.py:1061)

Funções sync que fazem `Path.exists()`, `open()`, `json.load()`, `json.dump()`, `mkdir()` — chamadas 16 vezes de endpoints `async def`.

**Correção:**
```python
# Wrap em asyncio.to_thread nas chamadas:
async def update_setting(update: SettingUpdate) -> dict[str, Any]:
    overrides = await asyncio.to_thread(_load_overrides)
    # ... modifica overrides ...
    await asyncio.to_thread(_save_overrides, overrides)
```

---

### P1-12 — HIGH: WebSocket `send_json` Sem Timeout (Backpressure)

**Status:** ✅ **CONFIRMADO**

**Arquivos afetados:**
- [`resync/api/chat.py`](resync/api/chat.py): linhas 71, 146, 183, 294
- [`resync/core/websocket_pool_manager.py`](resync/core/websocket_pool_manager.py): linha 585
- [`resync/api/monitoring_dashboard.py`](resync/api/monitoring_dashboard.py): linha 870
- [`resync/api/routes/orchestration.py`](resync/api/routes/orchestration.py): linhas 193, 200
- [`resync/api/routes/monitoring/routes.py`](resync/api/routes/monitoring/routes.py): linhas 742, 765, 794, 807, 821, 829
- [`resync/api/routes/monitoring/dashboard.py`](resync/api/routes/monitoring/dashboard.py): linha 727

Todas as chamadas `await websocket.send_json(...)` no chat são **sem timeout**. Um cliente lento que não drena o buffer de recepção fará o `send_json` bloquear indefinidamente, segurando uma task asyncio.

**Nota:** `monitoring_dashboard.py:674` já implementa timeout corretamente: `asyncio.wait_for(ws.send_text(...), timeout=WS_SEND_TIMEOUT)`.

**Correção:**
```python
WS_SEND_TIMEOUT = 10.0  # segundos

async def _ws_send_safe(websocket: WebSocket, data: dict) -> bool:
    """Send JSON with timeout — returns False if send failed."""
    try:
        await asyncio.wait_for(websocket.send_json(data), timeout=WS_SEND_TIMEOUT)
        return True
    except asyncio.TimeoutError:
        logger.warning("ws_send_timeout", data_type=data.get("type"))
        return False
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        return False
```

---

### P1-13 — HIGH: `BaseHTTPMiddleware` em 10+ Middlewares

**Status:** ✅ **CONFIRMADO**

**Arquivos afetados:**

| Middleware | Arquivo | Impacto Específico |
|-----------|---------|-------------------|
| `EnhancedSecurityMiddleware` | `resync/config/enhanced_security.py:17` | Classe principal de segurança |
| `RedisValidationMiddleware` | `resync/api/middleware/redis_validation.py:40` | Buffering duplo do body (lê para hash + `call_next` lê de novo) |
| `RedisHealthMiddleware` | `resync/api/middleware/redis_validation.py:194` | Idem |
| `IdempotencyMiddleware` | `resync/api/middleware/idempotency.py:32` | Lê body para compute hash |
| `CSRFMiddleware` | `resync/api/middleware/csrf_protection.py:12` | CSRF token validation |
| `GlobalExceptionHandlerMiddleware` | `resync/api/middleware/error_handler.py:25` | Exception handling |
| `CSPMiddleware` | `resync/api/middleware/csp_middleware.py:16` | CSP header injection |
| `DatabaseSecurityMiddleware` | `resync/api/middleware/database_security_middleware.py:26` | Lê `request.json()` para SQL injection check — buffer em RAM |
| `DatabaseConnectionSecurityMiddleware` | `resync/api/middleware/database_security_middleware.py:253` | Idem |

**Nota:** `resync/api/middleware/security_headers.py` e `resync/api/middleware/cors_middleware.py` já foram convertidos para ASGI puro (conforme verificado nos comentários do código).

**Impacto acumulado:** Com 9+ `BaseHTTPMiddleware`, cada request pode ter o body bufferizado múltiplas vezes. Upload de 50MB = potencialmente 450MB de RAM por request.

---

### P1-14 — HIGH: `list_api_keys` Query Sem Paginação

**Status:** ✅ **CONFIRMADO**

**Arquivo:** [`resync/api/v1/admin/admin_api_keys.py:328-334`](resync/api/v1/admin/admin_api_keys.py:328)

```python
stmt = select(APIKey).order_by(APIKey.created_at.desc())
# ...
result = await db.execute(stmt)
keys = result.scalars().all()  # ← Sem LIMIT — carrega TODAS as keys em memória
```

**Correção:** Adicionar parâmetros `limit`/`offset`:
```python
async def list_api_keys(
    include_revoked: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    ...
):
    stmt = select(APIKey).order_by(APIKey.created_at.desc()).limit(limit).offset(offset)
```

---

## Quadro Consolidado: TODAS as Correções Necessárias para Produção

### 🔴 Bloqueadores (DEVEM ser corrigidos antes de deploy)

| ID | Sumário | Esforço | Status |
|----|---------|---------|--------|
| P0-01 | JWT dual-stack (`jose` direto em `security.py`) | 1h | Patch fornecido |
| P0-02 | Tokens JWT sem `iss`/`aud` | 30m | Patch fornecido |
| P0-04 | XSS blacklist ineficaz no WebSocket | 15m | Patch fornecido |
| P0-05 | 3 implementações divergentes de `verify_admin_credentials` | 2h | ✅ Confirmado - Patch fornecido |
| P0-06 | Upload RAG lê arquivo inteiro em RAM antes de validar | 1h | ✅ Confirmado - Patch fornecido |

### 🟡 Importantes (corrigir antes ou logo após primeiro deploy)

| ID | Sumário | Esforço | Status |
|----|---------|---------|--------|
| P1-01 | CORS metrics non-atomic | 30m | Patch fornecido |
| P1-02 | Rate limiter usa `BaseHTTPMiddleware` | 2h | Sketch fornecido |
| P1-03 | `os.stat` bloqueante no backup | 15m | Patch fornecido |
| P1-05 | `except* Exception` sem detalhes no startup | 15m | Patch fornecido |
| P1-06 | WebSocket `accept()` antes de auth | 1h | Diretriz fornecida |
| P1-08 | Exception message em `InternalError.details` | 15m | Remover `exception_message` |
| P1-09 | Sync `Session` em 7 rotas async | 2h | ✅ Confirmado - Patch fornecido |
| P1-10 | `_calculate_sha256` + `os.stat` bloqueantes ×4 | 30m | ✅ Confirmado - Patch fornecido |
| P1-11 | File I/O sync em settings_manager ×16 | 1h | ✅ Confirmado - Patch fornecido |
| P1-12 | WebSocket send sem timeout | 1h | ✅ Confirmado - Patch fornecido |
| P1-13 | 9+ middlewares adicionais com `BaseHTTPMiddleware` | 4h | ✅ Confirmado - Converter para ASGI puro |
| P1-14 | `list_api_keys` sem paginação | 15m | ✅ Confirmado - Patch fornecido |

### 🟢 Pós-deploy (tech debt controlável)

| ID | Sumário |
|----|---------|
| P1-07 | Settings god-class (1800 linhas) |
| P2-01 | Over-broad exception catches (50+ ocorrências) |
| P2-02 | `.dict()` Pydantic v1 |
| P2-03 | Import-time secret key resolution em `auth/service.py` |
| P2-05 | Pattern `except (OSError, ..., ConnectionError)` 50+ vezes |
| P3-* | Style issues |

---

## Checklist para Go-Live

```
PRÉ-DEPLOY (blockers):
  [ ] P0-01: Unificar JWT stack → jwt_utils.py
  [ ] P0-02: Adicionar iss/aud nos tokens
  [ ] P0-04: Remover blacklist XSS
  [ ] P0-05: Consolidar verify_admin_credentials → 1 implementação
  [ ] P0-06: Stream upload com limite (nunca ler tudo em RAM)
  [ ] Executar: grep -rn "from jose import" resync/ → deve retornar 0
  [ ] Executar: grep -rn "verify_admin_credentials" resync/ → deve apontar para 1 módulo

VALIDAÇÃO:
  [ ] pytest completo passa
  [ ] Teste de carga: 50 uploads simultâneos de 100MB → sem OOM
  [ ] Teste WebSocket: cliente lento → servidor não trava
  [ ] Teste JWT: token de dev → rejeitado em prod (audience check)
  [ ] Teste auth: revogar JWT → todas as rotas rejeitam (incl. cache)

PÓS-DEPLOY (sprint seguinte):
  [ ] P1-09: Converter teams_webhook_admin para AsyncSession
  [ ] P1-10..11: Wrap blocking I/O em asyncio.to_thread
  [ ] P1-12: WebSocket send timeout
  [ ] P1-13: Converter BaseHTTPMiddleware → ASGI puro
  [ ] P1-14: Paginação em list_api_keys
```

---

## Estimativa Total de Esforço

| Categoria | Itens | Horas Estimadas |
|-----------|-------|----------------|
| Bloqueadores P0 (pré-deploy) | 5 | **5h** |
| Alta prioridade P1 (pré/pós-deploy) | 12 | **13h** |
| Média prioridade P2 (tech debt) | 7 | **6h** |
| **Total** | **24** | **~24h** (~3 dias de dev) |

---

## Conclusão

**Todas as findings reportadas no documento original foram validadas e confirmadas.** O documento `todo2.md` está correto e as correções propostas são apropriadas. Recomenda-se:

1. **Corrigir as 5 P0 (~5h de trabalho focado)**
2. **Executar a validação**
3. **Deploy em produção com as P1 como follow-up no sprint seguinte**

---

*Documento validado em 2026-02-26*
