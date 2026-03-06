# Revisão técnica — Resync v6.3.0 (lote 4: API core/middleware/models)

## Escopo auditado (40 arquivos)
- `resync/api/__init__.py`
- `resync/api/agent_evolution_api.py`
- `resync/api/agents.py`
- `resync/api/auth/__init__.py`
- `resync/api/auth/models.py`
- `resync/api/auth/repository.py`
- `resync/api/auth/service.py`
- `resync/api/auth_legacy.py`
- `resync/api/chat.py`
- `resync/api/core/__init__.py`
- `resync/api/core/config.py`
- `resync/api/core/security.py`
- `resync/api/dependencies.py`
- `resync/api/dependencies_v2.py`
- `resync/api/document_kg_admin.py`
- `resync/api/enhanced_endpoints.py`
- `resync/api/exception_handlers.py`
- `resync/api/graphrag_admin.py`
- `resync/api/middleware/__init__.py`
- `resync/api/middleware/compression.py`
- `resync/api/middleware/correlation_id.py`
- `resync/api/middleware/cors_config.py`
- `resync/api/middleware/cors_middleware.py`
- `resync/api/middleware/cors_monitoring.py`
- `resync/api/middleware/csp_middleware.py`
- `resync/api/middleware/csrf_protection.py`
- `resync/api/middleware/database_security_middleware.py`
- `resync/api/middleware/error_handler.py`
- `resync/api/middleware/idempotency.py`
- `resync/api/middleware/security_headers.py`
- `resync/api/middleware/valkey_validation.py`
- `resync/api/models/__init__.py`
- `resync/api/models/agents.py`
- `resync/api/models/auth.py`
- `resync/api/models/base.py`
- `resync/api/models/health.py`
- `resync/api/models/links.py`
- `resync/api/models/rag.py`
- `resync/api/models/requests.py`
- `resync/api/models/responses.py`

## Findings confirmados e status

_Status da auditoria: 2026-03-06 (itens abaixo foram corrigidos nesta rodada)._ 

1) **[🔴 Crítico] `APIRouter` usado antes do import em `document_kg_admin.py`** — **Corrigido**.
- Risco: `NameError` em import do módulo/registro de rotas.
- Correção: import ordenado e `router` definido após import.

2) **[🔴 Crítico] `settings` indefinido em `csrf_protection.py`** — **Corrigido**.
- Risco: `NameError` em runtime ao setar cookie CSRF.
- Correção: import explícito de `settings`.

3) **[⚪ Qualidade/Confiabilidade] Dependências FastAPI marcadas por B008 em pontos centrais** — **Tratado**.
- Correção: uso de dependências em constantes de módulo (`*_dependency`) ou anotação explícita quando intencional.

4) **[⚪ Qualidade] Alertas de segurança falsos positivos para literal OAuth2 `bearer`** — **Tratado**.
- Correção: `# noqa` pontual com justificativa nos campos de `token_type`.

## Observações
- A rodada também revalidou alertas fora do escopo estrito da diff anterior e aplicou correções quando havia risco prático de runtime.
