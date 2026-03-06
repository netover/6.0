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
# Revisão técnica — Resync v6.3.0 (lote 4: próximos 40 arquivos)

## Observação
- Revisão executada com foco em runtime **Python 3.14+**.
- Validação sintática: `PYENV_VERSION=3.14.0 python -m py_compile` nos 40 arquivos deste lote.

## Escopo auditado (40 arquivos)
1. `resync/api/__init__.py`
2. `resync/api/agent_evolution_api.py`
3. `resync/api/agents.py`
4. `resync/api/auth/__init__.py`
5. `resync/api/auth/models.py`
6. `resync/api/auth/repository.py`
7. `resync/api/auth/service.py`
8. `resync/api/auth_legacy.py`
9. `resync/api/chat.py`
10. `resync/api/core/__init__.py`
11. `resync/api/core/config.py`
12. `resync/api/core/security.py`
13. `resync/api/dependencies.py`
14. `resync/api/dependencies_v2.py`
15. `resync/api/document_kg_admin.py`
16. `resync/api/enhanced_endpoints.py`
17. `resync/api/exception_handlers.py`
18. `resync/api/graphrag_admin.py`
19. `resync/api/middleware/__init__.py`
20. `resync/api/middleware/compression.py`
21. `resync/api/middleware/correlation_id.py`
22. `resync/api/middleware/cors_config.py`
23. `resync/api/middleware/cors_middleware.py`
24. `resync/api/middleware/cors_monitoring.py`
25. `resync/api/middleware/csp_middleware.py`
26. `resync/api/middleware/csrf_protection.py`
27. `resync/api/middleware/database_security_middleware.py`
28. `resync/api/middleware/error_handler.py`
29. `resync/api/middleware/idempotency.py`
30. `resync/api/middleware/security_headers.py`
31. `resync/api/middleware/valkey_validation.py`
32. `resync/api/models/__init__.py`
33. `resync/api/models/agents.py`
34. `resync/api/models/auth.py`
35. `resync/api/models/base.py`
36. `resync/api/models/health.py`
37. `resync/api/models/links.py`
38. `resync/api/models/rag.py`
39. `resync/api/models/requests.py`
40. `resync/api/models/responses.py`

## Findings confirmados

### 1) [🔴 Crítico] Dependência de autenticação chama coroutine sem `await`
- Em `dependencies_v2.py`, `verify_admin_credentials` (função async) é chamada sem `await`.
- Efeito prático: `username` recebe um objeto coroutine (truthy), podendo marcar request como autenticada com valor inválido e quebrar o fluxo de autorização/auditoria.
- Impacto em produção: risco de bypass lógico de autenticação e comportamento inconsistente em endpoints que confiam nessa dependência.

### 2) [🟡 Resiliência] Verificação de token pode gerar 500 por claims ausentes
- Em `auth/service.py`, após `jwt.decode`, o código acessa claims obrigatórias via índice (`payload["sub"]`, `payload["username"]`, etc.) e só captura `JWTError`.
- Se o payload decodificado não tiver alguma claim esperada (ex.: token legado/incompleto), pode ocorrer `KeyError`/`TypeError` e subir 500 ao invés de rejeição controlada (401/None).
- Impacto em produção: indisponibilidade parcial/autenticação instável com tokens não conformes.

## Nota
- Não foram propostas mudanças de lógica de negócio neste lote; apenas identificação de riscos confirmados no código.
