# Plano de correção sequencial — issues estáticas API (`resync/api`)

## Validação dos apontamentos

Resumo da análise dos tipos de issue:

1. **"Add an explicit default value to this optional field" (Pydantic)**
   - **Correto na maior parte dos casos**: os campos estão tipados como `T | None` mas sem `= None`, ficando obrigatórios (aceitam `null`, mas continuam obrigatórios).
   - A correção deve ser aplicada quando a intenção é campo realmente opcional no payload.

2. **"Use asynchronous features in this function or remove the `async` keyword"**
   - **Tecnicamente correto como code smell**, mas **contextual**: em FastAPI é válido manter `async def` mesmo sem `await` por padronização, assinatura uniforme e evolução futura.
   - Decisão recomendada: manter `async` apenas onde houver necessidade real/esperada de I/O assíncrono; converter para `def` onde for função puramente síncrona estável.

3. **"missing arguments em `create_validation_problem_detail`"**
   - **Correto e crítico**: chamada está sem argumento `title` obrigatório.

4. **"if/elif com blocos idênticos"**
   - **Correto**: lógica redundante com mesmo efeito em todos os ramos.

5. **"`asyncio.CancelledError` deve ser re-raise"**
   - **Correto**: `CancelledError` está sendo engolido por `break`, o que pode impedir cancelamento cooperativo adequado.

6. **"usar API de arquivo assíncrona em função async"**
   - **Correto como melhoria de robustez**: há escrita síncrona (`open`/`Path.open`) em função async; deve ir para `aiofiles` ou `anyio.to_thread.run_sync` com encapsulamento consistente.

---

## Plano detalhado de correção (arquivo por arquivo, sequencial)

### Fase 1 — Blockers e Bugs (ordem obrigatória)

1. **`resync/api/exception_handlers.py`**
   - Corrigir chamada de `create_validation_problem_detail(...)` adicionando `title`.
   - Validar contrato com `resync/api/models/responses.py`.
   - Executar teste local de handler de validação (erro 400) para garantir serialização correta.

2. **`resync/api/routes/admin/connectors.py`**
   - Refatorar `if/elif` com blocos idênticos (linha apontada) para fluxo único.
   - Se necessário, manter diferenciação apenas para logging/mensagens específicas por tipo.

3. **`resync/api/routes/enterprise/gateway.py`**
   - Nos workers de background, em `except asyncio.CancelledError`, executar cleanup e **re-levantar (`raise`)**.
   - Garantir que o shutdown encerra tasks corretamente sem mascarar cancelamento.

4. **`resync/api/routes/knowledge/ingest_api.py`**
   - Substituir uso de `open()` síncrono por abordagem assíncrona consistente.
   - Preferência: manter `anyio.to_thread.run_sync` sem lambda com `open` inline, encapsulando operações de arquivo em função dedicada.

5. **`resync/api/routes/rag/query.py`**
   - Trocar `Path.open("wb")` por escrita assíncrona (`aiofiles`) ou `anyio.to_thread.run_sync`.
   - Preservar tratamento atual de exceções HTTP.

---

### Fase 2 — Campos opcionais Pydantic sem default explícito

> Regra de correção: para cada campo `campo: X | None` que deve ser opcional, usar `campo: X | None = None`.

6. **`resync/api/agent_evolution_api.py`**
7. **`resync/api/routes/admin/backup.py`**
8. **`resync/api/routes/admin/connectors.py`**
9. **`resync/api/routes/admin/feedback_curation.py`**
10. **`resync/api/routes/admin/prompts.py`**
11. **`resync/api/routes/admin/semantic_cache.py`**
12. **`resync/api/routes/admin/teams_notifications_admin.py`**
13. **`resync/api/routes/admin/teams_webhook_admin.py`**
14. **`resync/api/routes/admin/users.py`**
15. **`resync/api/routes/enterprise/enterprise.py`**
16. **`resync/api/routes/monitoring/admin_monitoring.py`**
17. **`resync/api/routes/monitoring/ai_monitoring.py`**
18. **`resync/api/routes/monitoring/observability.py`**
19. **`resync/api/v1/admin/admin_api_keys.py`**

Para cada arquivo:
- Revisar semântica de cada campo (opcional real vs obrigatório anulável).
- Ajustar schema e validar impacto em OpenAPI.
- Ajustar testes de payload/schemas onde necessário.

---

### Fase 3 — `async def` sem operação assíncrona (padronização)

> Regra de decisão por função:
> - Converter para `def` quando for puramente síncrona e estável.
> - Manter `async def` se fizer parte de contrato async, endpoint FastAPI padronizado, ou previsão concreta de I/O assíncrono iminente.

20. **`resync/api/auth/repository.py`**
21. **`resync/api/auth_legacy.py`**
22. **`resync/api/core/security.py`**
23. **`resync/api/dependencies.py`**
24. **`resync/api/dependencies_v2.py`**
25. **`resync/api/enhanced_endpoints.py`**
26. **`resync/api/exception_handlers.py`** (itens de smell além do bug)
27. **`resync/api/middleware/error_handler.py`**
28. **`resync/api/middleware/idempotency.py`**
29. **`resync/api/monitoring_dashboard.py`**
30. **`resync/api/routes/audit.py`**
31. **`resync/api/routes/cache.py`**
32. **`resync/api/routes/enterprise/gateway.py`** (smells além de cancelamento)
33. **`resync/api/routes/monitoring/dashboard.py`**
34. **`resync/api/routes/teams_webhook.py`**
35. **`resync/api/security/__init__.py`**

Para cada arquivo/função:
- Confirmar se há `await`, `async with`, `async for` ou chamada de API async.
- Se não houver, converter para `def` e ajustar call-sites/tipagem.
- Se houver motivo arquitetural para manter `async`, registrar justificativa em comentário curto/ADR técnica.

---

## Estratégia de execução segura

1. Criar branch de correção incremental.
2. Aplicar Fase 1 em commits pequenos por arquivo crítico.
3. Aplicar Fase 2 em lotes por domínio (`admin`, `monitoring`, `enterprise`).
4. Aplicar Fase 3 por módulo para reduzir risco de quebra de assinatura.
5. Rodar validações após cada lote:
   - lint estático (ruff/mypy/pyright, conforme projeto)
   - testes unitários/integrados do módulo alterado
   - verificação de OpenAPI (campos opcionais)
6. Fechar com revisão final de contratos públicos (request/response) e changelog técnico.
