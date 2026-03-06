# Revisão técnica — Resync v6.3.0 (lote 2: próximos 30 arquivos)

## Escopo auditado (30 arquivos)
1. `resync/api/routes/admin/environment.py`
2. `resync/api/routes/admin/teams.py`
3. `resync/api/routes/admin/connectors.py`
4. `resync/api/routes/admin/users.py`
5. `resync/api/routes/admin/main.py`
6. `resync/api/routes/admin/teams_notifications_admin.py`
7. `resync/api/routes/admin/routing.py`
8. `resync/api/routes/admin/settings_manager.py`
9. `resync/api/routes/admin/config.py`
10. `resync/api/routes/admin/threshold_tuning.py`
11. `resync/api/routes/admin/skills.py`
12. `resync/api/routes/admin/notification_admin.py`
13. `resync/api/routes/admin/semantic_cache.py`
14. `resync/api/routes/admin/prompts.py`
15. `resync/api/routes/admin/llm_metrics.py`
16. `resync/api/routes/admin/rag_stats.py`
17. `resync/api/routes/admin/feedback_curation.py`
18. `resync/api/routes/admin/litellm_health.py`
19. `resync/api/routes/admin/backup.py`
20. `resync/api/routes/admin/litellm_config.py`
21. `resync/api/routes/admin/tws_instances.py`
22. `resync/api/routes/admin/rag_reranker.py`
23. `resync/api/routes/admin/tasks.py`
24. `resync/api/routes/admin/teams_webhook_admin.py`
25. `resync/api/routes/admin/v2.py`
26. `resync/api/routes/health.py`
27. `resync/api/routes/performance.py`
28. `resync/api/routes/audit.py`
29. `resync/api/routes/orchestration.py`
30. `resync/api/routes/feedback_submit.py`

## Findings confirmados

### 1) [🔴 Crítico] Erro de runtime no rate-limit de feedback por símbolos não importados
- `feedback_submit.py` usa `with_timeout` e `classify_exception`, mas não importa esses símbolos no módulo.
- Impacto em produção: caminho de rate-limit pode disparar `NameError`, gerando falha 500 no endpoint de feedback.

### 2) [🟠 Segurança] Path traversal/leitura arbitrária em streaming de logs
- Em `/admin/logs/stream`, o parâmetro `file` entra direto em `logs_dir / file` sem validação de containment (por exemplo `../` ou caminho absoluto).
- Impacto em produção: leitura não autorizada de arquivos locais se endpoint estiver acessível a credenciais comprometidas.

### 3) [🟠 Segurança] Exposição de detalhes internos em health endpoints
- `health.py` retorna `detail` com interpolação da exceção (`{e}`) para falhas de DB/Valkey/LLM deep check.
- Impacto em produção: vazamento de informações internas (mensagens de driver, hostnames, erro de autenticação), aumentando superfície de reconhecimento.

### 4) [⚪ Qualidade] Duplicidade de import `from __future__ import annotations`
- O mesmo import aparece duas vezes no topo de `feedback_submit.py`.
- Impacto: não quebra runtime, mas indica falha de higiene de código e revisão estática.

## Pesquisa de versões (internet + lock local)
Fonte local: `requirements.txt` e `requirements-dev.txt`.
Fonte externa: consulta direta ao PyPI JSON API.

Principais observações:
- Alinhados com lock local: `fastapi 0.135.1`, `starlette 0.52.1`, `uvicorn 0.41.0`, `sqlalchemy 2.0.48`, `alembic 1.18.4`, `asyncpg 0.31.0`, `valkey 6.1.1`, `langgraph 1.0.10`, `langchain 1.2.10`.
- Defasagem relevante observada no lock local vs PyPI atual:
  - `openai`: local `1.63.2` vs PyPI `2.26.0`
  - `anthropic`: local `0.79.0` vs PyPI `0.84.0`
  - `cohere`: local `5.15.0` vs PyPI `5.20.7`
  - `google-generativeai`: local `0.8.3` vs PyPI `0.8.6`
  - `httpx`: local `0.27.2` vs PyPI `0.28.1`
  - `pydantic`: local `2.12.0` vs PyPI `2.12.5`
  - `pydantic-settings`: local `>=2.10.1,<3.0.0` vs PyPI `2.13.1`

Nota: defasagem de versão não é bug por si só; requer decisão de risco/compatibilidade antes de upgrade.
