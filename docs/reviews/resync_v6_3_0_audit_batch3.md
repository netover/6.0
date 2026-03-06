# Revisão técnica — Resync v6.3.0 (lote 3: próximos 40 arquivos)

## Observação de runtime

- Revisão executada considerando **Python 3.14+**.
- Validação sintática local realizada com `PYENV_VERSION=3.14.0 python -m py_compile` sobre os 40 arquivos do escopo.

## Escopo auditado (40 arquivos)
1. `resync/api/routes/__init__.py`
2. `resync/api/routes/a2a.py`
3. `resync/api/routes/agents/__init__.py`
4. `resync/api/routes/agents/agents.py`
5. `resync/api/routes/audit.py`
6. `resync/api/routes/cache.py`
7. `resync/api/routes/core/__init__.py`
8. `resync/api/routes/core/auth.py`
9. `resync/api/routes/core/chat.py`
10. `resync/api/routes/core/health.py`
11. `resync/api/routes/core/ip_utils.py`
12. `resync/api/routes/core/status.py`
13. `resync/api/routes/cors_monitoring.py`
14. `resync/api/routes/endpoints.py`
15. `resync/api/routes/enterprise/__init__.py`
16. `resync/api/routes/enterprise/enterprise.py`
17. `resync/api/routes/enterprise/gateway.py`
18. `resync/api/routes/feedback_submit.py`
19. `resync/api/routes/health.py`
20. `resync/api/routes/knowledge/__init__.py`
21. `resync/api/routes/knowledge/ingest_api.py`
22. `resync/api/routes/monitoring/__init__.py`
23. `resync/api/routes/monitoring/admin_monitoring.py`
24. `resync/api/routes/monitoring/ai_monitoring.py`
25. `resync/api/routes/monitoring/dashboard.py`
26. `resync/api/routes/monitoring/metrics.py`
27. `resync/api/routes/monitoring/metrics_dashboard.py`
28. `resync/api/routes/monitoring/observability.py`
29. `resync/api/routes/monitoring/prometheus_exporter.py`
30. `resync/api/routes/monitoring/routes.py`
31. `resync/api/routes/orchestration.py`
32. `resync/api/routes/performance.py`
33. `resync/api/routes/rag/__init__.py`
34. `resync/api/routes/rag/query.py`
35. `resync/api/routes/rag/upload.py`
36. `resync/api/routes/rfc_examples.py`
37. `resync/api/routes/system/__init__.py`
38. `resync/api/routes/system/config.py`
39. `resync/api/routes/teams_webhook.py`
40. `resync/api/dependencies.py`

## Findings confirmados

_Status da auditoria: 2026-03-06 (snapshot; itens podem ter sido corrigidos depois)._ 

### 1) [🔴 Crítico] Falha de runtime no rate-limit de feedback (símbolos não importados)

- `feedback_submit.py` usa `with_timeout` e `classify_exception`, mas o módulo não importa esses símbolos.
- Impacto: em caminho de rate-limit/erro do Valkey, pode ocorrer `NameError` e resposta 500 em produção.
- Referências: `resync/api/routes/feedback_submit.py` linhas 28-33 e 106-109 (estado no momento da auditoria).

### 2) [🟠 Segurança] Exposição de detalhes internos no endpoint de readiness/LLM health

- O endpoint retorna `detail=f"...{e}"` para erros de banco, Valkey e LLM.
- Impacto: vaza mensagens internas de infraestrutura/driver para clientes, facilitando reconnaissance.
- Referências: `resync/api/routes/health.py` linhas 39-43, 57-58 e 86-87 (estado no momento da auditoria).

### 3) [🟠 Segurança] Exposição de detalhes internos no health core

- `core/health.py` também inclui exceção original em `detail` (`Health check system error: ...`).
- Impacto: mesma classe de vazamento de informação operacional/sensível em superfície pública de health.
- Referências: `resync/api/routes/core/health.py` linha 166 (estado no momento da auditoria).

### 4) [⚪ Qualidade] Import duplicado em `feedback_submit.py`

- Há duplicação de `from __future__ import annotations`.
- Impacto: não quebra runtime, mas reduz qualidade e sinaliza falha de lint/revisão.
- Referências: `resync/api/routes/feedback_submit.py` linhas 17-18 (estado no momento da auditoria).

### 5) [⚪ Qualidade/Observabilidade] Logging com PII no webhook de Teams

- Evento registra `user_email` e preview da mensagem em log estruturado.
- Impacto: aumenta risco de exposição de dados pessoais em pipeline de logs/monitoramento.
- Referências: `resync/api/routes/teams_webhook.py` linhas 213-219.

## Pesquisa de versões (internet)
Consulta direta ao PyPI JSON API para componentes centrais:
- Latest confirmados: fastapi `0.135.1`, sqlalchemy `2.0.48`, asyncpg `0.31.0`, pydantic `2.12.5`, uvicorn `0.41.0`, httpx `0.28.1`, langchain `1.2.10`, langgraph `1.0.10`, valkey `6.1.1`.
- Classifier explícito `Python :: 3.14` encontrado em parte dos pacotes (ex.: fastapi/asyncpg/pydantic/uvicorn/langchain), mas não em todos (ex.: sqlalchemy/httpx/langgraph/valkey). Isso **não** prova incompatibilidade por si só; apenas ausência de marcador explícito no metadata do PyPI.
