# Plano de Migração Enterprise para `tws_unified` (Zero-Pendência)

## Objetivo
Migrar o ecossistema TWS para um padrão enterprise (contrato canônico + resiliência + segurança de segredos + concorrência segura em uvloop/FastAPI), **sem mascaramento de problemas**, e encerrar cada rodada somente com **zero pendência** nas duas camadas obrigatórias.

## Escopo coberto
- `resync/services/tws_unified.py`
- `resync/services/tws_service.py`
- Callsites em `api/routes/*`, `core/startup.py`, `core/*monitor*`, `core/langgraph/*`.
- Testes e mocks (`MockTWSClient`, integração e concorrência)
- Instrumentação, métricas e rollback operacional.

---

## Princípios mandatórios
1. **Sem supressão global nova** (`# mypy: ignore-errors`, `# pylint: skip-file`).
2. **Compatibilidade progressiva** via adapter/feature flag.
3. **Falha segura**: segredo protegido, estado limpo após erro de conexão.
4. **Rodada só encerra com zero pendência** em duas camadas:
   - Pattern scan amplo (ruff/mypy/pylint + regex);
   - Checklist semântico (5 categorias obrigatórias).

---

## Fase 0 — Baseline e congelamento de risco

### Entregáveis
- Inventário de contratos TWS atuais (métodos expostos reais x métodos esperados de domínio).
- Métricas base (erro/latência/retry/circuit-open).
- Feature flag: `TWS_UNIFIED_V2_ENABLED`.

### Critérios de aceite
- Rollback pronto via flag (sem rollback de deploy).
- Matriz de compatibilidade de métodos documentada.

---

## Fase 1 — Contrato canônico e adapter

### Ações
- Definir `ITWSUnifiedClient` (contrato estável do domínio):
  - `get_system_status`
  - `get_engine_info`
  - `get_jobs`
  - `get_job`
  - `get_job_status`
  - `get_workstations`
  - `get_plan`
- Implementar `OptimizedTWSClientAdapter` mapeando APIs reais (`query_*`, `get_jobstream`, etc.) para o contrato canônico.
- Tipagem explícita nos retornos (`dict[str, Any]`, `list[dict[str, Any]]` ou modelos).

### Critérios de aceite
- 100% dos métodos canônicos implementados no adapter.
- Sem `AttributeError` por mismatch de métodos nos smoke tests.

---

## Fase 2 — `UnifiedTWSClient` v2 resiliente e concorrente

### Ações
- Reescrever wrapper com:
  - `CircuitBreaker.call`
  - `RetryWithBackoff.execute`
  - `TimeoutManager.with_timeout`
- Lazy singleton seguro para uvloop:
  - lock inicializado no loop corrente;
  - reset seguro para testes.
- Garantir limpeza em falha de conexão:
  - `_client = None`
  - `_state = ERROR`.

### Critérios de aceite
- Teste de concorrência (N tarefas simultâneas) sem double-init.
- Sem deadlock em connect/disconnect.

---

## Fase 3 — Config e segredos (Pydantic v2)

### Ações
- Migrar config para `BaseSettings` + `SettingsConfigDict`.
- `password: SecretStr` + helper restrito para uso no boundary de conexão.
- Remover logging de segredo em qualquer caminho de erro.

### Critérios de aceite
- Testes de log/repr não exibem senha.
- Precedência de configuração validada (env/.env/settings).

---

## Fase 4 — Migração de callsites por ondas

### Onda A (crítica)
- `core/startup.py`
- rotas de status/health/monitoring

### Onda B (operacional)
- agentes/langgraph/workflows

### Onda C (complementar)
- admin e utilitários

### Critérios de aceite por onda
- Cobertura de testes da onda > baseline.
- Sem regressão funcional nos endpoints críticos.

---

## Fase 5 — Qualidade estática real (sem mascaramento)

### Ações
- Remover supressões em módulos migrados.
- Corrigir backlog real de lint/type:
  - imports/lint
  - nullable boolean semantics
  - assert runtime anti-pattern
  - factories sync com await em exemplos/docs/código.

### Critérios de aceite
- `ruff`, `mypy`, `pylint -E` verdes no escopo da onda.

---

## Fase 6 — Observabilidade/SLO

### Ações
- Publicar métricas:
  - requests, success/failure, retries, latency, circuit-open.
- Alarmes e dashboards (erro > threshold, circuit-open spike).

### Critérios de aceite
- Alertas acionando em ambiente de teste de falha.
- Dashboard validado pelo time de operação.

---

## Fase 7 — Rollout e remoção legado

### Ações
- Canário progressivo: 5% → 25% → 50% → 100%.
- Remover caminhos legados após janela de estabilidade.

### Critérios de aceite
- 0 incidentes críticos atribuíveis à migração em janela definida.
- Feature flag v2 default ON.

---

## Gate de qualidade por rodada (não negociável)

## Camada 1 — Pattern scan amplo
1. `ruff check --target-version py314`
2. `python -m mypy --follow-imports=skip <escopo_rodada>`
3. `pylint -E <escopo_rodada>`
4. Regex de governança:
   - bloquear novas supressões globais;
   - bloquear `await` em sync factory/fallback.

## Camada 2 — Checklist semântico obrigatório
Categorias (todas devem ser **0**):
1. async callback blocking
2. awaitable-returning sync factory/fallback
3. validação de parâmetros de runtime (`assert` em produção)
4. nullable boolean semantics
5. imports/lint

### Regra de encerramento
- Se qualquer categoria > 0: **rodada não encerra**.
- Se qualquer ferramenta falhar: **rodada não encerra**.

---

## Backlog operacional em PRs (sugestão)
- PR-01: contrato canônico + adapter
- PR-02: `UnifiedTWSClient` v2 + singleton seguro
- PR-03: config Pydantic v2 + SecretStr
- PR-04: migração callsites Onda A
- PR-05: Onda B
- PR-06: Onda C
- PR-07: remoção supressões + fixes estáticos reais
- PR-08: rollout canário + remoção legado

Cada PR deve anexar evidências dos gates de duas camadas com contagem explícita e tabela de pendências = 0.
