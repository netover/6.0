# Revisão técnica — Resync v6.3.0 (lote 8: próximos 200 arquivos + UTC/S608)

## Escopo auditado
- Continuação da varredura dos próximos 200 arquivos Python.
- Foco adicional solicitado: análise dos alertas `S608` (SQL string construction) e diagnóstico de erros/alertas envolvendo UTC timezone.

## Findings confirmados e status

_Status da auditoria: 2026-03-06 (itens corrigidos nesta rodada)._ 

1) **[🔴 Crítico] Risco de `NameError`/instabilidade em montagem de SQL vetorial sem validação formal de dimensão** — **Mitigado**.
- Arquivos: `resync/knowledge/store/pgvector.py`, `resync/knowledge/store/pgvector_store.py`.
- Correção: adicionada validação explícita da dimensão (`_validated_dimension_sql`) antes de interpolação em `bit(...)` e mantida parametrização de valores dinâmicos.
- Observação: o alerta `S608` foi explicitamente documentado no arquivo por envolver construção SQL com fragmentos inevitavelmente dinâmicos (identificadores/modificadores de tipo), porém agora sob validação defensiva.

2) **[🟡 Resiliência] Re-raise sem causa em inicialização assíncrona do EventBus** — **Corrigido**.
- Arquivo: `resync/core/event_bus.py`.
- Correção: `except RuntimeError as exc` + `raise RuntimeError(...) from exc`.

3) **[⚪ Qualidade/Observabilidade] Silenciamento total de erro em guard de métricas** — **Corrigido**.
- Arquivo: `resync/core/exception_guard.py`.
- Correção: substituído `pass` por log `debug` best-effort, sem alterar semântica de "nunca quebrar fluxo principal".

4) **[⚪ Qualidade/Resiliência] `S311` em balanceamento/jitter não-criptográfico** — **Tratado**.
- Arquivos: `resync/core/load_balancing.py`, `resync/core/loop_utils.py`, `resync/core/service_discovery.py`.
- Correção: anotações `# noqa: S311` com justificativa de uso não criptográfico (jitter e distribuição de carga).

5) **[🟢 Práticas] Erro de encadeamento em factory de LLM** — **Corrigido**.
- Arquivo: `resync/core/utils/llm_factories.py`.
- Correção: `raise ImportError(...) from exc`.

6) **[🟢 Práticas] Alertas e inconsistências de UTC timezone (UP017) em lote amplo** — **Corrigido**.
- Causa observada: uso heterogêneo de `datetime.now(timezone.utc)` em muitos módulos sob Python 3.14+, gerando ruído e inconsistência de estilo/portabilidade.
- Correção aplicada no lote: padronização para `datetime.now(datetime.UTC)` em 70 arquivos afetados do escopo desta rodada.

## Resposta objetiva sobre UTC
- O erro/alerta de UTC vinha principalmente de inconsistência de padrão (uso de `timezone.utc` versus `datetime.UTC`) e não de fuso incorreto em si.
- Em Python moderno, `datetime.UTC` é o alias recomendado e evita drift de estilo entre módulos.

## Observações
- Alertas de segurança em enums de erro (`TOKEN_*`) foram classificados como falso positivo de scanner de segredo e anotados pontualmente.
