# Revisão técnica — Resync v6.3.0 (lote 7: próximos 200 arquivos)

## Escopo auditado
- Varredura de 200 arquivos subsequentes ao lote anterior (com foco em código Python executável).
- Priorização: falhas de runtime, resiliência, segurança prática e robustez de fluxos LangGraph/RAG.

## Findings confirmados e status

_Status da auditoria: 2026-03-06 (itens corrigidos nesta rodada)._ 

1) **[🔴 Crítico] `wrap_langgraph_node` não definido em grafos LangGraph** — **Corrigido**.
- Arquivos: `incident_response.py` e `parallel_graph.py`.
- Impacto real: `NameError` em runtime na montagem do grafo, quebrando execução dos fluxos.
- Correção: import explícito de `wrap_langgraph_node` de `resync.core.langgraph.state_delta`.

2) **[🟡 Resiliência] Re-raise sem encadeamento em persistência de aprovação** — **Corrigido**.
- Arquivo: `resync/core/langgraph/nodes.py`.
- Impacto real: perda de causalidade da exceção original, dificultando diagnóstico em produção.
- Correção: `raise RuntimeError(...) from e`.

3) **[🟡 Resiliência] Re-raise sem encadeamento no timeout de startup** — **Corrigido**.
- Arquivo: `resync/core/startup.py`.
- Impacto real: reduz observabilidade do erro-fonte durante timeout de inicialização.
- Correção: `except TimeoutError as exc` + `raise ConfigurationError(...) from exc`.

4) **[⚪ Qualidade/Confiabilidade] `zip()` sem `strict` em ingestão e retriever híbrido** — **Corrigido**.
- Arquivos: `resync/knowledge/ingestion/ingest.py`, `resync/knowledge/retrieval/hybrid_retriever.py`.
- Correção: `zip(..., strict=False)` explícito.

5) **[🟢 Práticas] Ajuste de tipagem moderna em singleton de background tasks** — **Corrigido**.
- Arquivo: `resync/core/background_tasks.py`.
- Correção: remoção de anotações em string para `ConversationMemory` (com `from __future__ import annotations` já presente).

## Observações
- Foram sinalizados alertas `S608` em queries SQL montadas via f-string em módulos pgvector. Nesta rodada, não foi aplicado hardening adicional porque os pontos analisados já fazem validação de chaves/filtros e usam placeholders parametrizados para valores; recomendada revisão dedicada de segurança SQL para comprovação formal de ausência de vetor explorável.
