## 2026-03-06 - v5 production hardening and Valkey-only cleanup

- wrapped LangGraph node registration with `wrap_langgraph_node()` so runtime graph execution now passes copies of `state` and normalizes legacy full-state returns into delta updates without changing business logic
- extended the LangGraph hardening across main graph, diagnostic graph, incident response graph, parallel graph, ROMA graph, and reusable subgraphs
- removed project-controlled Redis naming from factories, middleware, admin settings, startup validation, auth/IP helpers, memory stores, monitoring helpers, tests, and `.env.example`, standardizing on Valkey throughout the repository and internal documentation
- kept only unavoidable protocol / third-party references such as LiteLLM `redis_url=` arguments and Valkey `INFO` fields like `redis_version`
- added explicit `response_model` contracts to health/status, system maintenance, audit review, and RAG query/file endpoints to reduce accidental field leakage and tighten API contracts
- replaced the middleware package import path to the canonical `valkey_validation` module
- modernized `CacheConfig` to Pydantic v2 `ConfigDict(frozen=True)`
- removed legacy Valkey/Redis alias exports from `resync/core/cache/valkey_config.py` and `resync/core/factories/__init__.py` so the public factory surface is Valkey-only
- changed system-config cache clearing to avoid calling a private `_flush_buffer()` method directly
- validation run: `python -m compileall -q resync` completed successfully after the changes

## 2026-03-06 - v4 final Valkey cleanup

- removed remaining project-controlled Redis naming in code, configs, middleware, cache routes, health checkers, and docs
- renamed internal store classes and health checker symbols from Redis* to Valkey* where controlled by the project
- renamed health checker module to `valkey_health_checker.py`
- cleaned user-facing/admin strings to reference Valkey instead of Redis
- preserved unavoidable third-party references such as package names (`redisvl`, telemetry package names) and Lua `redis.call(...)` syntax where required

# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2026-03-06] - Production-readiness fixes (non-security)

### Fixed
- LangGraph: removed in-place mutation of the incoming state and normalized nodes to return *delta updates* (prevents reducer/checkpoint corruption and racey behavior).
- LangGraph: fixed `HumanApprovalNode` to return delta updates instead of mutating and returning the incoming state (prevents checkpoint/reducer inconsistencies).
  - File: `resync/core/langgraph/nodes.py`
  - Files: `resync/core/langgraph/nodes.py`, `resync/core/langgraph/agent_graph.py`
- LangGraph: corrected `output_critique_node` to be async-safe and to avoid broad exception swallowing.
  - File: `resync/core/langgraph/agent_graph.py`
- Hybrid retriever: ensured async initialization is awaited and made BM25 index build concurrency-safe with a build lock.
  - File: `resync/knowledge/retrieval/hybrid_retriever.py`
- Postgres/BM25: corrected the SQL to use the schema column `chunk_id` (keeping compatibility by aliasing to `chunk_index`).
  - File: `resync/knowledge/store/pgvector_store.py`
- Ingestion: ensured basic ingest always persists textual content (`content`) so full-text/hybrid search is effective.
  - File: `resync/knowledge/ingestion/ingest.py`
- Ingestion (multi-view): moved sha256 dedup to be computed per `view_content` (avoids collisions across views) and aligned dedup strategy for future production use.
  - File: `resync/knowledge/ingestion/ingest.py`
- Retriever singleton: fixed `get_retriever()` to instantiate `RagRetriever` with required dependencies.
  - File: `resync/knowledge/retrieval/retriever.py`
- Chat route lazy-init: prevented storing a coroutine where a store instance is expected by using the sync accessor.
  - File: `resync/api/routes/core/chat.py`
- Lifespan/shutdown: ensured background tasks are cancelled with a timeout during shutdown (prevents hang).
  - File: `resync/core/startup.py`
- EventBus lifecycle: removed starting EventBus outside the lifespan-managed TaskGroup.
  - File: `resync/core/wiring.py`
- Checkpointer: fixed a TOCTOU bug in lock bootstrap that could create multiple locks under concurrency.
  - File: `resync/core/langgraph/checkpointer.py`
- Settings docs: moved class docstring to correct position so it becomes the class `__doc__`.
  - File: `resync/settings.py`

### Notes
- Security-related topics were intentionally excluded from this change set per project request.
