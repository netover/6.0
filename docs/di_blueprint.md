# Resync DI Blueprint (Enterprise / Critical)

This project uses **FastAPI native dependency injection** as the **only** mechanism in the HTTP request path.

## Goals
- Explicit, auditable wiring
- Correct lifecycle management (startup/shutdown + request teardown)
- Zero "magic container" in HTTP
- Fail-fast when dependencies are missing/mis-wired

---

## Canonical categories

### 1) Domain singletons (created at startup, stored on `app.state`)
**Created in:** `resync/lifespan.py` via `resync.core.wiring.init_domain_singletons(app)`  
**Accessed in HTTP via:** dependencies that read from `request.app.state`

**Current domain singletons:**
- `ConnectionManager` → `app.state.connection_manager`
- `ContextStore` (knowledge graph) → `app.state.knowledge_graph`
- `ITWSClient` → `app.state.tws_client`
- `AgentManager` → `app.state.agent_manager`
- `HybridRouter` → `app.state.hybrid_router`
- `IdempotencyManager` → `app.state.idempotency_manager`
- `LLMService` → `app.state.llm_service`

**Rationale:** explicit lifecycle, predictable behaviour, easy introspection.

### 2) Request-scoped resources (created per request, teardown via `yield`)
**Defined as:** `Depends` providers that `yield` a resource and clean it up in `finally`.

Examples (template):
- DB session / Unit-of-Work
- request context (trace IDs, correlation IDs)
- per-request caches

**Where to implement:**
- Add providers in `resync.core.wiring` or `resync.api.dependencies`
- Use `yield` provider pattern

### 3) Config singletons (safe caching with `@lru_cache`)
**Only for:** settings/config or pure objects with no teardown/lifecycle.

Current:
- `get_settings()` in `resync.core.wiring`

---

## Guardrails

### Banned imports
- `resync.core.di_container` is **banned** in `resync/api/**` (enforced by Ruff + CI).

### Dependency signatures must be typed
- Public dependency providers must not use `Any` and must include annotations.
- Enforced by `resync/tools/check_public_dependencies.py` + pytest gate.

### Concurrency isolation test
- `resync/tests/test_request_scope_isolation.py` validates request-scoped dependency isolation under concurrency.

---

## Adding a new dependency (checklist)
1. Decide category:
   - domain singleton → init in lifespan/app.state
   - request-scoped → `yield` provider
   - config singleton → `@lru_cache`
2. Add provider function (typed, no `Any`)
3. Add/extend tests
4. Ensure no `di_container` import in `resync/api/**`
