# Security & Performance Audit — Applied Corrections

**Patch applied:** 2026-02-24  
**Files modified:** 12  
**Issues resolved:** 4 × P0 (critical), 7 × P1 (high), 6 × P2 (medium)

---

## P0 — Critical (blocking production)

### P0-01 · Blocking ML inference on event loop
**File:** `resync/core/cache/semantic_cache.py`

`self.vectorizer.embed()` calls `sentence-transformers model.encode()` — CPU-intensive
inference (50–200 ms) — directly in async methods `get()`, `set()`, and `check_intent()`.

**Fix:** All three call sites now wrapped in `asyncio.to_thread()`.

---

### P0-02 · Unbounded `alerts` list — OOM risk
**File:** `resync/core/tws_monitor.py`

`self.alerts: list[Alert]` grew indefinitely; never pruned.

**Fix:**
- Replaced with `deque(maxlen=1_000)` — O(1) append, automatic eviction.
- Added `_should_emit_alert()` suppression guard (5-minute window per category)
  to prevent deque exhaustion during sustained threshold breaches.
- `_measure_memory_usage()` converted to `async` with `asyncio.to_thread` (P2-07 bundled).

---

### P0-03 · `psutil.cpu_percent(interval=1)` blocks event loop
**File:** `resync/core/health/performance_metrics_collector.py`

The 1-second blocking sleep inside `cpu_percent` stalled every in-flight coroutine
during each health-check cycle.

**Fix:** New `get_system_performance_metrics_async()` async method wraps all psutil
calls in `asyncio.to_thread()`. Synchronous fallback kept for non-async callers with
a deprecation notice.

---

### P0-04 · Lock held during network I/O
**File:** `resync/core/connection_manager.py`

`await websocket.send_text()` was called inside `async with self._lock`, allowing a
single slow client to serialise all WebSocket operations system-wide.

**Fix:** Snapshot the websocket reference under the lock, release it immediately, then
perform all I/O outside the lock.

---

## P1 — High priority

### P1-01 · TOCTOU race in `broadcast` / `broadcast_json`
**File:** `resync/core/websocket_pool_manager.py`

`broadcast()` captured `client_ids = list(connections.keys())` then
`_send_message_with_error_handling` re-fetched `connections.get(client_id)` without
holding `_lock`. A concurrent `_cleanup_loop` could remove the connection between
those two points.

**Fix:** Both `broadcast()` and `broadcast_json()` now snapshot `(client_id, conn_info)`
pairs atomically under `_lock` and pass `conn_info` directly to the send helpers.

---

### P1-02 · Duplicate `_remove_connection_safe` method
**File:** `resync/core/websocket_pool_manager.py`

`_remove_connection_safe` was a byte-for-byte copy of `_remove_connection` (both
acquired the lock), creating a maintenance hazard.

**Fix:** `_remove_connection_safe` reduced to a one-line forwarding alias pointing to
`_remove_connection`.

---

### P1-03 · `MetricHistogram.samples.pop(0)` — O(n) in hot path
**File:** `resync/core/metrics.py`

`list.pop(0)` shifts every element left — O(n) — under high-throughput metric recording.

**Fix:** `samples` changed to `deque(maxlen=max_samples)`. The `deque` auto-evicts the
oldest entry in O(1); the manual `pop(0)` guard was removed.

---

### P1-04 · Streaming response echoes unsanitised user content (XSS)
**File:** `resync/api/websocket/handlers.py`

`"message": f"Processando: {content}"` reflected raw user input. If the frontend
renders this field via `innerHTML`, it is a stored/reflected XSS vector.

**Fix:** Replaced with the static string `"Processando sua mensagem..."`.  
Bundled: `__import__("time")` dynamic import replaced with module-level `import time`.

---

### P1-05 · Deprecated OpenTelemetry Jaeger exporter
**File:** `resync/core/distributed_tracing.py`

`opentelemetry-exporter-jaeger` was archived and removed from the OTel Python
ecosystem in 2023. Import silently fell back to a stub class.

**Fix:** Replaced with `OTLPSpanExporter` (gRPC). Jaeger collectors that accept OTLP
(Jaeger ≥ 1.35 / Grafana Tempo / OpenTelemetry Collector) are the supported path.
The old `JAEGER_AVAILABLE` flag is retained as an alias for backward compatibility.

---

### P1-06 · Sentry SDK not integrated
**Files:** `resync/core/startup.py`, `resync/settings.py`

`sentry-sdk` was listed as a dependency but never imported or initialised.

**Fix:**
- Added `sentry_dsn`, `sentry_traces_sample_rate`, and `sentry_profiles_sample_rate`
  fields to `Settings` (accept env vars `SENTRY_DSN` / `APP_SENTRY_DSN`).
- Sentry is initialised early in the lifespan startup with `FastApiIntegration`,
  `StarletteIntegration`, and `AsyncioIntegration`. If the DSN is absent or
  `sentry-sdk` is not installed, startup continues normally.

---

### P1-07 · `CircuitBreaker._get_lock()` lazy-init race
**File:** `resync/core/resilience.py`

Check `if self._lock is None and not self._lock_initialized` is not atomic. Under
free-threaded Python 3.13+ two concurrent coroutines can both pass the guard.

**Fix:** `asyncio.Lock` is now eagerly initialised in `__init__` — safe since
Python 3.10+ binds the lock to the running loop on first `await`, not at construction.
`_lock_initialized` flag removed. `_get_lock()` reduced to `return self._lock`.

Bundled: `CircuitBreakerConfig.__post_init__` added to validate
`failure_threshold >= 1` and `recovery_timeout >= 1` at construction time (P2-09).

---

## P2 — Medium priority

### P2-04 · `hmac.new()` digestmod as string
**File:** `resync/api/middleware/csrf_protection.py`

`digestmod="sha256"` string disables static-analysis coverage of the digest function.

**Fix:** `digestmod=hashlib.sha256` (callable) used at both call sites.

### P2-06 · `_try_restore_redis` concurrent suppression race
**File:** `resync/core/cache/semantic_cache.py`

Multiple coroutines could all pass the 5-second cooldown check simultaneously,
causing a thundering-herd of Redis pings during outages.

**Fix:** `_restore_lock: asyncio.Lock` guards `_try_restore_redis`; a second check
inside the lock prevents redundant restores after another coroutine already succeeded.

### P2-07 · `_measure_memory_usage()` blocks event loop
**File:** `resync/core/tws_monitor.py`

`psutil.Process().memory_info()` is a blocking syscall.

**Fix:** Method converted to `async def` using `asyncio.to_thread` (applied with P0-02).

### P2-08 · `__import__("time")` dynamic import in hot WebSocket handler
**File:** `resync/api/websocket/handlers.py`

**Fix:** Module-level `import time` added (applied with P1-04).

### P2-09 · `CircuitBreakerConfig.recovery_timeout=0` not validated
**File:** `resync/core/resilience.py`

**Fix:** `__post_init__` validation added (applied with P1-07).

### P2-10 · Triple-state `_redis_stack_available: bool | None`
**File:** `resync/core/cache/semantic_cache.py`

Implicit `None`-falsy handling made conditional branches unpredictable.

**Fix:** Field type changed to `bool`, initialised to `False`.

---

## Not addressed in this patch

| ID | Reason |
|----|--------|
| P2-01 · 133 stdlib `json` usages | Large blast radius; recommend a dedicated migration sprint with orjson shim |
| P2-02 · `Optional[T]` / `Union[A,B]` | Cosmetic; no runtime impact — address via ruff auto-fix |
| P2-03 · `health_check()` dict read without lock | Read-only; race is benign under CPython GIL; re-evaluate if GIL is removed |

---

*Generated by audit pass · Python 3.14 / FastAPI / Pydantic v2 stack*
