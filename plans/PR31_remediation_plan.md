# PR #31 Remediation Plan

## Current Branch: `feature/chat-to-new1`

---

## 🚨 Critical Blockers (Must Fix Immediately)

### 1. sanitize_input Import Error
- **File:** `resync/api/chat.py`
- **Issue:** `NameError: name 'sanitize_input' is not defined`
- **Cause:** `sanitize_input` is imported from `resync.core.security` but not exported by `resync/core/security/__init__.py`
- **Fix:** Export `sanitize_input` in `resync/core/security/__init__.py`

### 2. Dependency Injection Failure in Dashboard
- **File:** Dashboard metrics endpoint
- **Issue:** `get_dashboard_metrics` bypasses FastAPI DI for `store`
- **Fix:** Ensure `store` is properly injected via `Depends()`

### 3. Missing @property on specialist_team
- **File:** `BaseHandler` class
- **Issue:** `DiagnosticHandler` accesses `self.specialist_team.process(...)` but it's a method, not a property
- **Fix:** Add `@property` decorator to `specialist_team` in `BaseHandler`

---

## 🔒 Security Fixes

### 4. Router Cache Cross-User Leakage
- **File:** Router cache logic
- **Issue:** Cache key is just `message`, allowing users to hit intents cached by others
- **Fix:** Scope cache key with `user_id` or `session_id`

### 5. XSS Vulnerabilities in Admin SPA
- **Files:**
  - `resync/static/js/admin-feedback.js`
  - `resync/static/js/admin-tuning.js`
  - `resync/static/js/admin-audit.js`
  - `resync/static/js/admin-resources.js`
  - `resync/static/js/admin-backup.js`
  - `resync/static/js/chat-client.js`
- **Issue:** Widespread use of `innerHTML` with unescaped data
- **Fix:** Replace with `textContent` or use a sanitization helper

### 6. WebSocket Auth Fail-Closed (Verify)
- **File:** WebSocket broadcast
- **Issue:** `except Exception: pass` and potential auth bypass
- **Status:** Partially addressed, needs verification of `broadcast` race condition fix

---

## ⚡ Concurrency & Logic

### 7. WebSocket connected_clients Race Condition
- **Issue:** Iterating over list while modifying it; no lock
- **Fix:** Use `set`, `asyncio.Lock`, and iterate over copy

### 8. Singleton Race in _get_router_cache
- **Issue:** Lazy initialization not thread-safe
- **Fix:** Add `asyncio.Lock` for initialization

### 9. Prometheus Metric Labeling Bug
- **Issue:** `routing_decisions_total.get()` returns 0 because it's labeled
- **Fix:** Use `sum(metric.collect()[0].samples[...])` or equivalent to sum all labels

---

## 🧹 Code Quality & Cleanup

### 10. Remove Test Output Files
- **Files to delete:**
  - `debug_test_output.txt`
  - `final_pytest_output*.txt`
  - `pytest_output*.txt`
  - `test_output.txt`
- **Fix:** Delete files and add to `.gitignore`

### 11. Duplicate Code
- **File:** `agent_router.py`
- **Issue:** `_compiled_patterns` defined twice
- **Fix:** Remove duplicate

### 12. Dashboard Code Duplication
- **Files:** `metrics_dashboard.py`, `dashboard.py`
- **Issue:** Overlap between the two
- **Priority:** Lower - consolidate if possible, or leave for refactor phase

---

## Execution Order

1. **Critical Blockers** (Items 1-3)
2. **Security Fixes** (Items 4-6)
3. **Concurrency & Logic** (Items 7-9)
4. **Cleanup** (Items 10-12)
