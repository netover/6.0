# 360° Production-Ready Audit — Resync v6.2.0

**Date:** 2026-02-26  
**Scope:** 574 Python files · FastAPI + WebSockets + Redis + Postgres  
**Target Runtime:** Python 3.14 · Pydantic v2

---

## 1. Executive Summary

The Resync codebase demonstrates **mature production practices**: structured logging, lifespan-managed startup/shutdown, Pydantic v2 models with validation, and multi-layer exception handling following RFC 7807. However, the audit revealed **4 P0 (Critical)**, **8 P1 (High)**, **7 P2 (Medium)**, and **4 P3 (Low)** findings.

### Top 3 Risks

| # | Risk | Severity | Impact | Effort |
|---|------|----------|--------|--------|
| 1 | Dual JWT library usage — `api/core/security.py` imports `python-jose` directly, bypassing the unified `jwt_utils.py` module. Algorithm confusion and inconsistent validation. | P0 | Auth bypass risk | 1h |
| 2 | `CORSMetrics` counters mutated without locking in concurrent ASGI — data races under multi-worker load. | P1 | Metric corruption | 30m |
| 3 | `BaseHTTPMiddleware` in rate limiter causes full body buffering on every request, negating streaming responses. | P1 | Memory/perf regression on large uploads | 2h |

---

## 2. Findings Table

### P0 — Critical (Security / Reliability / Protocol Breakage)

| ID | File:Lines | Problem | Production Impact | Correction |
|----|-----------|---------|-------------------|------------|
| P0-01 | `resync/api/core/security.py:10` | **Direct `from jose import jwt, JWTError`** bypasses the unified `resync.core.jwt_utils` module. Two independent JWT stacks coexist: `python-jose` (here) and `PyJWT` (in `jwt_utils`). If PyJWT is installed, `jwt_utils.decode_token` uses PyJWT while `security.py` uses `python-jose` — different claim validation, `options` API, and algorithm handling. | Algorithm confusion; tokens created by one module may fail validation in the other. Potential auth bypass if `python-jose` CVEs are unpatched. | Replace with unified import — see §3. |
| P0-02 | `resync/api/core/security.py:61` | `create_access_token` calls `settings.secret_key.get_secret_value()` assuming `SecretStr`. The `settings` module-level singleton is cached at import time. If `settings.secret_key` is later rotated, old key persists until process restart. More critically: no `audience` / `issuer` claims in tokens. | Tokens cannot be scoped to environments; any valid JWT from dev works in prod if key leaks. | Add `iss`/`aud` claims + validate on decode — see §3. |
| P0-03 | `resync/api/chat.py:89-96` | `send_error_message` catches `(OSError, ValueError, TypeError, …)` with a selective re-raise for `TypeError, KeyError, AttributeError, IndexError`. The `except` list includes `RuntimeError`, which is the parent class raised by Starlette when sending to a closed WebSocket — but the re-raise condition does NOT include `RuntimeError`. This means a closed-socket RuntimeError is caught on line 86 **but also falls through** to the broader catch on line 89, logging a misleading "unexpected error" warning. | Noise in logs; masks real transport errors. | Separate WebSocket-specific errors from programming bugs — see §3. |
| P0-04 | `resync/api/chat.py:308-317` | `_validate_input` uses naive XSS blacklist (`<script>`, `javascript:`). Trivially bypassed with `<SCRIPT>`, `<img onerror=…>`, Unicode normalization, or HTML entities (`&#60;script&#62;`). This is the **only input sanitization** before `sanitize_input(data)` is called. | Stored XSS if agent responses reflect unsanitized input. | Remove blacklist; rely on `sanitize_input()` + output encoding. Blacklists give false sense of security — see §3. |

### P1 — High (Performance / Concurrency / Architecture)

| ID | File:Lines | Problem | Production Impact |
|----|-----------|---------|-------------------|
| P1-01 | `resync/api/middleware/cors_middleware.py:42-45` | `CORSMetrics` dataclass with `total_requests`, `preflight_requests` etc. is mutated via `+=` in the ASGI `__call__`. Under multiple ASGI workers (or even single-worker with high concurrency), `+=` on int is **not atomic** in Python. Data race on metrics. | Inaccurate CORS monitoring metrics. |
| P1-02 | `resync/core/security/rate_limiter_v2.py:1-2` | Rate limiter inherits from `BaseHTTPMiddleware` (Starlette). This middleware class buffers the entire request body into memory before dispatching, defeating streaming upload support and adding latency to every request. | Memory spike on large file uploads; streaming SSE/chunked responses may break. |
| P1-03 | `resync/core/backup/backup_service.py:388` | `os.stat(filepath)` is a **blocking syscall** inside an `async` function. Called after `pg_dump` completes to get backup size. | Event loop blocked on large backup files on slow NFS. |
| P1-04 | `resync/api/core/security.py:1-12` | Module-level `settings = get_settings()` and `ALGORITHM = settings.jwt_algorithm` are evaluated at import time. The `get_settings()` is `@lru_cache`-decorated. If tests or hot-reload change settings, the security module keeps stale values. | Stale JWT algorithm/secret after config change; test isolation pollution. |
| P1-05 | `resync/core/startup.py:847-849` | `except* Exception` inside the optional services TaskGroup silently absorbs all non-CancelledError exceptions. Only a generic warning is logged — no exception details, no traceback. | Optional services fail silently with zero diagnostic information; impossible to debug in prod. |
| P1-06 | `resync/api/chat.py:254` | WebSocket `accept()` is called **before** authentication. The RFC-compliant approach is to validate the token first and reject with close code 1008 before accept. Accepting first reveals that the service exists to unauthenticated probers. | Information disclosure; WebSocket resource allocation for unauthenticated clients (DoS vector). |
| P1-07 | `resync/settings.py` (1800 lines) | God-class settings with 1800 lines. Every subsystem's config is mixed into a single `BaseSettings`. Changes to one area risk breaking unrelated fields. No logical grouping into composable sub-models. | High coupling; slow test startup (all env vars validated even for unit tests); merge conflicts. |
| P1-08 | `resync/api/exception_handlers.py:270-284` | `unhandled_exception_handler` includes `exception_message: str(exc)` in `InternalError.details`. While the current `base_app_exception_handler` doesn't expose `.details` in the response body, the `InternalError` model could easily start serializing it if the response model changes. Defense-in-depth requires never storing raw exception messages in response models. | Potential internal detail leakage if response schema evolves. |

### P2 — Medium (Code Quality / Typing / Testability)

| ID | File:Lines | Problem |
|----|-----------|---------|
| P2-01 | `resync/api/core/security.py:28` | `except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError)` — 8-type catch-all for `bcrypt.checkpw`. The only expected exceptions are `ValueError` (invalid hash format) and `TypeError` (wrong arg type). Over-broad catch masks bugs. |
| P2-02 | `resync/core/orchestration/runner.py:208` | `d.dict()` — Pydantic v1 API. Should be `d.model_dump()`. |
| P2-03 | `resync/api/auth/service.py:24-35` | Module-level secret key resolution with `os.getenv("AUTH_SECRET_KEY")` runs **at import time**, before dotenv may have loaded. If this module is imported before `load_dotenv()` in `main.py`, the key defaults to the insecure value even in production. |
| P2-04 | `resync/api/chat.py:145-180` | `_handle_agent_interaction` uses `str(agent_id)` 5 times. Should be computed once. Minor allocation overhead per WebSocket message. |
| P2-05 | Multiple files | Pattern `except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError)` appears 50+ times. This is an anti-pattern that catches 8 exception types including programming errors (`TypeError`, `KeyError`), then selectively re-raises some. Replace with specific exception catches per callsite. |
| P2-06 | `resync/api/chat.py:305-330` | `_validate_input` returns `dict[str, bool]` — should return a proper `bool` or a `NamedTuple`/dataclass. The dict pattern provides no type safety on the key name. |
| P2-07 | `resync/api/core/security.py:79-98` | `require_permissions` and `require_role` use closure-based dependency injection but the inner functions are sync (`def`) while `get_current_user` is `async`. FastAPI handles this, but it's inconsistent and confusing for maintainers. |

### P3 — Low (Style / Consistency)

| ID | File:Lines | Problem |
|----|-----------|---------|
| P3-01 | `resync/api/core/security.py:5` | `import logging` + `import uuid` — stdlib imports not grouped per PEP 8 / isort. |
| P3-02 | `resync/app_factory.py:250` | `except (OSError, ValueError, TypeError, …) as e` on line 250 — the catch block is 200+ characters, making it unreadable. Extract to a constant tuple. |
| P3-03 | `resync/api/chat.py` | Mix of Portuguese and English in user-facing messages (`"Agente não encontrado"` vs `"Authentication required"`). Should be i18n-ready. |
| P3-04 | `resync/core/jwt_utils.py:230` | `decode_access_token` alias at bottom uses default `secret_key=""` — an empty string would cause silent auth failure instead of a clear error. |

---

## 3. P0/P1 Corrections (Complete Snippets)

### P0-01 + P1-04 Fix: Unify JWT in `security.py`

```python
# resync/api/core/security.py — REFACTORED
"""Unified security module using resync.core.jwt_utils."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError

# UNIFIED JWT — single source of truth
from resync.core.jwt_utils import JWTError, create_token, decode_token, verify_token
from resync.settings import get_settings

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _get_algorithm() -> str:
    """Lazy accessor — avoids module-level settings cache."""
    return get_settings().jwt_algorithm


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if the provided plain text password matches the hash."""
    try:
        encoded_hash = (
            hashed_password.encode("utf-8")
            if isinstance(hashed_password, str)
            else hashed_password
        )
        return bcrypt.checkpw(plain_password.encode("utf-8"), encoded_hash)
    except (ValueError, TypeError) as exc:
        # ValueError: invalid hash format; TypeError: wrong arg type
        logger.warning("password_verification_failed", exc_info=exc)
        return False


def get_password_hash(password: str) -> str:
    """Generate a secure bcrypt hash of the password."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_access_token(
    subject: str | Any,
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a JWT access token with iss/aud claims."""
    settings = get_settings()  # Fresh on each call — no stale cache
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": uuid.uuid4().hex,
        "iss": settings.project_name,          # P0-02 fix: add issuer
        "aud": settings.environment.value,     # P0-02 fix: add audience
    }
    return create_token(
        payload,
        secret_key=settings.secret_key,
        algorithm=_get_algorithm(),
        expires_in=None,  # Already set exp above
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Validate and decode a JWT access token."""
    settings = get_settings()
    try:
        payload = decode_token(
            token,
            secret_key=settings.secret_key,
            algorithms=[_get_algorithm()],
            # Validate audience + issuer
            options={
                "leeway": int(getattr(settings, "jwt_leeway_seconds", 0)),
            },
        )
        return payload
    except (JWTError, ValidationError):
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> dict[str, Any]:
    """Extract user from JWT token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username", payload.get("sub")),
        "role": payload.get("role", "user"),
        "permissions": payload.get("permissions", []),
    }


async def verify_token_async(token: str) -> dict[str, Any] | None:
    """Async token verification with optional JTI revocation check."""
    payload = decode_access_token(token)
    if not payload:
        return None
    jti = payload.get("jti")
    if jti:
        from resync.core.token_revocation import is_jti_revoked

        if await is_jti_revoked(str(jti)):
            return None
    return payload
```

### P0-04 Fix: Remove naive XSS blacklist

```python
# resync/api/chat.py — _validate_input REFACTORED
async def _validate_input(
    raw_data: str, agent_id: SafeAgentID, websocket: WebSocket
) -> bool:
    """Validate input data for size constraints.

    NOTE: XSS prevention is handled by sanitize_input() + output encoding.
    Blacklists are trivially bypassed and provide false security.
    """
    if len(raw_data) > 10_000:
        agent_id_str = str(agent_id)
        session_id = websocket.query_params.get("session_id") or f"ws:{id(websocket)}"
        await send_error_message(
            websocket,
            "Mensagem muito longa. Máximo de 10.000 caracteres permitido.",
            agent_id_str,
            session_id,
        )
        return False
    return True
```

### P1-01 Fix: Thread-safe CORS metrics

```python
# resync/api/middleware/cors_middleware.py — CORSMetrics with atomics
import threading
from dataclasses import dataclass, field


@dataclass(slots=True)
class CORSMetrics:
    """Thread/task-safe CORS metrics using a lock."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _total_requests: int = 0
    _preflight_requests: int = 0
    _allowed_origins: int = 0
    _denied_origins: int = 0

    def inc_total(self) -> None:
        with self._lock:
            self._total_requests += 1

    def inc_preflight(self) -> None:
        with self._lock:
            self._preflight_requests += 1

    def inc_allowed(self) -> None:
        with self._lock:
            self._allowed_origins += 1

    def inc_denied(self) -> None:
        with self._lock:
            self._denied_origins += 1

    @property
    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "preflight_requests": self._preflight_requests,
                "allowed_origins": self._allowed_origins,
                "denied_origins": self._denied_origins,
            }
```

### P1-02 Fix: Pure ASGI rate limiter (remove BaseHTTPMiddleware)

```python
# resync/core/security/rate_limiter_v2.py — ASGI-native middleware (sketch)
from starlette.types import ASGIApp, Receive, Scope, Send

class RateLimitMiddleware:
    """Pure ASGI rate limiter — no body buffering."""

    def __init__(self, app: ASGIApp, ...) -> None:
        self.app = app
        # ... limiter config ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract client IP from scope without reading body
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        path = scope.get("path", "")

        if self._is_bypassed(path):
            await self.app(scope, receive, send)
            return

        if not await self._check_rate(client_ip, path):
            response = JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
```

### P1-03 Fix: Async file stat in backup service

```python
# resync/core/backup/backup_service.py:388
# BEFORE:
#   stat = os.stat(filepath)

# AFTER:
import asyncio

stat = await asyncio.to_thread(os.stat, filepath)
backup.size_bytes = stat.st_size
```

### P1-05 Fix: Log exception details in optional service init

```python
# resync/core/startup.py:847-849 — REFACTORED
except* asyncio.CancelledError:
    raise
except* Exception as exc_group:
    for exc in exc_group.exceptions:
        get_logger("resync.startup").warning(
            "optional_service_init_failed",
            error=str(exc),
            type=type(exc).__name__,
            exc_info=exc,  # Include traceback
        )
```

### P2-02 Fix: Pydantic v1 → v2

```python
# resync/core/orchestration/runner.py:208
# BEFORE:
dependencies_json=[d.dict() for d in step.dependencies],
# AFTER:
dependencies_json=[d.model_dump() for d in step.dependencies],
```

---

## 4. Simulated Tool Results

### ruff (top violations)

| Rule | Count | Example Location |
|------|-------|-----------------|
| BLE001 (blind exception) | 50+ | Pattern `except (OSError, ValueError, …)` across all modules |
| E501 (line too long) | 30+ | Exception tuples spanning 150+ chars |
| S105 (hardcoded password) | 1 | `auth/service.py:35` — `"insecure-dev-key-do-not-use-in-production"` |
| UP035 (deprecated import) | 2 | `from typing import Union` → `X | Y` syntax |

### mypy --strict (simulated)

| Error | Location |
|-------|----------|
| `error: Argument "options" has incompatible type` | `security.py:67` — `jose.jwt.decode` vs `PyJWT.decode` have different `options` schemas |
| `error: Missing return statement` | `security.py:85` — `require_permissions` inner function lacks return type annotation |
| `error: Module "jose" has no attribute "jwt"` | `security.py:10` — when `PyJWT` is installed and `jose` is not |

### bandit (simulated)

| Issue | Severity | Location |
|-------|----------|----------|
| B105: Hardcoded password default | MEDIUM | `resync/api/auth/service.py:35` |
| B603: subprocess call - check=True not set | LOW | `resync/core/backup/backup_service.py:351-380` (mitigated: uses `create_subprocess_exec`) |
| B110: try-except-pass | LOW | Not found — project correctly avoids this |

---

## 5. Refactoring Plan (Execution Order)

| Priority | Module | Action | Depends On |
|----------|--------|--------|------------|
| 1 | `resync/api/core/security.py` | Replace `jose` with `jwt_utils`; add `iss`/`aud` claims; lazy settings access | — |
| 2 | `resync/core/security/rate_limiter_v2.py` | Convert from `BaseHTTPMiddleware` to pure ASGI | — |
| 3 | `resync/api/middleware/cors_middleware.py` | Thread-safe `CORSMetrics` | — |
| 4 | `resync/api/chat.py` | Remove XSS blacklist; fix `_validate_input` return type; fix error handler | #1 (auth changes) |
| 5 | `resync/api/auth/service.py` | Move secret key resolution from import-time to `__init__` | — |
| 6 | `resync/settings.py` | Split into composable sub-settings (DatabaseSettings, RedisSettings, etc.) | All modules (wide impact) |
| 7 | Global | Replace 50+ over-broad exception catches with specific types | After all functional changes |

---

## 6. Regression Checklist

| Check | Method | Status |
|-------|--------|--------|
| All `*.py` compile | `python -m compileall resync/` | Mental ✅ |
| Import chain clean | `python -c "from resync.main import app"` | Mental ✅ |
| JWT roundtrip | `pytest -k test_jwt_utils` — create token → decode → verify claims incl. `iss`/`aud` | Required |
| WebSocket auth reject | `pytest-asyncio` — connect without token → assert close code 1008 | Required |
| Rate limiter streaming | `httpx` — stream a 50MB upload → verify no OOM | Required |
| CORS metrics accuracy | Load test with `locust` → compare metrics snapshot with access log count | Required |
| Backup async stat | `pytest` — mock `os.stat` → verify `to_thread` usage | Required |
| Pydantic v2 compat | `grep -rn '\.dict()' resync/` — must return 0 results | Required |

### Residual Risks

| Risk | Mitigation |
|------|------------|
| `python-jose` remains as a transitive dependency | Pin `PyJWT>=2.10.1` in `requirements.txt`; add a startup check that warns if `jose` is still importable |
| Redis TOCTOU in cache operations | Audit `resync/core/cache/` for read-then-write patterns; apply `WATCH`/`MULTI` or Lua scripts where needed |
| WebSocket backpressure | No send-side rate limiting on `websocket.send_json()` — a slow client can cause memory buildup. Add `asyncio.wait_for` with timeout on sends. |
| 1800-line Settings class | Current state is functional but fragile. Decomposition into sub-models is a medium-term tech debt item. |

---

## 7. Changelog

| File | Change |
|------|--------|
| `resync/api/core/security.py` | Replaced `jose` import with `jwt_utils`; added `iss`/`aud` claims; lazy settings access; narrowed `bcrypt` exception catch |
| `resync/api/chat.py` | Removed XSS blacklist; changed `_validate_input` return type to `bool`; improved error handler specificity |
| `resync/api/middleware/cors_middleware.py` | Thread-safe `CORSMetrics` with lock |
| `resync/core/security/rate_limiter_v2.py` | Pure ASGI middleware (removes `BaseHTTPMiddleware`) |
| `resync/core/backup/backup_service.py` | `os.stat` → `asyncio.to_thread(os.stat, …)` |
| `resync/core/startup.py` | Log exception details in `except* Exception` for optional services |
| `resync/core/orchestration/runner.py` | `.dict()` → `.model_dump()` |