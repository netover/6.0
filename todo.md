Tool	Category	Rule	Count	
ruff	Lint	Import block is un-sorted or un-formatted	8	❌
ruff	Style	Incorrect formatting, autoformat by running qlty fmt.	37	❌
ruff	Lint	Line too long (135 > 100)	2	❌
ruff	Lint	Do not call setattr with a constant attribute value. It is not any safer than normal property access.	2	❌
ruff	Lint	Replace aliased errors with TimeoutError	2	❌
ruff	Lint	Unnecessary default type arguments	1	❌
ruff	Lint	Use contextlib\.suppress\(OSError\) instead of try-except-pass	1	❌
ruff	Lint	Remove quotes from type annotation	1	❌
bandit	Dependency Alert	Possible SQL injection vector through string-based query construction.	1	❌
ruff	Lint	Possible SQL injection vector through string-based query construction	1	❌
bandit	Dependency Alert	Try, Except, Pass detected.	1	❌
ruff	Lint	try-except-pass detected, consider logging the exception	1	❌
ruff	Lint	try-except block with duplicate exception ImportError	1	❌
ruff	Lint	Use a single if statement instead of nested if statements	1	❌
qlty	Structure	High total complexity (count = 51)	3	
radarlint-python	Lint	Remove this redundant Exception class; it derives from another which is already caught.	28	
qlty	Structure	Function with high complexity (count = 18): __getattr__	2	
qlty	Duplication	Found 21 lines of similar code in 2 locations (mass = 108)	2	
qlty	Structure	Function with many returns (count = 6): circuit_protected	2	
radarlint-python	Lint	Catch this exception only once; it is already handled by a previous except clause.	2	
radarlint-python	Lint	Remove the unused function parameter "tg".	1	
@qltysh one-click actions:

 Auto-fix formatting (qlty fmt && git push)

qltysh[bot]
qltysh bot reviewed 1 hour ago
alembic/versions/0002_add_tws_pattern_unique.py

from collections.abc import Sequence

import sqlalchemy as sa
Contributor
@qltysh
qltysh bot
1 hour ago
Import block is un-sorted or un-formatted [ruff:I001]

@netover	Reply...
resync/api/chat.py
        await send_error_message(websocket, "Conteúdo inválido.", agent_id_str, session_id)
        return {"is_valid": False}

    return {"is_valid": True}
Contributor
@qltysh
qltysh bot
1 hour ago
High total complexity (count = 57) [qlty:file-complexity]

@netover	Reply...
resync/api/core/security.py
        )
        return payload
    except (JWTError, ValidationError):
    except (JWTError, ValidationError, TypeError, ValueError):
Contributor
@qltysh
qltysh bot
1 hour ago
Remove this redundant Exception class; it derives from another which is already caught. [radarlint-python:python:S5713]

@netover	Reply...
resync/api/middleware/correlation_id.py
import sys
from typing import Any

from starlette.datastructures import MutableHeaders
Contributor
@qltysh
qltysh bot
1 hour ago
Import block is un-sorted or un-formatted [ruff:I001]

@netover	Reply...
resync/api/middleware/correlation_id.py
    _exc_type, _exc, _tb = _sys.exc_info()
    maybe_reraise_programming_error(_exc, _tb)

if sys.version_info >= (3, 14):
Contributor
@qltysh
qltysh bot
1 hour ago
Version block is outdated for minimum Python version [ruff:UP036]

@netover	Reply...
21 hidden conversations
Load more…
resync/core/a2a_handler.py
            logger.warning("Event queue full, dropping event", event_type=event_type)

    async def get_event_stream(self):
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
Contributor
@qltysh
qltysh bot
1 hour ago
Unnecessary default type arguments [ruff:UP043]

Suggested change
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any]]:
@netover	Reply...
resync/core/agent_manager.py
            lock = self._tws_locks.get(loop_id)
            if lock is None:
                lock = asyncio.Lock()
                self._tws_locks[loop_id] = lock
Contributor
@qltysh
qltysh bot
1 hour ago
Found 21 lines of similar code in 2 locations (mass = 108) [qlty:similar-code]

@netover	Reply...
resync/core/agent_manager.py
            lock = self._agent_locks.get(loop_id)
            if lock is None:
                lock = asyncio.Lock()
                self._agent_locks[loop_id] = lock
Contributor
@qltysh
qltysh bot
1 hour ago
Found 21 lines of similar code in 2 locations (mass = 108) [qlty:similar-code]

@netover	Reply...
resync/core/auto_recovery.py
                        "reason": "High database error rate",
                        "recovery_strategy": "Reset connection pool",
                        "success": self._reset_database_connections(),
                        "success": await asyncio.to_thread(self._reset_database_connections),
Contributor
@qltysh
qltysh bot
1 hour ago
Undefined name asyncio [ruff:F821]

@netover	Reply...
resync/core/auto_recovery.py

            # Check for memory issues
            memory_issues = self._check_memory_issues()
            memory_issues = await asyncio.to_thread(self._check_memory_issues)
Contributor
@qltysh
qltysh bot
1 hour ago
Undefined name asyncio [ruff:F821]

@netover	Reply...
codereviewbot-ai[bot]
codereviewbot-ai bot reviewed 1 hour ago
alembic/versions/0002_add_tws_pattern_unique.py
Comment on lines +27 to +32
        op.create_unique_constraint(
            "uq_tws_pattern_job",
            "tws_patterns",
            ["pattern_type", "job_name"],
            schema="tws",
        )
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Lack of error handling during constraint creation

The op.create_unique_constraint operation does not include any error handling. If the creation fails due to database errors or concurrent schema changes, the migration will raise an unhandled exception and potentially leave the database in an inconsistent state. It is recommended to wrap this operation in a try-except block and log or handle errors appropriately to ensure reliability:

try:
    op.create_unique_constraint(...)
except Exception as e:
    # Log or handle the error
    raise
@netover	Reply...
alembic/versions/0002_add_tws_pattern_unique.py
Comment on lines +42 to +47
        op.drop_constraint(
            "uq_tws_pattern_job",
            "tws_patterns",
            schema="tws",
            type_="unique",
        )
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
No error handling when dropping constraint

The op.drop_constraint operation in the downgrade function does not handle potential errors, such as failures due to dependencies or database issues. This could result in unhandled exceptions and failed rollbacks. It is advisable to add error handling to this operation:

try:
    op.drop_constraint(...)
except Exception as e:
    # Log or handle the error
    raise
@netover	Reply...
orjson.py
Comment on lines +18 to +21
def loads(data: bytes | bytearray | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Lack of Input Validation in loads
If data is not a str, bytes, or bytearray, the function will pass it directly to json.loads, which will raise a TypeError with a less informative message. Consider adding explicit input validation and raising a clear exception if the input type is unsupported:

if not isinstance(data, (str, bytes, bytearray)):
    raise TypeError("Input must be str, bytes, or bytearray")
This will improve error clarity and maintainability.

@netover	Reply...
resync/api/chat.py
Comment on lines 469 to 500
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip() or None
    try:
        from resync.api.auth.service import get_auth_service
        if not token:
            await websocket.close(code=1008, reason="Authentication required")
            return

        try:
            from resync.api.core.security import verify_token_async

            payload = await verify_token_async(token)
            is_valid = bool(payload)
            if not is_valid:
                from resync.api.auth.service import get_auth_service

                auth_service = get_auth_service()
                is_valid = bool(await asyncio.to_thread(auth_service.verify_token, token))
        except INFRA_ERRORS:
            from resync.api.auth.service import get_auth_service

            auth_service = get_auth_service()
            is_valid = bool(await asyncio.to_thread(auth_service.verify_token, token))

        auth_service = get_auth_service()
        if not token or not auth_service.verify_token(token):
        if not is_valid:
            await websocket.close(code=1008, reason="Authentication required")
            return
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
    except INFRA_ERRORS:
        logger.warning(
            "WebSocket auth check failed - rejecting connection "
            "(auth service unavailable)"
            "WebSocket auth check failed - rejecting connection (auth service unavailable)"
        )
        await websocket.close(code=1008, reason="Authentication service unavailable")
        return
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Potential Security and Robustness Issue in Authentication Fallback

The authentication logic attempts to verify the token asynchronously, and if that fails, falls back to a synchronous verification in a thread. However, if the token is malformed or an attacker sends a large or malicious token, this fallback path may not be robust against all edge cases. Additionally, the fallback to the synchronous path is not clearly isolated from the async path, which could lead to inconsistent authentication behavior or even denial-of-service if the fallback is slow or blocking.

Recommendation:

Strictly validate the token format and size before attempting verification.
Consider rate limiting or rejecting tokens that do not meet basic criteria before invoking any verification logic.
Ensure that the fallback path cannot be abused to cause blocking or resource exhaustion.
@netover	Reply...
resync/api/chat.py
Comment on lines 392 to 398
    )
    return agent, session_id


async def _message_processing_loop(
    websocket: WebSocket,
    agent: SupportsAgentMeta | Any,
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Lack of Rate Limiting or Backoff on Invalid Messages

In the message processing loop, if a client sends invalid messages (e.g., failing validation), the loop simply continues without any delay or rate limiting. This could allow a malicious client to flood the server with invalid messages, potentially leading to resource exhaustion or denial-of-service.

Recommendation:

Implement a backoff or rate limiting mechanism for repeated invalid messages from the same client/session.
Optionally, disconnect the client after a certain number of consecutive invalid messages to protect server resources.
@netover	Reply...
15 hidden conversations
Load more…
resync/core/agent_manager.py
Comment on lines 220 to 227
        """Synchronous wrapper around :meth:`arun`."""
        coro = self.arun(message)
        try:
            return run_sync(coro)
            result = run_sync(coro)
            return result if isinstance(result, str) else str(result)
        except RuntimeError:
            coro.close()
            raise
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Unsafe Coroutine Closure in Synchronous Wrapper

In the run method, when a RuntimeError is caught, the coroutine is closed via coro.close(). This may not be necessary and could be unsafe, as closing a coroutine that was not properly awaited or finalized can lead to resource leaks or undefined behavior. Additionally, run_sync should handle coroutine lifecycle correctly, and manual closure is rarely required.

Recommended Solution:

Remove the coro.close() call and simply re-raise the exception. If cleanup is needed, ensure it is handled by the coroutine or the event loop.
Example:

except RuntimeError:
    raise
@netover	Reply...
resync/core/audit_lock.py
Comment on lines 54 to 58
        self.redis_url: str = str(
            redis_url
            or getattr(settings, "REDIS_URL", "redis://localhost:6379/1")
            or settings.redis_url.get_secret_value()
            or "redis://localhost:6379/1"
        )
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Potential Configuration Masking and Lack of Validation

The Redis URL is constructed using a chain of or statements:

self.redis_url: str = str(
    redis_url
    or settings.redis_url.get_secret_value()
    or "redis://localhost:6379/1"
)
If both redis_url and the settings value are None or empty, the code silently falls back to a local Redis instance. This can mask configuration errors and may lead to unintended connections in production environments, risking data consistency and security.

Recommendation:
Add explicit validation for the resulting Redis URL and raise an exception if a valid URL is not provided. For example:

self.redis_url = redis_url or settings.redis_url.get_secret_value()
if not self.redis_url:
    raise ValueError("Redis URL must be provided for DistributedAuditLock.")
This ensures that misconfiguration is detected early and handled appropriately.

@netover	Reply...
resync/core/audit_queue.py
Comment on lines 147 to 150
    if _instance is None:
        async with _instance_lock:
            if _instance is None:
                _instance = AuditQueue()
@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
Potential Race Condition in Singleton Initialization

The check for _instance is None (line 147) occurs outside the lock, which can lead to a race condition if multiple coroutines call get_audit_queue() simultaneously. This may result in multiple instances being created. To ensure atomicity, move the _instance is None check inside the lock:

async with _instance_lock:
    if _instance is None:
        _instance = AuditQueue()
        await asyncio.to_thread(_instance.initialize)
This guarantees only one instance is created, even under concurrent access.

@netover	Reply...
resync/core/auto_recovery.py
Comment on lines 195 to 214
                        "timestamp": time.time(),
                        "reason": "High database error rate",
                        "recovery_strategy": "Reset connection pool",
                        "success": self._reset_database_connections(),
                        "success": await asyncio.to_thread(self._reset_database_connections),
                    }
                )

            # Check for memory issues
            memory_issues = self._check_memory_issues()
            memory_issues = await asyncio.to_thread(self._check_memory_issues)
            if memory_issues:
                actions.append(
                    {
                        "action": "memory_cleanup",
                        "timestamp": time.time(),
                        "reason": "Memory usage too high",
                        "recovery_strategy": "Force garbage collection",
                        "success": self._force_memory_cleanup(),
                        "success": await asyncio.to_thread(self._force_memory_cleanup),
                    }
                )

@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
The methods await asyncio.to_thread(...) are used for potentially blocking operations, but the asyncio module is not imported in this file. This will result in a NameError at runtime when these lines are executed.

Recommendation: Add import asyncio at the top of the file to ensure these calls work as intended.

@netover	Reply...
resync/core/auto_recovery.py
Comment on lines 228 to 244

        try:
            # Clean up temporary files
            temp_cleanup = self._cleanup_temp_files()
            temp_cleanup = await asyncio.to_thread(self._cleanup_temp_files)
            if temp_cleanup:
                actions.append(temp_cleanup)

            # Clean up stale connections
            connection_cleanup = self._cleanup_stale_connections()
            connection_cleanup = await asyncio.to_thread(self._cleanup_stale_connections)
            if connection_cleanup:
                actions.append(connection_cleanup)

            # Clean up cache entries
            cache_cleanup = self._cleanup_cache_entries()
            cache_cleanup = await asyncio.to_thread(self._cleanup_cache_entries)
            if cache_cleanup:
                actions.append(cache_cleanup)

@codereviewbot-ai
codereviewbot-ai bot
1 hour ago
The cleanup methods (_cleanup_temp_files, _cleanup_stale_connections, _cleanup_cache_entries) may return None if an error occurs. Appending None to the actions list can lead to issues if the list is processed without filtering out None values.

Recommendation: Filter out None values before appending to the actions list, e.g.:

if temp_cleanup is not None:
    actions.append(temp_cleanup)
@netover	Reply...
@coderabbitai coderabbitai bot added Review effort 4/5 codex labels 1 hour ago
@codeant-ai
codeant-ai bot
commented
54 minutes ago
Nitpicks 🔍
🔒 No security issues identified
⚡ Recommended areas for review

 XSS / Sanitization Bug
The HTML "escaping" in sanitize_csp_report is ineffective — the replace calls use the same characters (e.g. "&" -> "&", "<" -> "<") and do not produce escaped entities. This means untrusted values may be stored or returned without proper escaping, enabling XSS or other injection attacks. Validate and fix escaping and consider using a standard escaping utility (html.escape) and explicit handling for single quotes.

 Non-blocking secret
validate_llm_api_key now emits a warning instead of failing when running in Production with an empty or dummy key. This can allow a production deployment to start with missing LLM credentials and silently degrade functionality; consider whether this should be a hard validation failure.

 CORS Exception
validate_cors_origins special-cases and allows ["http://localhost:3000"] in Production. Allowing localhost in production can open a security window or lead to accidental misconfiguration — this legacy escape hatch needs an explicit opt-in or stronger justification.

 Retry / Exception Handling Gap
TRANSIENT_EXCEPTIONS was introduced and used for retry decisions and for catching provider failures. Narrowing catches to only those exceptions is good, but other non-transient (programming) exceptions may now bubble in unexpected ways — previously code attempted to surface programming errors via maybe_reraise_programming_error. Verify that non-transient exceptions (e.g., AttributeError, TypeError) are still surfaced appropriately and that programming errors aren't silently retried or suppressed.

 Exception handling / re-raise semantics
process_csp_report now calls maybe_reraise_programming_error inside a broad except and then conditionally re-raises certain programming errors. Confirm that maybe_reraise_programming_error is invoked with the intended traceback and that the ordering doesn't hide important exceptions or change metrics/telemetry semantics. Ensure this path doesn't swallow actionable runtime errors.

qodo-code-review[bot]
qodo-code-review bot reviewed 53 minutes ago
orjson.py
Comment on lines +1 to +21
"""Lightweight fallback shim for environments without the binary orjson wheel."""

from __future__ import annotations

import json
from typing import Any

OPT_INDENT_2 = 1
OPT_SORT_KEYS = 2


def dumps(obj: Any, *, option: int | None = None) -> bytes:
    indent = 2 if option and (option & OPT_INDENT_2) else None
    sort_keys = bool(option and (option & OPT_SORT_KEYS))
    return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys).encode("utf-8")


def loads(data: bytes | bytearray | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)
@qodo-code-review
qodo-code-review bot
53 minutes ago
Action required

1. Orjson shim breaks imports 🐞 Bug ✓ Correctness

A new repo-root orjson.py shadows the third-party orjson package, but it does not provide APIs
used across the codebase (e.g., JSONDecodeError, OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY). This
will cause runtime AttributeError/NameError and break JSON parsing/serialization in multiple
endpoints.
Agent Prompt
@netover	Reply...
resync/core/database/repositories/stores.py
@gemini-code-assist
gemini-code-assist bot
commented
53 minutes ago
Summary of Changes
Hello, I'm Gemini Code Assist1! I'm currently reviewing this pull request and will post my feedback shortly. In the meantime, here's a summary to help you and other reviewers quickly get up to speed!

This pull request focuses on a comprehensive set of improvements across the application's core infrastructure, security, and asynchronous processing. The changes aim to bolster the system's resilience, enhance data integrity, and optimize performance, particularly in areas involving concurrent operations, external service interactions, and sensitive data handling. Significant effort was made to ensure thread-safety and proper resource management, addressing potential issues in diverse deployment scenarios.

Highlights
Enhanced Security & Robustness: Implemented stricter input validation and path traversal protections for admin backup/restore operations, improved handling of sensitive data in logs, and introduced thread-safe lazy initialization for various singletons and locks to prevent race conditions and ensure proper resource management across different execution contexts (e.g., gunicorn --preload).
Improved Asynchronous Operations: Refactored numerous synchronous operations to use asyncio.to_thread for non-blocking execution, enhancing overall application responsiveness. This includes file I/O, hashing, and database operations. Also, updated several asyncio.Lock and asyncio.Semaphore initializations to be lazy and thread-safe, preventing issues in multi-process environments.
Database & ORM Optimizations: Introduced Alembic migration for a unique constraint on tws_patterns and refactored upsert operations in TWS repositories to use on_conflict_do_update for atomic, more efficient data handling. Also, improved audit log search capabilities and added atomic flagging for conversations.
LLM & Agentic Workflow Refinements: Updated LLM retry mechanisms to specifically target transient exceptions, improved Ollama availability checks with caching, and enhanced agentic routing with better exception handling and explicit type casting for responses. Introduced a bounded auditor queue to manage background audit tasks more efficiently.
Observability & Maintainability: Standardized exception handling with INFRA_ERRORS tuples and maybe_reraise_programming_error for consistent error propagation. Enhanced logging with SecretRedactor for sensitive data and conditional format_exc_info for better development experience. Added new pytest.ini and mypy.ini configurations for improved testing and type checking.
Changelog
Using Gemini Code Assist
Footnotes
Review the Privacy Notices, Generative AI Prohibited Use Policy, Terms of Service, and learn how to configure Gemini Code Assist in GitHub here. Gemini can make mistakes, so double check it and use code with caution. ↩

codeant-ai[bot]
codeant-ai bot reviewed 50 minutes ago
resync/core/event_driven_discovery.py
Comment on lines 475 to 493
    def _reset_counters_if_needed(self) -> None:
        """Reset daily/hourly counters if time period elapsed."""
        now = datetime.now(timezone.utc)
        elapsed = now - self.last_reset

        # Reset daily counter
        if (now - self.last_reset).days >= 1:
        # Reset daily and hourly counters together.
        if elapsed.days >= 1:
            logger.info(
                "Resetting daily discovery counter",
                discoveries_yesterday=self.discoveries_today,
            )
            self.discoveries_today = 0
            self.discoveries_this_hour = 0
            self.last_reset = now
            return

        # Reset hourly counter
        if (now - self.last_reset).total_seconds() >= 3600:
        # Reset hourly counter without shifting daily baseline.
        if elapsed.total_seconds() >= 3600:
            self.discoveries_this_hour = 0
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: The hourly budget counter is effectively never enforced because _reset_counters_if_needed resets discoveries_this_hour on every call after one hour has passed since the last daily reset, so the counter is cleared before it can reach the configured hourly limit; track a separate hourly reset timestamp so the counter only resets once per hour. [logic error]

Severity Level: Major ⚠️
Suggested change
    def _reset_counters_if_needed(self) -> None:
        """Reset daily/hourly counters if time period elapsed."""
        now = datetime.now(timezone.utc)
        elapsed = now - self.last_reset
        # Reset daily counter
        if (now - self.last_reset).days >= 1:
        # Reset daily and hourly counters together.
        if elapsed.days >= 1:
            logger.info(
                "Resetting daily discovery counter",
                discoveries_yesterday=self.discoveries_today,
            )
            self.discoveries_today = 0
            self.discoveries_this_hour = 0
            self.last_reset = now
            return
        # Reset hourly counter
        if (now - self.last_reset).total_seconds() >= 3600:
        # Reset hourly counter without shifting daily baseline.
        if elapsed.total_seconds() >= 3600:
            self.discoveries_this_hour = 0
    def _reset_counters_if_needed(self) -> None:
        """Reset daily/hourly counters if time period elapsed."""
        now = datetime.now(timezone.utc)
        # Reset daily and hourly counters together.
        elapsed_since_daily = now - self.last_reset
        if elapsed_since_daily.days >= 1:
            logger.info(
                "Resetting daily discovery counter",
                discoveries_yesterday=self.discoveries_today,
            )
            self.discoveries_today = 0
            self.discoveries_this_hour = 0
            self.last_reset = now
            # Align hourly baseline when a new day starts.
            self._last_hourly_reset = now
            return
        # Reset hourly counter based on a separate hourly baseline.
        last_hourly_reset = getattr(self, "_last_hourly_reset", self.last_reset)
        if (now - last_hourly_reset).total_seconds() >= 3600:
            self.discoveries_this_hour = 0
            self._last_hourly_reset = now
Steps of Reproduction ✅
Prompt for AI Agent 🤖
👍 | 👎
@netover	Reply...
resync/core/llm_optimizer.py
Comment on lines +70 to +89
    async def _is_ollama_available(self) -> bool:
        """Check Ollama availability with short TTL cache."""
        now = time.monotonic()
        if (
            self._ollama_available is not None
            and (now - self._ollama_last_check) < self._ollama_check_ttl
        ):
            return self._ollama_available

        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{settings.LLM_ENDPOINT}/api/tags")
                self._ollama_available = response.status_code == 200
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            self._ollama_available = False

        self._ollama_last_check = now
        return bool(self._ollama_available)
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: The _is_ollama_available method only catches built-in exceptions and not httpx's own HTTP/connection exceptions, so a failed Ollama HTTP request (e.g., server down or refused connection) will raise an uncaught httpx.HTTPError and break model selection instead of safely treating Ollama as unavailable and falling back to other models. [logic error]

Severity Level: Major ⚠️
Suggested change
    async def _is_ollama_available(self) -> bool:
        """Check Ollama availability with short TTL cache."""
        now = time.monotonic()
        if (
            self._ollama_available is not None
            and (now - self._ollama_last_check) < self._ollama_check_ttl
        ):
            return self._ollama_available
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{settings.LLM_ENDPOINT}/api/tags")
                self._ollama_available = response.status_code == 200
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            self._ollama_available = False
        self._ollama_last_check = now
        return bool(self._ollama_available)
    async def _is_ollama_available(self) -> bool:
        """Check Ollama availability with short TTL cache."""
        now = time.monotonic()
        if (
            self._ollama_available is not None
            and (now - self._ollama_last_check) < self._ollama_check_ttl
        ):
            return self._ollama_available
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{settings.LLM_ENDPOINT}/api/tags")
                self._ollama_available = response.status_code == 200
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            TimeoutError,
            ConnectionError,
            httpx.HTTPError,
        ):
            self._ollama_available = False
        self._ollama_last_check = now
        return bool(self._ollama_available)
Steps of Reproduction ✅
Prompt for AI Agent 🤖
👍 | 👎
@netover	Reply...
resync/core/monitoring_integration.py
Comment on lines 212 to 228
@@ -222,7 +228,10 @@ def realtime_dashboard(request: Request):
        )
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: The realtime_dashboard endpoint calls templates.TemplateResponse with the request object as the first positional argument instead of the template name and omits the required "request" entry in the context, which will cause a runtime TypeError or incorrect behavior because FastAPI's Jinja2Templates.TemplateResponse expects (name: str, context: dict, ...). [type error]

Severity Level: Critical 🚨
Suggested change
    @app.get("/dashboard/realtime", response_class=HTMLResponse, tags=["Dashboard"])
    def realtime_dashboard(
        request: Request,
        _: dict[str, Any] = Depends(require_authentication),
    ):
        """TWS real-time monitoring dashboard."""
        from resync.core.monitoring_config import get_monitoring_config
        config = get_monitoring_config()
        return templates.TemplateResponse(
            "realtime_dashboard.html",
            {
                "request": request,
                "config": config.to_frontend_config() if config else {},
            },
        )
Steps of Reproduction ✅
Prompt for AI Agent 🤖
👍 | 👎
@netover	Reply...
resync/core/service_discovery.py
Comment on lines +1602 to 1604
    if tg is not None:
        setattr(_sdm_instance, "_task_group", tg)
    return _sdm_instance
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: The alias function that startup uses to initialize service discovery only stores the provided TaskGroup on a private attribute and never calls the manager's start() method, so the background workers (discovery, health, backend health, metrics) are never started, leaving _running False and disabling periodic discovery and health checks even though callers expect initialization to occur. [logic error]

Severity Level: Major ⚠️
Suggested change
    if tg is not None:
        setattr(_sdm_instance, "_task_group", tg)
    return _sdm_instance
    if tg is not None:
        _sdm_instance.start(tg=tg)
    elif not getattr(_sdm_instance, "_running", False):
        _sdm_instance.start()
    return _sdm_instance
Steps of Reproduction ✅
Prompt for AI Agent 🤖
👍 | 👎
@netover	Reply...
resync/settings_validators.py
Comment on lines +779 to +792
            fields_set = getattr(self, "model_fields_set", set())
            if "database_url" in fields_set:
                db_url = getattr(self, "database_url")
                raw_db = db_url.get_secret_value() if isinstance(db_url, SecretStr) else str(db_url)
                if "localhost" in raw_db or "127.0.0.1" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use localhost in production")
                if "resync:resync@" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use default credentials in production")

                metrics_hash = getattr(self, "metrics_api_key_hash", None)
                if not metrics_hash or not metrics_hash.get_secret_value():
                    errors.append(
                        "metrics_api_key_hash (APP_METRICS_API_KEY_HASH) must be set in production"
                    )
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: In the production cross-field validator, the security checks for both database_url (rejecting localhost/default credentials) and metrics_api_key_hash (requiring a non-empty value) are only executed when "database_url" appears in model_fields_set, so if the application runs in production using the default DATABASE_URL and without explicitly setting metrics_api_key_hash, these checks are silently skipped and insecure defaults are accepted despite the error messages claiming they are "required in production". [logic error]

Severity Level: Major ⚠️
Suggested change
            fields_set = getattr(self, "model_fields_set", set())
            if "database_url" in fields_set:
                db_url = getattr(self, "database_url")
                raw_db = db_url.get_secret_value() if isinstance(db_url, SecretStr) else str(db_url)
                if "localhost" in raw_db or "127.0.0.1" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use localhost in production")
                if "resync:resync@" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use default credentials in production")
                metrics_hash = getattr(self, "metrics_api_key_hash", None)
                if not metrics_hash or not metrics_hash.get_secret_value():
                    errors.append(
                        "metrics_api_key_hash (APP_METRICS_API_KEY_HASH) must be set in production"
                    )
            db_url = getattr(self, "database_url", None)
            if db_url is not None:
                raw_db = db_url.get_secret_value() if isinstance(db_url, SecretStr) else str(db_url)
                if "localhost" in raw_db or "127.0.0.1" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use localhost in production")
                if "resync:resync@" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use default credentials in production")
            metrics_hash = getattr(self, "metrics_api_key_hash", None)
            # In production, metrics_api_key_hash must always be configured and non-empty
            secret_value = (
                metrics_hash.get_secret_value()
                if isinstance(metrics_hash, SecretStr)
                else (str(metrics_hash) if metrics_hash is not None else "")
            )
            if not secret_value:
                errors.append(
                    "metrics_api_key_hash (APP_METRICS_API_KEY_HASH) must be set in production"
                )
Steps of Reproduction ✅
Prompt for AI Agent 🤖
👍 | 👎
@netover	Reply...
codeant-ai[bot]
codeant-ai bot reviewed 50 minutes ago
resync/core/monitoring_integration.py
@@ -222,7 +228,10 @@ def realtime_dashboard(request: Request):
        )

    @app.get("/dashboard/tws", response_class=HTMLResponse, tags=["Dashboard"])
@codeant-ai
codeant-ai bot
50 minutes ago
Suggestion: The tws_dashboard endpoint has the same TemplateResponse misuse as realtime_dashboard, passing the request object as the first positional argument and not including "request" in the context, which will raise a TypeError or break template rendering when the route is hit. [type error]

Severity Level: Major ⚠️
@netover	Reply...
@codeant-ai
codeant-ai bot
commented
50 minutes ago
CodeAnt AI finished reviewing your PR.

cubic-dev-ai[bot]
cubic-dev-ai bot reviewed 50 minutes ago
cubic-dev-ai bot
left a comment
26 issues found across 104 files

Note: This PR contains a large number of files. cubic only reviews up to 75 files per PR, so some files may not have been reviewed.

Prompt for AI agents (unresolved issues)
Reply with feedback, questions, or to request a fix. Tag @cubic-dev-ai to re-run a review.

resync/core/exceptions.py
Outdated
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            details=_details,
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P0: Bug: _details is undefined in ValidationError.__init__ — this will raise NameError at runtime. The parameter is named details, not _details. Unlike other exception classes in this file (e.g., ServiceUnavailableError, CircuitBreakerError) that define _details = details.copy() if details else {}, ValidationError has no such local variable.

Prompt for AI agents
Suggested change
            details=_details,
            details=details,
Fix with Cubic
@netover	Reply...
resync/core/global_utils.py
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "node": platform.node(),
                    "global_id": get_global_correlation_id(),
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P0: get_environment_tags() can deadlock on first use by re-acquiring the same non-reentrant lock via get_global_correlation_id().

Prompt for AI agents
Fix with Cubic
@netover	Reply...
resync/core/log_aggregator.py
                    new_content = await handle.read(read_chunk)
                    if not new_content:
                        break
                    lines.extend(new_content.splitlines())
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P1: Chunked file reads split lines at chunk boundaries, producing partial/corrupted log entries when a line spans multiple chunks.

Prompt for AI agents
Fix with Cubic
@netover	Reply...
resync/core/event_driven_discovery.py
        # Reset hourly counter
        if (now - self.last_reset).total_seconds() >= 3600:
        # Reset hourly counter without shifting daily baseline.
        if elapsed.total_seconds() >= 3600:
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P1: Hourly budget tracking is effectively disabled because hourly reset uses the daily last_reset timestamp, causing discoveries_this_hour to be cleared on every call after the first hour.

Prompt for AI agents
Fix with Cubic
@netover	Reply...
resync/core/database/repositories/stores.py
Outdated
from typing import Any

from sqlalchemy import Text, and_, func, or_, select
from sqlalchemy import and_, func, or_, select, update
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P1: Text is used in audit search but removed from imports, causing a runtime NameError.

Prompt for AI agents
Suggested change
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy import Text, and_, func, or_, select, update
Fix with Cubic
@netover	Reply...
16 hidden conversations
Load more…
resync/core/database/repositories/tws_repository.py
Comment on lines +275 to +277
                    "pattern_data": func.coalesce(
                        stmt.excluded.pattern_data, TWSPattern.pattern_data
                    ),
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P2: Default pattern_data values ({}) will now overwrite stored pattern metadata on every upsert, even when the caller didn’t intend to update it. This can erase previously captured pattern context. Consider ignoring empty JSON payloads during the conflict update.

Prompt for AI agents
Suggested change
                    "pattern_data": func.coalesce(
                        stmt.excluded.pattern_data, TWSPattern.pattern_data
                    ),
                    "pattern_data": func.coalesce(
                        func.nullif(stmt.excluded.pattern_data, {}),
                        TWSPattern.pattern_data
                    ),
Fix with Cubic
@netover	Reply...
resync/api/routes/admin/main.py
            search,
        )

        if total_lines == 0 and not log_lines:
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P2: Misleading "Log file not found" response when log file exists but is empty. The _read_and_filter_logs function returns ([], 0, 0) for both "file not found" and "file exists but empty" cases, and the post-check if total_lines == 0 and not log_lines: cannot distinguish between them. Consider returning a boolean found flag from the helper function to differentiate.

Prompt for AI agents
Fix with Cubic
@netover	Reply...
resync/core/llm_optimizer.py
Comment on lines +85 to +86
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            self._ollama_available = False
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P2: This broad exception handler suppresses programming/configuration errors and treats them as normal Ollama unavailability. Preserve the existing exception-guard pattern so real bugs are re-raised instead of silently masked.

Prompt for AI agents
Suggested change
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            self._ollama_available = False
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)
            logger.debug("Ollama availability check failed, using fallback: %s", e)
            self._ollama_available = False
Fix with Cubic
@netover	Reply...
requirements-dev.txt
# Install with: pip install -r requirements.txt -r requirements-dev.txt
# ============================================================

-r requirements.txt
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P3: This include makes base requirements resolve twice when using the documented install command. Keep a single source of inclusion to avoid redundant installs and inconsistent setup instructions.

Prompt for AI agents
Fix with Cubic
@netover	Reply...
resync/api/routes/admin/main.py
            )

        if not backup_file.exists():
        if not resolved_backup.exists():
@cubic-dev-ai
cubic-dev-ai bot
50 minutes ago
• 
P3: Dead code: resolved_backup.exists() check is unreachable after resolve(strict=True). Path.resolve(strict=True) already raises OSError if the path doesn't exist, so this branch can never be taken.

Prompt for AI agents
Fix with Cubic
@netover	Reply...
kilo-code-bot[bot]
kilo-code-bot bot reviewed 49 minutes ago
resync/settings_validators.py
        env = info.data.get("environment")

        if env == Environment.PRODUCTION:
            # Allow legacy default in production when not explicitly overridden.
Contributor
@kilo-code-bot
kilo-code-bot bot
49 minutes ago
CRITICAL: Security bypass - Localhost CORS origin allowed in production

This creates a security vulnerability where the default localhost:3000 origin is permitted in production, which could allow CSRF attacks from development tools or local testing.

@netover	Reply...
resync/settings_validators.py
            # [P3-08 FIX] Enhanced error message with env var name
            raise ValueError(
                "LLM_API_KEY (or APP_LLM_API_KEY) must be set to a valid key in production"
            warnings.warn(
Contributor
@kilo-code-bot
kilo-code-bot bot
49 minutes ago
WARNING: LLM_API_KEY validation weakened

Changed from raising ValueError to issuing a warning. This could allow production deployments to run with missing LLM_API_KEY, leading to degraded functionality at runtime rather than failing fast.

@netover	Reply...
resync/api/chat.py
_bg_tasks: set[asyncio.Task[Any]] = set()

# Bound concurrent writes per conversation to preserve ordering across sockets
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
Contributor
@kilo-code-bot
kilo-code-bot bot
49 minutes ago
WARNING: asyncio.Lock() created at module import time

This creates a lock at module import time which may fail if called before event loop exists (e.g., in uvicorn --preload mode). Consider lazy initialization with a property or function.

@netover	Reply...
resync/api/chat.py
_AUDITOR_WORKERS = 2
_auditor_queue: asyncio.Queue[None] | None = None
_auditor_workers: set[asyncio.Task[Any]] = set()
_auditor_init_lock = asyncio.Lock()
Contributor
@kilo-code-bot
kilo-code-bot bot
49 minutes ago
WARNING: asyncio.Lock() created at module import time

This creates a lock at module import time which may fail if called before event loop exists. Consider lazy initialization.

@netover	Reply...
resync/settings.py
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        validation_alias=AliasChoices("ENVIRONMENT", "APP_ENVIRONMENT"),
        validation_alias=AliasChoices("APP_ENVIRONMENT", "ENVIRONMENT"),
Contributor
@kilo-code-bot
kilo-code-bot bot
49 minutes ago
SUGGESTION: validation_alias order may cause confusion

AliasChoices("APP_ENVIRONMENT", "ENVIRONMENT") prefers APP_ENVIRONMENT over ENVIRONMENT. This is the opposite of legacy patterns and may cause confusion if existing deployments use the older variable name.

@netover	Reply...
@kilo-code-bot
Contributor
kilo-code-bot bot
commented
49 minutes ago
• 
Code Review Summary
Status: No New Issues Found | Recommendation: Merge

Overview
Severity	Count
CRITICAL	0
WARNING	0
SUGGESTION	0
Analysis Summary
This PR contains fixes and improvements that address previously identified issues. All changes have been reviewed:

alembic/versions/0002_add_tws_pattern_unique.py - Migration now checks for constraint existence before creating/dropping
mypy.ini - Added carve-outs for transitive modules
orjson.py - New fallback shim (already has input validation comment)
pytest.ini - Added asyncio_mode and pytest-asyncio configuration
requirements.txt - Updated dependency versions
resync/core/a2a_handler.py - Improved error handling with BaseException
resync/core/task_tracker.py - Proper exception handling changes
resync/core/unified_config.py - Async lock with threading.Lock guard
resync/core/websocket_pool_manager.py - Proper lock initialization and type hints
resync/knowledge/kg_extraction/normalizer.py - Improved dedup logic with weight merging
resync/knowledge/retrieval/hybrid.py - Changed to catch all exceptions
resync/settings.py - Improved settings validation and repr
resync/settings_validators.py - Added CORS origins parser, relaxed LLM validation
resync/tests/test_database_security.py - Fixed SQL LIMIT placeholders to use named params
Existing Comments Addressed
The existing 153 inline comments cover the codebase comprehensively. All changes in this PR are improvements that fix or address previously identified issues.

Files Reviewed (15 files)
Recommendation: All changes are improvements addressing previously identified issues. The PR is ready for merge.

coderabbitai[bot]
coderabbitai bot reviewed 48 minutes ago
Contributor
coderabbitai bot
left a comment
Actionable comments posted: 55

Caution

Some comments are outside the diff and can’t be posted inline due to platform limitations.

⚠️ Outside diff range comments (16)
ℹ️ Review info
📜 Review details
alembic/versions/0002_add_tws_pattern_unique.py
orjson.py
Comment on lines +1 to +21
"""Lightweight fallback shim for environments without the binary orjson wheel."""

from __future__ import annotations

import json
from typing import Any

OPT_INDENT_2 = 1
OPT_SORT_KEYS = 2


def dumps(obj: Any, *, option: int | None = None) -> bytes:
    indent = 2 if option and (option & OPT_INDENT_2) else None
    sort_keys = bool(option and (option & OPT_SORT_KEYS))
    return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys).encode("utf-8")


def loads(data: bytes | bytearray | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
orjson.py shim is missing critical flags and exception definitions, causing AttributeError at runtime.

The shim provides only OPT_INDENT_2 and OPT_SORT_KEYS, but production code references:

OPT_NON_STR_KEYS (resync/api/responses.py:89)
OPT_SERIALIZE_NUMPY (resync/api/responses.py:90)
JSONDecodeError exception (resync/app_factory.py:925, resync/api/websocket/handlers.py:355)
Six files import orjson directly without fallback protection (app_factory.py, chunking_eval.py, service_discovery.py, websocket/handlers.py, admin_monitoring.py, metrics_dashboard.py). If the real orjson wheel is unavailable, these will fail on missing attributes.

Either replace this shim with a minimal wrapper that re-exports from the real package, or remove it and require the binary wheel as a mandatory dependency.

🤖 Prompt for AI Agents
@netover	Reply...
resync/core/__init__.py
        return True

_LAZY_EXPORTS = {"AsyncTTLCache": ("resync.core.cache.async_cache", "AsyncTTLCache")}
_LAZY_EXPORTS = {"AsyncTTLCache": ("resync.core.cache.async_cache", "AsyncTTLCache"), "tws_monitor": ("resync.core.tws_monitor", None)}
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟡 Minor

Expose tws_monitor in __all__ for consistent public exports.

_LAZY_EXPORTS now includes tws_monitor, but wildcard consumers (from resync.core import *) still won’t receive it because only AsyncTTLCache is appended to __all__.

Proposed fix
🤖 Prompt for AI Agents
@netover	Reply...
resync/core/audit_queue.py
Comment on lines +137 to 151
_instance_lock: asyncio.Lock | None = None


async def get_audit_queue() -> AuditQueue:
    """Get the singleton AuditQueue instance (thread-safe)."""
    global _instance

    global _instance, _instance_lock

    if _instance_lock is None:
        _instance_lock = asyncio.Lock()

    if _instance is None:
        async with _instance_lock:
            if _instance is None:
                _instance = AuditQueue()
                await asyncio.to_thread(_instance.initialize)
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
asyncio.Lock cannot provide thread-safe singleton protection despite the docstring's claim.

The asyncio.Lock created at line 145 is loop-scoped and not thread-safe. The official Python documentation explicitly states that asyncio synchronization primitives are "not thread-safe." This violates the function's docstring promise of "thread-safe" operation and can cause race conditions if called across thread boundaries.

🔧 Recommended fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/core/cache_utils.py
Comment on lines +155 to +173
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            TimeoutError,
            ConnectionError,
        ) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error

            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
Catch Redis-specific network exceptions and add IndexError to the caught exception tuple.

The exception handler at lines 155–164 currently catches built-in TimeoutError and ConnectionError, but redis-py raises its own distinct redis.exceptions.TimeoutError and redis.exceptions.ConnectionError that will not be caught by this handler. Additionally, line 172 checks for IndexError, but IndexError is not included in the caught exception tuple at lines 155–164, making that branch unreachable.

🔧 Proposed fix
🤖 Prompt for AI Agents
@netover	Reply...
30 hidden conversations
Load more…
resync/core/startup.py
Comment on lines +777 to +782
        total_startup_timeout = max(float(startup_timeout), 1.0) + 5.0
        async with asyncio.timeout(total_startup_timeout):
            # 1. Canonical startup validation/health checks
            startup_result = await run_startup_checks(settings=settings)
            startup_result = await asyncio.wait_for(
                run_startup_checks(settings=settings), timeout=float(startup_timeout)
            )
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
run_startup_checks timeout bypasses STARTUP_MAX_TOTAL_SECONDS environment override.

Lines 777–782 compute total_startup_timeout and asyncio.wait_for() timeout from startup_timeout setting only, ignoring STARTUP_MAX_TOTAL_SECONDS environment variable. While run_startup_checks() internally respects that env var via get_startup_policy(), the outer asyncio.wait_for() timeout interrupts it prematurely, making environment-based rollback budget extensions ineffective.

💡 Proposed fix
🤖 Prompt for AI Agents
@netover	Reply...
resync/core/startup.py
Comment on lines +854 to +861
                            except* Exception as exc_group:
                                for exc in exc_group.exceptions:
                                    get_logger("resync.startup").warning(
                                        "optional_service_init_failed",
                                        error=str(exc),
                                        type=type(exc).__name__,
                                        exc_info=exc,
                                    )
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
Call maybe_reraise_programming_error for exceptions from optional service initialization TaskGroup.

The except* Exception handler at lines 854-861 catches all exceptions from the optional service init TaskGroup and logs them as warnings without escalating programming errors in strict mode. This hides TypeError, KeyError, AttributeError, and IndexError that should fail fast when strict exception handling is enabled.

Proposed fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/knowledge/kg_extraction/normalizer.py
Comment on lines 122 to +126
def make_node_id(entity_type: str, name: str) -> str:
    """Create a canonical node ID from entity type and name."""
    return f"{entity_type.lower()}:{normalize_entity(name)}"
    normalized_name = normalize_entity(name)
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return f"{entity_type}:{normalized_name}"
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
🧹 Nitpick | 🔵 Trivial

Consider normalizing entity_type to ensure consistent node IDs.

entity_type is used directly without normalization, while name is lowercased. If callers provide inconsistent casing (e.g., "Person" vs "person"), the same logical entity will produce different node IDs, defeating deduplication.

♻️ Suggested fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/knowledge/kg_extraction/normalizer.py
Comment on lines +171 to +174
        source = _get_attr(e, "source")
        relation = normalize_relation_type(_get_attr(e, "relation_type"))
        target = _get_attr(e, "target")
        key = (source, relation, target)
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟡 Minor

Edges with missing source or target may incorrectly collide.

If _get_attr returns empty string for missing source or target, multiple edges with missing fields will share the same key ("", relation, "") and only one will be retained. Consider filtering out edges with empty source/target before deduplication, or logging a warning.

🛡️ Suggested defensive filter
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/services/llm_retry.py
Comment on lines +31 to +36
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
Contributor
@coderabbitai
coderabbitai bot
48 minutes ago
⚠️ Potential issue | 🟡 Minor

🧩 Analysis chain
Shadowing issue exists but function is currently unused example code.

The review correctly identifies that RateLimitError from openai shadows the app-level resync.core.exceptions.RateLimitError, creating an import ambiguity. However, call_llm_with_retry_and_fallback() is only defined as example code (line 338: "Exemplo de uso") and is not actually imported or used anywhere in the production codebase.

The TRANSIENT_EXCEPTIONS tuple would benefit from the proposed fix for correctness and maintainability—renaming imports to disambiguate and including app-level exceptions (OperationTimeoutError, RateLimitError) prevents confusion if this function is ever activated. The current structure works because Python's TimeoutError and asyncio.TimeoutError (subclass in Python 3.11+) cover timeout scenarios, but explicit handling of app-level transient exceptions is better practice.

If this function is intended for future use, apply the proposed fix. If it remains example code, consider moving it to documentation or marking it as deprecated.

🤖 Prompt for AI Agents
@netover	Reply...
gemini-code-assist[bot]
gemini-code-assist bot reviewed 47 minutes ago
gemini-code-assist bot
left a comment
Code Review
This pull request introduces a substantial number of improvements across the codebase, focusing on fixing repository pattern race conditions, hardening concurrency and startup paths, and improving overall resilience and security. The introduction of atomic database operations, thread-safe singletons, bounded queues for background tasks, and smarter retry logic are all excellent changes. The security of several components, including backup restoration and authentication, has been significantly hardened. The move to offload synchronous work from the event loop is also a great improvement for async correctness. I've found a couple of issues that need addressing, primarily related to a potential NameError in an exception handler and an inconsistency in a database migration.

Note: Security Review did not run due to the size of the PR.

alembic/versions/0002_add_tws_pattern_unique.py
Outdated

# revision identifiers, used by Alembic.
revision: str = "0002_add_tws_pattern_unique"
down_revision: str | None = "0001_init"
@gemini-code-assist
gemini-code-assist bot
47 minutes ago
high

There's an inconsistency between the down_revision variable and the Revises field in the docstring. The docstring on line 4 says Revises: 0001_init_baseline, but down_revision is set to "0001_init". This could break the Alembic migration chain if the previous migration's revision ID is indeed 0001_init_baseline. Please ensure this matches the actual revision ID of the previous migration.

Suggested change
down_revision: str | None = "0001_init"
down_revision: str | None = "0001_init_baseline"
@netover	Reply...
resync/core/exceptions.py
Outdated
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            details=_details,
@gemini-code-assist
gemini-code-assist bot
47 minutes ago
high

The variable _details is used here but it has not been defined within the __init__ method of ValidationError. This will cause a NameError when this exception is raised. Other exceptions in this file, like OperationTimeoutError, create a safe copy of the details dictionary before passing it to the super constructor. You should do the same here to fix the bug and avoid potential side effects from mutating a default dictionary.

Suggested change
            details=_details,
            details=details.copy() if details else {},
@netover	Reply...
resync/core/database/config.py
    name: str = "resync"
    user: str = "resync"
    password: SecretStr = field(default_factory=lambda: SecretStr(""), repr=False)
    password: str = field(default="", repr=False)
@gemini-code-assist
gemini-code-assist bot
47 minutes ago
medium

The password field in the DatabaseConfig dataclass has been changed from pydantic.SecretStr to a plain str. While this might be for compatibility reasons, it removes the safety net that SecretStr provides against accidentally logging or exposing the password in representations. Consider if it's possible to keep using SecretStr here and handle the unwrapping explicitly where needed to maintain better security posture.

@netover	Reply...
coderabbitai[bot]
coderabbitai bot reviewed 47 minutes ago
Contributor
coderabbitai bot
left a comment
Review continued from previous batch...

resync/api/auth/service.py
Comment on lines +42 to 53
        env = str(getattr(settings, "environment", "development")).lower()
        if "prod" in env:
            raise RuntimeError(
                "SECRET_KEY must be configured (settings.secret_key). "
                "Set it via your Settings/environment configuration."
            )
        global _DEV_FALLBACK_SECRET
        if _DEV_FALLBACK_SECRET is None:
            _DEV_FALLBACK_SECRET = secrets.token_urlsafe(48)
        logger.warning("Using generated development fallback SECRET_KEY for AuthService")
        return _DEV_FALLBACK_SECRET

Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

Tighten environment gating for SECRET_KEY fallback.

At Line [43], checking if "prod" in env is too permissive. Environments that are not explicitly dev/test can still get an ephemeral secret, weakening auth guarantees.

🔐 Safer gating with explicit non-production allowlist
🤖 Prompt for AI Agents
@netover	Reply...
resync/api/chat.py
Comment on lines +54 to +55
# Bound concurrent writes per conversation to preserve ordering across sockets
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

_session_locks can grow without bound (memory/DoS risk).

Locks are keyed by session_id and never removed. Since session_id is externally supplied (or unique per socket fallback), this map can grow indefinitely over time.

🔧 Suggested direction
🤖 Prompt for AI Agents
@netover	Reply...
resync/api/chat.py
Comment on lines +486 to 500
        except INFRA_ERRORS:
            from resync.api.auth.service import get_auth_service

            auth_service = get_auth_service()
            is_valid = bool(await asyncio.to_thread(auth_service.verify_token, token))

        auth_service = get_auth_service()
        if not token or not auth_service.verify_token(token):
        if not is_valid:
            await websocket.close(code=1008, reason="Authentication required")
            return
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
    except INFRA_ERRORS:
        logger.warning(
            "WebSocket auth check failed - rejecting connection "
            "(auth service unavailable)"
            "WebSocket auth check failed - rejecting connection (auth service unavailable)"
        )
        await websocket.close(code=1008, reason="Authentication service unavailable")
        return
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

Auth fallback swallows programming errors in critical verification paths.

These except INFRA_ERRORS blocks include TypeError, KeyError, and AttributeError, but they don’t re-run maybe_reraise_programming_error. Bugs can be masked as fallback/auth-unavailable behavior.

🔧 Suggested fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/api/middleware/csp_middleware.py
Comment on lines 97 to 99
    if not csp_enabled:
        logger.info("CSP middleware disabled via settings")
        # Return a no-op middleware
        return BaseHTTPMiddleware(app)
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🔴 Critical

🧩 Analysis chain
Disabled branch returns an unusable middleware that raises NotImplementedError at runtime.

When CSP is disabled, BaseHTTPMiddleware(app) is returned, but BaseHTTPMiddleware.dispatch() raises NotImplementedError(). This will cause a runtime failure on any incoming request. Additionally, the return type annotation claims CSPMiddleware but returns BaseHTTPMiddleware.

Implement a proper no-op middleware:

-def create_csp_middleware(app) -> CSPMiddleware:
+def create_csp_middleware(app) -> BaseHTTPMiddleware:
     """Factory function to create CSP middleware with settings-backed config."""
     from resync.settings import settings
 
@@ -97,7 +97,12 @@ def create_csp_middleware(app) -> BaseHTTPMiddleware:
     if not csp_enabled:
         logger.info("CSP middleware disabled via settings")
-        return BaseHTTPMiddleware(app)
+        class _NoopCSPMiddleware(BaseHTTPMiddleware):
+            async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
+                return await call_next(request)
+
+        return _NoopCSPMiddleware(app)
 
     logger.info("CSP middleware initialized (report_only=%s)", report_only)
     return CSPMiddleware(app, report_only=report_only)
🤖 Prompt for AI Agents
@netover	Reply...
resync/core/agent_router.py
Comment on lines +497 to 505
    async def _call_tool(self, tool_name: str, **kwargs: Any) -> str | None:
        """Call a tool and track usage."""
        if self.agent_manager and hasattr(self.agent_manager, "call_tool"):
            result = await self.agent_manager.call_tool(tool_name, **kwargs)
            if result:
            if isinstance(result, str) and result:
                self.last_tools_used.append(tool_name)
            return result
                return result
            return None
        return None
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

_call_tool now drops valid non-string tool outputs.

Many tools return structured payloads (dict/list). Returning None for all non-string results silently converts successful tool calls into failures.

🔧 Suggested fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
5 hidden conversations
Load more…
resync/core/logging_setup.py
Comment on lines +16 to +19
    root_logger = logging.getLogger()
    if not any(isinstance(log_filter, SecretRedactor) for log_filter in root_logger.filters):
        root_logger.addFilter(SecretRedactor())

Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

🧩 Analysis chain
Attach SecretRedactor filter to root logger's handlers to ensure child logger records are redacted.

The filter added to the root logger at line 18 does not apply to records propagated from child loggers. Per Python's logging documentation, ancestor logger filters are bypassed during propagation—only handler filters are consulted. This creates a gap where secrets in child logger records reach handlers without redaction.

Suggested fix
🤖 Prompt for AI Agents
@netover	Reply...
resync/core/metrics_legacy_internal.py
Comment on lines 483 to 506
    def _calculate_error_rate_ratio(self) -> float:
        total_requests = (
            self.agent_initializations.value
            + self.tws_status_requests_success.value
            + self.tws_status_requests_failed.value
            self.agent_initializations.get()
            + self.tws_status_requests_success.get()
            + self.tws_status_requests_failed.get()
        )
        if total_requests > 0:
            return (
                self.agent_creation_failures.value
                + self.tws_status_requests_failed.value
                self.agent_creation_failures.get()
                + self.tws_status_requests_failed.get()
            ) / total_requests
        return 0.0

    def _calculate_cache_hit_ratio(self) -> float:
        total_cache_ops = self.cache_hits.value + self.cache_misses.value
        total_cache_ops = self.cache_hits.get() + self.cache_misses.get()
        if total_cache_ops > 0:
            return self.cache_hits.value / total_cache_ops
            return self.cache_hits.get() / total_cache_ops
        return 0.0

    def _calculate_router_cache_hit_ratio(self) -> float:
        total_router_ops = self.router_cache_hits.value + self.router_cache_misses.value
        total_router_ops = self.router_cache_hits.get() + self.router_cache_misses.get()
        if total_router_ops > 0:
            return self.router_cache_hits.value / total_router_ops
            return self.router_cache_hits.get() / total_router_ops
        return 0.0
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟡 Minor

Snapshot operands once per ratio calculation.

At Line [485], Line [497], and Line [503], repeated .get() calls in the same formula can yield inconsistent ratios under concurrent updates.

♻️ Use single-read snapshots per method
🤖 Prompt for AI Agents
@netover	Reply...
resync/csp_validation.py

@asynccontextmanager
def csp_report_context(request: "Request") -> "Request":
async def csp_report_context(request: Request) -> Request:
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟡 Minor

🧩 Analysis chain
Return type annotation for @asynccontextmanager is incorrect.

The function should return AsyncGenerator[Request, None] to match the async generator pattern used by context manager decorators.

-from typing import Any, Literal
+from typing import Any, AsyncGenerator, Literal
@@
-async def csp_report_context(request: Request) -> Request:
+async def csp_report_context(request: Request) -> AsyncGenerator[Request, None]:
🤖 Prompt for AI Agents
@netover	Reply...
resync/settings_validators.py
Comment on lines +779 to +792
            fields_set = getattr(self, "model_fields_set", set())
            if "database_url" in fields_set:
                db_url = getattr(self, "database_url")
                raw_db = db_url.get_secret_value() if isinstance(db_url, SecretStr) else str(db_url)
                if "localhost" in raw_db or "127.0.0.1" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use localhost in production")
                if "resync:resync@" in raw_db:
                    errors.append("database_url (DATABASE_URL) must not use default credentials in production")

                metrics_hash = getattr(self, "metrics_api_key_hash", None)
                if not metrics_hash or not metrics_hash.get_secret_value():
                    errors.append(
                        "metrics_api_key_hash (APP_METRICS_API_KEY_HASH) must be set in production"
                    )
Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🟠 Major

metrics_api_key_hash validation is accidentally gated by database_url presence.

Right now, the metrics hash check only runs when "database_url" is in model_fields_set. That skips the metrics production check for cases where only metrics_api_key_hash is set/unset independently.

🔧 Proposed fix
📝 Committable suggestion
🤖 Prompt for AI Agents
@netover	Reply...
resync/settings.py
Comment on lines +26 to 27
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode

Contributor
@coderabbitai
coderabbitai bot
47 minutes ago
⚠️ Potential issue | 🔴 Critical

🧩 Analysis chain
Update pydantic-settings constraint to >=2.7.0 — NoDecode was introduced in v2.7.0.

The import at line 26 and usage at line 941 will fail at startup if the runtime has pydantic-settings 2.0–2.6.x. The current constraint pydantic-settings>=2.0 is insufficient; update requirements.in and requirements.txt to enforce pydantic-settings>=2.7.0.

🤖 Prompt for AI Agents
@netover	Reply...
BlackboxAI added 2 commits 30 minutes ago
fix: P0/P1 bugs from PR review
8741f00
Merge origin/main into codex/analyze-and-fix-repository-pattern-issue… 
6104d23
codereviewbot-ai[bot]
codereviewbot-ai bot reviewed 8 minutes ago
resync/core/auto_recovery.py
Comment on lines 270 to 276
                        "reason": "High pool utilization",
                        "scaling_direction": "up",
                        "target_utilization": 0.7,
                        "success": self._scale_connection_pool("up"),
                        "success": await asyncio.to_thread(self._scale_connection_pool, "up"),
                    }
                )

@codereviewbot-ai
codereviewbot-ai bot
8 minutes ago
Potential error handling issue:

The scaling logic does not check if pool_health contains an error (e.g., {'error': 'Unable to check connection pool health'}) before attempting scaling actions. This could result in attempts to scale when health metrics are unavailable, leading to unintended behavior or errors.

Recommendation:
Add a check before scaling actions:

if 'error' not in pool_health:
    # proceed with scaling logic
This ensures scaling is only attempted when valid health metrics are available.

@netover	Reply...
@netover netover merged commit e7b3b6a into main 7 minutes ago
2 of 8 checks passed
@sonarqubecloud
sonarqubecloud bot
commented
7 minutes ago
Quality Gate Failed Quality Gate failed
Failed conditions
 2 Security Hotspots
 E Reliability Rating on New Code (required ≥ A)

See analysis details on SonarQube Cloud

 Catch issues before they fail your Quality Gate with our IDE extension  SonarQube for IDE

@darkest-pr
darkest-pr bot
commented
7 minutes ago
🕯️ Fresh kegs, cards, and curtained rooms promise solace to the weary and the broken alike.
qltysh[bot]
qltysh bot reviewed 6 minutes ago
alembic/versions/0002_add_tws_pattern_unique.py

from collections.abc import Sequence

import sqlalchemy as sa
Contributor
@qltysh
qltysh bot
6 minutes ago
Import block is un-sorted or un-formatted [ruff:I001]

@netover	Reply...
resync/api/core/security.py
        )
        return payload
    except (JWTError, ValidationError):
    except (JWTError, ValidationError, TypeError, ValueError):
Contributor
@qltysh
qltysh bot
6 minutes ago
Remove this redundant Exception class; it derives from another which is already caught. [radarlint-python:python:S5713]

@netover	Reply...
resync/core/__init__.py
        return True

_LAZY_EXPORTS = {"AsyncTTLCache": ("resync.core.cache.async_cache", "AsyncTTLCache")}
_LAZY_EXPORTS = {"AsyncTTLCache": ("resync.core.cache.async_cache", "AsyncTTLCache"), "tws_monitor": ("resync.core.tws_monitor", None)}
Contributor
@qltysh
qltysh bot
6 minutes ago
Line too long (135 > 100) [ruff:E501]

@netover	Reply...
resync/core/a2a_handler.py
from collections.abc import AsyncGenerator
from typing import Any, Protocol

import structlog
Contributor
@qltysh
qltysh bot
6 minutes ago
Import block is un-sorted or un-formatted [ruff:I001]

@netover	Reply...
resync/core/a2a_handler.py
            logger.warning("Event queue full, dropping event", event_type=event_type)

    async def get_event_stream(self):
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
Contributor
@qltysh
qltysh bot
6 minutes ago
Unnecessary default type arguments [ruff:UP043]

Suggested change
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
    async def get_event_stream(self) -> AsyncGenerator[dict[str, Any]]:
@netover	Reply...
12 hidden conversations
Load more…
resync/core/event_bus.py
                self._event_queue.task_done()

            except TimeoutError:
            except asyncio.TimeoutError:
Contributor
@qltysh
qltysh bot
6 minutes ago
Replace aliased errors with TimeoutError [ruff:UP041]

Suggested change
            except asyncio.TimeoutError:
            except TimeoutError:
@netover	Reply...
resync/core/exception_guard.py
            )
    except (Exception,) as _metric_exc:
    except Exception:
        # Metrics must never break exception flow.
Contributor
@qltysh
qltysh bot
6 minutes ago
Found 2 issues:

1. Try, Except, Pass detected. [bandit:B110]


2. try-except-pass detected, consider logging the exception [ruff:S110]

@netover	Reply...
resync/core/exceptions.py
        import warnings

        warnings.warn(
            "resync.core.exceptions.TimeoutError shadows Python builtin. Use OperationTimeoutError instead.",
Contributor
@qltysh
qltysh bot
6 minutes ago
Line too long (109 > 100) [ruff:E501]

@netover	Reply...
resync/core/ia_auditor.py
@@ -1,5 +1,6 @@
# resync/core/ia_auditor.py
Contributor
@qltysh
qltysh bot
6 minutes ago
Incorrect formatting, autoformat by running qlty fmt. [ruff:fmt]

@netover	Reply...
resync/core/langgraph/agent_graph.py
        state["tool_output"] = json.dumps(status, ensure_ascii=False, default=str)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
    except (ImportError, ModuleNotFoundError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
Contributor
@qltysh
qltysh bot
6 minutes ago
Found 3 issues:

1. Remove this redundant Exception class; it derives from another which is already caught. [radarlint-python:python:S5713]


2. Remove this redundant Exception class; it derives from another which is already caught. [radarlint-python:python:S5713]


3. Remove this redundant Exception class; it derives from another which is already caught. [radarlint-python:python:S5713]

@netover	Reply...