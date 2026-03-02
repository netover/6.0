# Changelog (Audit fixes)

## Resync hardened4

### Concurrency / cancellation safety
- Ensure `asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit` handlers re-raise immediately (do not swallow).
- Ensure `except Exception` blocks re-raise `asyncio.CancelledError` (guard inserted) where applicable.

### Redis atomicity
- Replace `exists()+setex()` with atomic `SET ... NX EX` in idempotency storage warm-path.

### Typing
- Added `-> None` return annotations for many simple functions where no value is returned (mypy(strict) readiness).

### Syntax/import hygiene
- Fixed `auto_refactor.py` broken `subprocess.run(...)` call.
- Fixed `deploy/gunicorn_conf.py` `from __future__` placement.
- Removed misplaced `from __future__ import annotations` lines that were embedded inside blocks/classes.

## Resync hardened5
- Added feedback PII redaction + mandatory API key in production.
- Added optional feedback retention background worker.
- Added logging compatibility helper.
- Added async file IO helpers and patched obvious open() patterns in request handlers.
