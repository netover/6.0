"""Token revocation (enterprise-grade).

Implements a lightweight JWT revocation list keyed by JTI.
- If Redis is available, store revoked JTIs with TTL until token expiry.
- If Redis is not configured, revocation is disabled (fail-open) and should be
  compensated by short-lived access tokens.

Configuration:
- TOKEN_REVOCATION_ENABLED=true|false (default: true in production)
- TOKEN_REVOCATION_PREFIX (default: "revoked:jti:")
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

_PREFIX = os.getenv("TOKEN_REVOCATION_PREFIX", "revoked:jti:")

def _enabled() -> bool:
    env = os.getenv("TOKEN_REVOCATION_ENABLED")
    if env is None:
        # default: enabled in prod, disabled in dev is risky, so enable if ENV says prod
        return os.getenv("ENVIRONMENT", "development").lower() in {"production", "prod", "staging"}
    return env.lower() in {"1", "true", "yes", "on"}

async def is_jti_revoked(jti: str) -> bool:
    if not _enabled():
        return False
    try:
        from resync.core.redis_init import get_redis_client

        redis = get_redis_client()
        if not redis:
            return False
        key = f"{_PREFIX}{jti}"
        val = await redis.get(key)
        return val is not None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Fail-open on infra errors to avoid full outage.
        logger.warning("token_revocation_check_failed", error=str(e))
        return False

async def revoke_jti(jti: str, exp_unix: int | None = None) -> None:
    """Revoke a token by its JTI until expiration."""
    if not _enabled():
        return
    try:
        from resync.core.redis_init import get_redis_client

        redis = get_redis_client()
        if not redis:
            return
        key = f"{_PREFIX}{jti}"
        ttl = None
        if exp_unix is not None:
            now = int(datetime.now(timezone.utc).timestamp())
            ttl = max(exp_unix - now, 1)
        if ttl:
            await redis.set(key, "1", ex=ttl)
        else:
            # default 1 hour if exp not known
            await redis.set(key, "1", ex=3600)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("token_revocation_store_failed", error=str(e))
