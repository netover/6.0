"""Exception handling utilities.

This module exists to reduce the risk of *silently* swallowing programming bugs
(TypeError/KeyError/AttributeError/IndexError) inside broad exception handlers.

By default, behavior is backward compatible: errors are still caught.
If Settings.strict_exception_handling is enabled, programming errors are re-raised
with their original traceback to fail fast in staging/canary.

IMPORTANT:
- Do NOT rely on CPython GIL "atomicity". This code is safe for free-threaded builds.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Final

_PROGRAMMING_ERRORS: Final[tuple[type[BaseException], ...]] = (
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
)


from resync.core.metrics_compat import Counter

# Low-cardinality counter (safe for production dashboards)
_programming_errors_caught_total = Counter(
    "resync_programming_errors_caught_total",
    "Count of programming errors (TypeError/KeyError/AttributeError/IndexError) caught by broad exception handlers",
    labels=["exc_type", "module"],
)

# Higher-cardinality counter (file:line label) - keep disabled in production.
_programming_errors_caught_detailed_total = Counter(
    "resync_programming_errors_caught_detailed_total",
    "Detailed count of programming errors caught by broad exception handlers (label site=file:line)",
    labels=["exc_type", "site"],
)


def maybe_reraise_programming_error(
    exc: BaseException | None,
    tb: TracebackType | None,
) -> None:
    """Re-raise programming errors when strict handling is enabled.

    Must be called from within an ``except ...`` block. Pass the active
    exception and traceback (e.g., from ``sys.exc_info()``).

    When strict handling is disabled, this is a no-op.
    """
    if exc is None or not isinstance(exc, _PROGRAMMING_ERRORS):
        return

    # Record metric for visibility even when strict handling is disabled.
    try:
        import traceback as _traceback
        from resync.settings import get_settings

        settings = get_settings()

        module = "unknown"
        site = "unknown"
        if tb is not None:
            frames = _traceback.extract_tb(tb)
            if frames:
                last = frames[-1]
                module = os.path.basename(last.filename) or "unknown"
                site = f"{module}:{last.lineno}"

        exc_type = type(exc).__name__

        # Safe-by-default metric (low cardinality)
        _programming_errors_caught_total.inc(
            1,
            labels={
                "exc_type": exc_type,
                "module": module,
            },
        )

        # Detailed metric only when explicitly enabled (staging/canary)
        if bool(getattr(settings, "programming_error_metrics_detailed_site", False)):
            _programming_errors_caught_detailed_total.inc(
                1,
                labels={
                    "exc_type": exc_type,
                    "site": site,
                },
            )
    except (Exception,) as _metric_exc:
        # Metrics must never break exception flow.
        pass


    # Lazy import to avoid import cycles; only executed on exceptions.
    from resync.settings import get_settings

    if not bool(get_settings().strict_exception_handling):
        return

    # Re-raise preserving traceback as much as possible.
    raise exc.with_traceback(tb)
