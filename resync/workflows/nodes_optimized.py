# pylint
"""LangGraph Workflow Nodes.

This module implements the shared node functions used by:

* Predictive Maintenance workflow
* Capacity Forecasting workflow

The original repository shipped with stub implementations. This version keeps
backwards compatible wrappers (that accept and return a ``state`` dict) while
also exposing explicit, typed functions used directly by the workflow graphs.

Design goals (no architecture change):
* Prefer DB history when available; fallback to TWS REST reads when not.
* Provide statistical degradation detection (rolling z-score / trend checks).
* Provide metric correlation via pandas when available (fallback to pure python).
* Provide timeline prediction with confidence intervals (simple regression).
* Combine heuristic recommendations with optional LLM enrichment.
* Support operator notification via existing Teams integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


try:
    from resync.settings import get_settings

    _SETTINGS_AVAILABLE = True
except Exception:
    get_settings = None  # type: ignore
    _SETTINGS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional heavy deps (keep project runnable without data-science extras)
# ---------------------------------------------------------------------------

try:
    import numpy as _np  # type: ignore

    _NUMPY_AVAILABLE = True
except Exception:
    _np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    import pandas as _pd  # type: ignore

    _PANDAS_AVAILABLE = True
except Exception:
    _pd = None  # type: ignore
    _PANDAS_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    _scipy_stats = None  # type: ignore
    _SCIPY_AVAILABLE = False


try:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from resync.core.database.models.stores import (
        TWSJobStatus,
        TWSWorkstationStatus,
    )

    _SQLA_AVAILABLE = True
except Exception:
    # Keep import-time failures non-fatal for environments that run workflows
    # without DB connectivity.
    AsyncSession = Any  # type: ignore
    select = None  # type: ignore
    TWSJobStatus = None  # type: ignore
    TWSWorkstationStatus = None  # type: ignore
    _SQLA_AVAILABLE = False


try:
    from resync.core.teams_notifier import TeamsNotificationManager

    _TEAMS_AVAILABLE = True
except Exception:
    TeamsNotificationManager = None  # type: ignore
    _TEAMS_AVAILABLE = False


try:
    # This is our REST client for read-only TWS queries.
    from resync.services.tws_service import OptimizedTWSClient

    _TWS_CLIENT_AVAILABLE = True
except Exception:
    OptimizedTWSClient = None  # type: ignore
    _TWS_CLIENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _predictive_enabled() -> bool:
    """Central gate for predictive workflows.

    We keep workflows importable in environments that don't want predictions yet.
    """

    if not _SETTINGS_AVAILABLE or get_settings is None:
        return False
    try:
        return bool(get_settings().enable_predictive_workflows)
    except Exception:
        return False


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        return None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if not start or not end:
        return None
    try:
        return max(0.0, (end - start).total_seconds())
    except Exception:
        return None


def _pearson_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    if _NUMPY_AVAILABLE:
        try:
            return float(_np.corrcoef(_np.array(x), _np.array(y))[0, 1])
        except Exception:
            return None
    # pure python fallback
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x) ** 0.5
    deny = sum((b - my) ** 2 for b in y) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def _rolling_zscore(values: list[float], window: int = 10) -> list[float]:
    """Compute rolling z-score.

    Used for detecting anomalies/degradation in runtime/failure rates.
    """
    if len(values) < window + 1:
        return [0.0 for _ in values]
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window)
        chunk = values[start:i] if i > start else []
        if len(chunk) < 3:
            out.append(0.0)
            continue
        mu = sum(chunk) / len(chunk)
        var = sum((v - mu) ** 2 for v in chunk) / max(1, len(chunk) - 1)
        sd = var**0.5
        if sd == 0:
            out.append(0.0)
        else:
            out.append((values[i] - mu) / sd)
    return out


@dataclass
class DegradationResult:
    detected: bool
    type: str | None
    severity: float
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# 1) Fetch job history (DB first; fallback to TWS plan query)
# ---------------------------------------------------------------------------


async def fetch_job_history(
    *,
    db: AsyncSession,
    job_name: str,
    days: int = 30,
    tws_client: OptimizedTWSClient | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch historical job execution history.

    Primary source: Postgres (tws.tws_job_status). Fallback: TWS REST current plan
    query (best-effort when DB history is unavailable).
    """
    from resync.workflows.statistical_analysis import (
        fetch_job_history_from_db,
        fetch_job_history_from_tws,
    )

    if not _predictive_enabled():
        logger.info("predictive_disabled.fetch_job_history")
        return []

    since = _utcnow() - timedelta(days=max(1, days))
    job_name = (job_name or "").strip()
    if not job_name:
        return []

    if _SQLA_AVAILABLE:
        try:
            rows = await fetch_job_history_from_db(db, job_name, since, limit)
            if rows:
                return rows
        except Exception as e:
            logger.warning(
                "fetch_job_history_db_failed", job_name=job_name, error=str(e)
            )

    if _TWS_CLIENT_AVAILABLE and tws_client is not None:
        try:
            return await fetch_job_history_from_tws(tws_client, job_name, limit)
        except Exception as e:
            logger.warning(
                "fetch_job_history_tws_failed", job_name=job_name, error=str(e)
            )

    return []


async def fetch_workstation_metrics(
    *,
    db: AsyncSession,
    workstation: str | None,
    days: int = 30,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Fetch workstation metrics history from DB.

    The project already stores workstation status/metrics in tws.tws_workstation_status.
    """

    if not _predictive_enabled():
        logger.info("predictive_disabled.fetch_workstation_metrics")
        return []
    if not workstation:
        return []
    since = _utcnow() - timedelta(days=max(1, days))
    if not _SQLA_AVAILABLE:
        return []
    try:
        stmt = (
            select(TWSWorkstationStatus)
            .where(TWSWorkstationStatus.workstation_name == workstation)
            .where(TWSWorkstationStatus.timestamp >= since)
            .order_by(TWSWorkstationStatus.timestamp.desc())
            .limit(limit)
        )
        res = await db.execute(stmt)
        rows = res.scalars().all()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "workstation": r.workstation_name,
                    "status": r.status,
                    "cpu_usage": r.cpu_usage,
                    "memory_usage": r.memory_usage,
                    "active_jobs": r.active_jobs,
                    "timestamp": r.timestamp,
                    "source": "db",
                }
            )
        return list(reversed(out))
    except Exception as e:
        logger.warning(
            "fetch_workstation_metrics_db_failed",
            workstation=workstation,
            error=str(e),
        )
        return []


# ---------------------------------------------------------------------------
# 2) Detect degradation (statistical)
# ---------------------------------------------------------------------------


async def detect_degradation(
    *,
    job_history: list[dict[str, Any]],
    llm: Any | None = None,
    window: int = 10,
    runtime_growth_threshold: float = 0.10,
) -> dict[str, Any]:
    """Detect degradation patterns.

    Heuristics:
    * runtime trend increases > 10% over the recent window vs previous window
    * failure rate increases materially (z-score on failures)
    """
    from resync.workflows.statistical_analysis import (
        calculate_failure_metrics,
        calculate_runtime_growth,
        extract_runtimes_and_failures,
        rolling_zscore,
    )

    if not _predictive_enabled():
        logger.info("predictive_disabled.detect_degradation")
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "details": {"reason": "disabled"},
        }
    if not job_history:
        return {"detected": False, "type": None, "severity": 0.0, "details": {}}

    runtimes, failures = extract_runtimes_and_failures(job_history)

    if len(runtimes) < max(8, window * 2):
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "details": {"reason": "insufficient_history", "samples": len(runtimes)},
        }

    mu_recent, mu_prior, growth = calculate_runtime_growth(runtimes, window)
    z_fail = rolling_zscore(failures, window=window)
    fail_recent, fail_prior, fail_growth, _ = calculate_failure_metrics(
        failures, window
    )

    detected = False
    dtype: str | None = None
    severity = 0.0

    if growth >= runtime_growth_threshold:
        detected = True
        dtype = "runtime_growth"
        severity = min(1.0, max(0.0, growth / (runtime_growth_threshold * 3)))

    if max(z_fail[-window:]) >= 2.5 or fail_growth >= 0.20:
        detected = True
        dtype = dtype or "failure_rate"
        sev_fail = min(1.0, max(0.0, max(z_fail[-window:]) / 6.0))
        severity = max(severity, sev_fail)

    details = {
        "runtime_mean_recent": mu_recent,
        "runtime_mean_prior": mu_prior,
        "runtime_growth": growth,
        "failure_rate_recent": fail_recent,
        "failure_rate_prior": fail_prior,
        "failure_rate_growth": fail_growth,
        "z_fail_max_recent": max(z_fail[-window:]) if z_fail else 0.0,
        "window": window,
    }

    # Optional LLM enrichment: explain in natural language
    if llm is not None:
        try:
            prompt = (
                "You are an SRE assistant for IBM Workload Scheduler (TWS/HWA). "
                "Given summary stats, explain whether degradation "
                "is happening and why. "
                f"Stats: {details}. "
                "Return a short JSON with keys: summary, likely_causes (list)."
            )
            msg = await llm.ainvoke(prompt)
            details["llm"] = getattr(msg, "content", str(msg))
        except Exception as e:
            logger.debug("detect_degradation_llm_failed", error=str(e))

    return {
        "detected": detected,
        "type": dtype,
        "severity": float(severity),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 3) Correlate metrics (pandas + optional LLM)
# ---------------------------------------------------------------------------


async def correlate_metrics(
    *,
    job_history: list[dict[str, Any]],
    workstation_metrics: list[dict[str, Any]],
    degradation_type: str | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Correlate job runtime/failures with workstation metrics."""
    from resync.workflows.statistical_analysis import (
        aggregate_by_hour,
        calculate_correlations,
        extract_job_rows,
        extract_workstation_rows,
        interpret_factor,
        select_best_factor,
    )

    if not _predictive_enabled():
        logger.info("predictive_disabled.correlate_metrics")
        return {
            "found": False,
            "root_cause": None,
            "factors": [],
            "details": {"reason": "disabled"},
        }

    if not job_history or not workstation_metrics:
        return {
            "found": False,
            "root_cause": None,
            "factors": [],
            "details": {"reason": "missing_data"},
        }

    job_rows = extract_job_rows(job_history)
    ws_rows = extract_workstation_rows(workstation_metrics)

    dur_by_t = aggregate_by_hour(job_rows, "duration")
    fail_by_t = aggregate_by_hour(job_rows, "failed")
    cpu_by_t = aggregate_by_hour(ws_rows, "cpu")
    mem_by_t = aggregate_by_hour(ws_rows, "mem")
    aj_by_t = aggregate_by_hour(ws_rows, "active_jobs")

    common = sorted(set(dur_by_t) & set(cpu_by_t))
    if len(common) < 6:
        return {
            "found": False,
            "root_cause": None,
            "factors": [],
            "details": {"reason": "insufficient_overlap", "samples": len(common)},
        }

    duration = [dur_by_t[t] for t in common]
    cpu = [cpu_by_t.get(t, 0.0) for t in common]
    mem = [mem_by_t.get(t, 0.0) for t in common]
    active_jobs_list = [aj_by_t.get(t, 0.0) for t in common]
    failures = [fail_by_t.get(t, 0.0) for t in common]

    corr_details = calculate_correlations(
        duration, cpu, mem, active_jobs_list, failures
    )
    corr_dur = corr_details.get("corr_duration", {})

    best_key, best_val, found = select_best_factor(corr_dur)
    root_cause, factors = interpret_factor(best_key) if found else (None, [])

    if llm is not None:
        try:
            prompt = (
                "You are diagnosing a TWS job degradation. "
                f"Degradation type: {degradation_type}. "
                f"Correlation summary: best_factor={best_key}, value={best_val}. "
                "Provide a concise root cause hypothesis and 3 contributing factors."
            )
            msg = await llm.ainvoke(prompt)
            corr_details["llm"] = getattr(msg, "content", str(msg))
        except Exception as e:
            logger.debug("correlate_metrics_llm_failed", error=str(e))

    return {
        "found": found,
        "root_cause": root_cause,
        "factors": factors,
        "details": {**corr_details, "best": {"factor": best_key, "value": best_val}},
    }


# ---------------------------------------------------------------------------
# 4) Predict timeline (simple extrapolation + CI)
# ---------------------------------------------------------------------------


async def predict_timeline(
    *,
    job_history: list[dict[str, Any]],
    degradation_detected: bool,
    degradation_severity: float,
    llm: Any | None = None,
    horizon_days: int = 28,
) -> dict[str, Any]:
    """Predict failure timeline.

    Uses a simple linear trend on runtime to estimate when it crosses a
    "danger" threshold (median + 3*mad). Also computes a confidence band.
    """
    from resync.workflows.statistical_analysis import (
        calculate_confidence_interval,
        calculate_confidence_score,
        calculate_danger_threshold,
        calculate_failure_probability,
        extract_runtime_series,
        linear_regression,
    )

    if not _predictive_enabled():
        logger.info("predictive_disabled.predict_timeline")
        return {
            "failure_probability": 0.0,
            "estimated_failure_date": None,
            "confidence": 0.0,
            "details": {"reason": "disabled"},
        }

    if not degradation_detected or not job_history:
        return {
            "failure_probability": 0.0,
            "estimated_failure_date": None,
            "confidence": 0.0,
            "details": {"reason": "no_degradation"},
        }

    series = extract_runtime_series(job_history)
    if len(series) < 10:
        return {
            "failure_probability": min(1.0, 0.25 + degradation_severity * 0.5),
            "estimated_failure_date": None,
            "confidence": 0.2,
            "details": {"reason": "insufficient_runtime_series"},
        }

    t0 = series[0][0]
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in series]
    ys = [v for _, v in series]

    med, mad, danger = calculate_danger_threshold(ys)

    a, b, s = linear_regression(xs, ys)

    if b == 0:
        return {
            "failure_probability": min(1.0, 0.3 + degradation_severity * 0.6),
            "estimated_failure_date": None,
            "confidence": 0.3,
            "details": {"reason": "no_time_variance"},
        }

    est_days: float | None = None
    if b > 0:
        est_days = (danger - a) / b
        if est_days < 0:
            est_days = None
    est_date = (t0 + timedelta(days=est_days)) if est_days is not None else None

    n = len(xs)
    xbar = sum(xs) / n
    sxx = sum((x - xbar) ** 2 for x in xs)

    ci = calculate_confidence_interval(
        slope=b,
        intercept=a,
        x_values=xs,
        y_values=ys,
        residual_std=s,
        x_mean=xbar,
        sxx=sxx,
        n=n,
        predicted_x=est_days,
    )

    if est_days is not None and b != 0 and ci:
        dt_days = ci.get("delta_days", 0)
        ci = {
            "lower": (t0 + timedelta(days=max(0.0, est_days - dt_days))).isoformat(),
            "upper": (t0 + timedelta(days=est_days + dt_days)).isoformat(),
            **ci,
        }

    ybar = sum(ys) / n
    prob = calculate_failure_probability(
        degradation_severity, est_days, horizon_days, 0.0, ybar, s
    )
    confidence = calculate_confidence_score(
        degradation_severity, s, ybar, est_date is not None
    )

    details = {
        "model": "linear",
        "a": a,
        "b": b,
        "danger_threshold": danger,
        "median": med,
        "mad": mad,
        "ci": ci,
        "samples": n,
    }

    if llm is not None:
        try:
            prompt = (
                "Summarize this failure prediction for a TWS job in 3 bullets. "
                f"Details: {details}."
            )
            msg = await llm.ainvoke(prompt)
            details["llm"] = getattr(msg, "content", str(msg))
        except Exception as e:
            logger.debug("predict_timeline_llm_failed", error=str(e))

    return {
        "failure_probability": float(prob),
        "estimated_failure_date": est_date,
        "confidence": float(confidence),
        "details": details,
    }


# ---------------------------------------------------------------------------
# 5) Recommendations (heuristics + optional LLM)
# ---------------------------------------------------------------------------


async def generate_recommendations(
    *,
    job_name: str,
    degradation_type: str | None,
    degradation_severity: float,
    correlation: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Generate recommendations based on analysis."""
    from resync.workflows.statistical_analysis import build_recommendations

    if not _predictive_enabled():
        logger.info("predictive_disabled.generate_recommendations")
        return {"recommendations": [], "actions": [], "details": {"reason": "disabled"}}

    recs, actions = build_recommendations(
        job_name, degradation_type, degradation_severity, correlation, prediction
    )

    if llm is not None:
        try:
            prompt = (
                "You are an SRE assistant. Convert findings "
                "into 3 actionable steps. "
                f"Job: {job_name}. Degradation: {degradation_type} "
                f"severity={degradation_severity}. "
                f"Correlation: {correlation}. Prediction: {prediction}."
            )
            msg = await llm.ainvoke(prompt)
            recs.append(
                {
                    "title": "LLM recommended steps",
                    "priority": "medium",
                    "rationale": getattr(msg, "content", str(msg)),
                    "job": job_name,
                    "confidence": 0.55,
                }
            )
        except Exception as e:
            logger.debug("generate_recommendations_llm_failed", error=str(e))

    return {"recommendations": recs, "actions": actions}


# ---------------------------------------------------------------------------
# 6) Notify operators (Teams, with placeholders for Email/Slack)
# ---------------------------------------------------------------------------


async def notify_operators(
    *,
    workflow_id: str,
    job_name: str,
    recommendations: list[dict[str, Any]],
    failure_probability: float,
    estimated_failure_date: datetime | None,
    db: AsyncSession | None = None,
    instance_name: str = "default",
) -> dict[str, Any]:
    """Notify operators.

    This integrates with the existing Teams notification manager when a DB session
    is available and Teams is configured.
    """
    if not _predictive_enabled():
        logger.info("predictive_disabled.notify_operators")
        return {
            "sent": False,
            "timestamp": _utcnow().isoformat(),
            "details": {"reason": "disabled"},
        }
    payload = {
        "workflow_id": workflow_id,
        "job_name": job_name,
        "failure_probability": failure_probability,
        "estimated_failure_date": estimated_failure_date.isoformat()
        if estimated_failure_date
        else None,
        "recommendations": recommendations,
    }

    sent = False
    errors: list[str] = []

    if db is not None and _TEAMS_AVAILABLE:
        try:
            # AsyncSession exposes a sync_session for legacy sync SQLAlchemy code.
            mgr = TeamsNotificationManager(db.sync_session)  # type: ignore[attr-defined]

            # Use ABEND as a generic severity for proactive alerting.
            summary = "Predictive alert for {job_name} (p={failure_probability:.2f})"
            if estimated_failure_date:
                summary += f" | ETA: {estimated_failure_date.date().isoformat()}"

            # Send as a job notification card
            sent = await mgr.send_job_notification(
                job_name=job_name,
                job_status="WARNING" if failure_probability < 0.75 else "FAILED",
                instance_name=instance_name,
                return_code=None,
                error_message=summary,
                timestamp=_utcnow().isoformat(),
            )
        except Exception as e:
            errors.append(f"teams:{e}")
            logger.warning("notify_operators_teams_failed", error=str(e))

    # Slack/email hooks can be implemented using existing connector settings.
    # We keep these as placeholders to avoid architectural changes.
    if not sent:
        logger.info("notify_operators_noop", payload=payload)

    return {
        "notification_sent": bool(sent),
        "notification_timestamp": _utcnow().isoformat(),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Backwards compatible wrappers (state-dict style)
# ---------------------------------------------------------------------------


async def fetch_job_history_state(
    state: dict[str, Any], db: AsyncSession, days: int = 30
) -> dict[str, Any]:
    history = await fetch_job_history(
        db=db, job_name=state.get("job_name", ""), days=days
    )
    return {**state, "job_history": history, "fetch_timestamp": _utcnow().isoformat()}


async def fetch_workstation_metrics_state(
    state: dict[str, Any], db: AsyncSession, days: int = 30
) -> dict[str, Any]:
    metrics = await fetch_workstation_metrics(
        db=db, workstation=state.get("workstation"), days=days
    )
    return {**state, "workstation_metrics": metrics}


async def detect_degradation_state(
    state: dict[str, Any], llm: Any | None = None
) -> dict[str, Any]:
    res = await detect_degradation(job_history=state.get("job_history", []), llm=llm)
    return {**state, "degradation": res}


__all__ = [
    # explicit functions (used by workflows)
    "fetch_job_history",
    "fetch_workstation_metrics",
    "detect_degradation",
    "correlate_metrics",
    "predict_timeline",
    "generate_recommendations",
    "notify_operators",
    # state-style wrappers (kept for compatibility)
    "fetch_job_history_state",
    "fetch_workstation_metrics_state",
    "detect_degradation_state",
]
