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
from typing import Any, Iterable

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
    except Exception:
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

    if not _predictive_enabled():
        logger.info("predictive_disabled.fetch_job_history")
        return []

    since = _utcnow() - timedelta(days=max(1, days))
    job_name = (job_name or "").strip()
    if not job_name:
        return []

    # 1) DB history
    if _SQLA_AVAILABLE:
        try:
            stmt = (
                select(TWSJobStatus)
                .where(TWSJobStatus.job_name == job_name)
                .where(TWSJobStatus.timestamp >= since)
                .order_by(TWSJobStatus.timestamp.desc())
                .limit(limit)
            )
            res = await db.execute(stmt)
            rows = res.scalars().all()
            if rows:
                out: list[dict[str, Any]] = []
                for r in rows:
                    out.append(
                        {
                            "job_name": r.job_name,
                            "job_stream": r.job_stream,
                            "workstation": r.workstation,
                            "status": r.status,
                            "run_number": r.run_number,
                            "start_time": r.start_time,
                            "end_time": r.end_time,
                            "duration_seconds": _duration_seconds(r.start_time, r.end_time),
                            "return_code": r.return_code,
                            "timestamp": r.timestamp,
                            "metadata": r.metadata_ or {},
                            "source": "db",
                        }
                    )
                return list(reversed(out))  # oldest->newest
        except Exception as e:
            logger.warning("fetch_job_history_db_failed", job_name=job_name, error=str(e))

    # 2) Fallback to TWS (current plan only)
    if _TWS_CLIENT_AVAILABLE and tws_client is not None:
        try:
            plan = await tws_client.query_current_plan_jobs(q=job_name, limit=min(200, limit))
            # IBM API typically returns an object with a list field; be defensive.
            items = plan.get("jobs") if isinstance(plan, dict) else None
            if items is None and isinstance(plan, dict):
                # try a few common keys
                for k in ("items", "results", "data"):
                    if isinstance(plan.get(k), list):
                        items = plan[k]
                        break
            if not isinstance(items, list):
                items = []
            out: list[dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                # Heuristic mapping for plan job objects
                st = it.get("status") or it.get("jobStatus")
                ws = it.get("workstation") or it.get("workStation") or it.get("workstationName")
                ts = it.get("timestamp") or it.get("startTime") or it.get("lastUpdate")
                out.append(
                    {
                        "job_name": it.get("name") or it.get("jobName") or job_name,
                        "job_stream": it.get("jobStream") or it.get("jobstream"),
                        "workstation": ws,
                        "status": st,
                        "run_number": it.get("runNumber") or 1,
                        "start_time": it.get("startTime"),
                        "end_time": it.get("endTime"),
                        "duration_seconds": _safe_float(it.get("duration")),
                        "return_code": it.get("returnCode"),
                        "timestamp": ts,
                        "metadata": it,
                        "source": "tws",
                    }
                )
            return out
        except Exception as e:
            logger.warning("fetch_job_history_tws_failed", job_name=job_name, error=str(e))

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

    if not _predictive_enabled():
        logger.info("predictive_disabled.detect_degradation")
        return {"detected": False, "type": None, "severity": 0.0, "details": {"reason": "disabled"}}
    if not job_history:
        return {"detected": False, "type": None, "severity": 0.0, "details": {}}

    # Build runtime series
    runtimes: list[float] = []
    failures: list[float] = []
    for r in job_history:
        dur = r.get("duration_seconds")
        if dur is None:
            dur = _duration_seconds(r.get("start_time"), r.get("end_time"))
        dur_f = _safe_float(dur)
        if dur_f is None:
            continue
        runtimes.append(dur_f)
        st = (r.get("status") or "").upper()
        failures.append(1.0 if st in {"ABEND", "ERROR", "FAILED", "FAIL"} else 0.0)

    if len(runtimes) < max(8, window * 2):
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "details": {"reason": "insufficient_history", "samples": len(runtimes)},
        }

    # Compare recent vs prior window
    recent = runtimes[-window:]
    prior = runtimes[-2 * window : -window]
    mu_recent = sum(recent) / len(recent)
    mu_prior = sum(prior) / len(prior)
    growth = (mu_recent - mu_prior) / max(1e-9, mu_prior)

    # Failure anomaly
    z_fail = _rolling_zscore(failures, window=window)
    fail_recent = sum(failures[-window:]) / window
    fail_prior = sum(failures[-2 * window : -window]) / window
    fail_growth = (fail_recent - fail_prior)

    detected = False
    dtype: str | None = None
    severity = 0.0

    if growth >= runtime_growth_threshold:
        detected = True
        dtype = "runtime_growth"
        severity = min(1.0, max(0.0, growth / (runtime_growth_threshold * 3)))

    # If failures spiked, upgrade severity
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
                "Given summary stats, explain whether degradation is happening and why. "
                f"Stats: {details}. "
                "Return a short JSON with keys: summary, likely_causes (list)."
            )
            msg = await llm.ainvoke(prompt)
            details["llm"] = getattr(msg, "content", str(msg))
        except Exception as e:
            logger.debug("detect_degradation_llm_failed", error=str(e))

    return {"detected": detected, "type": dtype, "severity": float(severity), "details": details}


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

    if not _predictive_enabled():
        logger.info("predictive_disabled.correlate_metrics")
        return {"found": False, "root_cause": None, "factors": [], "details": {"reason": "disabled"}}
    if not job_history or not workstation_metrics:
        return {"found": False, "root_cause": None, "factors": [], "details": {"reason": "missing_data"}}

    # Build aligned time buckets (hourly) to merge
    def hour_bucket(ts: Any) -> datetime | None:
        if isinstance(ts, datetime):
            return ts.replace(minute=0, second=0, microsecond=0)
        return None

    job_rows: list[dict[str, Any]] = []
    for r in job_history:
        ts = r.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        dur = r.get("duration_seconds")
        dur_f = _safe_float(dur)
        if dur_f is None:
            dur_f = _safe_float(_duration_seconds(r.get("start_time"), r.get("end_time")))
        if dur_f is None:
            continue
        st = (r.get("status") or "").upper()
        job_rows.append(
            {
                "t": hour_bucket(ts),
                "duration": dur_f,
                "failed": 1.0 if st in {"ABEND", "ERROR", "FAILED", "FAIL"} else 0.0,
            }
        )

    ws_rows: list[dict[str, Any]] = []
    for m in workstation_metrics:
        ts = m.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        ws_rows.append(
            {
                "t": hour_bucket(ts),
                "cpu": _safe_float(m.get("cpu_usage")),
                "mem": _safe_float(m.get("memory_usage")),
                "active_jobs": _safe_float(m.get("active_jobs")),
            }
        )

    # Aggregate per hour
    def agg_mean(rows: list[dict[str, Any]], key: str) -> dict[datetime, float]:
        buckets: dict[datetime, list[float]] = {}
        for r in rows:
            t = r.get("t")
            v = r.get(key)
            if t is None or v is None:
                continue
            buckets.setdefault(t, []).append(float(v))
        return {t: sum(vs) / len(vs) for t, vs in buckets.items() if vs}

    dur_by_t = agg_mean(job_rows, "duration")
    fail_by_t = agg_mean(job_rows, "failed")
    cpu_by_t = agg_mean(ws_rows, "cpu")
    mem_by_t = agg_mean(ws_rows, "mem")
    aj_by_t = agg_mean(ws_rows, "active_jobs")

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
    active_jobs = [aj_by_t.get(t, 0.0) for t in common]
    failures = [fail_by_t.get(t, 0.0) for t in common]

    # Use pandas correlation when available for convenience
    corr_details: dict[str, Any] = {}
    if _PANDAS_AVAILABLE:
        df = _pd.DataFrame(
            {
                "duration": duration,
                "cpu": cpu,
                "mem": mem,
                "active_jobs": active_jobs,
                "failures": failures,
            }
        )
        corr = df.corr(method="pearson")
        corr_details["corr_matrix"] = corr.to_dict()
        corr_dur = corr["duration"].to_dict()
    else:
        corr_dur = {
            "cpu": _pearson_corr(duration, cpu),
            "mem": _pearson_corr(duration, mem),
            "active_jobs": _pearson_corr(duration, active_jobs),
            "failures": _pearson_corr(duration, failures),
        }
        corr_details["corr_duration"] = corr_dur

    # Pick strongest factor by absolute correlation
    scored = [(k, v) for k, v in corr_dur.items() if isinstance(v, (int, float)) and v is not None]
    scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
    best = scored[0] if scored else (None, None)
    best_key, best_val = best
    found = best_key is not None and best_val is not None and abs(best_val) >= 0.45

    factors: list[str] = []
    root_cause: str | None = None
    if found:
        if best_key == "cpu":
            root_cause = "CPU saturation correlates with increased job runtime"
            factors.append("cpu")
        elif best_key == "mem":
            root_cause = "Memory pressure correlates with increased job runtime"
            factors.append("memory")
        elif best_key == "active_jobs":
            root_cause = "High concurrency correlates with increased job runtime"
            factors.append("active_jobs")
        elif best_key == "failures":
            root_cause = "Failures correlate with increased runtime (possible retries/resource issues)"
            factors.append("failures")

    # Optional LLM enrichment
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
        "found": bool(found),
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

    # Extract runtimes by time
    series: list[tuple[datetime, float]] = []
    for r in job_history:
        ts = r.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        dur = _safe_float(r.get("duration_seconds"))
        if dur is None:
            dur = _safe_float(_duration_seconds(r.get("start_time"), r.get("end_time")))
        if dur is None:
            continue
        series.append((ts, dur))
    series.sort(key=lambda x: x[0])
    if len(series) < 10:
        return {
            "failure_probability": min(1.0, 0.25 + degradation_severity * 0.5),
            "estimated_failure_date": None,
            "confidence": 0.2,
            "details": {"reason": "insufficient_runtime_series"},
        }

    t0 = series[0][0]
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in series]  # days
    ys = [v for _, v in series]

    # Robust threshold based on MAD
    med = sorted(ys)[len(ys) // 2]
    mad = sorted([abs(v - med) for v in ys])[len(ys) // 2] or 1.0
    danger = med + 3.0 * mad

    # Fit linear regression y = a + b*x
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        return {
            "failure_probability": min(1.0, 0.3 + degradation_severity * 0.6),
            "estimated_failure_date": None,
            "confidence": 0.3,
            "details": {"reason": "no_time_variance"},
        }
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    a = ybar - b * xbar

    # Predict crossing
    est_days: float | None = None
    if b > 0:
        est_days = (danger - a) / b
        if est_days < 0:
            est_days = None
    est_date = (t0 + timedelta(days=est_days)) if est_days is not None else None

    # Confidence interval for forecast (simple)
    # Compute residual std
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    sse = sum(r**2 for r in residuals)
    dof = max(1, n - 2)
    s = (sse / dof) ** 0.5

    # Estimate uncertainty of crossing time: approximate via delta method
    # Var(y_hat) at x is s^2 * (1 + 1/n + (x-xbar)^2/sxx)
    def _pred_se(x: float) -> float:
        return s * (1.0 + 1.0 / n + ((x - xbar) ** 2) / sxx) ** 0.5

    # Choose critical value
    if _SCIPY_AVAILABLE:
        tcrit = float(_scipy_stats.t.ppf(0.975, dof))
    else:
        tcrit = 1.96

    ci: dict[str, Any] = {}
    if est_days is not None:
        se_at_est = _pred_se(est_days)
        # Translate runtime CI into time CI approximately (divide by slope)
        if b != 0:
            dt_days = abs(tcrit * se_at_est / b)
            ci = {
                "lower": (t0 + timedelta(days=max(0.0, est_days - dt_days))).isoformat(),
                "upper": (t0 + timedelta(days=est_days + dt_days)).isoformat(),
                "tcrit": tcrit,
                "slope": b,
            }

    # Failure probability heuristic
    # If slope positive and estimated crossing in horizon => high probability
    prob = min(1.0, 0.25 + 0.55 * degradation_severity)
    if est_date is not None:
        days_to = (est_date - _utcnow()).total_seconds() / 86400.0
        if days_to <= horizon_days:
            prob = min(1.0, max(prob, 0.75))
        elif days_to <= horizon_days * 2:
            prob = min(1.0, max(prob, 0.55))

    confidence = min(1.0, max(0.1, 0.35 + 0.5 * degradation_severity - 0.1 * (s / max(1.0, ybar))))
    if est_date is None:
        confidence *= 0.7

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

    if not _predictive_enabled():
        logger.info("predictive_disabled.generate_recommendations")
        return {"recommendations": [], "actions": [], "details": {"reason": "disabled"}}
    recs: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    root = (correlation or {}).get("root_cause")
    factors = (correlation or {}).get("factors") or []
    prob = (prediction or {}).get("failure_probability") or 0.0

    # Heuristics
    if degradation_type == "runtime_growth":
        recs.append(
            {
                "title": "Investigate runtime growth trend",
                "priority": "high" if degradation_severity >= 0.6 else "medium",
                "rationale": "Job runtime increased materially vs baseline",
                "job": job_name,
                "confidence": min(0.95, 0.5 + degradation_severity / 2),
            }
        )

    if "cpu" in factors:
        recs.append(
            {
                "title": "Check CPU saturation on workstation",
                "priority": "high" if prob >= 0.7 else "medium",
                "rationale": root or "CPU correlates with runtime degradation",
                "job": job_name,
                "confidence": 0.8,
            }
        )
        actions.append({"type": "workstation_capacity_review", "target": "cpu", "job": job_name})

    if "memory" in factors:
        recs.append(
            {
                "title": "Check memory pressure / swap",
                "priority": "high" if prob >= 0.7 else "medium",
                "rationale": root or "Memory correlates with runtime degradation",
                "job": job_name,
                "confidence": 0.75,
            }
        )
        actions.append({"type": "workstation_capacity_review", "target": "memory", "job": job_name})

    if prob >= 0.75:
        recs.append(
            {
                "title": "Schedule proactive validation run / controlled restart",
                "priority": "high",
                "rationale": "High probability of failure within horizon",
                "job": job_name,
                "confidence": 0.7,
            }
        )
        actions.append({"type": "proactive_validation", "job": job_name})

    # Optional LLM enrichment
    if llm is not None:
        try:
            prompt = (
                "You are an SRE assistant. Convert the following findings into 3 actionable steps. "
                f"Job: {job_name}. Degradation: {degradation_type} severity={degradation_severity}. "
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
        "estimated_failure_date": estimated_failure_date.isoformat() if estimated_failure_date else None,
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


async def fetch_job_history_state(state: dict[str, Any], db: AsyncSession, days: int = 30) -> dict[str, Any]:
    history = await fetch_job_history(db=db, job_name=state.get("job_name", ""), days=days)
    return {**state, "job_history": history, "fetch_timestamp": _utcnow().isoformat()}


async def fetch_workstation_metrics_state(
    state: dict[str, Any], db: AsyncSession, days: int = 30
) -> dict[str, Any]:
    metrics = await fetch_workstation_metrics(
        db=db, workstation=state.get("workstation"), days=days
    )
    return {**state, "workstation_metrics": metrics}


async def detect_degradation_state(state: dict[str, Any], llm: Any | None = None) -> dict[str, Any]:
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
