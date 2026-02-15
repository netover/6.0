"""
LangGraph Workflow Nodes

Shared node functions used by predictive maintenance and capacity forecasting workflows.
"""

from datetime import datetime, timezone, timedelta
from typing import Any

import structlog
from sqlalchemy import text

logger = structlog.get_logger(__name__)

_JOB_EXECUTION_HISTORY_LIMIT = 2000
_METRICS_HISTORY_LIMIT = 5000


async def fetch_job_history(
    db: Any,
    job_name: str,
    days: int = 30
) -> list[dict[str, Any]]:
    """Fetch historical job data from TWS.

    Tries multiple sources in order:
    1. PostgreSQL job_execution_history table (if exists)
    2. TWS API direct query
    3. TWS current plan iteration over dates

    Args:
        db: Database session (AsyncSession)
        job_name: Name of the job to fetch history for
        days: Number of days to look back (default 30)

    Returns:
        List of job execution records with:
        - timestamp (ISO format)
        - job_name
        - workstation
        - status (SUCC/FAIL/ABEND/etc)
        - return_code
        - runtime_seconds
        - scheduled_time
        - actual_start_time
        - completed_time
    """
    import os
    from datetime import timedelta
    from sqlalchemy import select, text
    
    # Feature flag protection
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning(
            "fetch_job_history.feature_disabled",
            job_name=job_name,
            message="Predictive workflows disabled. Set ENABLE_PREDICTIVE_WORKFLOWS=true to enable."
        )
        return []
    
    logger.info("fetch_job_history.started", job_name=job_name, days=days)
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    job_history = []
    
    try:
        # OPTION 1: Try PostgreSQL first (fastest)
        try:
            # Check if table exists
            check_table_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'job_execution_history'
                )
            """)
            result = await db.execute(check_table_query)
            table_exists = result.scalar()
            
            if table_exists:
                logger.debug("fetch_job_history.using_postgresql", job_name=job_name)
                
                # Query job history
                query = text("""
                    SELECT 
                        timestamp,
                        job_name,
                        workstation,
                        status,
                        return_code,
                        runtime_seconds,
                        scheduled_time,
                        actual_start_time,
                        completed_time
                    FROM job_execution_history
                    WHERE job_name = :job_name
                    AND timestamp >= :cutoff_date
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)
                
                result = await db.execute(
                    query,
                    {"job_name": job_name, "cutoff_date": cutoff_date}
                )
                rows = result.mappings().fetchall()
                
                if rows:
                    job_history = [
                        {
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                            "job_name": row["job_name"],
                            "workstation": row["workstation"] or "UNKNOWN",
                            "status": row["status"] or "UNKNOWN",
                            "return_code": row["return_code"] or 0,
                            "runtime_seconds": row["runtime_seconds"] or 0,
                            "scheduled_time": row["scheduled_time"].isoformat() if row["scheduled_time"] else None,
                            "actual_start_time": row["actual_start_time"].isoformat() if row["actual_start_time"] else None,
                            "completed_time": row["completed_time"].isoformat() if row["completed_time"] else None,
                        }
                        for row in rows
                    ]
                    
                    logger.info(
                        "fetch_job_history.postgresql_success",
                        job_name=job_name,
                        records=len(job_history)
                    )
                    return job_history
        except Exception as db_error:
            logger.warning(
                "fetch_job_history.postgresql_failed",
                job_name=job_name,
                error=str(db_error)
            )
        
        # OPTION 2: Try TWS API
        try:
            from resync.core.factories.tws_factory import get_tws_client_singleton
            
            logger.debug("fetch_job_history.using_tws_api", job_name=job_name)
            
            tws_client = get_tws_client_singleton()
            
            # v5.9.3: Removed N+1 loop over days. Fetch current plan jobs for the specific job name once.
            try:
                # Use query_current_plan_jobs with a filter for the specific job
                # NOTE: TWS current plan API doesn't support historical queries by date change in this context
                response = await tws_client.query_current_plan_jobs(q=job_name, limit=100)
                
                # The API returns a list of jobs or a dict with 'jobs' key
                jobs = response if isinstance(response, list) else response.get("jobs", [])
                
                for job in jobs:
                    # Parse job execution data
                    status = job.get("status", "UNKNOWN")
                    
                    # Calculate runtime
                    start_time_str = job.get("startTime") or job.get("actualStartTime")
                    end_time_str = job.get("completedTime") or job.get("endTime")
                    
                    runtime_seconds = job.get("duration", 0)
                    if not runtime_seconds and start_time_str and end_time_str:
                        try:
                            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                            runtime_seconds = int((end_time - start_time).total_seconds())
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                "fetch_job_history.runtime_parse_failed",
                                job_name=job_name,
                                start_time=start_time_str,
                                end_time=end_time_str,
                                error=str(e),
                            )
                    
                    job_history.append({
                        "timestamp": end_time_str or start_time_str or datetime.now(timezone.utc).isoformat(),
                        "job_name": job_name,
                        "workstation": job.get("workstation", "UNKNOWN"),
                        "status": status,
                        "return_code": job.get("returnCode", 0),
                        "runtime_seconds": int(runtime_seconds),
                        "scheduled_time": job.get("scheduledTime"),
                        "actual_start_time": start_time_str,
                        "completed_time": end_time_str,
                    })
            except Exception as tws_api_error:
                logger.debug(
                    "fetch_job_history.tws_api_failed",
                    job_name=job_name,
                    error=str(tws_api_error)
                )
            
            if job_history:
                logger.info(
                    "fetch_job_history.tws_success",
                    job_name=job_name,
                    records=len(job_history)
                )
                return job_history
        except Exception as tws_error:
            logger.warning(
                "fetch_job_history.tws_failed",
                job_name=job_name,
                error=str(tws_error)
            )
        
        # OPTION 3: Return empty with clear message
        logger.warning(
            "fetch_job_history.no_data",
            job_name=job_name,
            message="No job history found in PostgreSQL or TWS API"
        )
        return []
        
    except Exception as e:
        logger.error(
            "fetch_job_history.failed",
            job_name=job_name,
            error=str(e),
            exc_info=True
        )
        return []


async def fetch_workstation_metrics(
    db: Any,
    workstation: str,
    days: int = 30
) -> list[dict[str, Any]]:
    """Fetch workstation metrics (CPU, memory, disk).

    Tries multiple sources:
    1. PostgreSQL workstation_metrics_history table
    2. TWS API workstation status
    3. Mock data for testing (if enabled)

    Args:
        db: Database session (AsyncSession)
        workstation: Workstation name to fetch metrics for
        days: Number of days to look back (default 30)

    Returns:
        List of metrics records with:
        - timestamp (ISO format)
        - workstation
        - cpu_percent (0-100)
        - memory_percent (0-100)
        - disk_percent (0-100)
        - network_mbps (optional)
        - active_jobs (optional)
    """
    import os
    from datetime import timedelta
    from sqlalchemy import text
    
    # Feature flag protection
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning(
            "fetch_workstation_metrics.feature_disabled",
            workstation=workstation,
            message="Predictive workflows disabled"
        )
        return []
    
    logger.info("fetch_workstation_metrics.started", workstation=workstation, days=days)
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    metrics = []
    
    try:
        # OPTION 1: Try PostgreSQL first
        try:
            check_table_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'workstation_metrics_history'
                )
            """)
            result = await db.execute(check_table_query)
            table_exists = result.scalar()
            
            if table_exists:
                logger.debug("fetch_workstation_metrics.using_postgresql", workstation=workstation)
                
                query = text("""
                    SELECT 
                        timestamp,
                        workstation,
                        cpu_percent,
                        memory_percent,
                        disk_percent,
                        network_mbps,
                        active_jobs
                    FROM workstation_metrics_history
                    WHERE workstation = :workstation
                    AND timestamp >= :cutoff_date
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)
                
                result = await db.execute(
                    query,
                    {"workstation": workstation, "cutoff_date": cutoff_date}
                )
                rows = result.mappings().fetchall()
                
                if rows:
                    metrics = [
                        {
                            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                            "workstation": row["workstation"],
                            "cpu_percent": float(row["cpu_percent"]) if row["cpu_percent"] is not None else 0.0,
                            "memory_percent": float(row["memory_percent"]) if row["memory_percent"] is not None else 0.0,
                            "disk_percent": float(row["disk_percent"]) if row["disk_percent"] is not None else 0.0,
                            "network_mbps": float(row["network_mbps"]) if row["network_mbps"] is not None else 0.0,
                            "active_jobs": int(row["active_jobs"]) if row["active_jobs"] is not None else 0,
                        }
                        for row in rows
                    ]
                    
                    logger.info(
                        "fetch_workstation_metrics.postgresql_success",
                        workstation=workstation,
                        records=len(metrics)
                    )
                    return metrics
        except Exception as db_error:
            logger.warning(
                "fetch_workstation_metrics.postgresql_failed",
                workstation=workstation,
                error=str(db_error)
            )
        
        # OPTION 2: Try TWS API for current metrics
        try:
            from resync.core.factories.tws_factory import get_tws_client_singleton
            
            logger.debug("fetch_workstation_metrics.using_tws_api", workstation=workstation)
            
            tws_client = get_tws_client_singleton()
            
            # Get workstation status
            ws_status = await tws_client.get_workstation(workstation)
            
            if ws_status:
                # TWS typically provides current status only, not historical
                # Create a single point with current data
                metrics.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "workstation": workstation,
                    "cpu_percent": float(ws_status.get("cpuUsage", 0)),
                    "memory_percent": float(ws_status.get("memoryUsage", 0)),
                    "disk_percent": float(ws_status.get("diskUsage", 0)),
                    "network_mbps": 0.0,
                    "active_jobs": int(ws_status.get("activeJobs", 0)),
                })
                
                logger.info(
                    "fetch_workstation_metrics.tws_success",
                    workstation=workstation,
                    records=1
                )
                return metrics
        except Exception as tws_error:
            logger.warning(
                "fetch_workstation_metrics.tws_failed",
                workstation=workstation,
                error=str(tws_error)
            )
        
        # OPTION 3: Generate synthetic data for testing (if TEST_MODE enabled)
        if os.getenv("GENERATE_SYNTHETIC_METRICS", "").lower() == "true":
            logger.warning(
                "fetch_workstation_metrics.generating_synthetic",
                workstation=workstation
            )
            
            import random
            current_time = datetime.now(timezone.utc)
            
            # Generate hourly datapoints for the period
            for i in range(days * 24):
                timestamp = current_time - timedelta(hours=i)
                
                # Simulate increasing load over time (for degradation detection)
                base_cpu = 30 + (i / (days * 24)) * 40  # Grows from 30% to 70%
                base_mem = 40 + (i / (days * 24)) * 30  # Grows from 40% to 70%
                
                metrics.append({
                    "timestamp": timestamp.isoformat(),
                    "workstation": workstation,
                    "cpu_percent": base_cpu + random.uniform(-10, 10),
                    "memory_percent": base_mem + random.uniform(-10, 10),
                    "disk_percent": 50 + random.uniform(-5, 15),
                    "network_mbps": random.uniform(10, 100),
                    "active_jobs": random.randint(5, 20),
                })
            
            metrics.reverse()  # Chronological order
            
            logger.info(
                "fetch_workstation_metrics.synthetic_generated",
                workstation=workstation,
                records=len(metrics)
            )
            return metrics
        
        # No data available
        logger.warning(
            "fetch_workstation_metrics.no_data",
            workstation=workstation,
            message="No metrics found. Consider enabling GENERATE_SYNTHETIC_METRICS=true for testing."
        )
        return []
        
    except Exception as e:
        logger.error(
            "fetch_workstation_metrics.failed",
            workstation=workstation,
            error=str(e),
            exc_info=True
        )
        return []


async def detect_degradation(
    job_history: list[dict[str, Any]],
    llm: Any
) -> dict[str, Any]:
    """Detect performance degradation patterns using statistical analysis + LLM.

    Methods:
    1. Linear regression for runtime trend
    2. Failure rate analysis (first half vs second half)
    3. Anomaly detection via z-score
    4. LLM contextual analysis

    Args:
        job_history: List of job execution records
        llm: LLM instance for analysis (ChatAnthropic)

    Returns:
        Dict with:
        - detected: bool
        - type: "runtime_increase" | "failure_rate_increase" | "anomaly" | None
        - severity: float (0.0-1.0)
        - evidence: str (LLM-generated explanation)
        - metrics: dict (statistical details)
    """
    import os
    import numpy as np
    from scipy import stats
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Feature flag protection
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("detect_degradation.feature_disabled")
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "evidence": "Feature disabled",
            "metrics": {}
        }
    
    logger.info("detect_degradation.started", executions=len(job_history))

    # Validate minimum data
    if not job_history or len(job_history) < 7:
        logger.warning(
            "detect_degradation.insufficient_data",
            executions=len(job_history) if job_history else 0
        )
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "evidence": f"Insufficient data for analysis (need 7+, got {len(job_history)})",
            "metrics": {"executions": len(job_history) if job_history else 0}
        }
    
    try:
        # Extract time series data
        runtimes = []
        timestamps = []
        statuses = []
        
        for job in job_history:
            if job.get("runtime_seconds") and job["runtime_seconds"] > 0:
                runtimes.append(job["runtime_seconds"])
                
                timestamp_str = job.get("timestamp") or job.get("completed_time")
                if timestamp_str:
                    try:
                        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except (ValueError, TypeError):
                        timestamps.append(datetime.now(timezone.utc))
                else:
                    timestamps.append(datetime.now(timezone.utc))
            
            statuses.append(job.get("status", "UNKNOWN"))
        
        if not runtimes:
            return {
                "detected": False,
                "type": None,
                "severity": 0.0,
                "evidence": "No valid runtime data",
                "metrics": {}
            }
        
        # === ANALYSIS 1: Runtime Trend (Linear Regression) ===
        x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps[:len(runtimes)]])
        y = np.array(runtimes)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate percent change (baseline vs current)
        baseline_size = min(7, len(runtimes))
        baseline_runtime = np.mean(runtimes[:baseline_size])
        current_runtime = np.mean(runtimes[-baseline_size:])
        percent_change = ((current_runtime - baseline_runtime) / baseline_runtime) * 100
        
        # Detect runtime degradation
        runtime_degrading = False
        if percent_change > 10 and p_value < 0.05:  # 10% slower + statistically significant
            runtime_degrading = True
        
        # === ANALYSIS 2: Failure Rate ===
        success_statuses = {"SUCC", "SUCCESS", "COMPLETED", "OK"}
        failures = sum(1 for s in statuses if s not in success_statuses)
        failure_rate = failures / len(statuses) if statuses else 0.0
        
        # Compare first half vs second half
        mid = len(statuses) // 2
        if mid > 0:
            first_half_failures = sum(1 for s in statuses[:mid] if s not in success_statuses)
            second_half_failures = sum(1 for s in statuses[mid:] if s not in success_statuses)
            
            failure_rate_first = first_half_failures / mid
            failure_rate_second = second_half_failures / (len(statuses) - mid)
            
            failure_rate_increasing = failure_rate_second > (failure_rate_first * 1.5)  # 50% increase
        else:
            failure_rate_increasing = False
        
        # === ANALYSIS 3: Anomaly Detection (Z-Score) ===
        if len(runtimes) >= 10:
            runtime_mean = np.mean(runtimes)
            runtime_std = np.std(runtimes)
            
            # Recent anomalies (last 20% of data)
            recent_count = max(3, len(runtimes) // 5)
            recent_runtimes = runtimes[-recent_count:]
            
            anomaly_count = 0
            for rt in recent_runtimes:
                z_score = abs((rt - runtime_mean) / runtime_std) if runtime_std > 0 else 0
                if z_score > 2:  # 2 standard deviations
                    anomaly_count += 1
            
            has_anomalies = anomaly_count >= 2
        else:
            has_anomalies = False
            anomaly_count = 0
        
        # === DECISION: Detected degradation? ===
        detected = runtime_degrading or failure_rate_increasing or (failure_rate > 0.15) or has_anomalies
        
        # Determine degradation type and severity
        if runtime_degrading:
            degradation_type = "runtime_increase"
            severity = min(abs(percent_change) / 100, 1.0)
        elif failure_rate_increasing or failure_rate > 0.15:
            degradation_type = "failure_rate_increase"
            severity = min(failure_rate * 2, 1.0)
        elif has_anomalies:
            degradation_type = "anomaly"
            severity = min(anomaly_count / recent_count, 1.0) if recent_count > 0 else 0.0
        else:
            degradation_type = None
            severity = 0.0
        
        # === LLM CONTEXTUAL ANALYSIS ===
        if detected:
            job_name = job_history[0].get("job_name", "UNKNOWN")
            time_span_days = (timestamps[-1] - timestamps[0]).days
            
            prompt = f"""You are an expert at analyzing TWS job execution patterns.

Job: {job_name}
Executions analyzed: {len(job_history)}
Time period: {time_span_days} days

Statistical Analysis Results:
- Baseline runtime: {baseline_runtime:.1f} seconds ({baseline_runtime/60:.1f} minutes)
- Current runtime: {current_runtime:.1f} seconds ({current_runtime/60:.1f} minutes)
- Percent change: {percent_change:+.1f}%
- Trend slope: {slope:.2f} seconds/day
- Trend significance (p-value): {p_value:.4f}
- R² (trend strength): {r_value**2:.3f}
- Failure rate: {failure_rate:.1%}
- Failure trend: {"Increasing" if failure_rate_increasing else "Stable"}
- Recent anomalies: {anomaly_count}

Degradation Type Detected: {degradation_type}

Based on this statistical analysis, provide a concise technical explanation:
1. What degradation pattern was detected (2-3 sentences)
2. Likely root causes (list 2-3 specific hypotheses)
3. Urgency assessment (Low/Medium/High) with brief justification

Be specific, technical, and actionable. Avoid generic statements."""

            try:
                messages = [
                    SystemMessage(content="You are a TWS performance analyst providing technical diagnoses."),
                    HumanMessage(content=prompt)
                ]
                
                response = await llm.ainvoke(messages)
                evidence = response.content
                
                logger.debug("detect_degradation.llm_analysis_complete")
            except Exception as llm_error:
                logger.warning(
                    "detect_degradation.llm_failed",
                    error=str(llm_error)
                )
                evidence = "Degradation detected: {degradation_type}. Runtime changed {percent_change:+.1f}%, failure rate {failure_rate:.1%}."
        else:
            evidence = "No significant degradation detected. Job performance is stable within normal variation."
        
        # Compile metrics
        metrics = {
            "baseline_runtime_sec": float(baseline_runtime),
            "current_runtime_sec": float(current_runtime),
            "percent_change": float(percent_change),
            "failure_rate": float(failure_rate),
            "failure_rate_increasing": bool(failure_rate_increasing),
            "trend_slope": float(slope),
            "trend_p_value": float(p_value),
            "r_squared": float(r_value ** 2),
            "executions_analyzed": len(job_history),
            "runtime_points": len(runtimes),
            "anomaly_count": int(anomaly_count) if has_anomalies else 0,
        }
        
        logger.info(
            "detect_degradation.completed",
            detected=detected,
            type=degradation_type,
            severity=severity
        )
        
        return {
            "detected": detected,
            "type": degradation_type,
            "severity": severity,
            "evidence": evidence,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(
            "detect_degradation.failed",
            error=str(e),
            exc_info=True
        )
        return {
            "detected": False,
            "type": None,
            "severity": 0.0,
            "evidence": f"Analysis failed: {str(e)}",
            "metrics": {"error": str(e)}
        }


async def correlate_metrics(
    job_history: list[dict[str, Any]],
    workstation_metrics: list[dict[str, Any]],
    degradation_type: str | None,
    llm: Any
) -> dict[str, Any]:
    """Correlate job performance with resource metrics using Pandas + statistical analysis.

    Methods:
    1. Temporal alignment (merge_asof)
    2. Pearson correlation coefficients
    3. Lag correlation (check if metrics precede performance issues)
    4. LLM synthesis and interpretation

    Args:
        job_history: List of job execution records
        workstation_metrics: List of workstation metrics
        degradation_type: Type of degradation detected
        llm: LLM instance for analysis

    Returns:
        Dict with:
        - found: bool
        - root_cause: str
        - factors: list[str]
        - correlation_coefficient: float
        - explanation: str
    """
    import os
    import pandas as pd
    from langchain_core.messages import HumanMessage, SystemMessage
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("correlate_metrics.feature_disabled")
        return {
            "found": False,
            "root_cause": None,
            "factors": []
        }
    
    logger.info("correlate_metrics.started")

    # If no degradation, skip correlation
    if not degradation_type:
        return {
            "found": False,
            "root_cause": None,
            "factors": [],
            "correlation_coefficient": 0.0,
            "explanation": "No degradation detected, correlation analysis skipped"
        }
    
    # Validate data
    if not job_history or not workstation_metrics:
        return {
            "found": False,
            "root_cause": "Insufficient data",
            "factors": [],
            "correlation_coefficient": 0.0,
            "explanation": "Missing job history or workstation metrics"
        }
    
    try:
        # Convert to DataFrames
        jobs_df = pd.DataFrame(job_history)
        if 'timestamp' not in jobs_df.columns:
            return {
                "found": False,
                "root_cause": "Missing timestamp in job_history",
                "factors": [],
                "correlation_coefficient": 0.0,
                "explanation": "Job history missing timestamp field"
            }
        
        jobs_df['timestamp'] = pd.to_datetime(jobs_df['timestamp'])
        jobs_df = jobs_df.sort_values('timestamp')
        
        metrics_df = pd.DataFrame(workstation_metrics)
        if 'timestamp' not in metrics_df.columns:
            return {
                "found": False,
                "root_cause": "Missing timestamp in workstation_metrics",
                "factors": [],
                "correlation_coefficient": 0.0,
                "explanation": "Workstation metrics missing timestamp field"
            }
        
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df = metrics_df.sort_values('timestamp')
        
        # Temporal alignment: merge_asof (nearest timestamp within tolerance)
        merged = pd.merge_asof(
            jobs_df,
            metrics_df,
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('15min'),
            suffixes=('_job', '_metric')
        )
        
        # Remove rows without matching metrics
        merged = merged.dropna(subset=['cpu_percent', 'memory_percent'])
        
        if len(merged) < 5:
            return {
                "found": False,
                "root_cause": "Insufficient matching data points",
                "factors": [],
                "correlation_coefficient": 0.0,
                "explanation": f"Only {len(merged)} matching data points found (need 5+)"
            }
        
        # === CALCULATE CORRELATIONS ===
        correlations = {}
        
        if 'runtime_seconds' in merged.columns:
            # Correlation with CPU
            if 'cpu_percent' in merged.columns:
                corr_cpu = merged['runtime_seconds'].corr(merged['cpu_percent'])
                if not pd.isna(corr_cpu):
                    correlations['cpu'] = corr_cpu
            
            # Correlation with Memory
            if 'memory_percent' in merged.columns:
                corr_mem = merged['runtime_seconds'].corr(merged['memory_percent'])
                if not pd.isna(corr_mem):
                    correlations['memory'] = corr_mem
            
            # Correlation with Disk
            if 'disk_percent' in merged.columns:
                corr_disk = merged['runtime_seconds'].corr(merged['disk_percent'])
                if not pd.isna(corr_disk):
                    correlations['disk'] = corr_disk
            
            # Correlation with Active Jobs
            if 'active_jobs' in merged.columns:
                corr_jobs = merged['runtime_seconds'].corr(merged['active_jobs'])
                if not pd.isna(corr_jobs):
                    correlations['active_jobs'] = corr_jobs
        
        if not correlations:
            return {
                "found": False,
                "root_cause": "Unable to calculate correlations",
                "factors": [],
                "correlation_coefficient": 0.0,
                "explanation": "Missing runtime_seconds or resource metrics"
            }
        
        # Identify strongest correlation
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        metric_name, corr_value = strongest
        
        # Threshold: |correlation| > 0.6 = strong
        if abs(corr_value) > 0.6:
            found = True
            root_cause = f"{metric_name.upper()} usage strongly correlates with job runtime"
            factors = [f"{metric_name}_saturation"]
            
            # Additional analysis: check if metric is consistently high
            metric_col = f"{metric_name}_percent" if metric_name in ['cpu', 'memory', 'disk'] else metric_name
            if metric_col in merged.columns:
                avg_metric = merged[metric_col].mean()
                max_metric = merged[metric_col].max()
                
                if avg_metric > 80:
                    factors.append(f"{metric_name}_consistently_high")
                if max_metric > 95:
                    factors.append(f"{metric_name}_saturation_episodes")
        else:
            found = False
            root_cause = None
            factors = []
        
        # === LLM SYNTHESIS ===
        if found:
            job_name = job_history[0].get("job_name", "UNKNOWN")
            workstation = workstation_metrics[0].get("workstation", "UNKNOWN")
            
            # Prepare statistics
            metric_col = f"{metric_name}_percent" if metric_name in ['cpu', 'memory', 'disk'] else metric_name
            if metric_col in merged.columns:
                avg_metric = merged[metric_col].mean()
                max_metric = merged[metric_col].max()
                min_metric = merged[metric_col].min()
                
                # Recent trend (last 7 data points)
                recent_data = merged.tail(7)
                avg_recent_metric = recent_data[metric_col].mean() if len(recent_data) > 0 else avg_metric
            else:
                avg_metric = max_metric = min_metric = avg_recent_metric = 0
            
            prompt = f"""You are analyzing the root cause of TWS job performance degradation.

Job: {job_name}
Workstation: {workstation}
Degradation Type: {degradation_type}

Correlation Analysis:
- Strongest correlation: {metric_name.upper()} usage
- Correlation coefficient: {corr_value:.3f} ({"strong positive" if corr_value > 0.6 else "strong negative" if corr_value < -0.6 else "moderate"})
- Average {metric_name}: {avg_metric:.1f}%
- Peak {metric_name}: {max_metric:.1f}%
- Minimum {metric_name}: {min_metric:.1f}%
- Recent average {metric_name}: {avg_recent_metric:.1f}%

Data points analyzed: {len(merged)}

All correlations found:
{chr(10).join('- {k}: {v:.3f}' for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))}

Provide a technical root cause analysis:
1. Why this correlation is significant (2-3 sentences)
2. Specific root cause hypothesis (be precise about mechanism)
3. Recommended immediate action (1 concrete technical step)

Be specific, technical, and actionable."""

            try:
                messages = [
                    SystemMessage(content="You are a systems performance analyst providing root cause diagnoses."),
                    HumanMessage(content=prompt)
                ]
                
                response = await llm.ainvoke(messages)
                explanation = response.content
                
                logger.debug("correlate_metrics.llm_synthesis_complete")
            except Exception as llm_error:
                logger.warning(
                    "correlate_metrics.llm_failed",
                    error=str(llm_error)
                )
                explanation = "Strong correlation ({corr_value:.2f}) found between job runtime and {metric_name} usage. Average {metric_name}: {avg_metric:.1f}%, Peak: {max_metric:.1f}%."
        else:
            explanation = "No strong correlation found between job performance and workstation metrics. Degradation may be due to factors outside resource constraints: network latency, database performance, external dependencies, or algorithmic inefficiencies."
        
        logger.info(
            "correlate_metrics.completed",
            found=found,
            root_cause=root_cause,
            correlation=corr_value if found else 0.0
        )
        
        return {
            "found": found,
            "root_cause": root_cause,
            "factors": factors,
            "correlation_coefficient": float(corr_value) if found else 0.0,
            "explanation": explanation,
            "all_correlations": {k: float(v) for k, v in correlations.items()}
        }
        
    except Exception as e:
        logger.error(
            "correlate_metrics.failed",
            error=str(e),
            exc_info=True
        )
        return {
            "found": False,
            "root_cause": f"Analysis failed: {str(e)}",
            "factors": [],
            "correlation_coefficient": 0.0,
            "explanation": "Correlation analysis encountered an error"
        }


async def predict_timeline(
    job_history: list[dict[str, Any]],
    degradation_type: str | None,
    degradation_severity: float,
    llm: Any
) -> dict[str, Any]:
    """Predict failure timeline using trend extrapolation + LLM contextual assessment.

    Methods:
    1. Linear/exponential extrapolation
    2. Failure threshold estimation (2.5x baseline)
    3. Confidence intervals based on R²
    4. LLM risk assessment

    Args:
        job_history: List of job execution records
        degradation_type: Type of degradation
        degradation_severity: Severity score (0.0-1.0)
        llm: LLM instance for analysis

    Returns:
        Dict with:
        - probability: float (0.0-1.0)
        - date: datetime | None
        - confidence: float (0.0-1.0)
        - explanation: str
        - metrics: dict
    """
    import os
    import numpy as np
    from scipy import stats
    from datetime import timedelta
    from langchain_core.messages import HumanMessage, SystemMessage
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("predict_timeline.feature_disabled")
        return {
            "probability": 0.0,
            "date": None,
            "confidence": 0.0
        }
    
    logger.info("predict_timeline.started")

    # If no degradation, low failure probability
    if not degradation_type or degradation_severity < 0.1:
        return {
            "probability": 0.0,
            "date": None,
            "confidence": 0.9,
            "explanation": "No significant degradation detected, failure unlikely",
            "metrics": {}
        }
    
    # Validate minimum data
    if not job_history or len(job_history) < 10:
        return {
            "probability": 0.0,
            "date": None,
            "confidence": 0.0,
            "explanation": f"Insufficient data for prediction (need 10+, got {len(job_history)})",
            "metrics": {"executions": len(job_history)}
        }
    
    try:
        # Extract runtime data
        runtimes = []
        timestamps = []
        
        for job in job_history:
            if job.get("runtime_seconds") and job["runtime_seconds"] > 0:
                runtimes.append(job["runtime_seconds"])
                
                timestamp_str = job.get("timestamp") or job.get("completed_time")
                if timestamp_str:
                    try:
                        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except (ValueError, TypeError):
                        timestamps.append(datetime.now(timezone.utc))
                else:
                    timestamps.append(datetime.now(timezone.utc))
        
        if len(runtimes) < 10:
            return {
                "probability": 0.0,
                "date": None,
                "confidence": 0.0,
                "explanation": "Insufficient valid runtime data",
                "metrics": {}
            }
        
        # === TREND EXTRAPOLATION ===
        
        # Convert to days since start
        x = np.array([(t - timestamps[0]).total_seconds() / 86400 for t in timestamps])
        y = np.array(runtimes)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Define failure threshold (2.5x baseline = considered failure)
        baseline = np.mean(runtimes[:min(7, len(runtimes))])
        failure_threshold = baseline * 2.5
        
        # If trend is flat or improving, low failure probability
        if slope <= 0:
            return {
                "probability": 0.1,
                "date": None,
                "confidence": 0.7,
                "explanation": "Runtime trend is stable or improving, failure unlikely",
                "metrics": {
                    "baseline_runtime": float(baseline),
                    "trend_slope": float(slope),
                    "r_squared": float(r_value ** 2)
                }
            }
        
        # Calculate current runtime (at latest timestamp)
        current_runtime = slope * x[-1] + intercept
        
        # Days until failure threshold
        days_to_failure = (failure_threshold - current_runtime) / slope
        
        # Handle different scenarios
        if days_to_failure <= 0:
            # Already at or past threshold
            failure_date = datetime.now(timezone.utc) + timedelta(days=7)
            probability = 0.9
            confidence = 0.8
            urgency = "critical"
        elif days_to_failure < 7:
            # Imminent failure (< 1 week)
            failure_date = datetime.now(timezone.utc) + timedelta(days=int(days_to_failure))
            probability = 0.9
            confidence = min(r_value ** 2 * 1.2, 0.95)
            urgency = "high"
        elif days_to_failure < 14:
            # Near-term failure (1-2 weeks)
            failure_date = datetime.now(timezone.utc) + timedelta(days=int(days_to_failure))
            probability = 0.7
            confidence = min(r_value ** 2 * 1.1, 0.90)
            urgency = "high"
        elif days_to_failure < 30:
            # Medium-term failure (2-4 weeks)
            failure_date = datetime.now(timezone.utc) + timedelta(days=int(days_to_failure))
            probability = 0.5
            confidence = min(r_value ** 2, 0.85)
            urgency = "medium"
        else:
            # Long-term (> 1 month)
            failure_date = datetime.now(timezone.utc) + timedelta(days=int(days_to_failure))
            probability = 0.3
            confidence = min(r_value ** 2 * 0.9, 0.75)
            urgency = "low"
        
        # Adjust confidence based on data quality
        if p_value > 0.05:
            # Trend not statistically significant
            confidence *= 0.7
        
        if len(runtimes) < 20:
            # Limited data reduces confidence
            confidence *= 0.9
        
        # === LLM RISK ASSESSMENT ===
        job_name = job_history[0].get("job_name", "UNKNOWN")
        time_span_days = (timestamps[-1] - timestamps[0]).days
        
        prompt = f"""You are predicting when a TWS job will fail based on performance trends.

Job: {job_name}
Degradation Type: {degradation_type}
Severity: {degradation_severity:.2f}

Statistical Projection:
- Current runtime: {current_runtime:.1f} seconds ({current_runtime/60:.1f} minutes)
- Baseline runtime: {baseline:.1f} seconds ({baseline/60:.1f} minutes)
- Failure threshold: {failure_threshold:.1f} seconds ({failure_threshold/60:.1f} minutes, 2.5x baseline)
- Daily increase: {slope:.2f} seconds/day
- Days to failure: {days_to_failure:.1f} days
- Estimated failure date: {failure_date.strftime('%Y-%m-%d')}
- Trend strength (R²): {r_value**2:.3f}
- Statistical significance (p-value): {p_value:.4f}
- Urgency: {urgency.upper()}

Prediction confidence: {confidence:.1%}
Failure probability: {probability:.1%}

Provide a pragmatic risk assessment:
1. Interpretation of this prediction (is it realistic? what are key assumptions?)
2. Leading indicators to monitor (3 specific metrics)
3. Contingency plan recommendation (1 concrete action)

Be realistic about uncertainty. If confidence is low, acknowledge it."""

        try:
            messages = [
                SystemMessage(content="You are a predictive analyst providing risk assessments for production systems."),
                HumanMessage(content=prompt)
            ]
            
            response = await llm.ainvoke(messages)
            explanation = response.content
            
            logger.debug("predict_timeline.llm_assessment_complete")
        except Exception as llm_error:
            logger.warning(
                "predict_timeline.llm_failed",
                error=str(llm_error)
            )
            explanation = "Based on current trend ({slope:.1f}s/day increase), job is projected to reach failure threshold ({failure_threshold:.0f}s) in {days_to_failure:.0f} days. Confidence: {confidence:.1%}."
        
        logger.info(
            "predict_timeline.completed",
            failure_date=failure_date.isoformat() if failure_date else None,
            probability=probability,
            confidence=confidence,
            days_to_failure=days_to_failure
        )
        
        return {
            "probability": float(probability),
            "date": failure_date,
            "confidence": float(confidence),
            "explanation": explanation,
            "metrics": {
                "current_runtime": float(current_runtime),
                "baseline_runtime": float(baseline),
                "failure_threshold": float(failure_threshold),
                "days_to_failure": float(days_to_failure),
                "trend_slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "urgency": urgency,
                "data_points": len(runtimes),
                "time_span_days": time_span_days
            }
        }
        
    except Exception as e:
        logger.error(
            "predict_timeline.failed",
            error=str(e),
            exc_info=True
        )
        return {
            "probability": 0.0,
            "date": None,
            "confidence": 0.0,
            "explanation": f"Prediction failed: {str(e)}",
            "metrics": {"error": str(e)}
        }


async def generate_recommendations(
    root_cause: str | None,
    contributing_factors: list[str],
    failure_probability: float,
    estimated_failure_date: Any,
    llm: Any
) -> dict[str, Any]:
    """Generate actionable recommendations using heuristic rules + LLM synthesis.

    Strategy:
    1. Heuristic rules based on root cause
    2. Priority assignment based on failure probability
    3. LLM for contextual recommendations
    4. Preventive actions that can be automated

    Args:
        root_cause: Identified root cause of degradation
        contributing_factors: List of contributing factors
        failure_probability: Probability of failure (0.0-1.0)
        estimated_failure_date: Predicted failure date
        llm: LLM instance for synthesis

    Returns:
        Dict with:
        - recommendations: list[dict] with action, priority, description, estimated_impact
        - preventive_actions: list[dict] with action, when, description
        - urgency: "low" | "medium" | "high" | "critical"
        - llm_analysis: str
    """
    import os
    from langchain_core.messages import HumanMessage, SystemMessage
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("generate_recommendations.feature_disabled")
        return {
            "recommendations": [],
            "preventive_actions": []
        }
    
    logger.info("generate_recommendations.started", root_cause=root_cause, failure_probability=failure_probability)

    # If no problem, minimal recommendations
    if failure_probability < 0.1:
        return {
            "recommendations": [
                {
                    "action": "Continue monitoring",
                    "priority": "low",
                    "description": "Job performance is stable. Continue regular monitoring and review metrics monthly.",
                    "estimated_impact": "Preventive"
                }
            ],
            "preventive_actions": [],
            "urgency": "low",
            "llm_analysis": "No degradation detected. System is performing within normal parameters."
        }
    
    try:
        # === HEURISTIC RULES ===
        heuristic_recommendations = []
        
        # Rule set: CPU-related
        if root_cause and "cpu" in root_cause.lower():
            priority = "critical" if failure_probability > 0.7 else "high" if failure_probability > 0.4 else "medium"
            heuristic_recommendations.append({
                "action": "Increase CPU allocation for workstation",
                "priority": priority,
                "description": "Job runtime strongly correlates with CPU usage. Allocate additional CPU cores or move job to higher-capacity workstation.",
                "estimated_impact": "20-40% runtime reduction"
            })
            
            if any("saturation" in f.lower() for f in contributing_factors):
                heuristic_recommendations.append({
                    "action": "Offload concurrent workload",
                    "priority": priority,
                    "description": "CPU saturation detected. Reschedule conflicting jobs or implement job prioritization.",
                    "estimated_impact": "30-50% CPU availability increase"
                })
        
        # Rule set: Memory-related
        if root_cause and "memory" in root_cause.lower():
            priority = "critical" if failure_probability > 0.7 else "high"
            heuristic_recommendations.append({
                "action": "Increase memory allocation",
                "priority": priority,
                "description": "Performance correlates with memory pressure. Add RAM or optimize job memory footprint.",
                "estimated_impact": "15-30% runtime reduction"
            })
            
            heuristic_recommendations.append({
                "action": "Implement memory leak detection",
                "priority": "medium",
                "description": "Monitor job memory usage over time. Check for memory leaks in job scripts.",
                "estimated_impact": "Prevents future degradation"
            })
        
        # Rule set: Disk-related
        if root_cause and "disk" in root_cause.lower():
            heuristic_recommendations.append({
                "action": "Disk cleanup and optimization",
                "priority": "high" if failure_probability > 0.5 else "medium",
                "description": "High disk usage detected. Run cleanup scripts, archive old data, or migrate to faster storage (SSD).",
                "estimated_impact": "10-25% runtime reduction"
            })
        
        # Rule set: Active jobs contention
        if root_cause and "active_jobs" in root_cause.lower():
            heuristic_recommendations.append({
                "action": "Optimize job scheduling",
                "priority": "high",
                "description": "Performance correlates with workstation load. Implement job queuing or redistribute jobs across workstations.",
                "estimated_impact": "20-35% throughput increase"
            })
        
        # Rule set: Runtime degradation (general)
        if not root_cause or root_cause == "Unknown":
            heuristic_recommendations.append({
                "action": "Comprehensive performance audit",
                "priority": "high" if failure_probability > 0.6 else "medium",
                "description": "Root cause unclear. Conduct detailed analysis: network latency, database performance, job script efficiency.",
                "estimated_impact": "Identify bottleneck"
            })
        
        # Rule set: High failure probability (regardless of cause)
        if failure_probability > 0.8:
            heuristic_recommendations.insert(0, {
                "action": "Immediate intervention required",
                "priority": "critical",
                "description": "Failure probability {failure_probability:.0%}. Schedule maintenance window urgently. Consider temporary workarounds.",
                "estimated_impact": "Prevent imminent failure"
            })
        
        if failure_probability > 0.5:
            heuristic_recommendations.append({
                "action": "Implement retry logic with backoff",
                "priority": "high",
                "description": "Add resilience to job execution. Handle transient failures gracefully.",
                "estimated_impact": "Reduce failure rate by 30-50%"
            })
        
        # === DETERMINE URGENCY ===
        if failure_probability > 0.8:
            urgency = "critical"
        elif failure_probability > 0.6:
            urgency = "high"
        elif failure_probability > 0.3:
            urgency = "medium"
        else:
            urgency = "low"
        
        # === PREVENTIVE ACTIONS (Automatable) ===
        preventive_actions = []
        
        if failure_probability > 0.5:
            preventive_actions.append({
                "action": "Schedule maintenance window",
                "when": "Next weekend" if estimated_failure_date else "Within 7 days",
                "description": "Coordinate with team to implement optimizations during low-traffic period."
            })
        
        if root_cause and "cpu" in root_cause.lower():
            preventive_actions.append({
                "action": "Enable CPU monitoring alerts",
                "when": "Immediate",
                "description": "Set alert threshold at 85% CPU for 10+ minutes. Auto-notify ops team."
            })
        
        if root_cause and "memory" in root_cause.lower():
            preventive_actions.append({
                "action": "Enable memory pressure alerts",
                "when": "Immediate",
                "description": "Set alert threshold at 90% memory usage. Monitor for leaks."
            })
        
        if failure_probability > 0.7:
            preventive_actions.append({
                "action": "Increase monitoring frequency",
                "when": "Immediate",
                "description": "Switch from hourly to 15-minute monitoring. Enable real-time dashboards."
            })
        
        # === LLM SYNTHESIS ===
        failure_date_str = estimated_failure_date.strftime('%Y-%m-%d') if estimated_failure_date else "Unknown"
        
        prompt = f"""You are a TWS operations consultant providing actionable recommendations.

Situation:
- Root Cause: {root_cause or 'Unknown'}
- Contributing Factors: {', '.join(contributing_factors) if contributing_factors else 'None identified'}
- Failure Probability: {failure_probability:.1%}
- Estimated Failure: {failure_date_str}
- Urgency: {urgency.upper()}

Heuristic recommendations generated:
{chr(10).join(f"{i+1}. [{r['priority'].upper()}] {r['action']}: {r['description']}" for i, r in enumerate(heuristic_recommendations))}

Your task:
1. Validate these recommendations (are they appropriate for this scenario?)
2. Suggest 1-2 additional recommendations that heuristics might have missed
3. Prioritize: which action should be done FIRST and why?

Format:
**VALIDATION:**
[Brief assessment of heuristic recommendations]

**ADDITIONAL RECOMMENDATIONS:**
1. [Recommendation]
2. [Recommendation]

**PRIORITY ORDER:**
1. [Action to do first - from all recommendations]
2. [Action to do second]
...

**RATIONALE:**
[Why this priority order]

Be specific, technical, and pragmatic."""

        try:
            messages = [
                SystemMessage(content="You are a TWS operations consultant providing technical recommendations."),
                HumanMessage(content=prompt)
            ]
            
            response = await llm.ainvoke(messages)
            llm_analysis = response.content
            
            logger.debug("generate_recommendations.llm_synthesis_complete")
        except Exception as llm_error:
            logger.warning(
                "generate_recommendations.llm_failed",
                error=str(llm_error)
            )
            llm_analysis = "Recommendations generated based on {root_cause or 'general'} degradation pattern. Failure probability: {failure_probability:.1%}. Urgency: {urgency}."
        
        logger.info(
            "generate_recommendations.completed",
            recommendations_count=len(heuristic_recommendations),
            urgency=urgency
        )
        
        return {
            "recommendations": heuristic_recommendations,
            "preventive_actions": preventive_actions,
            "urgency": urgency,
            "llm_analysis": llm_analysis
        }
        
    except Exception as e:
        logger.error(
            "generate_recommendations.failed",
            error=str(e),
            exc_info=True
        )
        return {
            "recommendations": [
                {
                    "action": "Manual investigation required",
                    "priority": "high",
                    "description": f"Recommendation generation failed: {str(e)}. Conduct manual analysis."
                }
            ],
            "preventive_actions": [],
            "urgency": "medium",
            "llm_analysis": f"Error generating recommendations: {str(e)}"
        }


async def notify_operators(
    workflow_id: str,
    job_name: str,
    recommendations: list[dict[str, Any]],
    failure_probability: float,
    estimated_failure_date: Any
) -> dict[str, Any]:
    """Notify operators of findings and recommendations via multiple channels.

    Channels:
    1. Microsoft Teams webhook (if configured)
    2. Email via SMTP (if configured)
    3. Console output (always enabled for development)
    4. Audit log (always enabled)

    Args:
        workflow_id: Workflow identifier
        job_name: Job name
        recommendations: List of recommendations
        failure_probability: Failure probability
        estimated_failure_date: Estimated failure date

    Returns:
        Dict with:
        - notification_sent: bool
        - channels_used: list[str]
        - timestamp: str
    """
    import os
    import httpx
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("notify_operators.feature_disabled")
        return {
            "notification_sent": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    logger.info("notify_operators.started", job_name=job_name, workflow_id=workflow_id)

    notification_timestamp = datetime.now(timezone.utc).isoformat()
    channels_used = []
    
    # === FORMAT MESSAGE ===
    
    # Emoji based on urgency
    urgency = "critical" if failure_probability > 0.8 else "high" if failure_probability > 0.6 else "medium"
    urgency_emoji = {
        "low": "ℹ️",
        "medium": "⚠️",
        "high": "🚨",
        "critical": "🔴"
    }.get(urgency, "ℹ️")
    
    message_title = f"{urgency_emoji} Predictive Maintenance Alert: {job_name}"
    
    failure_date_str = estimated_failure_date.strftime('%Y-%m-%d') if estimated_failure_date else "Unknown"
    
    message_body = f"""
**Job:** {job_name}
**Failure Probability:** {failure_probability:.1%}
**Urgency:** {urgency.upper()}
**Estimated Failure:** {failure_date_str}

**Top Recommendations ({len(recommendations)} total):**
"""
    
    # Top 3 recommendations
    for i, rec in enumerate(recommendations[:3], 1):
        priority = rec.get('priority', 'medium').upper()
        action = rec.get('action', 'Unknown')
        description = rec.get('description', '')
        
        message_body += f"\n{i}. **[{priority}] {action}**\n   {description}"
    
    if len(recommendations) > 3:
        message_body += f"\n\n_...and {len(recommendations) - 3} more recommendations_"
    
    message_body += f"\n\n**Workflow ID:** `{workflow_id}`"
    message_body += f"\n**Timestamp:** {notification_timestamp}"
    
    try:
        # === CHANNEL 1: Microsoft Teams ===
        teams_webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        
        if teams_webhook_url and teams_webhook_url.startswith("https://"):
            try:
                logger.debug("notify_operators.sending_teams", job_name=job_name)
                
                # Teams MessageCard format
                teams_payload = {
                    "@type": "MessageCard",
                    "@context": "https://schema.org/extensions",
                    "summary": message_title,
                    "themeColor": "FF0000" if urgency == "critical" else "FFA500" if urgency == "high" else "FFFF00" if urgency == "medium" else "0078D4",
                    "title": message_title,
                    "sections": [
                        {
                            "activityTitle": "ReSync Predictive Maintenance",
                            "activitySubtitle": notification_timestamp,
                            "activityImage": "https://via.placeholder.com/64x64.png?text=ReSync",
                            "text": message_body,
                            "markdown": True
                        }
                    ],
                    "potentialAction": [
                        {
                            "@type": "OpenUri",
                            "name": "View Workflow",
                            "targets": [
                                {
                                    "os": "default",
                                    "uri": f"https://resync.company.com/workflows/{workflow_id}"
                                }
                            ]
                        }
                    ]
                }
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        teams_webhook_url,
                        json=teams_payload
                    )
                    
                    if response.status_code == 200:
                        channels_used.append("teams")
                        logger.info(
                            "notify_operators.teams_success",
                            job_name=job_name,
                            urgency=urgency
                        )
                    else:
                        logger.error(
                            "notify_operators.teams_failed",
                            status_code=response.status_code,
                            response=response.text[:200]
                        )
            except Exception as teams_error:
                logger.error(
                    "notify_operators.teams_error",
                    error=str(teams_error),
                    exc_info=True
                )
        else:
            logger.debug("notify_operators.teams_not_configured")
        
        # === CHANNEL 2: Email (SMTP) ===
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = os.getenv("SMTP_PORT", "587")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        notification_email = os.getenv("NOTIFICATION_EMAIL")
        
        if all([smtp_host, smtp_user, smtp_password, notification_email]):
            try:
                logger.debug("notify_operators.sending_email", job_name=job_name)
                
                # Import SMTP libraries
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                # Create email
                msg = MIMEMultipart('alternative')
                msg['Subject'] = message_title
                msg['From'] = smtp_user
                msg['To'] = notification_email
                
                # Plain text version
                text_body = f"{message_title}\n\n{message_body}"
                part1 = MIMEText(text_body, 'plain')
                
                # HTML version (nicer formatting)
                # NOTE: Avoid using backslash escapes inside f-string expressions.
                # This keeps formatting simple and ensures the f-string compiles.
                formatted_body = message_body.replace("**", "").replace("\n", "<br>")
                html_body = f"""
                <html>
                    <body style="font-family: Arial, sans-serif;">
                        <h2>{message_title}</h2>
                        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px;">
                            {formatted_body}
                        </div>
                        <p style="margin-top: 20px; color: #666;">
                            This is an automated notification from ReSync Predictive Maintenance System.
                        </p>
                    </body>
                </html>
                """
                part2 = MIMEText(html_body, 'html')
                
                msg.attach(part1)
                msg.attach(part2)
                
                # Send email
                with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.send_message(msg)
                
                channels_used.append("email")
                logger.info(
                    "notify_operators.email_success",
                    job_name=job_name,
                    recipient=notification_email
                )
            except Exception as email_error:
                logger.error(
                    "notify_operators.email_error",
                    error=str(email_error),
                    exc_info=True
                )
        else:
            logger.debug("notify_operators.email_not_configured")
        
        # === CHANNEL 3: Console Output (Always Enabled) ===
        logger.info(
            "notify_operators.console_output",
            title=message_title,
            body=message_body,
        )
        channels_used.append("console")
        
        # === CHANNEL 4: Audit Log (Always Enabled) ===
        try:
            # This would integrate with audit_db.py if available
            logger.info(
                "predictive_maintenance.notification_sent",
                workflow_id=workflow_id,
                job_name=job_name,
                failure_probability=failure_probability,
                urgency=urgency,
                channels=channels_used,
                recommendations_count=len(recommendations),
                estimated_failure_date=failure_date_str
            )
            channels_used.append("audit_log")
        except Exception as audit_error:
            logger.warning(
                "notify_operators.audit_failed",
                error=str(audit_error)
            )
        
        # === SUCCESS ===
        success = len(channels_used) > 0
        
        logger.info(
            "notify_operators.completed",
            success=success,
            channels=channels_used,
            job_name=job_name
        )
        
        return {
            "notification_sent": success,
            "channels_used": channels_used,
            "timestamp": notification_timestamp
        }
        
    except Exception as e:
        logger.error(
            "notify_operators.failed",
            job_name=job_name,
            error=str(e),
            exc_info=True
        )
        return {
            "notification_sent": False,
            "channels_used": [],
            "timestamp": notification_timestamp,
            "error": str(e)
        }


async def fetch_job_execution_history(
    db: Any,
    workstation: str | None = None,
    days: int = 30
) -> list[dict[str, Any]]:
    """Fetch job execution history for capacity forecasting.

    Args:
        db: Database session (AsyncSession)
        workstation: Filter by workstation (None = all workstations)
        days: Number of days to look back

    Returns:
        List of job execution records
    """
    import os
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("fetch_job_execution_history.feature_disabled")
        return []
    
    logger.info("fetch_job_execution_history", workstation=workstation, days=days)

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    try:
        # Build query
        query_str = """
            SELECT
                timestamp,
                job_name,
                workstation,
                status,
                return_code,
                runtime_seconds,
                scheduled_time,
                actual_start_time,
                completed_time
            FROM job_execution_history
            WHERE timestamp >= :cutoff_date
        """
        params = {"cutoff_date": cutoff_date}

        if workstation:
            query_str += " AND workstation = :workstation"
            params["workstation"] = workstation

        query_str += f" ORDER BY timestamp DESC LIMIT {_JOB_EXECUTION_HISTORY_LIMIT}"

        result = await db.execute(text(query_str), params)
        rows = result.mappings().fetchall()

        job_history = [
            {
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "job_name": row["job_name"],
                "workstation": row["workstation"] or "UNKNOWN",
                "status": row["status"] or "UNKNOWN",
                "return_code": row["return_code"] or 0,
                "runtime_seconds": row["runtime_seconds"] or 0,
                "scheduled_time": row["scheduled_time"].isoformat() if row["scheduled_time"] else None,
                "actual_start_time": row["actual_start_time"].isoformat() if row["actual_start_time"] else None,
                "completed_time": row["completed_time"].isoformat() if row["completed_time"] else None,
            }
            for row in rows
        ]

        logger.info(
            "fetch_job_execution_history.success",
            workstation=workstation,
            records=len(job_history)
        )
        return job_history

    except Exception as e:
        logger.error("fetch_job_execution_history.failed", error=str(e))
        return []


async def fetch_workstation_metrics_history(
    db: Any,
    workstation: str | None = None,
    days: int = 30
) -> list[dict[str, Any]]:
    """Fetch workstation metrics history for capacity forecasting.

    Args:
        db: Database session (AsyncSession)
        workstation: Filter by workstation (None = all workstations)
        days: Number of days to look back

    Returns:
        List of metrics records with timestamp, CPU, memory, disk
    """
    import os
    
    if not os.getenv("ENABLE_PREDICTIVE_WORKFLOWS", "").lower() == "true":
        logger.warning("fetch_workstation_metrics_history.feature_disabled")
        return []
    
    logger.info("fetch_workstation_metrics_history", workstation=workstation, days=days)

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    try:
        # Build query
        query_str = """
            SELECT
                timestamp,
                workstation,
                cpu_percent,
                memory_percent,
                disk_percent,
                load_avg_1min,
                cpu_count,
                total_memory_gb,
                total_disk_gb
            FROM workstation_metrics_history
            WHERE timestamp >= :cutoff_date
        """
        params = {"cutoff_date": cutoff_date}

        if workstation:
            query_str += " AND workstation = :workstation"
            params["workstation"] = workstation

        query_str += f" ORDER BY timestamp DESC LIMIT {_METRICS_HISTORY_LIMIT}"

        result = await db.execute(text(query_str), params)
        rows = result.mappings().fetchall()

        metrics_history = [
            {
                "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                "workstation": row["workstation"],
                "cpu_percent": float(row["cpu_percent"]) if row["cpu_percent"] is not None else 0.0,
                "memory_percent": float(row["memory_percent"]) if row["memory_percent"] is not None else 0.0,
                "disk_percent": float(row["disk_percent"]) if row["disk_percent"] is not None else 0.0,
                "load_avg_1min": float(row["load_avg_1min"]) if row["load_avg_1min"] is not None else 0.0,
                "cpu_count": int(row["cpu_count"]) if row["cpu_count"] is not None else 0,
                "total_memory_gb": float(row["total_memory_gb"]) if row["total_memory_gb"] is not None else 0.0,
                "total_disk_gb": float(row["total_disk_gb"]) if row["total_disk_gb"] is not None else 0.0,
            }
            for row in rows
        ]

        logger.info(
            "fetch_workstation_metrics_history.success",
            workstation=workstation,
            records=len(metrics_history)
        )
        return metrics_history

    except Exception as e:
        logger.error("fetch_workstation_metrics_history.failed", error=str(e))
        return []


__all__ = [
    "fetch_job_history",
    "fetch_workstation_metrics",
    "fetch_job_execution_history",
    "fetch_workstation_metrics_history",
    "detect_degradation",
    "correlate_metrics",
    "predict_timeline",
    "generate_recommendations",
    "notify_operators",
]
