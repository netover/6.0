# ruff: noqa: E501
# pylint
"""
Event-Driven Auto-Discovery for Knowledge Graph

Automatically discovers job relationships, dependencies, and error patterns
from TWS logs using LLM extraction. Runs in background without blocking users.

Author: Resync Team
Version: 5.9.9
"""

import json
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.prompts import PromptTemplate

from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)


def load_config_from_file():
    """Load GraphRAG configuration from TOML file."""
    try:
        from pathlib import Path

        import toml

        config_file = Path(__file__).parent.parent.parent / "config" / "graphrag.toml"

        if config_file.exists():
            config = toml.load(config_file)
            return config.get("graphrag", {})
        logger.warning("GraphRAG config file not found: %s", config_file)
        return {}
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to load GraphRAG config: %s", e, exc_info=True)
        return {}


class DiscoveryConfig:
    """
    Configuration for auto-discovery behavior.

    Values are loaded from config/graphrag.toml on startup.
    Changes via API are persisted to file.
    """

    # Load from file or use defaults
    _config = load_config_from_file()

    # Budget controls
    MAX_DISCOVERIES_PER_DAY = _config.get("budget", {}).get(
        "max_discoveries_per_day", 5
    )
    MAX_DISCOVERIES_PER_HOUR = _config.get("budget", {}).get(
        "max_discoveries_per_hour", 2
    )

    # Cache TTL - Dependências TWS são ESTÁTICAS!
    DISCOVERY_CACHE_DAYS = _config.get("cache", {}).get("ttl_days", 90)

    # Triggers
    DISCOVER_ON_NEW_ERROR = _config.get("triggers", {}).get(
        "discover_on_new_error", True
    )
    DISCOVER_ON_RECURRING_FAILURE = _config.get("triggers", {}).get(
        "discover_on_recurring_failure", True
    )
    MIN_FAILURES_TO_TRIGGER = _config.get("triggers", {}).get(
        "min_failures_to_trigger", 3
    )

    # Critical jobs (customize per deployment)
    CRITICAL_JOBS = set(
        _config.get("critical_jobs", {}).get(
            "jobs",
            [
                "PAYROLL_NIGHTLY",
                "BACKUP_DB",
                "ETL_CUSTOMER",
                "REPORT_SALES",
            ],
        )
    )

    @classmethod
    def reload_from_file(cls):
        """Reload configuration from file."""
        cls._config = load_config_from_file()

        cls.MAX_DISCOVERIES_PER_DAY = cls._config.get("budget", {}).get(
            "max_discoveries_per_day", 5
        )
        cls.MAX_DISCOVERIES_PER_HOUR = cls._config.get("budget", {}).get(
            "max_discoveries_per_hour", 2
        )
        cls.DISCOVERY_CACHE_DAYS = cls._config.get("cache", {}).get("ttl_days", 90)
        cls.MIN_FAILURES_TO_TRIGGER = cls._config.get("triggers", {}).get(
            "min_failures_to_trigger", 3
        )

        critical_jobs = cls._config.get("critical_jobs", {}).get("jobs", [])
        cls.CRITICAL_JOBS = set(critical_jobs) if critical_jobs else set()

        logger.info("DiscoveryConfig reloaded from file")


class EventDrivenDiscovery:
    """
    Automatically discovers job relationships from events.

    Uses LLM to extract entities and relationships from job logs
    when specific events occur (failures, delays, etc).
    Runs entirely in background - zero user wait time.
    """

    # LLM extraction prompt
    EXTRACTION_PROMPT = PromptTemplate.from_template("""
Extract job relationships and error patterns from these logs.

Job Name: {job_name}
Error Code: {error_code}
Logs:
{logs}

Return ONLY valid JSON (no markdown, no preamble):
{{
  "dependencies": [
    {{"source": "JOB_A", "relation": "WAITS_FOR", "target": "JOB_B", "confidence": 0.9}},
    {{"source": "JOB_A", "relation": "DEPENDS_ON", "target": "JOB_C", "confidence": 0.85}}
  ],
  "errors": [
    {{"job": "JOB_A", "error_type": "DATABASE_TIMEOUT", "description": "...", "confidence": 0.95}}
  ],
  "root_causes": [
    {{"error": "DATABASE_TIMEOUT", "cause": "Backup running concurrently", "confidence": 0.8}}
  ]
}}
""")

    def __init__(self, llm_service, knowledge_graph, tws_client, redis_client=None):
        """
        Initialize event-driven discovery.

        Args:
            llm_service: LLM service for extraction
            knowledge_graph: Knowledge graph instance
            tws_client: TWS client for fetching logs
            redis_client: Redis for caching (optional)
        """
        self.llm = llm_service
        self.kg = knowledge_graph
        self.tws = tws_client
        self.redis = redis_client

        # Counters for budget control
        self.discoveries_today = 0
        self.discoveries_this_hour = 0
        self.last_reset = datetime.now(timezone.utc)

        logger.info("EventDrivenDiscovery initialized")

    async def on_job_failed(self, job_name: str, event_details: dict):
        """
        Called when a job fails (ABEND).

        Decides whether to discover relationships and triggers
        background discovery if appropriate.

        Args:
            job_name: Name of failed job
            event_details: Event metadata (error_code, etc)
        """
        # Reset counters if needed
        self._reset_counters_if_needed()

        # Quick filters (no I/O)
        if not self._quick_filter(job_name, event_details):
            return

        # Async filters (with I/O)
        if not await self._should_discover(job_name, event_details):
            return

        # ✅ Trigger background discovery
        await create_tracked_task(self._discover_and_store(job_name, event_details))

        logger.info(
            "Discovery triggered",
            job_name=job_name,
            error_code=event_details.get("return_code"),
            discoveries_today=self.discoveries_today,
        )

    def _quick_filter(self, job_name: str, event_details: dict) -> bool:
        """
        Fast filters without I/O.

        Returns:
            True if job passes quick filters
        """
        # Budget exceeded?
        if self.discoveries_today >= DiscoveryConfig.MAX_DISCOVERIES_PER_DAY:
            logger.warning("Daily discovery budget exceeded")
            return False

        if self.discoveries_this_hour >= DiscoveryConfig.MAX_DISCOVERIES_PER_HOUR:
            logger.debug("Hourly discovery budget exceeded")
            return False

        # Only critical jobs
        if job_name not in DiscoveryConfig.CRITICAL_JOBS:
            logger.debug("Job %s not critical, skipping discovery", job_name)
            return False

        # Only severe errors (ABEND)
        severity = event_details.get("severity", "")
        if severity.upper() not in ("CRITICAL", "ERROR", "HIGH"):
            logger.debug("Event severity %s too low, skipping", severity)
            return False

        return True

    async def _should_discover(self, job_name: str, event_details: dict) -> bool:
        """
        Async filters with I/O (cache checks, graph queries).

        Returns:
            True if job should be discovered
        """
        # Already discovered recently?
        if self.redis:
            cache_key = f"discovered:{job_name}"
            if await self.redis.exists(cache_key):
                logger.debug("Job %s discovered recently, skipping", job_name)
                return False

        # Error pattern already in graph?
        error_code = event_details.get("return_code")
        if error_code and await self._has_known_solution(job_name, error_code):
            logger.debug("Error %s already mapped for %s", error_code, job_name)
            return False

        # Wait for recurring failures
        if DiscoveryConfig.DISCOVER_ON_RECURRING_FAILURE:
            failures = await self._count_recent_failures(job_name, days=7)
            if failures < DiscoveryConfig.MIN_FAILURES_TO_TRIGGER:
                logger.debug(
                    f"Job {job_name} failures ({failures}) below threshold, skipping"
                )
                return False

        return True

    async def _discover_and_store(self, job_name: str, event_details: dict):
        """
        Background task: extract relationships and store in graph.

        This runs asynchronously - user never waits for this!
        """
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Fetch job logs
            logs = await self._fetch_logs(job_name)

            if not logs:
                logger.warning("No logs available for %s", job_name)
                return

            # 2. Extract relationships using LLM
            relations = await self._extract_relations(job_name, event_details, logs)

            if not relations:
                logger.warning("No relations extracted for %s", job_name)
                return

            # 3. Store in knowledge graph
            stored_count = await self._store_relations(job_name, relations)

            # 4. Mark as discovered (cache)
            if self.redis:
                cache_key = f"discovered:{job_name}"
                await self.redis.setex(
                    cache_key, DiscoveryConfig.DISCOVERY_CACHE_DAYS * 86400, "1"
                )

            # 5. Update counters
            self.discoveries_today += 1
            self.discoveries_this_hour += 1

            # 6. Log success
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                "Discovery completed",
                job_name=job_name,
                relations_stored=stored_count,
                duration_seconds=duration,
                discoveries_today=self.discoveries_today,
            )

        except Exception as e:
            logger.error(
                f"Discovery failed for {job_name}: {e}",
                exc_info=True,
                job_name=job_name,
            )

    async def _fetch_logs(self, job_name: str, lines: int = 500) -> str:
        """Fetch job logs from TWS."""
        try:
            # Use existing TWS client
            logs = await self.tws.get_job_logs(job_name, lines=lines)

            # Limit size (avoid huge LLM prompts)
            if isinstance(logs, list):
                logs = "\n".join(logs[:500])
            elif isinstance(logs, str):
                logs = logs[:10000]  # Max 10KB

            return logs

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to fetch logs for %s: %s", job_name, e)
            return ""

    async def _extract_relations(
        self, job_name: str, event_details: dict, logs: str
    ) -> dict[str, Any] | None:
        """
        Extract relationships using LLM.

        Returns:
            Dict with dependencies, errors, root_causes
        """
        try:
            # Format prompt
            prompt = self.EXTRACTION_PROMPT.format(
                job_name=job_name,
                error_code=event_details.get("return_code", "Unknown"),
                logs=logs,
            )

            # Call LLM
            response = await self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,  # Deterministic extraction
            )

            # Parse JSON response
            # Remove markdown fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]

            return json.loads(clean_response.strip())

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Response was: %s", response[:200])
            return None
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("LLM extraction failed: %s", e, exc_info=True)
            return None

    async def _store_relations(self, job_name: str, relations: dict) -> int:
        """
        Store extracted relations in knowledge graph.

        Returns:
            Number of relations stored
        """
        stored = 0

        try:
            # Store dependencies
            for dep in relations.get("dependencies", []):
                await self.kg.add_relationship(
                    source=dep["source"],
                    relation=dep["relation"],
                    target=dep["target"],
                    properties={
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                        "confidence": dep.get("confidence", 0.8),
                        "source": "auto_discovery",
                    },
                )
                stored += 1

            # Store error patterns
            for error in relations.get("errors", []):
                await self.kg.add_node(
                    node_type="Error",
                    node_id=f"{error['job']}:{error['error_type']}",
                    properties={
                        "job": error["job"],
                        "error_type": error["error_type"],
                        "description": error.get("description", ""),
                        "confidence": error.get("confidence", 0.8),
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                stored += 1

            # Store root causes
            for cause in relations.get("root_causes", []):
                await self.kg.add_relationship(
                    source=cause["error"],
                    relation="CAUSED_BY",
                    target=cause["cause"],
                    properties={
                        "confidence": cause.get("confidence", 0.8),
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                stored += 1

            return stored

        except Exception as e:
            logger.error("Failed to store relations: %s", e, exc_info=True)
            return stored

    async def _has_known_solution(self, job_name: str, error_code: int) -> bool:
        """Check if error pattern already exists in graph."""
        try:
            # Query knowledge graph
            cypher = """
            MATCH (j:Job {name: $job_name})-[:FAILED_WITH]->(e:Error {return_code: $error_code})
            OPTIONAL MATCH (e)-[:SOLVED_BY]->(s:Solution)
            RETURN count(s) as solution_count
            """

            result = await self.kg.execute_cypher(
                cypher, {"job_name": job_name, "error_code": error_code}
            )

            return bool(result and result[0].get("solution_count", 0) > 0)

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to check known solutions: %s", e)
            return False

    async def _count_recent_failures(self, job_name: str, days: int = 7) -> int:
        """Count how many times job failed in last N days."""
        try:
            # Query knowledge graph or TWS history
            cypher = """
            MATCH (j:Job {name: $job_name})-[:HAS_EXECUTION]->(e:Execution)
            WHERE (e.status = 'FAILED' OR e.status = 'ABEND')
              AND e.timestamp > datetime() - duration({days: $days})
            RETURN count(e) as failure_count
            """

            result = await self.kg.execute_cypher(
                cypher, {"job_name": job_name, "days": days}
            )

            if result:
                return result[0].get("failure_count", 0)

            return 0

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to count failures: %s", e)
            return 0

    def _reset_counters_if_needed(self):
        """Reset daily/hourly counters if time period elapsed."""
        now = datetime.now(timezone.utc)

        # Reset daily counter
        if (now - self.last_reset).days >= 1:
            logger.info(
                "Resetting daily discovery counter",
                discoveries_yesterday=self.discoveries_today,
            )
            self.discoveries_today = 0
            self.last_reset = now

        # Reset hourly counter
        if (now - self.last_reset).total_seconds() >= 3600:
            self.discoveries_this_hour = 0

    def get_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            "discoveries_today": self.discoveries_today,
            "discoveries_this_hour": self.discoveries_this_hour,
            "budget_daily": DiscoveryConfig.MAX_DISCOVERIES_PER_DAY,
            "budget_hourly": DiscoveryConfig.MAX_DISCOVERIES_PER_HOUR,
            "critical_jobs_count": len(DiscoveryConfig.CRITICAL_JOBS),
            "cache_ttl_days": DiscoveryConfig.DISCOVERY_CACHE_DAYS,
            "last_reset": self.last_reset.isoformat(),
        }

    async def invalidate_discovery_cache(self, job_name: str | None = None):
        """
        Invalidate discovery cache for one job or all jobs.

        Use this when TWS plan changes (dependencies modified, new jobs added, etc).

        Args:
            job_name: Specific job to invalidate, or None for all jobs

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            logger.warning("Cannot invalidate cache - Redis not available")
            return 0

        try:
            if job_name:
                # Invalidate specific job
                cache_key = f"discovered:{job_name}"
                deleted = await self.redis.delete(cache_key)
                logger.info("Invalidated discovery cache for %s", job_name)
                return deleted
            # Invalidate all discoveries
            pattern = "discovered:*"
            keys = await self.redis.keys(pattern)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("Invalidated %s discovery cache entries", deleted)
                return deleted

            return 0

        except Exception as e:
            logger.error("Failed to invalidate cache: %s", e, exc_info=True)
            return 0


def get_discovery_service(llm_service, knowledge_graph, tws_client, redis_client):
    """
    Factory function to get EventDrivenDiscovery instance.

    Args:
        llm_service: LLM service instance
        knowledge_graph: Knowledge graph instance
        tws_client: TWS client instance
        redis_client: Redis client instance (optional)

    Returns:
        EventDrivenDiscovery instance
    """
    return EventDrivenDiscovery(llm_service, knowledge_graph, tws_client, redis_client)
