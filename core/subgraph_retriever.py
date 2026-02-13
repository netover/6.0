"""
Subgraph Retrieval for GraphRAG

Retrieves structured knowledge subgraphs from Neo4j/Apache AGE instead of
unstructured text chunks. Provides rich contextual relationships for LLM reasoning.

Author: Resync Team
Version: 5.9.9
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SubgraphRetriever:
    """
    Retrieves knowledge subgraphs for GraphRAG applications.

    Instead of returning isolated documents, returns connected subgraphs
    containing entities, relationships, and their neighborhoods.
    """

    def __init__(self, knowledge_graph):
        """
        Initialize subgraph retriever.

        Args:
            knowledge_graph: IKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.cache = {}  # Simple in-memory cache

    async def retrieve_job_context(
        self,
        job_name: str,
        depth: int = 2,
        include_history: bool = True,
        include_solutions: bool = True
    ) -> dict[str, Any]:
        """
        Retrieve comprehensive job context as subgraph.

        Args:
            job_name: Name of the job
            depth: Graph traversal depth (1-3)
            include_history: Include execution history
            include_solutions: Include known solutions

        Returns:
            Dict with structured context:
            {
                "job": {...},
                "dependencies": [...],
                "errors": [...],
                "solutions": [...],
                "history": [...],
                "related_jobs": [...]
            }
        """
        cache_key = f"{job_name}:{depth}:{include_history}:{include_solutions}"

        if cache_key in self.cache:
            logger.debug("Subgraph cache hit", job_name=job_name)
            return self.cache[cache_key]

        try:
            # Build Cypher query based on parameters
            cypher = self._build_job_context_query(
                job_name=job_name,
                depth=depth,
                include_history=include_history,
                include_solutions=include_solutions
            )

            # Execute query
            result = await self.kg.execute_cypher(cypher, {"job_name": job_name})

            # Structure the result
            context = self._structure_job_context(result)

            # Cache for 1 hour
            self.cache[cache_key] = context

            logger.info(
                "Subgraph retrieved",
                job_name=job_name,
                dependencies=len(context.get("dependencies", [])),
                errors=len(context.get("errors", [])),
                solutions=len(context.get("solutions", []))
            )

            return context

        except Exception as e:
            logger.error("Subgraph retrieval failed: %s", e, exc_info=True)
            return self._empty_context(job_name)

    def _build_job_context_query(
        self,
        job_name: str,
        depth: int,
        include_history: bool,
        include_solutions: bool
    ) -> str:
        """Build Cypher query for job context."""

        # Base query - job and dependencies
        query_parts = [
            "MATCH (j:Job {name: $job_name})",
            "OPTIONAL MATCH (j)-[:DEPENDS_ON]->(dep:Job)",
            "OPTIONAL MATCH (j)-[:RUNS_ON]->(ws:Workstation)",
        ]

        # Add error patterns
        query_parts.append("OPTIONAL MATCH (j)-[:FAILED_WITH]->(err:Error)")

        # Add solutions if requested
        if include_solutions:
            query_parts.append("OPTIONAL MATCH (err)-[:SOLVED_BY]->(sol:Solution)")

        # Add execution history if requested
        if include_history:
            query_parts.append(
                "OPTIONAL MATCH (j)-[:HAS_EXECUTION]->(exec:Execution) "
                "WHERE exec.timestamp > datetime() - duration('P7D') "  # Last 7 days
            )

        # Return clause
        query_parts.append(
            "RETURN j as job, "
            "collect(DISTINCT dep) as dependencies, "
            "ws as workstation, "
            "collect(DISTINCT err) as errors"
        )

        if include_solutions:
            query_parts[-1] += ", collect(DISTINCT sol) as solutions"

        if include_history:
            query_parts[-1] += ", collect(DISTINCT exec) as executions"

        return "\n".join(query_parts)

    def _structure_job_context(self, result: list[dict]) -> dict[str, Any]:
        """Structure query result into usable context."""

        if not result:
            return {}

        row = result[0]

        return {
            "job": row.get("job", {}),
            "dependencies": [
                self._node_to_dict(dep)
                for dep in row.get("dependencies", [])
                if dep
            ],
            "workstation": self._node_to_dict(row.get("workstation")),
            "errors": [
                self._node_to_dict(err)
                for err in row.get("errors", [])
                if err
            ],
            "solutions": [
                self._node_to_dict(sol)
                for sol in row.get("solutions", [])
                if sol
            ],
            "executions": [
                self._node_to_dict(exec_)
                for exec_ in row.get("executions", [])
                if exec_
            ]
        }


    def _node_to_dict(self, node) -> dict | None:
        """Convert graph node to dictionary."""
        if node is None:
            return None

        if hasattr(node, "_properties"):
            return dict(node._properties)

        return dict(node) if isinstance(node, dict) else {}

    def _empty_context(self, job_name: str) -> dict[str, Any]:
        """Return empty context structure."""
        return {
            "job": {"name": job_name},
            "dependencies": [],
            "workstation": None,
            "errors": [],
            "solutions": [],
            "executions": []
        }

    def format_for_llm(self, context: dict[str, Any]) -> str:
        """
        Format subgraph context for LLM consumption.

        Converts structured graph data into natural language description
        that LLMs can reason over.

        Args:
            context: Structured context from retrieve_job_context()

        Returns:
            Formatted text for LLM prompt
        """
        lines = []

        # Job info
        job = context.get("job", {})
        lines.append(f"Job: {job.get('name', 'Unknown')}")

        if job.get("description"):
            lines.append(f"Description: {job['description']}")

        # Dependencies
        deps = context.get("dependencies", [])
        if deps:
            lines.append(f"\nDependencies ({len(deps)}):")
            for dep in deps[:5]:  # Limit to 5
                lines.append(f"  - {dep.get('name', 'Unknown')}")

        # Workstation
        ws = context.get("workstation")
        if ws:
            lines.append(f"\nWorkstation: {ws.get('name', 'Unknown')}")
            if ws.get("status"):
                lines.append(f"  Status: {ws['status']}")

        # Known errors
        errors = context.get("errors", [])
        if errors:
            lines.append(f"\nKnown Errors ({len(errors)}):")
            for err in errors[:3]:  # Limit to 3
                lines.append(f"  - {err.get('message', 'Unknown error')}")
                if err.get("return_code"):
                    lines.append(f"    RC: {err['return_code']}")

        # Solutions
        solutions = context.get("solutions", [])
        if solutions:
            lines.append(f"\nKnown Solutions ({len(solutions)}):")
            for sol in solutions[:3]:  # Limit to 3
                lines.append(f"  - {sol.get('description', 'Unknown solution')}")

        # Recent executions
        executions = context.get("executions", [])
        if executions:
            lines.append(f"\nRecent Executions ({len(executions)}):")
            for exec_ in executions[:5]:  # Last 5
                status = exec_.get("status", "UNKNOWN")
                timestamp = exec_.get("timestamp", "")
                lines.append(f"  - {timestamp}: {status}")

        return "\n".join(lines)

    def clear_cache(self):
        """Clear subgraph cache."""
        self.cache.clear()
        logger.info("Subgraph cache cleared")


def get_subgraph_retriever(knowledge_graph):
    """
    Factory function to get SubgraphRetriever instance.

    Args:
        knowledge_graph: IKnowledgeGraph instance

    Returns:
        SubgraphRetriever instance
    """
    return SubgraphRetriever(knowledge_graph)
