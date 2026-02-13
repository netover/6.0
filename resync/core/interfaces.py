"""Interfaces for Resync components."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Import model types for protocol type hints
from resync.models.agents import AgentConfig


@runtime_checkable
class IKnowledgeGraph(Protocol):
    """
    Interface for the Knowledge Graph service.
    Defines methods for interacting with the knowledge graph.
    """

    def add_content(self, content: str, metadata: dict[str, Any]) -> str:
        """Adds a piece of content (e.g., a document chunk) to the knowledge graph."""
        ...

    def add_content_sync(self, content: str, metadata: dict[str, Any]) -> str:
        """Adds a piece of content (e.g., a document chunk) to the knowledge graph."""
        ...

    def add_conversation(
        self,
        user_query: str,
        agent_response: str,
        agent_id: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Stores a conversation between a user and an agent."""
        ...

    def add_conversation_sync(
        self,
        user_query: str,
        agent_response: str,
        agent_id: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Stores a conversation between a user and an agent."""
        ...

    def search_similar_issues(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Searches the knowledge graph for similar past issues and solutions."""
        ...

    def search_similar_issues_sync(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Searches the knowledge graph for similar past issues and solutions."""
        ...

    def search_conversations(
        self,
        query: str = "type:conversation",
        limit: int = 100,
        sort_by: str = "created_at",
        _sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Optimized search method for conversations."""
        ...

    def search_conversations_sync(
        self,
        query: str = "type:conversation",
        limit: int = 100,
        sort_by: str = "created_at",
        _sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Optimized search method for conversations."""
        ...

    def add_solution_feedback(self, memory_id: str, feedback: str, rating: int) -> None:
        """Adds user feedback to a specific memory."""
        ...

    def add_solution_feedback_sync(self, memory_id: str, feedback: str, rating: int) -> None:
        """Adds user feedback to a specific memory."""
        ...

    def get_all_recent_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieves all recent conversation-type memories for auditing."""
        ...

    def get_all_recent_conversations_sync(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieves all recent conversation-type memories for auditing."""
        ...

    def get_relevant_context(self, user_query: str) -> str:
        """Retrieves a structured text summary of relevant knowledge from the graph."""
        ...

    def get_relevant_context_sync(self, user_query: str) -> str:
        """Retrieves a structured text summary of relevant knowledge from the graph."""
        ...

    def is_memory_flagged(self, memory_id: str) -> bool:
        """Checks if a memory has been flagged by the IA Auditor."""
        ...

    def is_memory_flagged_sync(self, memory_id: str) -> bool:
        """Checks if a memory has been flagged by the IA Auditor."""
        ...

    def is_memory_approved(self, memory_id: str) -> bool:
        """Checks if a memory has been approved by an admin."""
        ...

    def is_memory_approved_sync(self, memory_id: str) -> bool:
        """Checks if a memory has been approved by an admin."""
        ...

    def delete_memory(self, memory_id: str) -> None:
        """Deletes a memory from the knowledge graph."""
        ...

    def delete_memory_sync(self, memory_id: str) -> None:
        """Deletes a memory from the knowledge graph."""
        ...

    def add_observations(self, memory_id: str, observations: list[str]) -> None:
        """Adds observations to a memory in the knowledge graph."""
        ...

    def add_observations_sync(self, memory_id: str, observations: list[str]) -> None:
        """Adds observations to a memory in the knowledge graph."""
        ...

    def is_memory_already_processed(self, memory_id: str) -> bool:
        """Atomically checks if a memory has already been processed."""
        ...

    def is_memory_already_processed_sync(self, memory_id: str) -> bool:
        """Atomically checks if a memory has already been processed."""
        ...

    def atomic_check_and_flag(self, memory_id: str, reason: str, confidence: float) -> bool:
        """Atomically checks if memory is already processed, and if not, flags it."""
        ...

    def atomic_check_and_flag_sync(self, memory_id: str, reason: str, confidence: float) -> bool:
        """Atomically checks if memory is already processed, and if not, flags it."""
        ...

    def atomic_check_and_delete(self, memory_id: str) -> bool:
        """Atomically checks if memory is already processed, and if not, deletes it."""
        ...

    def atomic_check_and_delete_sync(self, memory_id: str) -> bool:
        """Atomically checks if memory is already processed, and if not, deletes it."""
        ...

    @property
    def client(self) -> Any:
        """Gets the underlying database client (SQLite connection for ContextStore)."""
        ...


@runtime_checkable
class IFileIngestor(Protocol):
    """
    Interface for the File Ingestor service.
    Defines methods for handling file uploads, saving, and processing for RAG.
    """

    def save_uploaded_file(self, file_name: str, file_content: Any) -> Path:
        """Saves an uploaded file to the RAG directory."""
        ...

    def ingest_file(self, file_path: Path) -> bool:
        """Ingests a single file into the knowledge graph."""
        ...


@runtime_checkable
class IAgentManager(Protocol):
    """Interface for managing AI agents."""

    def load_agents_from_config(self) -> None:
        """Loads agent configurations."""
        ...

    def get_agent(self, agent_id: str) -> Any:
        """Retrieves an agent by its ID."""
        ...

    def get_all_agents(self) -> list["AgentConfig"]:
        """Returns the configuration of all loaded agents."""
        ...


@runtime_checkable
class IConnectionManager(Protocol):
    """Interface for managing WebSocket connections."""

    def connect(self, websocket: Any, client_id: str) -> None:
        """Handles a new WebSocket connection."""
        ...

    def disconnect(self, client_id: str) -> None:
        """Handles a disconnected WebSocket client."""
        ...

    def broadcast(self, message: str) -> None:
        """Broadcasts a message to all connected clients."""
        ...

    def send_personal_message(self, message: str, client_id: str) -> None:
        """Sends a message to a specific client."""
        ...


@runtime_checkable
class IAuditQueue(Protocol):
    """Interface for an audit queue."""

    def add_audit_record(self, record: dict[str, Any]) -> None:
        """Adds an audit record to the queue."""
        ...

    def add_audit_record_sync(self, record: dict[str, Any]) -> None:
        """Adds an audit record to the queue."""
        ...

    def get_all_audits(self) -> list[dict[str, Any]]:
        """Retrieves all audit records."""
        ...

    def get_all_audits_sync(self) -> list[dict[str, Any]]:
        """Synchronously retrieves all audit records."""
        ...

    def get_audits_by_status(self, status: str) -> list[dict[str, Any]]:
        """Retrieves audit records by status."""
        ...

    def get_audits_by_status_sync(self, status: str) -> list[dict[str, Any]]:
        """Synchronously retrieves audit records by status."""
        ...

    def update_audit_status(self, audit_id: str, status: str) -> bool:
        """Updates audit status."""
        ...

    def update_audit_status_sync(self, audit_id: str, status: str) -> bool:
        """Synchronously updates audit status."""
        ...

    def get_audit_metrics(self) -> dict[str, Any]:
        """Retrieves audit metrics."""
        ...

    def get_audit_metrics_sync(self) -> dict[str, Any]:
        """Synchronously retrieves audit metrics."""
        ...


@runtime_checkable
class ITWSClient(Protocol):
    """Interface for the TWS client."""

    def get_system_status(self) -> dict[str, Any]:
        """Retrieves the current TWS system status."""
        ...

    def get_workstations_status(self) -> list[dict[str, Any]]:
        """Retrieves the status of all workstations."""
        ...

    def get_jobs_status(self) -> list[dict[str, Any]]:
        """Retrieves the status of all jobs."""
        ...

    def get_critical_path_status(self) -> list[dict[str, Any]]:
        """Retrieves the status of jobs on the critical path."""
        ...

    def check_connection(self) -> bool:
        """Checks if the connection to TWS is active."""
        ...

    @property
    def is_connected(self) -> bool:
        """Checks if the TWS client is currently connected."""
        ...

    def validate_connection(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> dict[str, bool]:
        """Validates TWS connection parameters."""
        ...

    def close(self) -> None:
        """Closes the TWS client connection."""
        ...

    def invalidate_system_cache(self) -> None:
        """Invalidates system-level cache."""
        ...

    def invalidate_all_jobs(self) -> None:
        """Invalidates all job-related cache."""
        ...

    def invalidate_all_workstations(self) -> None:
        """Invalidates all workstation-related cache."""
        ...

    def get_job_status_batch(self, job_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieves status for multiple jobs in batch."""
        ...

    # Note: These are simple attributes in the implementation, not properties
    host: str
    port: int
    user: str
    password: str
