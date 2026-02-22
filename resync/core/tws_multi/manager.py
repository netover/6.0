# pylint: skip-file
# mypy: ignore-errors
"""
TWS Instance Manager.

Central manager for all TWS instances, providing:
- Instance CRUD operations
- Connection management
- Learning data coordination
- Session management
"""

from resync.core.task_tracker import track_task
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from resync.core.exceptions import IntegrationError

from .instance import TWSEnvironment, TWSInstance, TWSInstanceConfig, TWSInstanceStatus
from .learning import TWSLearningStore
from .session import TWSSession, get_session_manager

logger = logging.getLogger(__name__)


class TWSInstanceManager:
    """
    Manages multiple TWS instances.

    Provides centralized management of all TWS connections,
    each with isolated configuration and learning data.

    Example instances:
        SAZ → tws.saz.com.br:31116
        NAZ → tws.naz.com:31116
        MAZ → tws.maz.com:31116
    """

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config/tws_instances.json")

        # Instance storage
        self._instances: dict[str, TWSInstance] = {}

        # Learning stores (one per instance)
        self._learning_stores: dict[str, TWSLearningStore] = {}

        # HTTP clients (one per instance)
        self._clients: dict[str, httpx.AsyncClient] = {}

        # Session manager
        self._session_manager = get_session_manager()

        # Load saved instances
        self._load_instances()

    def _load_instances(self):
        """Load instances from configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                for instance_data in data.get("instances", []):
                    config = TWSInstanceConfig.from_dict(instance_data)
                    instance = TWSInstance(config=config)
                    self._instances[config.id] = instance

                    # Create learning store
                    self._learning_stores[config.id] = TWSLearningStore(config.id)

                logger.info("Loaded %s TWS instances", len(self._instances))
            except Exception as e:
                logger.error("Error loading instances: %s", e)
        else:
            # Create default instances
            self._create_default_instances()

    def _create_default_instances(self):
        """Create default example instances."""
        defaults = [
            TWSInstanceConfig(
                name="SAZ",
                display_name="São Paulo - SAZ",
                description="Production TWS - South America Zone",
                host="tws.saz.com.br",
                port=31116,
                datacenter="SAZ",
                region="South America",
                environment=TWSEnvironment.PRODUCTION,
                color="#10B981",  # Green
                sort_order=1,
            ),
            TWSInstanceConfig(
                name="NAZ",
                display_name="New York - NAZ",
                description="Production TWS - North America Zone",
                host="tws.naz.com",
                port=31116,
                datacenter="NAZ",
                region="North America",
                environment=TWSEnvironment.PRODUCTION,
                color="#3B82F6",  # Blue
                sort_order=2,
            ),
            TWSInstanceConfig(
                name="MAZ",
                display_name="Miami - MAZ",
                description="Production TWS - Mid America Zone",
                host="tws.maz.com",
                port=31116,
                datacenter="MAZ",
                region="North America",
                environment=TWSEnvironment.PRODUCTION,
                color="#F59E0B",  # Amber
                sort_order=3,
            ),
        ]

        for config in defaults:
            instance = TWSInstance(config=config)
            self._instances[config.id] = instance
            self._learning_stores[config.id] = TWSLearningStore(config.id)

        self._save_instances()
        logger.info("Created default TWS instances")

    def _save_instances(self):
        """Save instances to configuration file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "instances": [
                inst.config.to_dict()
                for inst in sorted(
                    self._instances.values(), key=lambda x: x.config.sort_order
                )
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %s TWS instances", len(self._instances))

    # Instance CRUD

    def add_instance(self, config: TWSInstanceConfig) -> TWSInstance:
        """Add a new TWS instance."""
        if any(i.config.name == config.name for i in self._instances.values()):
            raise ValueError(f"Instance with name '{config.name}' already exists")

        instance = TWSInstance(config=config)
        self._instances[config.id] = instance

        # Create learning store
        self._learning_stores[config.id] = TWSLearningStore(config.id)

        self._save_instances()
        logger.info("Added TWS instance: %s", config.name)

        return instance

    def get_instance(self, instance_id: str) -> TWSInstance | None:
        """Get instance by ID."""
        return self._instances.get(instance_id)

    def get_instance_by_name(self, name: str) -> TWSInstance | None:
        """Get instance by name."""
        for instance in self._instances.values():
            if instance.config.name.lower() == name.lower():
                return instance
        return None

    def get_all_instances(self) -> list[TWSInstance]:
        """Get all instances sorted by sort_order."""
        return sorted(self._instances.values(), key=lambda x: x.config.sort_order)

    def update_instance(
        self, instance_id: str, updates: dict[str, Any]
    ) -> TWSInstance | None:
        """Update instance configuration."""
        instance = self._instances.get(instance_id)
        if not instance:
            return None

        config = instance.config
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now(timezone.utc)
        self._save_instances()

        logger.info("Updated TWS instance: %s", config.name)
        return instance

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an instance."""
        if instance_id not in self._instances:
            return False

        instance = self._instances[instance_id]
        name = instance.config.name

        # Close any active connections
        if instance_id in self._clients:
            track_task(self._clients[instance_id].aclose(), name="clients")
            del self._clients[instance_id]

        # Remove learning store
        if instance_id in self._learning_stores:
            del self._learning_stores[instance_id]

        del self._instances[instance_id]
        self._save_instances()

        logger.info("Deleted TWS instance: %s", name)
        return True

    # Connection Management

    async def connect_instance(self, instance_id: str) -> bool:
        """Establish connection to a TWS instance."""
        instance = self._instances.get(instance_id)
        if not instance:
            return False

        config = instance.config
        instance.status = TWSInstanceStatus.CONNECTING

        client: httpx.AsyncClient | None = None
        try:
            # Create HTTP client
            client = httpx.AsyncClient(
                base_url=instance.connection_url,
                timeout=httpx.Timeout(
                    connect=config.connect_timeout,
                    read=config.read_timeout,
                ),
                verify=config.ssl_verify,
            )

            # Test connection
            response = await client.get("/twsd/health")

            if response.status_code == 200:
                instance.status = TWSInstanceStatus.CONNECTED
                instance.last_connected = datetime.now(timezone.utc)
                instance.error_count = 0
                instance.last_error = None
                self._clients[instance_id] = client
                client = None  # Ownership transferred — don't close in finally

                logger.info("Connected to TWS instance: %s", config.name)
                return True
            raise IntegrationError(f"Health check failed: {response.status_code}")

        except Exception as e:
            instance.status = TWSInstanceStatus.ERROR
            instance.error_count += 1
            instance.last_error = str(e)

            logger.error("Failed to connect to %s: %s", config.name, e)
            return False
        finally:
            # Close client if it was NOT stored (i.e. connection failed)
            if client is not None:
                await client.aclose()

    async def disconnect_instance(self, instance_id: str):
        """Disconnect from a TWS instance."""
        instance = self._instances.get(instance_id)
        if not instance:
            return

        if instance_id in self._clients:
            await self._clients[instance_id].aclose()
            del self._clients[instance_id]

        instance.status = TWSInstanceStatus.DISCONNECTED
        logger.info("Disconnected from TWS instance: %s", instance.config.name)

    async def test_connection(self, instance_id: str) -> dict[str, Any]:
        """Test connection to an instance."""
        instance = self._instances.get(instance_id)
        if not instance:
            return {"success": False, "error": "Instance not found"}

        config = instance.config
        start_time = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=10),
                verify=config.ssl_verify,
            ) as client:
                response = await client.get(f"{instance.connection_url}/twsd/health")

                latency = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "latency_ms": round(latency, 2),
                    "url": instance.connection_url,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": instance.connection_url,
            }

    # Learning Store Access

    def get_learning_store(self, instance_id: str) -> TWSLearningStore | None:
        """Get learning store for an instance."""
        return self._learning_stores.get(instance_id)

    # Session Management

    def create_session(
        self,
        instance_id: str,
        user_id: str,
        username: str,
    ) -> TWSSession | None:
        """Create a session for an operator on an instance."""
        instance = self._instances.get(instance_id)
        if not instance:
            return None

        session = self._session_manager.create_session(
            instance_id=instance_id,
            instance_name=instance.config.name,
            user_id=user_id,
            username=username,
        )

        instance.active_sessions += 1
        return session

    def close_session(self, session_id: str):
        """Close an operator session."""
        session = self._session_manager.get_session(session_id)
        if session:
            instance = self._instances.get(session.instance_id)
            if instance:
                instance.active_sessions = max(0, instance.active_sessions - 1)

        self._session_manager.close_session(session_id)

    def get_user_sessions(self, user_id: str) -> list[TWSSession]:
        """Get all sessions for a user."""
        return self._session_manager.get_user_sessions(user_id)

    # Summary & Stats

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all instances."""
        instances = self.get_all_instances()

        return {
            "total_instances": len(instances),
            "connected": len(
                [i for i in instances if i.status == TWSInstanceStatus.CONNECTED]
            ),
            "disconnected": len(
                [i for i in instances if i.status == TWSInstanceStatus.DISCONNECTED]
            ),
            "error": len([i for i in instances if i.status == TWSInstanceStatus.ERROR]),
            "total_sessions": sum(i.active_sessions for i in instances),
            "instances": [
                {
                    "id": i.config.id,
                    "name": i.config.name,
                    "display_name": i.config.display_name,
                    "status": i.status.value,
                    "color": i.config.color,
                    "active_sessions": i.active_sessions,
                }
                for i in instances
            ],
        }


# Global manager instance
_manager: TWSInstanceManager | None = None


def get_tws_manager() -> TWSInstanceManager:
    """Get global TWS instance manager."""
    global _manager
    if _manager is None:
        _manager = TWSInstanceManager()
    return _manager
