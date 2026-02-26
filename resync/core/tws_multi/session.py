"""
TWS Session Management.

Manages operator sessions for TWS instances.
Each operator can have multiple sessions open to different instances.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class TWSSession:
    """
    Represents an operator's session to a TWS instance.

    Each session maintains:
    - Connection state
    - User context
    - Activity history
    - Current view/filters
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instance_id: str = ""
    instance_name: str = ""

    # User
    user_id: str = ""
    username: str = ""

    # State
    connected: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # View state (preserved per session)
    current_view: str = "dashboard"  # dashboard, jobs, schedules, logs
    filters: dict[str, Any] = field(default_factory=dict)
    selected_job_stream: str | None = None

    # Activity
    actions_count: int = 0
    last_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "user_id": self.user_id,
            "username": self.username,
            "connected": self.connected,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_view": self.current_view,
            "filters": self.filters,
            "selected_job_stream": self.selected_job_stream,
            "actions_count": self.actions_count,
            "last_action": self.last_action,
        }

    def record_activity(self, action: str):
        """Record user activity."""
        self.last_activity = datetime.now(timezone.utc)
        self.actions_count += 1
        self.last_action = action

    def set_view(self, view: str, filters: dict[str, Any] | None = None):
        """Update current view and filters."""
        self.current_view = view
        if filters:
            self.filters = filters
        self.record_activity(f"view:{view}")

    @property
    def is_active(self) -> bool:
        """Check if session is still active (activity in last 30 minutes)."""
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds() < 1800

class SessionManager:
    """
    Manages all active sessions across TWS instances.
    """

    def __init__(self):
        self._sessions: dict[str, TWSSession] = {}
        self._user_sessions: dict[str, list[str]] = {}  # user_id -> [session_ids]
        self._instance_sessions: dict[
            str, list[str]
        ] = {}  # instance_id -> [session_ids]

    def create_session(
        self,
        instance_id: str,
        instance_name: str,
        user_id: str,
        username: str,
    ) -> TWSSession:
        """Create a new session."""
        session = TWSSession(
            instance_id=instance_id,
            instance_name=instance_name,
            user_id=user_id,
            username=username,
            connected=True,
        )

        self._sessions[session.id] = session

        # Track by user
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        self._user_sessions[user_id].append(session.id)

        # Track by instance
        if instance_id not in self._instance_sessions:
            self._instance_sessions[instance_id] = []
        self._instance_sessions[instance_id].append(session.id)

        logger.info(
            "Session created: %s for user %s on %s", session.id, username, instance_name
        )

        return session

    def get_session(self, session_id: str) -> TWSSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_user_sessions(self, user_id: str) -> list[TWSSession]:
        """Get all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    def get_instance_sessions(self, instance_id: str) -> list[TWSSession]:
        """Get all sessions for an instance."""
        session_ids = self._instance_sessions.get(instance_id, [])
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    def close_session(self, session_id: str):
        """Close a session."""
        session = self._sessions.get(session_id)
        if not session:
            return

        session.connected = False

        # Remove from tracking
        if session.user_id in self._user_sessions:
            self._user_sessions[session.user_id] = [
                sid for sid in self._user_sessions[session.user_id] if sid != session_id
            ]

        if session.instance_id in self._instance_sessions:
            self._instance_sessions[session.instance_id] = [
                sid
                for sid in self._instance_sessions[session.instance_id]
                if sid != session_id
            ]

        del self._sessions[session_id]

        logger.info("Session closed: %s", session_id)

    def cleanup_inactive_sessions(self):
        """Clean up inactive sessions."""
        inactive = [
            sid for sid, session in self._sessions.items() if not session.is_active
        ]

        for session_id in inactive:
            self.close_session(session_id)

        if inactive:
            logger.info("Cleaned up %s inactive sessions", len(inactive))

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len([s for s in self._sessions.values() if s.is_active])

    def get_sessions_summary(self) -> dict[str, Any]:
        """Get summary of all sessions."""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": self.get_active_session_count(),
            "sessions_by_instance": {
                inst_id: len(sessions)
                for inst_id, sessions in self._instance_sessions.items()
            },
            "sessions_by_user": {
                user_id: len(sessions)
                for user_id, sessions in self._user_sessions.items()
            },
        }

# Global session manager
_session_manager: SessionManager | None = None

def get_session_manager() -> SessionManager:
    """Get global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
