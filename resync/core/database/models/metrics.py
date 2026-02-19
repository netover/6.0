from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Index, Integer, String
from resync.core.database import Base

class WorkstationMetricsHistory(Base):
    """
    Historical record of TWS workstation resource metrics (CPU, Memory, Disk).
    Data is collected via external monitoring scripts on FTAs/Workstations.
    """

    __tablename__ = "workstation_metrics_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workstation = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Core Resource Metrics
    cpu_percent = Column(Float, nullable=False)
    memory_percent = Column(Float, nullable=False)
    disk_percent = Column(Float, nullable=False)

    # Additional System Metrics
    load_avg_1min = Column(Float, nullable=True)
    cpu_count = Column(Integer, nullable=True)
    total_memory_gb = Column(Integer, nullable=True)
    total_disk_gb = Column(Integer, nullable=True)

    # Metadata
    os_type = Column(String(50), nullable=True)
    hostname = Column(String(255), nullable=True)
    collector_version = Column(String(20), nullable=True)

    # Reception Timestamp (Audit)
    received_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    __table_args__ = (
        Index(
            'ix_workstation_timestamp',
            'workstation',
            'timestamp',
            postgresql_using='btree'
        ),
        Index(
            'ix_received_at',
            'received_at',
            postgresql_using='btree'
        ),
    )

    def __repr__(self):
        return (
            f"<WorkstationMetricsHistory("
            f"workstation={self.workstation}, "
            f"timestamp={self.timestamp}, "
            f"cpu={self.cpu_percent}%, "
            f"mem={self.memory_percent}%, "
            f"disk={self.disk_percent}%"
            f")>"
        )
