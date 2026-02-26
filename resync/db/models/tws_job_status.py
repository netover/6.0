"""
resync.db.models.tws_job_status
================================
Legacy import shim.  The canonical model is in
:mod:`resync.core.database.models.stores`.
"""
from __future__ import annotations

from resync.core.database.models.stores import TWSJobStatus

__all__ = ["TWSJobStatus"]
