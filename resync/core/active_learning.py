"""
Active Learning Module - PostgreSQL Implementation.

This module re-exports from continual_learning for backward compatibility.
"""

from resync.core.continual_learning.active_learning import (
    ActiveLearner,
    get_active_learner,
)

__all__ = ["ActiveLearner", "get_active_learner"]
