"""
TWS Multi-Instance Management Package.

Allows connecting to multiple TWS/HWA servers simultaneously,
each with isolated configuration, learning data, and sessions.

Example:
    SAZ → tws.saz.com.br:31116
    NAZ → tws.naz.com:31116
    MAZ → tws.maz.com:31116
"""

from .instance import TWSInstance, TWSInstanceConfig
from .learning import TWSLearningStore
from .manager import TWSInstanceManager, get_tws_manager
from .session import TWSSession

__all__ = [
    "TWSInstance",
    "TWSInstanceConfig",
    "TWSInstanceManager",
    "get_tws_manager",
    "TWSLearningStore",
    "TWSSession",
]
