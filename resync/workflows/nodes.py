# pylint: disable=all
# mypy: no-rerun
"""
Unified Workflow Nodes - Unified Mode Implementation

ReSync v6.1.0 uses a centralized configuration system:

1. OPTIMIZED MODE (default):
   - Fast, production-ready implementation
   - Controlled via NODE_LOG_LEVEL=NORMAL

2. VERBOSE MODE:
   - Detailed logging for debugging
   - Controlled via NODE_LOG_LEVEL=VERBOSE or NODE_LOG_LEVEL=DEBUG

3. PROFILING:
   - Enable performance profiling
   - Controlled via NODE_ENABLE_PROFILING=true

Usage:
    # Optimized mode (default)
    export NODE_LOG_LEVEL=NORMAL

    # Verbose mode
    export NODE_LOG_LEVEL=VERBOSE

    # Debug mode with profiling
    export NODE_LOG_LEVEL=DEBUG
    export NODE_ENABLE_PROFILING=true

The mode can be switched at runtime via environment variables
or programmatically using configure_nodes().
"""

import logging
from typing import Literal

from .node_config import get_node_config

logger = logging.getLogger(__name__)

# Get centralized configuration
config = get_node_config()

# Legacy: map to old WORKFLOW_MODE for compatibility
WORKFLOW_MODE: Literal["optimized", "verbose"] = config.legacy_mode.value

# Import appropriate implementation
if WORKFLOW_MODE == "verbose":
    logger.info("Loading VERBOSE workflow implementation (1813 lines, v5.11.0)")
    from .nodes_verbose import (
        fetch_job_history,
        fetch_workstation_metrics,
        detect_degradation,
        correlate_metrics,
        predict_timeline,
        generate_recommendations,
        notify_operators,
        # verbose-only
        fetch_job_execution_history,
        fetch_workstation_metrics_history,
    )

    # Export all workflow functions
    __all__ = [
        # Data fetching functions
        "fetch_job_history",
        "fetch_job_execution_history",
        "fetch_workstation_metrics",
        "fetch_workstation_metrics_history",
        # Analysis functions
        "detect_degradation",
        "correlate_metrics",
        "predict_timeline",
        # Action functions
        "generate_recommendations",
        "notify_operators",
        # Configuration
        "WORKFLOW_MODE",
    ]
else:
    logger.info("Loading OPTIMIZED workflow implementation (998 lines, v6.0.3)")
    from .nodes_optimized import (
        fetch_job_history,
        fetch_workstation_metrics,
        detect_degradation,
        correlate_metrics,
        predict_timeline,
        generate_recommendations,
        notify_operators,
    )

    # Export all workflow functions
    __all__ = [
        # Data fetching functions
        "fetch_job_history",
        "fetch_workstation_metrics",
        # Analysis functions
        "detect_degradation",
        "correlate_metrics",
        "predict_timeline",
        # Action functions
        "generate_recommendations",
        "notify_operators",
        # Configuration
        "WORKFLOW_MODE",
    ]

# Log the active mode on import
logger.info("Workflow nodes loaded in %s mode", WORKFLOW_MODE.upper())
