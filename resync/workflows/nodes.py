"""
Unified Workflow Nodes - Dual Mode Implementation

ReSync v6.1.0 Unified Edition supports two workflow implementations:

1. OPTIMIZED MODE (default):
   - Fast, production-ready implementation (998 lines)
   - Streamlined code from v6.0.3
   - Ideal for production environments
   
2. VERBOSE MODE:
   - Detailed, educational implementation (1813 lines)
   - Comprehensive code from v5.11.0
   - Ideal for learning and debugging

Usage:
    # Optimized mode (default)
    export WORKFLOW_MODE=optimized
    
    # Verbose mode
    export WORKFLOW_MODE=verbose

The mode can be switched at runtime without code changes.
All functions maintain the same interface regardless of mode.
"""

import os
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Determine workflow mode from environment
WORKFLOW_MODE: Literal["optimized", "verbose"] = os.getenv("WORKFLOW_MODE", "optimized").lower()

# Validate mode
if WORKFLOW_MODE not in ("optimized", "verbose"):
    logger.warning(
        f"Invalid WORKFLOW_MODE '{WORKFLOW_MODE}'. Defaulting to 'optimized'. "
        f"Valid values: 'optimized', 'verbose'"
    )
    WORKFLOW_MODE = "optimized"

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
        'fetch_job_history',
        'fetch_job_execution_history',
        'fetch_workstation_metrics',
        'fetch_workstation_metrics_history',
        
        # Analysis functions
        'detect_degradation',
        'correlate_metrics',
        'predict_timeline',
        
        # Action functions
        'generate_recommendations',
        'notify_operators',
        
        # Configuration
        'WORKFLOW_MODE',
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
        'fetch_job_history',
        'fetch_workstation_metrics',
        
        # Analysis functions
        'detect_degradation',
        'correlate_metrics',
        'predict_timeline',
        
        # Action functions
        'generate_recommendations',
        'notify_operators',
        
        # Configuration
        'WORKFLOW_MODE',
    ]

# Log the active mode on import
logger.info("Workflow nodes loaded in %s mode", WORKFLOW_MODE.upper())
