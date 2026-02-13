
import sys
import os
from unittest.mock import MagicMock

# Adapt path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock optional dependencies
sys.modules["langchain_anthropic"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.prompts"] = MagicMock()
sys.modules["langchain_core.output_parsers"] = MagicMock()
sys.modules["langchain_core.messages"] = MagicMock()
sys.modules["langchain_core.runnables"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()

try:
    print("Attempting to import resync.workflows.nodes...")
    import resync.workflows.nodes
    print("Success importing resync.workflows.nodes")
except ImportError as e:
    print(f"Failed importing resync.workflows.nodes: {e}")
    if "LLMCacheWrapper" in str(e):
        print("Confirmed LLMCacheWrapper import error!")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Failed with other error: {e}")
    import traceback
    traceback.print_exc()
