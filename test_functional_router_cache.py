"""
Updated functional test for Router Cache hit/miss logic in agent_graph.py.
This version mocks the SemanticCache to test the router_node integration logic.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from resync.core.langgraph.models import Intent, RouterOutput
# We need to import the module to patch the singleton getter
import resync.core.langgraph.agent_graph as agent_graph

async def run_functional_test():
    print("🧪 Functional Test: Router Cache Integration (With Mock Cache)\n")
    
    # Mock message
    message = "Qual o status do job PAYROLL?"
    state = {
        "message": message,
        "entities": {},
        "metadata": {}
    }

    # Mock RouterOutput
    mock_llm_result = RouterOutput(
        intent=Intent.STATUS,
        confidence=0.95,
        entities={"job_name": "PAYROLL"},
        reasoning="User asking for status"
    )

    # Mock Cache methods
    mock_cache = MagicMock()
    mock_cache.check_intent = AsyncMock()
    mock_cache.store_intent = AsyncMock()

    # Define behavior: 
    # 1st call -> miss (returns None)
    # 2nd call -> hit (returns dict)
    mock_cache.check_intent.side_effect = [None, {
        "intent": "status",
        "entities": {"job_name": "PAYROLL"},
        "confidence": 0.95
    }]

    with patch("resync.core.langgraph.agent_graph._get_router_cache", return_value=mock_cache), \
         patch("resync.core.utils.llm.call_llm_structured", new_callable=AsyncMock) as mock_llm:
        
        mock_llm.return_value = mock_llm_result
        
        # 1. First Call: Should be a MISS and call LLM
        print("1️⃣ Testing Cache MISS (First call)...")
        new_state = await agent_graph.router_node(state.copy())
        
        assert new_state["intent"] == Intent.STATUS
        assert new_state["entities"]["job_name"] == "PAYROLL"
        assert mock_llm.call_count == 1
        assert mock_cache.check_intent.call_count == 1
        assert mock_cache.store_intent.call_count == 1
        print("   ✅ Cache checked and LLM called on miss")
        print("   ✅ store_intent called after LLM classification")
        
        # 2. Second Call: Should be a HIT and skip LLM
        print("2️⃣ Testing Cache HIT (Second call)...")
        mock_llm.reset_mock()
        
        state_hit = {
            "message": message,
            "entities": {},
            "metadata": {}
        }
        
        new_state_hit = await agent_graph.router_node(state_hit)
        
        assert new_state_hit["intent"] == Intent.STATUS
        assert new_state_hit["entities"]["job_name"] == "PAYROLL"
        assert mock_llm.call_count == 0
        assert mock_cache.check_intent.call_count == 2
        print("   ✅ Cache HIT! LLM was skipped correctly.")

    print("\n" + "="*40)
    print("✅ FUNCTIONAL TEST PASSED!")
    print("="*40)
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(run_functional_test())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
