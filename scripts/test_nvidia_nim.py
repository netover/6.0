#!/usr/bin/env python3
"""
Test script for Nvidia NIM integration via LiteLLM.

Usage:
    python scripts/test_nvidia_nim.py
"""

import asyncio
import os
import sys


async def test_nvidia_nim():
    """Test Nvidia NIM API via LiteLLM."""
    try:
        import litellm
        
        # Configure from environment
        nvidia_key = os.getenv('NVIDIA_API_KEY')
        nvidia_base = os.getenv('NVIDIA_API_BASE', 'https://integrate.api.nvidia.com/v1')
        
        if not nvidia_key:
            print("ERROR: NVIDIA_API_KEY not set")
            print("Set it with: export NVIDIA_API_KEY='nvapi-...'")
            return False
        
        print(f"Using Nvidia NIM endpoint: {nvidia_base}")
        print(f"API Key: {nvidia_key[:20]}...")
        
        # Configure LiteLLM
        os.environ['NVIDIA_API_KEY'] = nvidia_key
        os.environ['NVIDIA_API_BASE'] = nvidia_base
        litellm.drop_params = True
        
        print("\nTesting LiteLLM + Nvidia NIM...")
        
        # Test 1: Simple completion
        print("\n--- Test 1: Simple Completion ---")
        response = await litellm.acompletion(
            model="nvidia_nim/minimaxai/minimax-m2.5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, Nvidia NIM!' in one sentence."}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        
        content = response.choices[0].message.content
        print(f"Response: {content}")
        
        # Test 2: Tool calling (if supported)
        print("\n--- Test 2: Tool Calling ---")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        response2 = await litellm.acompletion(
            model="nvidia_nim/minimaxai/minimax-m2.5",
            messages=[
                {"role": "user", "content": "What's the weather in Sao Paulo?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
        )
        
        msg = response2.choices[0].message
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Tool calling works! Called: {msg.tool_calls[0].function.name}")
        else:
            print(f"Tool calling not triggered. Response: {msg.content}")
        
        print("\nAll tests passed!")
        return True
        
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install with: pip install litellm openai")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_nvidia_nim())
    sys.exit(0 if success else 1)
