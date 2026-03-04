#!/usr/bin/env python3
"""Test script to validate NVIDIA API connection with MiniMax model - non-streaming version."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/root/6.0/.env")

# Get API key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    print("ERROR: NVIDIA_API_KEY not found in environment")
    exit(1)

print(f"Using API Key: {NVIDIA_API_KEY[:20]}...")
sys.stdout.flush()

# Now import OpenAI after env is loaded

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

print("\n=== Testing NVIDIA API with minimaxai/minimax-m2.5 ===\n")
sys.stdout.flush()

try:
    completion = client.chat.completions.create(
        model="minimaxai/minimax-m2.5",
        messages=[{"role": "user", "content": "Olá, como você está?"}],
        temperature=0.7,
        max_tokens=100,
    )

    response = completion.choices[0].message.content
    print(f"Response: {response}")
    print("\n✅ API test successful!")
    sys.stdout.flush()
    
except Exception as e:
    print(f"\n\n❌ API test failed: {type(e).__name__}: {e}")
    sys.stdout.flush()
    exit(1)
