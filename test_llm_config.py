#!/usr/bin/env python3
"""
Test script to validate LLM and embedding configuration.
Tests OpenRouter connectivity and model availability.
"""

import os
import sys
import asyncio

# Load environment
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def validate_env():
    """Validate .env configuration."""
    print_section("1. ENVIRONMENT CONFIGURATION")
    
    # LLM Settings
    llm_endpoint = os.getenv("APP_LLM_ENDPOINT", "")
    llm_model = os.getenv("APP_LLM_MODEL", "")
    llm_api_key = os.getenv("APP_LLM_API_KEY", "")
    
    print(f"APP_LLM_ENDPOINT: {llm_endpoint}")
    print(f"APP_LLM_MODEL:    {llm_model}")
    print(f"APP_LLM_API_KEY:  {'*' * 10}...{llm_api_key[-10:] if llm_api_key else 'NOT SET'}")
    
    # Embedding Settings
    embed_model = os.getenv("APP_EMBED_MODEL", "")
    embed_endpoint = os.getenv("APP_EMBEDDING_ENDPOINT", "")
    embed_api_key = os.getenv("APP_EMBEDDING_API_KEY", "")
    
    print(f"\nAPP_EMBED_MODEL:    {embed_model}")
    print(f"APP_EMBEDDING_ENDPOINT: {embed_endpoint}")
    print(f"APP_EMBEDDING_API_KEY:  {'*' * 10}...{embed_api_key[-10:] if embed_api_key else 'NOT SET'}")
    
    # RAG Settings
    rag_url = os.getenv("APP_RAG_SERVICE_URL", "")
    print(f"\nAPP_RAG_SERVICE_URL: {rag_url}")
    
    # Validate
    errors = []
    
    if not llm_endpoint:
        errors.append("APP_LLM_ENDPOINT is not set")
    if not llm_model:
        errors.append("APP_LLM_MODEL is not set")
    if not llm_api_key:
        errors.append("APP_LLM_API_KEY is not set")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for e in errors:
            print(f"   - {e}")
        return False
    
    print("\n✅ Basic configuration looks correct")
    return True


async def test_llm_connection():
    """Test LLM connection to OpenRouter."""
    print_section("2. TESTING LLM CONNECTION")
    
    import httpx
    
    llm_endpoint = os.getenv("APP_LLM_ENDPOINT", "")
    llm_model = os.getenv("APP_LLM_MODEL", "")
    llm_api_key = os.getenv("APP_LLM_API_KEY", "")
    
    if not llm_endpoint or not llm_model:
        print("❌ LLM not configured")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test OpenRouter models endpoint
            print(f"Testing OpenRouter at: {llm_endpoint}")
            
            # Try to list models
            resp = await client.get(
                f"{llm_endpoint}/models",
                headers={"Authorization": f"Bearer {llm_api_key}"}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models]
                
                print(f"✅ Connected to OpenRouter!")
                print(f"   Available models: {len(models)}")
                
                # Check if our model is available
                if llm_model in model_ids:
                    print(f"✅ Model '{llm_model}' is available!")
                else:
                    # Try partial match
                    matches = [m for m in model_ids if llm_model.split('/')[-1] in m]
                    if matches:
                        print(f"⚠️ Model '{llm_model}' not found, but similar found:")
                        for m in matches[:5]:
                            print(f"   - {m}")
                    else:
                        print(f"❌ Model '{llm_model}' not found in available models")
                        print(f"   First 10 models:")
                        for m in model_ids[:10]:
                            print(f"   - {m}")
                
                return True
            else:
                print(f"❌ OpenRouter returned status {resp.status_code}")
                print(f"   {resp.text[:200]}")
                return False
                
    except httpx.ConnectError as e:
        print(f"❌ Cannot connect to OpenRouter: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        return False


async def test_llm_completion():
    """Test LLM completion with the configured model."""
    print_section("3. TESTING LLM COMPLETION")
    
    import httpx
    
    llm_endpoint = os.getenv("APP_LLM_ENDPOINT", "")
    llm_model = os.getenv("APP_LLM_MODEL", "")
    llm_api_key = os.getenv("APP_LLM_API_KEY", "")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{llm_endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": llm_model,
                    "messages": [
                        {"role": "user", "content": "Hello! Say 'OK' if you can hear me."}
                    ],
                    "max_tokens": 50
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ LLM Response: {content}")
                return True
            else:
                print(f"❌ Completion failed: {resp.status_code}")
                print(f"   {resp.text[:500]}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_embedding():
    """Test embedding service."""
    print_section("4. TESTING EMBEDDING SERVICE")
    
    import httpx
    
    embed_endpoint = os.getenv("APP_EMBEDDING_ENDPOINT", "")
    embed_model = os.getenv("APP_EMBED_MODEL", "")
    embed_api_key = os.getenv("APP_EMBEDDING_API_KEY", "")
    
    if not embed_endpoint:
        print("❌ Embedding endpoint not configured")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{embed_endpoint}/embeddings",
                headers={
                    "Authorization": f"Bearer {embed_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": embed_model,
                    "input": ["Hello world"]
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("data", [])
                if embeddings:
                    dim = len(embeddings[0].get("embedding", []))
                    print(f"✅ Embedding service working!")
                    print(f"   Model: {embed_model}")
                    print(f"   Dimension: {dim}")
                    return True
                else:
                    print(f"❌ No embeddings returned")
                    return False
            else:
                print(f"❌ Embedding failed: {resp.status_code}")
                print(f"   {resp.text[:300]}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    print("="*60)
    print(" Resync Configuration Validation Test")
    print("="*60)
    
    # Step 1: Validate .env
    env_ok = validate_env()
    
    if not env_ok:
        print("\n❌ Configuration validation failed!")
        sys.exit(1)
    
    # Step 2: Test LLM connection
    llm_ok = await test_llm_connection()
    
    if llm_ok:
        # Step 3: Test actual completion
        await test_llm_completion()
    
    # Step 4: Test embedding
    await test_embedding()
    
    print("\n" + "="*60)
    print(" TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
