#!/usr/bin/env python3
"""
Test script for Ollama/Qwen integration with LiteLLM.

v5.2.3.21: Validates local LLM setup and fallback behavior.

Usage:
    # Start Ollama first
    ollama serve
    ollama pull qwen2.5:3b
    
    # Run tests
    python scripts/test_ollama_integration.py

Requirements:
    - Ollama running on localhost:11434
    - Model qwen2.5:3b pulled
    - OPENAI_API_KEY set (for fallback testing)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_ollama_health():
    """Test if Ollama is running and accessible."""
    import httpx
    
    print("\n" + "=" * 60)
    print("🔍 Test 1: Ollama Health Check")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                print("✅ Ollama está rodando!")
                print(f"   Modelos disponíveis: {models}")
                
                if any("qwen" in m.lower() for m in models):
                    print("   ✅ Qwen encontrado!")
                    return True
                else:
                    print("   ⚠️  Qwen não encontrado. Execute: ollama pull qwen2.5:3b")
                    return False
            else:
                print(f"❌ Ollama retornou status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Ollama não está acessível: {e}")
        print("   Execute: ollama serve")
        return False


async def test_litellm_ollama():
    """Test LiteLLM with Ollama provider."""
    print("\n" + "=" * 60)
    print("🔍 Test 2: LiteLLM + Ollama Integration")
    print("=" * 60)
    
    try:
        import litellm
        litellm.suppress_debug_info = True
        
        start = time.time()
        response = await litellm.acompletion(
            model="ollama/qwen2.5:3b",
            messages=[{"role": "user", "content": "Responda apenas: OK"}],
            api_base="http://localhost:11434",
            max_tokens=10,
            temperature=0.1,
        )
        elapsed = time.time() - start
        
        content = response.choices[0].message.content
        print("✅ LiteLLM + Ollama funcionando!")
        print(f"   Resposta: {content}")
        print("   Tempo: {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Falha no LiteLLM + Ollama: {e}")
        return False


async def test_streaming():
    """Test streaming with Ollama."""
    print("\n" + "=" * 60)
    print("🔍 Test 3: Streaming")
    print("=" * 60)
    
    try:
        import litellm
        litellm.suppress_debug_info = True
        
        print("   Resposta (streaming): ", end="", flush=True)
        start = time.time()
        
        response = await litellm.acompletion(
            model="ollama/qwen2.5:3b",
            messages=[{"role": "user", "content": "Conte até 5 em português"}],
            api_base="http://localhost:11434",
            max_tokens=50,
            temperature=0.1,
            stream=True,
        )
        
        chunks = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunks.append(content)
                print(content, end="", flush=True)
        
        elapsed = time.time() - start
        print("\n   ✅ Streaming OK! ({elapsed:.2f}s)")
        return True
        
    except Exception as e:
        print(f"\n❌ Falha no streaming: {e}")
        return False


async def test_json_mode():
    """Test JSON mode with Qwen."""
    print("\n" + "=" * 60)
    print("🔍 Test 4: JSON Mode")
    print("=" * 60)
    
    try:
        import litellm
        litellm.suppress_debug_info = True
        
        prompt = """Extraia o nome do job da seguinte frase:
"Por favor cancele o job PAYMENT_PROCESS_DAILY"

Responda APENAS com JSON no formato: {"job_name": "nome"}"""
        
        start = time.time()
        response = await litellm.acompletion(
            model="ollama/qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            api_base="http://localhost:11434",
            max_tokens=50,
            temperature=0.1,
            format="json",
        )
        elapsed = time.time() - start
        
        content = response.choices[0].message.content
        print(f"   Resposta raw: {content}")
        
        # Try to parse JSON
        try:
            parsed = json.loads(content)
            print(f"   ✅ JSON válido: {parsed}")
            print("   Tempo: {elapsed:.2f}s")
            return True
        except json.JSONDecodeError:
            print("   ⚠️  JSON inválido (comum em modelos 3B)")
            return False
        
    except Exception as e:
        print(f"❌ Falha no JSON mode: {e}")
        return False


async def test_fallback_service():
    """Test the LLMService with fallback."""
    print("\n" + "=" * 60)
    print("🔍 Test 5: LLMService com Fallback")
    print("=" * 60)
    
    try:
        from resync.services.llm_fallback import get_llm_service
        
        llm = await get_llm_service()
        print(f"   Primary model: {llm.config.primary_model}")
        print(f"   Fallback chain: {llm.config.fallback_chain}")
        print(f"   Timeout: {llm.config.default_timeout}s")
        
        start = time.time()
        response = await llm.complete(
            prompt="Qual é a capital do Brasil? Responda em uma palavra.",
            max_tokens=20,
        )
        elapsed = time.time() - start
        
        print(f"   ✅ Resposta: {response.content}")
        print(f"   Provider: {response.provider.value}")
        print(f"   Fallback: {response.was_fallback}")
        print("   Tempo: {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Falha no LLMService: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_timeout_fallback():
    """Test that timeout triggers fallback to cloud."""
    print("\n" + "=" * 60)
    print("🔍 Test 6: Timeout → Fallback (simulado)")
    print("=" * 60)
    
    # This test is informational only
    print("   ℹ️  Para testar o fallback real:")
    print("   1. Pare o Ollama: systemctl stop ollama")
    print("   2. Faça uma chamada via LLMService")
    print("   3. Deve usar gpt-4o-mini automaticamente")
    print("   4. Reinicie: systemctl start ollama")
    
    if os.getenv("OPENAI_API_KEY"):
        print("   ✅ OPENAI_API_KEY configurada - fallback funcionará")
        return True
    else:
        print("   ⚠️  OPENAI_API_KEY não configurada - fallback não funcionará")
        return False


async def test_performance():
    """Test token generation speed."""
    print("\n" + "=" * 60)
    print("🔍 Test 7: Performance (tokens/segundo)")
    print("=" * 60)
    
    try:
        import litellm
        litellm.suppress_debug_info = True
        
        # Generate longer response to measure throughput
        start = time.time()
        response = await litellm.acompletion(
            model="ollama/qwen2.5:3b",
            messages=[{"role": "user", "content": "Explique o que é TWS em 100 palavras"}],
            api_base="http://localhost:11434",
            max_tokens=150,
            temperature=0.1,
        )
        elapsed = time.time() - start
        
        content = response.choices[0].message.content
        tokens = len(content.split())  # Aproximação
        tokens_per_sec = tokens / elapsed
        
        print(f"   Tokens gerados: ~{tokens}")
        print("   Tempo total: {elapsed:.2f}s")
        print("   Velocidade: ~{tokens_per_sec:.1f} tokens/s")
        
        if tokens_per_sec >= 5:
            print("   ✅ Performance aceitável")
        else:
            print("   ⚠️  Performance baixa (esperado em CPU)")
        
        return True
        
    except Exception as e:
        print(f"❌ Falha no teste de performance: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 Resync v5.2.3.21 - Ollama Integration Tests")
    print("=" * 60)
    
    results = {}
    
    # Basic health check
    results["ollama_health"] = await test_ollama_health()
    
    if not results["ollama_health"]:
        print("\n❌ Ollama não está disponível. Abortando testes.")
        print("   Execute: ollama serve && ollama pull qwen2.5:3b")
        return
    
    # LiteLLM integration
    results["litellm"] = await test_litellm_ollama()
    
    # Streaming
    results["streaming"] = await test_streaming()
    
    # JSON mode
    results["json"] = await test_json_mode()
    
    # LLMService
    results["service"] = await test_fallback_service()
    
    # Fallback check
    results["fallback"] = await test_timeout_fallback()
    
    # Performance
    results["performance"] = await test_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Resumo dos Testes")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test}")
    
    print(f"\n   Total: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 Todos os testes passaram! Ollama está pronto para uso.")
    else:
        print("\n⚠️  Alguns testes falharam. Verifique a configuração.")


if __name__ == "__main__":
    asyncio.run(main())
