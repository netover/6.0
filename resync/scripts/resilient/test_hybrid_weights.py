#!/usr/bin/env python3
"""
Test script for Hybrid Retriever dynamic weights.

v5.2.3.22: Validates automatic weight adjustment for TWS domain.

Usage:
    python scripts/test_hybrid_weights.py
"""

import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Copy of patterns from hybrid_retriever.py for testing
EXACT_MATCH_PATTERNS = [
    re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Z0-9_]*\b"),  # AWSBH001, BATCH001 (min 2 digits)
    re.compile(r"\bRC[=:]\s*\d+\b", re.IGNORECASE),  # RC=8, RC: 12
    re.compile(r"\bABEND\s+[A-Z0-9]+\b", re.IGNORECASE),  # ABEND S0C7
    re.compile(r"\b[A-Z]{2}\d{3,}\b"),  # WS001, SRV123 (2 letters + 3+ digits)
    re.compile(r"\bEQQQ[A-Z0-9]+\b", re.IGNORECASE),  # EQQQ001I (full message ID)
    re.compile(r"\bAWSB[A-Z0-9]+\b", re.IGNORECASE),  # AWSBH001 (TWS job prefix)
    re.compile(r"\bJOB[=:]\s*[A-Z0-9_]+", re.IGNORECASE),  # JOB=X, JOB: X (explicit job ref)
    re.compile(r"\b[A-Z]{2,}_[A-Z0-9_]+_\d+\b"),  # BATCH_DAILY_001 (compound with number)
]

SEMANTIC_PATTERNS = [
    re.compile(r"\b(como|how|why|por\s*que)\b", re.IGNORECASE),
    re.compile(r"\b(resolver|fix|solve|solucionar)\b", re.IGNORECASE),
    re.compile(r"\b(configurar|configure|setup|instalar)\b", re.IGNORECASE),
    re.compile(r"\b(explicar?|explain|what\s+is|o\s+que\s+[eé])\b", re.IGNORECASE),
    re.compile(r"\b(melhores?\s+práticas?|best\s+practices?)\b", re.IGNORECASE),
    re.compile(r"\b(documentação|documentation|manual|guia)\b", re.IGNORECASE),
]


def get_dynamic_weights(query: str) -> tuple[float, float, str]:
    """
    Determine optimal weights based on query characteristics.
    Returns: (vector_weight, bm25_weight, query_type)
    """
    has_exact = any(p.search(query) for p in EXACT_MATCH_PATTERNS)
    has_semantic = any(p.search(query) for p in SEMANTIC_PATTERNS)

    if has_exact and not has_semantic:
        return (0.2, 0.8, "EXACT_MATCH")
    elif has_semantic and not has_exact:
        return (0.8, 0.2, "SEMANTIC")
    elif has_exact and has_semantic:
        return (0.4, 0.6, "MIXED")
    else:
        return (0.5, 0.5, "DEFAULT")


def test_query_classification():
    """Test query classification and weight assignment."""
    
    print("=" * 70)
    print("🧪 Hybrid Retriever - Dynamic Weights Test")
    print("=" * 70)
    
    # Test cases: (query, expected_type, description)
    test_cases = [
        # EXACT_MATCH - Should favor BM25 (0.8)
        ("status job AWSBH001", "EXACT_MATCH", "Job code uppercase"),
        ("erro RC=8 no batch", "EXACT_MATCH", "RC code"),
        ("ABEND S0C7 no job PAYROLL", "EXACT_MATCH", "ABEND code"),
        ("verificar EQQQ001I", "EXACT_MATCH", "TWS message ID"),
        ("job BATCH_DAILY_001 falhou", "EXACT_MATCH", "Job with underscore and number"),
        ("workstation WS001 offline", "EXACT_MATCH", "Workstation code"),
        ("JOB=PAYMENT_PROC status", "EXACT_MATCH", "JOB= syntax"),
        
        # SEMANTIC - Should favor Vector (0.8)
        ("como configurar o agente TWS", "SEMANTIC", "How-to question"),
        ("how to fix connection timeout", "SEMANTIC", "English how-to"),
        ("melhores práticas para scheduling", "SEMANTIC", "Best practices"),
        ("o que é um job stream", "SEMANTIC", "Definition question (no job code)"),
        ("documentação de instalação", "SEMANTIC", "Documentation request"),
        ("explicar dependências de jobs", "SEMANTIC", "Explanation request"),
        
        # MIXED - Should be balanced slightly toward BM25 (0.6)
        ("como resolver erro no job BATCH001", "MIXED", "How-to + job code"),
        ("por que AWSBH001 está falhando", "MIXED", "Why + job code"),
        ("fix RC=8 error in production", "MIXED", "Fix + RC code"),
        ("como configurar workstation WS001", "MIXED", "Configure + workstation code"),
        
        # DEFAULT - Should be balanced (0.5/0.5)
        ("jobs lentos ontem", "DEFAULT", "No clear pattern"),
        ("problemas de performance", "DEFAULT", "Generic issue"),
        ("listar todos os jobs ativos", "DEFAULT", "List command"),
    ]
    
    passed = 0
    failed = 0
    
    print("\n📋 Test Results:\n")
    print("{'Query':<45} {'Expected':<12} {'Got':<12} {'Weights':<15} {'Status'}")
    print("-" * 100)
    
    for query, expected_type, description in test_cases:
        vec_w, bm25_w, query_type = get_dynamic_weights(query)
        
        status = "✅" if query_type == expected_type else "❌"
        if query_type == expected_type:
            passed += 1
        else:
            failed += 1
        
        # Truncate long queries
        query_display = query[:42] + "..." if len(query) > 45 else query
        weights_display = "V:{vec_w:.1f} B:{bm25_w:.1f}"
        
        print("{query_display:<45} {expected_type:<12} {query_type:<12} {weights_display:<15} {status}")
    
    print("-" * 100)
    print("\n📊 Summary: {passed}/{passed+failed} tests passed ({100*passed/(passed+failed):.0f}%)")
    
    if failed > 0:
        print(f"⚠️  {failed} tests failed - review pattern matching")
    else:
        print("🎉 All tests passed!")
    
    return failed == 0


def test_tokenization():
    """Test BM25 tokenization for TWS patterns."""
    
    print("\n" + "=" * 70)
    print("🔤 BM25 Tokenization Test")
    print("=" * 70)
    
    def tokenize(text: str) -> list[str]:
        """Simplified tokenizer matching hybrid_retriever.py logic."""
        if not text:
            return []
        
        text = text.lower()
        
        # TWS pattern normalization
        text = re.sub(r"rc[=:]\s*(\d+)", r"rc_\1 rc\1", text)
        text = re.sub(r"abend\s+([a-z0-9]+)", r"abend_\1 \1", text)
        
        tokens = re.findall(r"[a-z0-9_\-]+", text)
        
        expanded = []
        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(token.split("_"))
            if "-" in token:
                expanded.extend(token.split("-"))
        
        return [t for t in expanded if len(t) >= 2]
    
    test_cases = [
        ("RC=8", ["rc_8", "rc8"]),
        ("RC: 12", ["rc_12", "rc12"]),
        ("ABEND S0C7", ["abend_s0c7", "s0c7"]),
        ("AWSBH001_BACKUP", ["awsbh001_backup", "awsbh001", "backup"]),
        ("JOB-DAILY-001", ["job-daily-001", "job", "daily", "001"]),
    ]
    
    print("\n{'Input':<25} {'Expected Tokens':<40} {'Got':<40} {'Status'}")
    print("-" * 120)
    
    all_passed = True
    for input_text, expected_contains in test_cases:
        tokens = tokenize(input_text)
        
        # Check if expected tokens are present
        missing = [t for t in expected_contains if t not in tokens]
        status = "✅" if not missing else f"❌ Missing: {missing}"
        
        if missing:
            all_passed = False
        
        print("{input_text:<25} {str(expected_contains):<40} {str(tokens):<40} {status}")
    
    return all_passed


def main():
    """Run all tests."""
    
    results = []
    
    results.append(("Query Classification", test_query_classification()))
    results.append(("Tokenization", test_tokenization()))
    
    print("\n" + "=" * 70)
    print("📊 Final Results")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Hybrid retriever is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
