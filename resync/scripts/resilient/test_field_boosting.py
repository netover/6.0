#!/usr/bin/env python3
"""
Test script for BM25 Field Boosting.

v5.2.3.23: Validates field-specific boost weights for TWS domain.

Usage:
    python scripts/test_field_boosting.py
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SIMPLIFIED BM25 INDEX FOR TESTING
# =============================================================================

@dataclass
class TestBM25Index:
    """Simplified BM25 index for testing field boosting."""

    field_boosts: dict[str, float] = field(default_factory=lambda: {
        "job_name": 4.0,
        "error_code": 3.5,
        "workstation": 3.0,
        "job_stream": 2.5,
        "message_id": 2.5,
        "resource": 2.0,
        "title": 1.5,
        "content": 1.0,
    })

    ERROR_CODE_PATTERNS = [
        re.compile(r"RC[=:]\s*(\d+)", re.IGNORECASE),
        re.compile(r"ABEND\s+([A-Z0-9]+)", re.IGNORECASE),
    ]

    def _tokenize(self, text: str) -> list[str]:
        """Basic tokenization."""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r"rc[=:]\s*(\d+)", r"rc_\1 rc\1", text)
        text = re.sub(r"abend\s+([a-z0-9]+)", r"abend_\1 \1", text)
        tokens = re.findall(r"[a-z0-9_\-]+", text)
        expanded = []
        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(token.split("_"))
        return [t for t in expanded if len(t) >= 2]

    def _extract_error_codes(self, text: str) -> list[str]:
        """Extract error codes."""
        codes = []
        for pattern in self.ERROR_CODE_PATTERNS:
            codes.extend(pattern.findall(text))
        return codes

    def _extract_message_ids(self, text: str) -> list[str]:
        """Extract message IDs."""
        pattern = re.compile(r"\b(EQQQ\w+|AWSB\w+)\b", re.IGNORECASE)
        return pattern.findall(text)

    def calculate_boosted_freq(self, doc: dict[str, Any]) -> dict[str, float]:
        """Calculate boosted term frequencies for a document."""
        boosted_freqs: dict[str, float] = defaultdict(float)
        metadata = doc.get("metadata", {}) or {}

        # Content
        content = doc.get("content", "") or ""
        content_boost = self.field_boosts.get("content", 1.0)
        for token in self._tokenize(content):
            boosted_freqs[token] += content_boost

        # Job name (highest boost)
        job_name = metadata.get("job_name", "") or ""
        if job_name:
            job_boost = self.field_boosts.get("job_name", 4.0)
            for token in self._tokenize(job_name):
                boosted_freqs[token] += job_boost

        # Workstation
        workstation = metadata.get("workstation", "") or ""
        if workstation:
            ws_boost = self.field_boosts.get("workstation", 3.0)
            for token in self._tokenize(workstation):
                boosted_freqs[token] += ws_boost

        # Error codes
        full_text = f"{content} {job_name}"
        error_codes = self._extract_error_codes(full_text)
        if error_codes:
            error_boost = self.field_boosts.get("error_code", 3.5)
            for code in error_codes:
                for token in self._tokenize(code):
                    boosted_freqs[token] += error_boost

        # Message IDs
        message_ids = self._extract_message_ids(full_text)
        if message_ids:
            msg_boost = self.field_boosts.get("message_id", 2.5)
            for msg_id in message_ids:
                for token in self._tokenize(msg_id):
                    boosted_freqs[token] += msg_boost

        return dict(boosted_freqs)


# =============================================================================
# TEST CASES
# =============================================================================

def test_field_boosting():
    """Test that field boosting applies correct weights."""

    print("=" * 70)
    print("🧪 BM25 Field Boosting Test")
    print("=" * 70)

    index = TestBM25Index()

    # Test document with various fields
    test_doc = {
        "content": "Job AWSBH001 falhou com RC=8 na execução",
        "metadata": {
            "job_name": "AWSBH001",
            "workstation": "WS001",
            "job_stream": "DAILY_BATCH",
        }
    }

    freqs = index.calculate_boosted_freq(test_doc)

    print("\n📄 Test Document:")
    print(f"   Content: {test_doc['content']}")
    print(f"   Job Name: {test_doc['metadata']['job_name']}")
    print(f"   Workstation: {test_doc['metadata']['workstation']}")

    print("\n📊 Boosted Term Frequencies:")
    print("-" * 50)

    # Sort by frequency descending
    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

    for term, freq in sorted_freqs[:15]:
        boost_source = "?"
        if term in ["awsbh001", "awsbh"]:
            boost_source = f"job_name ({index.field_boosts['job_name']}x)"
        elif term in ["ws001", "ws"]:
            boost_source = f"workstation ({index.field_boosts['workstation']}x)"
        elif term in ["rc_8", "rc8"]:
            boost_source = f"error_code ({index.field_boosts['error_code']}x)"
        elif term in ["daily_batch", "daily", "batch"]:
            boost_source = f"job_stream ({index.field_boosts['job_stream']}x)"
        else:
            boost_source = f"content ({index.field_boosts['content']}x)"

        print("   {term:<20} {freq:>6.1f}  ← {boost_source}")

    # Validate expected rankings
    print("\n✅ Validation:")

    # Job name should have highest boost
    job_name_freq = freqs.get("awsbh001", 0)
    content_freq = freqs.get("job", 0)

    if job_name_freq > content_freq:
        print("   ✅ job_name ('awsbh001': {job_name_freq:.1f}) > content ('job': {content_freq:.1f})")
    else:
        print("   ❌ FAIL: job_name should have higher boost than content")
        return False

    # Error code should be boosted
    rc_freq = freqs.get("rc_8", 0) + freqs.get("rc8", 0)
    if rc_freq > 0:
        print("   ✅ error_code ('rc_8/rc8': {rc_freq:.1f}) detected and boosted")
    else:
        print("   ❌ FAIL: error_code should be extracted and boosted")
        return False

    # Workstation should be boosted
    ws_freq = freqs.get("ws001", 0)
    if ws_freq >= index.field_boosts["workstation"]:
        print("   ✅ workstation ('ws001': {ws_freq:.1f}) has correct boost")
    else:
        print(f"   ❌ FAIL: workstation should have boost of {index.field_boosts['workstation']}")
        return False

    return True


def test_boost_ratios():
    """Test that boost ratios are applied correctly."""

    print("\n" + "=" * 70)
    print("🧪 Boost Ratio Test")
    print("=" * 70)

    index = TestBM25Index()

    # Document with term appearing in different fields
    doc_job_only = {
        "content": "",
        "metadata": {"job_name": "TESTJOB"}
    }

    doc_content_only = {
        "content": "TESTJOB mentioned in content",
        "metadata": {}
    }

    freqs_job = index.calculate_boosted_freq(doc_job_only)
    freqs_content = index.calculate_boosted_freq(doc_content_only)

    job_freq = freqs_job.get("testjob", 0)
    content_freq = freqs_content.get("testjob", 0)

    expected_ratio = index.field_boosts["job_name"] / index.field_boosts["content"]
    actual_ratio = job_freq / content_freq if content_freq > 0 else 0

    print("\n   Term 'testjob' in job_name field: {job_freq:.1f}")
    print("   Term 'testjob' in content field: {content_freq:.1f}")
    print("   Expected ratio (job_name/content): {expected_ratio:.1f}x")
    print("   Actual ratio: {actual_ratio:.1f}x")

    if abs(actual_ratio - expected_ratio) < 0.1:
        print("   ✅ Boost ratio is correct!")
        return True
    else:
        print("   ❌ FAIL: Boost ratio mismatch")
        return False


def test_error_code_extraction():
    """Test error code extraction and boosting."""

    print("\n" + "=" * 70)
    print("🧪 Error Code Extraction Test")
    print("=" * 70)

    index = TestBM25Index()

    test_cases = [
        ("RC=8", ["8"]),
        ("RC: 12", ["12"]),
        ("ABEND S0C7", ["S0C7"]),
        ("error RC=4 and RC=8", ["4", "8"]),
        ("ABEND U0001 occurred", ["U0001"]),
    ]

    all_passed = True

    print("\n{'Input':<30} {'Expected':<20} {'Got':<20} {'Status'}")
    print("-" * 80)

    for text, expected in test_cases:
        codes = index._extract_error_codes(text)
        status = "✅" if set(codes) == set(expected) else "❌"
        if status == "❌":
            all_passed = False
        print("{text:<30} {str(expected):<20} {str(codes):<20} {status}")

    return all_passed


def test_message_id_extraction():
    """Test message ID extraction."""

    print("\n" + "=" * 70)
    print("🧪 Message ID Extraction Test")
    print("=" * 70)

    index = TestBM25Index()

    test_cases = [
        ("EQQQ001I: Job started", ["EQQQ001I"]),
        ("AWSBH001 completed", ["AWSBH001"]),
        ("Messages EQQQ100W and EQQQ200E", ["EQQQ100W", "EQQQ200E"]),
        ("No message IDs here", []),
    ]

    all_passed = True

    print("\n{'Input':<40} {'Expected':<25} {'Got':<25} {'Status'}")
    print("-" * 100)

    for text, expected in test_cases:
        ids = index._extract_message_ids(text)
        # Case-insensitive comparison
        ids_upper = [i.upper() for i in ids]
        expected_upper = [e.upper() for e in expected]
        status = "✅" if set(ids_upper) == set(expected_upper) else "❌"
        if status == "❌":
            all_passed = False
        print("{text:<40} {str(expected):<25} {str(ids):<25} {status}")

    return all_passed


def test_ranking_simulation():
    """Simulate ranking to verify field boosting impact."""

    print("\n" + "=" * 70)
    print("🧪 Ranking Simulation Test")
    print("=" * 70)

    index = TestBM25Index()

    # Create test documents
    docs = [
        {
            "id": "doc1",
            "content": "Generic content about batch processing",
            "metadata": {}
        },
        {
            "id": "doc2",
            "content": "Some text about jobs",
            "metadata": {"job_name": "AWSBH001"}  # High boost
        },
        {
            "id": "doc3",
            "content": "AWSBH001 mentioned in content only",
            "metadata": {}
        },
    ]

    query_term = "awsbh001"

    print(f"\n   Query: '{query_term}'")
    print("\n   Document Scores:")
    print("-" * 50)

    scores = []
    for doc in docs:
        freqs = index.calculate_boosted_freq(doc)
        score = freqs.get(query_term, 0)
        scores.append((doc["id"], score, doc.get("metadata", {}).get("job_name", "N/A")))
        print("   {doc['id']}: score={score:.1f} (job_name={doc.get('metadata', {}).get('job_name', 'N/A')})")

    # doc2 should rank highest (term in job_name field)
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n   Ranking: {' > '.join([s[0] for s in scores])}")

    if scores[0][0] == "doc2":
        print("   ✅ Document with job_name match ranked highest!")
        return True
    else:
        print("   ❌ FAIL: Document with job_name should rank highest")
        return False


def main():
    """Run all tests."""

    results = []

    results.append(("Field Boosting", test_field_boosting()))
    results.append(("Boost Ratios", test_boost_ratios()))
    results.append(("Error Code Extraction", test_error_code_extraction()))
    results.append(("Message ID Extraction", test_message_id_extraction()))
    results.append(("Ranking Simulation", test_ranking_simulation()))

    print("\n" + "=" * 70)
    print("📊 Final Results")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All field boosting tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
