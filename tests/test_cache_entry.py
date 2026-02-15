"""
Unit tests for CacheEntry serialization logic.

The CacheEntry class is imported from resync.models.cache, which is a lightweight
module without heavy dependencies, allowing for clean unit testing.
"""

import json
from datetime import datetime, timezone
from resync.models.cache import CacheEntry

def test_cache_entry_to_dict_valid():
    """Test that CacheEntry correctly serializes to a dictionary."""
    embedding = [0.1, 0.2, 0.3]
    metadata = {"model": "gpt-4", "latency": 100}
    timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    entry = CacheEntry(
        query="test query",
        response="test response",
        embedding=embedding,
        timestamp=timestamp,
        hit_count=5,
        metadata=metadata
    )

    data = entry.to_dict()

    assert data["query"] == "test query"
    assert data["response"] == "test response"
    assert data["embedding"] == json.dumps(embedding)
    assert data["timestamp"] == timestamp.isoformat()
    assert data["hit_count"] == 5
    assert data["metadata"] == json.dumps(metadata)

def test_cache_entry_from_dict_valid():
    """Test that CacheEntry correctly reconstructs from a dictionary."""
    embedding = [0.1, 0.2, 0.3]
    metadata = {"model": "gpt-4", "latency": 100}
    timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    data = {
        "query": "test query",
        "response": "test response",
        "embedding": json.dumps(embedding),
        "timestamp": timestamp.isoformat(),
        "hit_count": "5",
        "metadata": json.dumps(metadata)
    }

    entry = CacheEntry.from_dict(data)

    assert entry.query == "test query"
    assert entry.response == "test response"
    assert entry.embedding == embedding
    assert entry.timestamp == timestamp
    assert entry.hit_count == 5
    assert entry.metadata == metadata

def test_cache_entry_serialization_roundtrip():
    """Test the full round-trip of serialization and deserialization."""
    embedding = [0.5, 0.6]
    metadata = {"key": "value"}
    entry = CacheEntry(
        query="roundtrip query",
        response="roundtrip response",
        embedding=embedding,
        metadata=metadata
    )

    data = entry.to_dict()
    new_entry = CacheEntry.from_dict(data)

    assert new_entry.query == entry.query
    assert new_entry.response == entry.response
    assert new_entry.embedding == entry.embedding
    assert new_entry.metadata == entry.metadata
    assert new_entry.hit_count == entry.hit_count
    assert new_entry.timestamp == entry.timestamp

def test_cache_entry_from_dict_defaults():
    """Test that from_dict handles missing fields with sensible defaults."""
    data = {
        "query": "minimal query",
        "response": "minimal response"
    }

    entry = CacheEntry.from_dict(data)

    assert entry.query == "minimal query"
    assert entry.response == "minimal response"
    assert entry.embedding == []
    assert isinstance(entry.timestamp, datetime)
    assert entry.hit_count == 0
    assert entry.metadata == {}
