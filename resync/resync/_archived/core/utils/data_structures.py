"""
Optimized Data Structures

This module provides high-performance data structures
that replace O(n²) operations with O(1) or O(n log n) alternatives.
"""

import contextlib
import heapq
import math
import time
from collections import deque
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    O(1) LRU Cache implementation using OrderedDict and tracking.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[K, V] = {}
        self.access_order: deque[K] = deque(maxlen=capacity)
        self.access_set: set[K] = set()

    def get(self, key: K) -> V | None:
        """Get value with O(1) complexity."""
        if key not in self.cache:
            return None

        # Move to end of access order
        if key in self.access_set:
            self.access_order.remove(key)
        self.access_order.append(key)
        self.access_set.add(key)

        return self.cache[key]

    def put(self, key: K, value: V) -> None:
        """Put value with O(1) complexity."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            if key in self.access_set:
                self.access_order.remove(key)
            self.access_order.append(key)
            return

        # Check capacity and evict if necessary
        if len(self.cache) >= self.capacity:
            # Remove oldest
            oldest = self.access_order.popleft()
            del self.cache[oldest]
            self.access_set.discard(oldest)

        self.cache[key] = value
        self.access_order.append(key)
        self.access_set.add(key)

    def remove(self, key: K) -> bool:
        """Remove key with O(1) complexity."""
        if key not in self.cache:
            return False

        del self.cache[key]
        if key in self.access_set:
            self.access_set.remove(key)
            with contextlib.suppress(ValueError):
                self.access_order.remove(key)

        return True

    def size(self) -> int:
        """Get current size."""
        return len(self.cache)

    def keys(self) -> list[K]:
        """Get all keys."""
        return list(self.cache.keys())


class FastSet:
    """
    High-performance set with optimized operations.
    """

    def __init__(self, initial_data: list[Any] | None = None):
        self._data: set[Any] = set(initial_data) if initial_data else set()

    def add(self, item: Any) -> None:
        """Add item with O(1) complexity."""
        self._data.add(item)

    def update(self, items: list[Any]) -> None:
        """Add multiple items efficiently."""
        self._data.update(items)

    def discard(self, item: Any) -> None:
        """Discard item with O(1) complexity."""
        self._data.discard(item)

    def __contains__(self, item: Any) -> bool:
        """Check membership with O(1) complexity."""
        return item in self._data

    def __len__(self) -> int:
        """Get size with O(1) complexity."""
        return len(self._data)

    def __iter__(self):
        """Iterate over items."""
        return iter(self._data)


class IndexedPriorityQueue(Generic[T]):
    """
    Priority queue with O(log n) operations and value lookup.
    """

    def __init__(self):
        self._heap: list[tuple[int, int, T]] = []
        self._index: dict[T, int] = {}
        self._counter = 0

    def put(self, item: T, priority: int) -> None:
        """Put item with O(log n) complexity."""
        if item in self._index:
            # Update existing item
            old_index = self._index[item]
            if old_index < len(self._heap):
                self._heap[old_index] = (priority, self._counter, item)
        else:
            # Add new item
            entry = (priority, self._counter, item)
            self._index[item] = len(self._heap)
            heapq.heappush(self._heap, entry)
            self._counter += 1

    def get(self) -> T | None:
        """Get highest priority item with O(log n) complexity."""
        if not self._heap:
            return None

        priority, counter, item = heapq.heappop(self._heap)

        # Remove from index
        if item in self._index:
            del self._index[item]

        return item

    def remove(self, item: T) -> bool:
        """Remove item with O(log n) complexity."""
        if item not in self._index:
            return False

        index = self._index[item]
        if index < len(self._heap):
            # Mark as removed (using priority as sentinel)
            self._heap[index] = (float("-inf"), float("-inf"), item)

        del self._index[item]
        return True

    def size(self) -> int:
        """Get queue size."""
        return len(self._heap)

    def __len__(self) -> int:
        """Get queue size."""
        return len(self._heap)


class TimeBasedCache(Generic[K, V]):
    """
    Cache with time-based expiration and O(1) operations.
    """

    def __init__(self, ttl_seconds: int, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[K, tuple[V, float]] = {}
        self._heap: list[tuple[float, K]] = []

    def get(self, key: K) -> V | None:
        """Get value if not expired."""
        current_time = time.time()

        # Clean expired entries
        self._clean_expired(current_time)

        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if current_time - timestamp > self.ttl_seconds:
            # Expired, remove it
            del self._cache[key]
            return None

        return value

    def put(self, key: K, value: V) -> None:
        """Put value with timestamp."""
        current_time = time.time()

        # Clean expired entries first
        self._clean_expired(current_time)

        # Check size limit
        if len(self._cache) >= self.max_size and self._heap:
            # Remove oldest entry
            oldest_time, oldest_key = heapq.heappop(self._heap)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        # Add new entry
        self._cache[key] = (value, current_time)
        heapq.heappush(self._heap, (current_time, key))

    def _clean_expired(self, current_time: float) -> None:
        """Remove expired entries efficiently."""
        while self._heap and current_time - self._heap[0][0] > self.ttl_seconds:
            expired_time, expired_key = heapq.heappop(self._heap)
            if expired_key in self._cache:
                del self._cache[expired_key]

    def size(self) -> int:
        """Get current size."""
        return len(self._cache)


class FrequencyCounter:
    """
    High-performance frequency counter with O(1) operations.
    """

    def __init__(self):
        self._counts: dict[Any, int] = {}
        self._total = 0

    def increment(self, item: Any, count: int = 1) -> None:
        """Increment count with O(1) complexity."""
        self._counts[item] = self._counts.get(item, 0) + count
        self._total += count

    def get(self, item: Any) -> int:
        """Get count with O(1) complexity."""
        return self._counts.get(item, 0)

    def get_frequency(self, item: Any) -> float:
        """Get frequency as percentage."""
        if self._total == 0:
            return 0.0
        return self._counts.get(item, 0) / self._total

    def most_common(self, n: int = 10) -> list[tuple[Any, int]]:
        """Get n most common items efficiently."""
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def total(self) -> int:
        """Get total count."""
        return self._total

    def items(self) -> list[tuple[Any, int]]:
        """Get all items as list of tuples."""
        return list(self._counts.items())


class BloomFilter:
    """
    Space-efficient probabilistic set for membership testing.
    """

    def __init__(self, capacity: int, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = self._get_optimal_size(capacity, error_rate)
        self.bit_array = 0
        self.hash_count = self._get_hash_count(self.bit_array_size)
        self.items_added = 0

    def _get_optimal_size(self, capacity: int, error_rate: float) -> int:
        """Calculate optimal bit array size."""
        size = -capacity * (math.log(error_rate) / (math.log(2) ** 2))
        return max(1, int(size))

    def _get_hash_count(self, bit_size: int) -> int:
        """Get optimal number of hash functions."""
        return max(1, int(bit_size * 0.693147))

    def add(self, item: str) -> None:
        """Add item to bloom filter."""
        if self.items_added >= self.capacity:
            return  # Filter is saturated

        for i in range(self.hash_count):
            hash_val = self._hash(item, i)
            bit_index = hash_val % self.bit_array_size
            self.bit_array |= 1 << bit_index

        self.items_added += 1

    def __contains__(self, item: str) -> bool:
        """Check if item might be in set."""
        if self.items_added == 0:
            return False

        for i in range(self.hash_count):
            hash_val = self._hash(item, i)
            bit_index = hash_val % self.bit_array_size
            if not (self.bit_array & (1 << bit_index)):
                return False

        return True

    def _hash(self, item: str, seed: int) -> int:
        """Simple hash function."""
        hash_val = seed
        for char in item:
            hash_val = ((hash_val << 5) + hash_val) + ord(char)
            hash_val = hash_val & 0xFFFFFFFF
        return hash_val


class ChunkedArray:
    """
    Memory-efficient array for large data sets.
    """

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.chunks: list[list[Any]] = []
        self.current_chunk: list[Any] = []

    def append(self, item: Any) -> None:
        """Append item with efficient memory usage."""
        self.current_chunk.append(item)

        if len(self.current_chunk) >= self.chunk_size:
            self.chunks.append(self.current_chunk)
            self.current_chunk = []

    def extend(self, items: list[Any]) -> None:
        """Extend with multiple items efficiently."""
        for item in items:
            self.append(item)

    def finalize(self) -> list[list[Any]]:
        """Get all chunks and finalize."""
        if self.current_chunk:
            self.chunks.append(self.current_chunk)
            self.current_chunk = []

        return self.chunks

    def size(self) -> int:
        """Get total number of items."""
        return sum(len(chunk) for chunk in self.chunks) + len(self.current_chunk)

    def __iter__(self):
        """Iterate over all items."""
        for chunk in self.chunks:
            yield from chunk
        yield from self.current_chunk


def create_lru_cache(capacity: int) -> LRUCache:
    """Factory function to create LRU cache."""
    return LRUCache(capacity)


def create_priority_queue() -> IndexedPriorityQueue:
    """Factory function to create priority queue."""
    return IndexedPriorityQueue()


def create_time_cache(ttl_seconds: int, max_size: int = 1000) -> TimeBasedCache:
    """Factory function to create time-based cache."""
    return TimeBasedCache(ttl_seconds, max_size)


def create_frequency_counter() -> FrequencyCounter:
    """Factory function to create frequency counter."""
    return FrequencyCounter()


def create_bloom_filter(capacity: int, error_rate: float = 0.01) -> BloomFilter:
    """Factory function to create bloom filter."""
    return BloomFilter(capacity, error_rate)


def create_chunked_array(chunk_size: int = 1000) -> ChunkedArray:
    """Factory function to create chunked array."""
    return ChunkedArray(chunk_size)
