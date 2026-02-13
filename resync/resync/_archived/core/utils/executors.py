"""
Optimized thread executors for different workload types.

Thread-safe singleton ensures a single set of pools is shared application-wide.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor


class OptimizedExecutors:
    """Optimized thread executors for different workload types."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                # Re-check after acquiring the lock (another thread may have created it)
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init_executors()
                    cls._instance = instance
        return cls._instance

    def _init_executors(self):
        """Initialize thread pools with optimal configurations."""
        cpu_count = os.cpu_count() or 4

        # CPU-bound operations: JSON parsing, crypto, data processing
        # Limited workers to avoid context switching overhead
        self.cpu_pool = ThreadPoolExecutor(
            max_workers=min(8, cpu_count), thread_name_prefix="cpu_worker"
        )

        # I/O-bound operations: file reading, network calls
        # More workers to handle concurrent I/O operations
        self.io_pool = ThreadPoolExecutor(
            max_workers=min(32, cpu_count + 16), thread_name_prefix="io_worker"
        )

        # Large file processing: sequential to avoid memory pressure
        # Very limited workers to prevent memory exhaustion
        self.large_file_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="large_file_worker"
        )

    def get_cpu_executor(self):
        """Get executor for CPU-bound operations."""
        return self.cpu_pool

    def get_io_executor(self):
        """Get executor for I/O-bound operations."""
        return self.io_pool

    def get_large_file_executor(self):
        """Get executor for large file processing."""
        return self.large_file_pool

    def shutdown_all(self):
        """Shutdown all executors gracefully."""
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        self.large_file_pool.shutdown(wait=True)

    def __del__(self):
        """Ensure executors are shutdown when object is destroyed."""
        self.shutdown_all()
