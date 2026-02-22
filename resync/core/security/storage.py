# pylint: skip-file
# mypy: ignore-errors
"""
Custom storage implementations for rate limiting.
Ensures no side effects (like threads) are created during import/initialization
in audit or strict environments.
"""

try:
    from limits.storage.memory import MemoryStorage

    import unittest.mock

    class SafeMemoryStorage(MemoryStorage):
        """
        Memory storage backend that disables the background cleanup thread.
        Use this for import-time Limiter instances to avoid side effects.
        """

        def __init__(self, uri: str, **options):
            # Temporarily block Thread.start during initialization
            # This ensures NO thread is ever spawned, even for a microsecond.
            # We use a localized patch valid only for this block.
            with unittest.mock.patch(
                "threading.Thread.start", new=lambda *a, **k: None
            ):
                super().__init__(uri, **options)

            # Ensure timer is dead
            self.timer = None

        def start_expiry_timer(self):
            # No-op: Do not start cleanup thread.
            pass

except ImportError:
    SafeMemoryStorage = None
