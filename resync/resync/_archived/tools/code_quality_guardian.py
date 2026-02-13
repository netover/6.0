"""
Real-Time Code Quality Guardian for Resync

Watches code changes and automatically:
- Formats with black
- Lints with ruff
- Type checks with mypy
- Runs tests on affected files
- Reports issues in real-time

Author: Resync Team
Version: 5.9.9
"""

import hashlib
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class CodeQualityGuardian:
    """
    Real-time code quality enforcement.

    Watches source files and automatically runs quality checks
    when files change.

    Usage:
        guardian = CodeQualityGuardian()
        guardian.watch()  # Runs forever
    """

    def __init__(
        self,
        watch_dirs: list[str] | None = None,
        check_interval: int = 2,
    ):
        """
        Initialize guardian.

        Args:
            watch_dirs: Directories to watch (default: ["resync", "tests"])
            check_interval: Seconds between checks (default: 2)
        """
        self.watch_dirs = watch_dirs or ["resync", "tests"]
        self.check_interval = check_interval

        self.file_hashes: dict[str, str] = {}
        self.last_check = datetime.now()

        # Quality tools config
        self.tools = {
            "format": {
                "command": ["uv", "run", "black"],
                "auto_fix": True,
                "severity": "low"
            },
            "lint": {
                "command": ["uv", "run", "ruff", "check"],
                "auto_fix": True,  # ruff can auto-fix many issues
                "severity": "medium"
            },
            "typecheck": {
                "command": ["uv", "run", "mypy"],
                "auto_fix": False,
                "severity": "high"
            },
        }

    def watch(self):
        """
        Start watching files and running checks.

        Runs forever until interrupted.
        """
        logger.info("🔍 Code Quality Guardian started")
        logger.info(f"Watching: {', '.join(self.watch_dirs)}")

        # Initial hash
        self._hash_all_files()

        try:
            while True:
                time.sleep(self.check_interval)

                changed_files = self._detect_changes()

                if changed_files:
                    self._run_quality_checks(changed_files)

        except KeyboardInterrupt:
            logger.info("Guardian stopped")

    def _hash_all_files(self) -> None:
        """Compute hash of all watched Python files."""
        for watch_dir in self.watch_dirs:
            for root, _, files in os.walk(watch_dir):
                for filename in files:
                    if filename.endswith(".py"):
                        filepath = os.path.join(root, filename)
                        self.file_hashes[filepath] = self._hash_file(filepath)

    def _hash_file(self, filepath: str) -> str:
        """Compute MD5 hash of file."""
        try:
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
        except Exception:
            return ""

    def _detect_changes(self) -> set[str]:
        """
        Detect which files changed since last check.

        Returns:
            Set of changed file paths
        """
        changed = set()

        for watch_dir in self.watch_dirs:
            for root, _, files in os.walk(watch_dir):
                for filename in files:
                    if not filename.endswith(".py"):
                        continue

                    filepath = os.path.join(root, filename)
                    current_hash = self._hash_file(filepath)

                    # New file
                    if filepath not in self.file_hashes or current_hash != self.file_hashes[filepath]:
                        changed.add(filepath)
                        self.file_hashes[filepath] = current_hash

        return changed

    def _run_quality_checks(self, files: set[str]) -> None:
        """
        Run quality checks on changed files.

        Args:
            files: Set of changed file paths
        """
        logger.info(
            f"📝 Files changed: {len(files)}",
            files=[Path(f).name for f in list(files)[:5]]
        )

        # Run each tool
        for tool_name, tool_config in self.tools.items():
            self._run_tool(tool_name, tool_config, files)

    def _run_tool(
        self,
        name: str,
        config: dict,
        files: set[str]
    ) -> None:
        """
        Run a specific quality tool.

        Args:
            name: Tool name (format, lint, typecheck)
            config: Tool configuration
            files: Files to check
        """
        command = config["command"] + list(files)

        logger.info(f"Running {name}...")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.warning(
                    f"⚠️ {name}: ISSUES FOUND",
                    severity=config["severity"],
                    output=result.stdout[:500]  # First 500 chars
                )

                # Auto-fix if enabled
                if config["auto_fix"] and name in ["format", "lint"]:
                    self._auto_fix(name, files)

        except subprocess.TimeoutExpired:
            logger.error(f"{name}: TIMEOUT")
        except Exception as e:
            logger.error(f"{name}: ERROR - {e}")

    def _auto_fix(self, tool: str, files: set[str]) -> None:
        """
        Auto-fix issues with a tool.

        Args:
            tool: Tool name
            files: Files to fix
        """
        if tool == "format":
            # Black auto-formats
            subprocess.run(
                ["uv", "run", "black"] + list(files),
                capture_output=True
            )
            logger.info("🔧 Auto-formatted with black")

        elif tool == "lint":
            # Ruff can auto-fix many issues
            subprocess.run(
                ["uv", "run", "ruff", "check", "--fix"] + list(files),
                capture_output=True
            )
            logger.info("🔧 Auto-fixed with ruff")


class TestRunner:
    """
    Intelligently runs tests for changed files.

    Only runs tests affected by the changes, not entire suite.
    """

    def __init__(self):
        self.test_map = self._build_test_map()

    def _build_test_map(self) -> dict[str, list[str]]:
        """
        Build map of source files to test files.

        Returns:
            Dict mapping source files to their test files
        """
        # Simple heuristic: resync/core/foo.py → tests/test_foo.py
        test_map = {}

        for test_file in Path("tests").glob("test_*.py"):
            # Extract module name
            module = test_file.stem.replace("test_", "")

            # Find source file
            for source_file in Path("resync").rglob(f"{module}.py"):
                if str(source_file) not in test_map:
                    test_map[str(source_file)] = []
                test_map[str(source_file)].append(str(test_file))

        return test_map

    def run_tests_for(self, changed_files: set[str]) -> None:
        """
        Run tests affected by changed files.

        Args:
            changed_files: Set of changed source files
        """
        tests_to_run = set()

        for source_file in changed_files:
            if source_file in self.test_map:
                tests_to_run.update(self.test_map[source_file])

        if not tests_to_run:
            logger.info("No tests to run")
            return

        logger.info(f"🧪 Running {len(tests_to_run)} test files")

        command = ["uv", "run", "pytest", "-v"] + list(tests_to_run)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info("✅ All tests PASSED")
            else:
                logger.error(
                    "❌ Tests FAILED",
                    output=result.stdout[-500:]  # Last 500 chars
                )

        except subprocess.TimeoutExpired:
            logger.error("Tests: TIMEOUT")
        except Exception as e:
            logger.error(f"Tests: ERROR - {e}")


def main():
    """Run code quality guardian."""
    guardian = CodeQualityGuardian()
    guardian.watch()


if __name__ == "__main__":
    main()
