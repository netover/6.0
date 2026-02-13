#!/usr/bin/env python3
"""
Refactor Helper - Core Migration Tools

Provides automated tools for:
- Moving files with git mv (preserves history)
- Updating imports across codebase
- Creating compatibility shims
- Validating migration
- Detecting circular dependencies

Usage:
    python scripts/refactor_helper.py analyze
    python scripts/refactor_helper.py move --old-path FILE --new-path NEW_FILE
    python scripts/refactor_helper.py update-imports
    python scripts/refactor_helper.py create-shims
    python scripts/refactor_helper.py validate
    python scripts/refactor_helper.py migrate-module --module platform

Version: 5.4.2
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


class RefactorHelper:
    """Helper for automated refactoring tasks"""

    def __init__(self, base_path: str = "resync"):
        self.base_path = Path(base_path)
        self.moved_files: dict[str, str] = {}  # old_path -> new_path
        self.import_updates: dict[str, str] = {}  # old_import -> new_import

        # Load existing moves from file if exists
        self._load_state()

    def _load_state(self):
        """Load migration state from file"""
        state_file = Path(".refactor_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.moved_files = state.get("moved_files", {})
                self.import_updates = state.get("import_updates", {})

    def _save_state(self):
        """Save migration state to file"""
        state = {
            "moved_files": self.moved_files,
            "import_updates": self.import_updates,
        }
        with open(".refactor_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _path_to_import(self, file_path: str) -> str:
        """Convert file path to Python import path"""
        # Remove .py and convert / to .
        return file_path.replace(".py", "").replace("/", ".").replace("\\", ".")

    def _import_to_path(self, import_path: str) -> str:
        """Convert Python import path to file path"""
        return import_path.replace(".", "/") + ".py"

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def git_mv(self, old_path: str, new_path: str, dry_run: bool = False) -> bool:
        """
        Move file using git mv (preserves history)

        Args:
            old_path: Current file path
            new_path: New file path
            dry_run: If True, only simulate

        Returns:
            True if successful
        """
        old_path = str(old_path)
        new_path = str(new_path)

        # Ensure old file exists
        if not os.path.exists(old_path):
            print(f"❌ Source file not found: {old_path}")
            return False

        # Create target directory if needed
        new_dir = Path(new_path).parent
        if not dry_run:
            new_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "mv", old_path, new_path]

        if dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Moved: {old_path} → {new_path}")

            # Record the move
            self.moved_files[old_path] = new_path
            self.import_updates[self._path_to_import(old_path)] = self._path_to_import(new_path)
            self._save_state()

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error moving {old_path}: {e.stderr.decode()}")
            return False

    def move_batch(self, moves: list[tuple[str, str]], dry_run: bool = False) -> int:
        """
        Move multiple files

        Args:
            moves: List of (old_path, new_path) tuples
            dry_run: If True, only simulate

        Returns:
            Number of successful moves
        """
        success_count = 0

        for old_path, new_path in moves:
            if self.git_mv(old_path, new_path, dry_run=dry_run):
                success_count += 1

        print(f"\n📊 Moved {success_count}/{len(moves)} files")
        return success_count

    # =========================================================================
    # IMPORT UPDATES
    # =========================================================================

    def update_imports_in_file(self, file_path: str, dry_run: bool = False) -> bool:
        """
        Update imports in a single file based on moved files

        Args:
            file_path: File to update
            dry_run: If True, only show changes

        Returns:
            True if file was modified
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Cannot read {file_path}: {e}")
            return False

        original_content = content

        # Update imports based on moved files
        for old_import, new_import in self.import_updates.items():
            # Pattern: from X import ... or import X
            patterns = [
                (rf"from\s+{re.escape(old_import)}\s+import", f"from {new_import} import"),
                (rf"import\s+{re.escape(old_import)}(?:\s|$|,)", f"import {new_import}"),
            ]

            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)

        # Check if changed
        if content != original_content:
            if dry_run:
                print(f"\n[DRY RUN] Would update: {file_path}")
                self._show_diff(original_content, content)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Updated imports in: {file_path}")
            return True

        return False

    def update_all_imports(self, dry_run: bool = False) -> int:
        """
        Update imports in all Python files

        Args:
            dry_run: If True, only show changes

        Returns:
            Number of files updated
        """
        print("\n📝 Updating imports in all Python files...")

        updated_count = 0

        for py_file in self.base_path.rglob("*.py"):
            if self.update_imports_in_file(str(py_file), dry_run=dry_run):
                updated_count += 1

        # Also check tests/
        tests_path = Path("tests")
        if tests_path.exists():
            for py_file in tests_path.rglob("*.py"):
                if self.update_imports_in_file(str(py_file), dry_run=dry_run):
                    updated_count += 1

        print(f"\n📊 Updated {updated_count} files")
        return updated_count

    def _show_diff(self, old: str, new: str, max_lines: int = 20):
        """Show diff between old and new content"""
        import difflib

        diff = list(
            difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True), lineterm=""
            )
        )

        for _i, line in enumerate(diff[:max_lines]):
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  \033[92m{line}\033[0m", end="")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  \033[91m{line}\033[0m", end="")
            else:
                print(f"  {line}", end="")

        if len(diff) > max_lines:
            print(f"\n  ... ({len(diff) - max_lines} more lines)")

    # =========================================================================
    # COMPATIBILITY SHIMS
    # =========================================================================

    def create_compatibility_shim(self, old_path: str, new_path: str) -> bool:
        """
        Create a compatibility shim at old location

        Allows old imports to continue working with deprecation warning.

        Args:
            old_path: Original file location
            new_path: New file location

        Returns:
            True if shim created
        """
        new_import = self._path_to_import(new_path)
        old_module = Path(old_path).stem

        shim_content = f'''"""
Compatibility shim for {old_module}

This module has been moved to: {new_import}

DEPRECATED: This shim will be removed in v6.0.0
Import from the new location instead.
"""

import warnings

warnings.warn(
    f"Importing from {{__name__}} is deprecated. "
    f"Use 'from {new_import} import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from {new_import} import *

# Also re-export __all__ if defined
try:
    from {new_import} import __all__
except ImportError:
    pass
'''

        try:
            # Create directory if needed
            Path(old_path).parent.mkdir(parents=True, exist_ok=True)

            with open(old_path, "w") as f:
                f.write(shim_content)

            print(f"✅ Created shim: {old_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating shim {old_path}: {e}")
            return False

    def create_all_shims(self) -> int:
        """
        Create shims for all moved files

        Returns:
            Number of shims created
        """
        print("\n🔗 Creating compatibility shims...")

        created = 0
        for old_path, new_path in self.moved_files.items():
            if self.create_compatibility_shim(old_path, new_path):
                created += 1

        print(f"\n📊 Created {created} shims")
        return created

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_imports(self) -> bool:
        """
        Validate that all imports work

        Returns:
            True if all imports valid
        """
        print("\n🔍 Validating imports...")

        try:
            # Try to import main modules
            test_imports = [
                "resync",
                "resync.core",
                "resync.settings",
            ]

            for module in test_imports:
                try:
                    __import__(module)
                    print(f"  ✅ {module}")
                except ImportError as e:
                    print(f"  ❌ {module}: {e}")
                    return False

            print("\n✅ All imports valid!")
            return True

        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            return False

    def check_circular_dependencies(self) -> list[list[str]]:
        """
        Check for circular import dependencies

        Returns:
            List of circular dependency chains
        """
        print("\n🔄 Checking for circular dependencies...")

        # Build import graph
        import_graph: dict[str, set[str]] = defaultdict(set)

        for py_file in self.base_path.rglob("*.py"):
            module = self._path_to_import(str(py_file))

            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find imports
                import_pattern = r"(?:from|import)\s+(resync[.\w]+)"
                for match in re.finditer(import_pattern, content):
                    imported = match.group(1)
                    if imported != module:
                        import_graph[module].add(imported)
            except Exception:
                pass

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in import_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)
            return False

        for node in import_graph:
            if node not in visited:
                dfs(node, [])

        if cycles:
            print(f"\n⚠️  Found {len(cycles)} circular dependencies:")
            for cycle in cycles[:5]:  # Show first 5
                print("  → " + " → ".join(cycle))
        else:
            print("\n✅ No circular dependencies found")

        return cycles

    # =========================================================================
    # MODULE MIGRATION
    # =========================================================================

    def get_module_files(self, module_name: str) -> list[str]:
        """
        Get list of files belonging to a module based on analysis

        Args:
            module_name: Module name (platform, observability, etc.)

        Returns:
            List of file paths
        """
        # Load analysis data
        analysis_file = Path("docs/core_analysis.json")
        if not analysis_file.exists():
            print("❌ Run analyze_core_structure.py first")
            return []

        with open(analysis_file) as f:
            data = json.load(f)

        return data.get("grouping", {}).get(module_name, [])

    def migrate_module(self, module_name: str, target_dir: str, dry_run: bool = False) -> int:
        """
        Migrate all files of a module to target directory

        Args:
            module_name: Module name (platform, observability, etc.)
            target_dir: Target directory path
            dry_run: If True, only simulate

        Returns:
            Number of files migrated
        """
        print(f"\n📦 Migrating module: {module_name} → {target_dir}")

        files = self.get_module_files(module_name)

        if not files:
            print(f"⚠️  No files found for module: {module_name}")
            return 0

        print(f"   Found {len(files)} files")

        # Generate moves
        moves = []
        for file_path in files:
            # Only move files from root
            if "/core/" in file_path and file_path.count("/") == 2:  # root level
                filename = Path(file_path).name
                new_path = f"{target_dir}/{filename}"
                moves.append((file_path, new_path))

        if not moves:
            print("   No root-level files to move")
            return 0

        print(f"   Moving {len(moves)} root-level files...")

        # Execute moves
        success = self.move_batch(moves, dry_run=dry_run)

        # Update imports
        if not dry_run and success > 0:
            self.update_all_imports(dry_run=dry_run)

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Core Refactoring Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze                         # Analyze current structure
  %(prog)s move --old-path X --new-path Y  # Move single file
  %(prog)s update-imports                  # Update all imports
  %(prog)s create-shims                    # Create compatibility shims
  %(prog)s validate                        # Validate imports
  %(prog)s migrate-module --module platform --target resync/core/platform
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "analyze",
            "move",
            "update-imports",
            "create-shims",
            "validate",
            "check-circular",
            "migrate-module",
            "status",
        ],
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--old-path", help="Source file path")
    parser.add_argument("--new-path", help="Target file path")
    parser.add_argument("--module", help="Module name to migrate")
    parser.add_argument("--target", help="Target directory for migration")

    args = parser.parse_args()

    helper = RefactorHelper()

    if args.command == "analyze":
        print("Running analysis... Use analyze_core_structure.py for full analysis")
        helper.check_circular_dependencies()

    elif args.command == "move":
        if not args.old_path or not args.new_path:
            print("Error: --old-path and --new-path required")
            sys.exit(1)
        success = helper.git_mv(args.old_path, args.new_path, dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    elif args.command == "update-imports":
        helper.update_all_imports(dry_run=args.dry_run)

    elif args.command == "create-shims":
        helper.create_all_shims()

    elif args.command == "validate":
        success = helper.validate_imports()
        sys.exit(0 if success else 1)

    elif args.command == "check-circular":
        cycles = helper.check_circular_dependencies()
        sys.exit(0 if not cycles else 1)

    elif args.command == "migrate-module":
        if not args.module or not args.target:
            print("Error: --module and --target required")
            sys.exit(1)
        count = helper.migrate_module(args.module, args.target, dry_run=args.dry_run)
        print(f"\n✅ Migrated {count} files")

    elif args.command == "status":
        print("\n📊 Migration Status:")
        print(f"   Files moved: {len(helper.moved_files)}")
        print(f"   Import mappings: {len(helper.import_updates)}")

        if helper.moved_files:
            print("\n   Recent moves:")
            for old, new in list(helper.moved_files.items())[-5:]:
                print(f"     {old} → {new}")


if __name__ == "__main__":
    main()
