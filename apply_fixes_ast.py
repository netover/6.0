"""
AST-based logger call transformer.

Transforms logger f-string calls to structured keyword-argument style.

Example::

    BEFORE: logger.info(f"User {user} logged in")
    AFTER:  logger.info("user_logged_in", user=user)

Usage::

    python apply_fixes_ast.py              # dry-run (safe, no writes)
    python apply_fixes_ast.py --apply      # write changes (loses comments!)

Caveats:
  - ``ast.unparse()`` removes ALL source comments.  Only use ``--apply``
    in a clean git working tree.  For production use prefer ``libcst``.
  - Complex f-string expressions (``f"{obj.attr}"``) are skipped to avoid
    partial/incorrect transforms.
"""

# ---------------------------------------------------------------------------
# Standard-library imports (ALL at module level — never inside functions)
# ---------------------------------------------------------------------------
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# AST transformer
# ---------------------------------------------------------------------------

class LoggerCallFixer(ast.NodeTransformer):
    """
    Transform ``logger.<level>(f"…")`` calls to structured-logging style.

    NOTE: ``ast.unparse()`` does NOT preserve source comments. Only use
    this transformer with ``apply=False`` (dry-run) to inspect changes.
    For comment-preserving transforms use ``libcst`` instead.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.fixes: int = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        """Visit a Call node and transform logger f-string calls."""
        # Match: logger.<method>(f"...", ...)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            return self.generic_visit(node)

        # First argument must be an f-string
        if not node.args or not isinstance(node.args[0], ast.JoinedStr):
            return self.generic_visit(node)

        fstring = node.args[0]
        new_keywords: list[ast.keyword] = []
        message_parts: list[str] = []
        skipped_complex = False

        for value in fstring.values:
            if isinstance(value, ast.Constant):
                message_parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                if isinstance(value.value, ast.Name):
                    var_name = value.value.id
                    new_keywords.append(ast.keyword(arg=var_name, value=value.value))
                    message_parts.append(f"{{{var_name}}}")
                else:
                    # Complex expression (obj.attr, func(), etc.) — skip entirely
                    # to avoid generating an incorrect partial transform.
                    skipped_complex = True
                    break

        if skipped_complex:
            return self.generic_visit(node)

        # Build snake_case event key from the text fragments
        raw_msg = "".join(message_parts).strip()
        slug = re.sub(r"[^a-z0-9_]", "", raw_msg.lower().replace(" ", "_").replace("{", "").replace("}", ""))
        slug = re.sub(r"_+", "_", slug).strip("_")
        if len(slug) > 50:
            slug = slug[:50].rstrip("_")

        if not slug:
            return self.generic_visit(node)

        node.args = [ast.Constant(value=slug)]
        node.keywords = new_keywords + node.keywords
        self.fixes += 1
        return node

# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def fix_file_ast(file_path: str, *, apply: bool = False) -> int:
    """
    Find (and optionally write) structured-logging transforms for one file.

    When *apply* is ``True``, ``ast.unparse()`` is used which discards ALL
    source comments.  Only use this flag in a clean git working tree.

    Args:
        file_path: Path to Python source file.
        apply:     If ``False`` (default), report changes without writing.
                   If ``True``, write the transformed source.

    Returns:
        Number of fixes found (dry-run) or applied.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {file_path}: file not found")
        return 0

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)

        fixer = LoggerCallFixer(source)
        new_tree = fixer.visit(tree)
        ast.fix_missing_locations(new_tree)

        if fixer.fixes == 0:
            return 0

        print(f"  {file_path}: {fixer.fixes} fix(es) found")

        if not apply:
            return fixer.fixes

        # ----- apply mode -----
        warnings.warn(
            f"apply_fixes_ast: ast.unparse() will remove ALL comments in "
            f"{file_path}. Ensure you have a clean git commit to revert to.",
            UserWarning,
            stacklevel=2,
        )

        new_source = ast.unparse(new_tree)

        # Verify syntax before touching the file
        try:
            compile(new_source, file_path, "exec")
        except SyntaxError as exc:
            print(f"  ERROR: generated invalid syntax for {file_path}: {exc}")
            return 0

        # Atomic write: backup → temp → fsync → rename
        backup = path.with_suffix(".py.bak")
        shutil.copy2(path, backup)

        tmp_path: Path | None = None  # pre-initialize so except block is safe
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(new_source)
                tmp.flush()
                os.fsync(tmp.fileno())

            tmp_path.replace(path)

            # Reformat with ruff if available (best-effort)
            try:
                subprocess.run(
                    ["ruff", "format", file_path],
                    check=True,
                    capture_output=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Clean up backup only after successful write
            backup.unlink(missing_ok=True)

        except Exception as exc:
            shutil.copy2(backup, path)  # Rollback
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write {file_path}") from exc

        return fixer.fixes

    except RuntimeError:
        raise  # Let caller see write failures
    except (SyntaxError, ValueError, OSError) as exc:
        # P1-05: Log parse/transform errors without swallowing them silently
        # These are recoverable errors - report and continue
        print(f"Error processing {file_path}: {exc}", file=sys.stderr)
        return 0

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Scan ``resync/`` for logger f-string calls and report/apply fixes."""
    apply = "--apply" in sys.argv

    resync_dir = Path("resync")
    if not resync_dir.is_dir():
        print(
            "ERROR: 'resync/' directory not found. "
            "Run this script from the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if apply:
        print(
            "WARNING: --apply mode uses ast.unparse() which REMOVES ALL source "
            "comments. Ensure your working tree is clean before proceeding.\n"
        )

    # Skip symlinks (infinite loop risk on Docker volumes) and migrations
    files = [
        p
        for p in resync_dir.rglob("*.py")
        if not p.is_symlink() and "migrations" not in str(p)
    ]

    total_fixes = 0
    for file_path in files:
        total_fixes += fix_file_ast(str(file_path), apply=apply)

    action = "applied" if apply else "found (dry-run — pass --apply to write)"
    print(f"\nTotal fixes {action}: {total_fixes}")
    sys.exit(0 if total_fixes == 0 or not apply else 0)

if __name__ == "__main__":
    main()
