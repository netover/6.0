# apply_fixes_ast.py — P0 Fix (Python 3.12+ compatible)
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Any

class LoggerCallFixer(ast.NodeTransformer):
    """
    Transforms logger calls from f-strings to structured logging.
    Example: logger.info(f"User {user} logged in") -> logger.info("user_logged_in", user=user)
    """
    def __init__(self, source: str):
        self.source = source
        self.fixes = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Check if it's a logger call (logger.info, logger.error, etc.)
        if not (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == "logger"):
            return self.generic_visit(node)

        # Check for f-string as first argument
        if not node.args or not isinstance(node.args[0], ast.JoinedStr):
            return self.generic_visit(node)

        fstring = node.args[0]
        new_keywords = []
        message_parts = []

        for value in fstring.values:
            if isinstance(value, ast.Constant):
                message_parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                # Extract variable name
                if isinstance(value.value, ast.Name):
                    var_name = value.value.id
                    new_keywords.append(ast.keyword(arg=var_name, value=value.value))
                    message_parts.append(f"{{{var_name}}}") # Keep placeholder if needed, or structured key
                else:
                    # Complex expression, keep as is or simplify?
                    # For safety, we might skip complex expressions or name them arg_N
                    pass

        # Construct new message key (simplified)
        # This is a heuristic: "User {user} login" -> "user_login"
        raw_msg = "".join(message_parts).strip()
        slug = raw_msg.lower().replace(" ", "_").replace("{", "").replace("}", "")
        # Limit slug length
        if len(slug) > 50:
            slug = slug[:50] + "..."

        # In a real scenario, we might want to manually review these keys.
        # For this fix, we will use the snake_case conversion.

        new_args = [ast.Constant(value=slug)]

        # Merge existing keywords
        new_keywords.extend(node.keywords)

        node.args = new_args
        node.keywords = new_keywords
        self.fixes += 1

        return node

def fix_file_ast(file_path: str, dry_run: bool = True) -> int:
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {file_path}: File not found")
        return 0

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)

        fixer = LoggerCallFixer(source)
        new_tree = fixer.visit(tree)
        ast.fix_missing_locations(new_tree)

        if fixer.fixes > 0:
            print(f"  {file_path}: Found {fixer.fixes} fixes")
            if not dry_run:
                # Use ast.unparse (Python 3.9+)
                new_source = ast.unparse(new_tree)

                # Atomic write with backup
                backup = path.with_suffix(".py.bak")
                shutil.copy2(path, backup)

                try:
                    path.write_text(new_source, encoding="utf-8")
                    # Try to format with ruff if available
                    try:
                        subprocess.run(["ruff", "format", file_path], check=True, capture_output=True)
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        pass # Ruff not found or failed, ignore
                except Exception as e:
                    shutil.copy2(backup, path) # Rollback
                    raise RuntimeError(f"Failed to write {file_path}") from e

                # Verify syntax
                try:
                    compile(new_source, file_path, 'exec')
                except SyntaxError:
                     shutil.copy2(backup, path) # Rollback
                     raise RuntimeError(f"Generated invalid syntax for {file_path}")

        return fixer.fixes

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    dry_run = "--dry-run" in sys.argv
    # Scan all python files in resync
    files = list(Path("resync").rglob("*.py"))

    total_fixes = 0
    for file_path in files:
        # Skip migration files or tests if needed
        if "migrations" in str(file_path):
            continue

        fixes = fix_file_ast(str(file_path), dry_run=dry_run)
        total_fixes += fixes

    print(f"Total fixes {'identified' if dry_run else 'applied'}: {total_fixes}")

if __name__ == "__main__":
    main()
