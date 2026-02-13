#!/usr/bin/env python3
"""
Code Quality Utilities - Comprehensive utility library for code analysis and automation.

This module provides utility functions for:
- File operations (backup, restore, safe write)
- Code analysis (parse Pyflakes output, categorize issues)
- Import analysis (detect missing/unused imports)
- String processing (regex fixes, f-string validation)
- Progress tracking and reporting
- Validation functions (syntax checking, import verification)
- Error handling and logging utilities

All functions include comprehensive type hints and docstrings.
"""

import ast
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Type definitions
@dataclass
class PyflakesIssue:
    """Represents a single Pyflakes issue."""

    file_path: str
    line: int
    column: int
    message: str
    issue_type: str
    fixed: bool = False


@dataclass
class CodeQualityReport:
    """Report structure for code quality analysis."""

    total_issues: int = 0
    issues_by_type: dict[str, int] = field(default_factory=dict)
    issues_by_file: dict[str, list[PyflakesIssue]] = field(
        default_factory=lambda: defaultdict(list)
    )
    errors: list[str] = field(default_factory=list)


# File operation utilities


def backup_file(file_path: str | Path, backup_suffix: str = ".backup") -> str | None:
    """
    Create a backup of the specified file.

    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix to append to create backup filename

    Returns:
        Path to the backup file if successful, None if failed

    Raises:
        FileNotFoundError: If the source file doesn't exist
        PermissionError: If backup cannot be created due to permissions
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {file_path}")

    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)

    try:
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    except Exception as e:
        raise PermissionError(f"Failed to create backup: {e}") from None


def restore_file(file_path: str | Path, backup_suffix: str = ".backup") -> bool:
    """
    Restore a file from its backup.

    Args:
        file_path: Path to the file to restore
        backup_suffix: Suffix used for the backup file

    Returns:
        True if restoration was successful, False otherwise
    """
    file_path = Path(file_path)
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)

    if not backup_path.exists():
        return False

    try:
        shutil.copy2(backup_path, file_path)
        backup_path.unlink()  # Remove backup after successful restore
        return True
    except Exception:
        return False


def safe_write_file(
    file_path: str | Path,
    content: str | list[str],
    backup: bool = True,
    encoding: str = "utf-8",
) -> bool:
    """
    Safely write content to a file with optional backup.

    Uses atomic write operations to prevent corruption.

    Args:
        file_path: Path to the file to write
        content: Content to write (string or list of lines)
        backup: Whether to create backup before writing
        encoding: File encoding

    Returns:
        True if write was successful, False otherwise
    """
    file_path = Path(file_path)

    # Create backup if requested
    backup_path = None
    if backup and file_path.exists():
        try:
            backup_path = backup_file(file_path)
        except Exception:
            return False

    try:
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            delete=False,
            dir=file_path.parent,
            suffix=".tmp",
        ) as temp_file:
            if isinstance(content, list):
                temp_file.writelines(content)
            else:
                temp_file.write(content)
            temp_path = temp_file.name

        # Atomic move to final location
        os.replace(temp_path, file_path)
        return True

    except Exception:
        # Restore backup if write failed
        if backup_path:
            try:
                restore_file(file_path)
            except Exception:
                pass  # Best effort
        return False


# Code analysis functions


def parse_pyflakes_output(pyflakes_output: str) -> list[PyflakesIssue]:
    """
    Parse Pyflakes output and return structured issues.

    Args:
        pyflakes_output: Raw Pyflakes output as string

    Returns:
        List of PyflakesIssue objects
    """
    issues = []
    pattern = r"^(.+?):(\d+):(\d+):\s*(.+)"

    issue_patterns = {
        "unused_import": re.compile(r"'.+' imported but unused"),
        "undefined_name": re.compile(r"undefined name '(.+)'"),
        "fstring_placeholder": re.compile(r"f-string is missing placeholders"),
        "forward_annotation": re.compile(r"syntax error in forward annotation"),
        "unused_variable": re.compile(r"local variable '.+' is assigned to but never used"),
        "redefinition": re.compile(r"redefinition of unused '(.+)' from line \d+"),
    }

    lines = pyflakes_output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            file_path, line_num, col_num, message = match.groups()
            try:
                line_num, col_num = int(line_num), int(col_num)
            except ValueError:
                continue

            # Determine issue type
            issue_type = "unknown"
            for type_name, pat in issue_patterns.items():
                if pat.search(message):
                    issue_type = type_name
                    break

            issue = PyflakesIssue(
                file_path=file_path,
                line=line_num,
                column=col_num,
                message=message,
                issue_type=issue_type,
            )
            issues.append(issue)

    return issues


def categorize_issues(issues: list[PyflakesIssue]) -> CodeQualityReport:
    """
    Categorize Pyflakes issues into a structured report.

    Args:
        issues: List of PyflakesIssue objects

    Returns:
        CodeQualityReport with categorized issues
    """
    report = CodeQualityReport(total_issues=len(issues))

    for issue in issues:
        report.issues_by_type[issue.issue_type] = report.issues_by_type.get(issue.issue_type, 0) + 1
        report.issues_by_file[issue.file_path].append(issue)

    return report


# Import analysis utilities


def detect_missing_imports(file_path: str | Path) -> list[str]:
    """
    Detect missing imports in a Python file using AST analysis.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        List of missing import names
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        # Collect all names used in the code
        used_names = set()
        imported_names = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
                self.generic_visit(node)

        visitor = NameVisitor()
        visitor.visit(tree)

        # Common built-ins that don't need imports
        builtins = {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "__import__",
            "__name__",
            "Exception",
            "BaseException",
            "ValueError",
            "TypeError",
            "AttributeError",
            "ImportError",
            "NameError",
            "KeyError",
            "IndexError",
            "StopIteration",
            "GeneratorExit",
            "KeyboardInterrupt",
            "SystemExit",
            "ArithmeticError",
            "AssertionError",
            "LookupError",
            "OSError",
            "EOFError",
            "RuntimeError",
            "NotImplementedError",
            "SyntaxError",
            "IndentationError",
            "TabError",
            "UnicodeError",
            "UnicodeDecodeError",
            "UnicodeEncodeError",
            "UnicodeTranslateError",
            "BufferError",
            "BlockingIOError",
            "ChildProcessError",
            "ConnectionError",
            "BrokenPipeError",
            "ConnectionAbortedError",
            "ConnectionRefusedError",
            "ConnectionResetError",
            "FileExistsError",
            "FileNotFoundError",
            "InterruptedError",
            "IsADirectoryError",
            "NotADirectoryError",
            "PermissionError",
            "ProcessLookupError",
            "TimeoutError",
            "Warning",
            "UserWarning",
            "DeprecationWarning",
            "PendingDeprecationWarning",
            "SyntaxWarning",
            "RuntimeWarning",
            "FutureWarning",
            "ImportWarning",
            "UnicodeWarning",
            "BytesWarning",
            "ResourceWarning",
            "True",
            "False",
            "None",
        }

        missing = used_names - imported_names - builtins
        return sorted(list(missing))

    except Exception:
        return []


def detect_unused_imports(file_path: str | Path) -> list[str]:
    """
    Detect unused imports in a Python file using AST analysis.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        List of unused import names
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        # Collect imported names and used names
        imported_names = set()
        used_names = set()

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split(".")[0])
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # Handle module.attribute usage
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                self.generic_visit(node)

        visitor = ImportVisitor()
        visitor.visit(tree)

        unused = imported_names - used_names
        return sorted(list(unused))

    except Exception:
        return []


# String processing functions


def fix_regex_patterns(text: str) -> str:
    """
    Fix common regex pattern issues in code.

    Args:
        text: Text containing potential regex patterns

    Returns:
        Text with fixed regex patterns
    """
    # Fix unquoted regex patterns in forward annotations
    # Pattern: '^[a-zA-Z0-9_.-]+$' -> "^[a-zA-Z0-9_.-]+$"
    pattern = r"'(\^[^']+\$)'"
    return re.sub(pattern, r'"\1"', text)


def validate_f_string(f_string: str) -> tuple[bool, str | None]:
    """
    Validate an f-string expression.

    Args:
        f_string: The f-string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Remove the f prefix and parse as regular string
        if f_string.startswith("f"):
            f_string[1:]
        else:
            return False, "Not an f-string"

        # Try to compile the f-string
        compile(f_string, "<string>", "eval")
        return True, None

    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Invalid f-string: {e}"


# Progress tracking and reporting utilities


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description

    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current += increment

    def get_progress(self) -> tuple[int, int, float]:
        """Get current progress as (current, total, percentage)."""
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        return self.current, self.total, percentage

    def print_progress(self) -> None:
        """Print current progress to console."""
        current, total, percentage = self.get_progress()
        print("{self.description}: {current}/{total} ({percentage:.1f}%)")


def generate_quality_report(issues: list[PyflakesIssue], output_file: str | None = None) -> str:
    """
    Generate a comprehensive code quality report.

    Args:
        issues: List of PyflakesIssue objects
        output_file: Optional file path to save report

    Returns:
        Report as string
    """
    report = CodeQualityReport(total_issues=len(issues))

    for issue in issues:
        report.issues_by_type[issue.issue_type] = report.issues_by_type.get(issue.issue_type, 0) + 1
        report.issues_by_file[issue.file_path].append(issue)

    report_lines = [
        "Code Quality Analysis Report",
        "=" * 40,
        f"Total issues found: {report.total_issues}",
        "",
        "Issues by type:",
    ]

    for issue_type, count in report.issues_by_type.items():
        report_lines.append(f"  {issue_type}: {count}")

    report_lines.append("")
    report_lines.append("Issues by file:")

    for file_path, file_issues in report.issues_by_file.items():
        report_lines.append(f"  {file_path}: {len(file_issues)} issues")

    if report.errors:
        report_lines.append("")
        report_lines.append("Errors encountered:")
        for error in report.errors:
            report_lines.append(f"  - {error}")

    report_text = "\n".join(report_lines)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
        except Exception:
            pass  # Best effort

    return report_text


# Validation functions


def check_syntax(file_path: str | Path) -> tuple[bool, str | None]:
    """
    Check Python file syntax.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        ast.parse(source, filename=str(file_path))
        return True, None

    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error checking syntax: {e}"


def verify_imports(file_path: str | Path) -> tuple[bool, list[str]]:
    """
    Verify that all imports in a file can be resolved.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (all_valid, list_of_failed_imports)
    """
    failed_imports = []

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        class ImportChecker(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    try:
                        __import__(module_name)
                    except ImportError:
                        failed_imports.append(module_name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split(".")[0]
                    try:
                        __import__(module_name)
                    except ImportError:
                        failed_imports.append(node.module)
                self.generic_visit(node)

        checker = ImportChecker()
        checker.visit(tree)

    except Exception:
        pass  # If we can't parse, we'll catch it elsewhere

    return len(failed_imports) == 0, failed_imports


# Error handling and logging utilities


class CodeQualityError(Exception):
    """Base exception for code quality operations."""


class FileOperationError(CodeQualityError):
    """Exception raised for file operation failures."""


class AnalysisError(CodeQualityError):
    """Exception raised for analysis failures."""


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """
    Setup logging for code quality operations.

    Args:
        level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("code_quality")

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_operation(
    logger: logging.Logger, operation: str, success: bool, details: str | None = None
) -> None:
    """
    Log an operation result.

    Args:
        logger: Logger instance
        operation: Description of the operation
        success: Whether operation succeeded
        details: Optional additional details
    """
    level = logging.INFO if success else logging.ERROR
    message = f"{operation}: {'SUCCESS' if success else 'FAILED'}"
    if details:
        message += f" - {details}"

    logger.log(level, message)
