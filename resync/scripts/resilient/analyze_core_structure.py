#!/usr/bin/env python3
"""
Core Structure Analyzer

Analyzes the current structure of resync/core/ and generates:
- Complete file inventory
- Dependency map
- Suggested thematic grouping
- Duplication detection

Part of the Core Refactoring Plan v5.4.2
"""

import ast
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class CoreAnalyzer:
    """Analyzer for core/ directory structure"""

    def __init__(self, core_path: str = "resync/core"):
        self.core_path = Path(core_path)
        self.files: list[dict] = []
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.dependencies: dict[str, list[str]] = defaultdict(list)

    def scan_files(self) -> list[dict]:
        """Scan all Python files in core/"""
        print("📂 Scanning core/ directory...")

        for py_file in self.core_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            file_info = {
                "path": str(py_file),
                "relative_path": str(py_file.relative_to(self.core_path)),
                "name": py_file.name,
                "directory": str(py_file.parent.relative_to(self.core_path))
                if py_file.parent != self.core_path
                else "root",
                "lines": 0,
                "imports": [],
                "internal_imports": [],
                "classes": [],
                "functions": [],
                "docstring": "",
            }

            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    file_info["lines"] = len(content.splitlines())

                # Parse AST
                tree = ast.parse(content)

                # Get docstring
                if ast.get_docstring(tree):
                    file_info["docstring"] = ast.get_docstring(tree)[:200]

                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_info["imports"].append(alias.name)
                            if "resync" in alias.name:
                                file_info["internal_imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        file_info["imports"].append(node.module)
                        if "resync" in node.module:
                            file_info["internal_imports"].append(node.module)

                    # Extract classes
                    if isinstance(node, ast.ClassDef):
                        file_info["classes"].append(node.name)

                    # Extract top-level functions
                    if isinstance(node, ast.FunctionDef) and isinstance(node, ast.FunctionDef):
                        # Only top-level (not inside classes)
                        file_info["functions"].append(node.name)

            except SyntaxError as e:
                print(f"⚠️  Syntax error in {py_file}: {e}")
            except Exception as e:
                print(f"⚠️  Error processing {py_file}: {e}")

            self.files.append(file_info)

        print(f"✅ Found {len(self.files)} Python files")
        return self.files

    def suggest_grouping(self) -> dict[str, list[str]]:
        """Suggest thematic grouping based on file names and content"""
        print("\n💡 Suggesting thematic grouping...")

        groups = {
            "tws": [],  # TWS integration
            "agents": [],  # AI agents, LLM, specialists
            "retrieval": [],  # RAG, knowledge graph, cache, embeddings
            "security": [],  # Auth, validation, encryption
            "observability": [],  # Logs, metrics, health, monitoring
            "platform": [],  # Config, DI, database, redis, resilience
            "unknown": [],
        }

        # Keywords for each group
        keywords = {
            "tws": ["tws", "workload", "hwa", "conman", "optman"],
            "agents": [
                "agent",
                "llm",
                "model",
                "specialist",
                "agno",
                "router",
                "langgraph",
                "litellm",
            ],
            "retrieval": [
                "rag",
                "knowledge",
                "graph",
                "cache",
                "embed",
                "vector",
                "retriev",
                "chunk",
                "ingest",
                "pgvector",
                "rerank",
            ],
            "security": [
                "auth",
                "security",
                "session",
                "token",
                "validation",
                "header",
                "csp",
                "csrf",
                "encrypt",
                "gdpr",
                "soc2",
                "compliance",
            ],
            "observability": [
                "log",
                "metric",
                "monitor",
                "health",
                "trace",
                "alert",
                "siem",
                "incident",
                "audit",
                "benchmark",
            ],
            "platform": [
                "container",
                "config",
                "database",
                "redis",
                "resilience",
                "di",
                "dependency",
                "pool",
                "connect",
                "retry",
                "circuit",
                "lifespan",
                "context",
                "setting",
                "exception",
            ],
        }

        for file_info in self.files:
            file_name = file_info["name"].lower()
            file_path = file_info["path"].lower()
            imports = " ".join(file_info["imports"]).lower()
            classes = " ".join(file_info["classes"]).lower()
            docstring = (file_info.get("docstring") or "").lower()

            # Combine all text for matching
            search_text = f"{file_name} {file_path} {imports} {classes} {docstring}"

            matched = False
            match_scores = {}

            for group, keys in keywords.items():
                score = sum(1 for key in keys if key in search_text)
                if score > 0:
                    match_scores[group] = score

            if match_scores:
                # Assign to highest scoring group
                best_group = max(match_scores, key=match_scores.get)
                groups[best_group].append(file_info["path"])
                matched = True

            if not matched:
                groups["unknown"].append(file_info["path"])

        # Print summary
        print("\n📊 Grouping Summary:")
        for group, files in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
            if files:
                print(f"  {group}: {len(files)} files")

        return groups

    def detect_duplications(self) -> list[dict]:
        """Detect potential duplications based on similar names"""
        print("\n🔍 Detecting potential duplications...")

        duplications = []
        file_names = defaultdict(list)

        # Group by base name (without common suffixes)
        for file_info in self.files:
            base_name = file_info["name"].replace(".py", "")

            # Remove common suffixes
            for suffix in [
                "_improved",
                "_enhanced",
                "_v2",
                "_new",
                "_legacy",
                "_refactored",
                "_simple",
                "_optimized",
                "_updated",
                "_pkg",
            ]:
                base_name = base_name.replace(suffix, "")

            file_names[base_name].append(file_info)

        # Find groups with multiple files
        for base_name, files in file_names.items():
            if len(files) > 1:
                duplications.append(
                    {
                        "base_name": base_name,
                        "count": len(files),
                        "files": [f["path"] for f in files],
                        "total_lines": sum(f["lines"] for f in files),
                    }
                )

        # Sort by count
        duplications.sort(key=lambda x: x["count"], reverse=True)

        print(f"✅ Found {len(duplications)} potential duplication groups")
        return duplications

    def get_existing_subdirs(self) -> dict[str, list[str]]:
        """Get files already organized in subdirectories"""
        print("\n📁 Analyzing existing subdirectories...")

        subdirs = defaultdict(list)

        for file_info in self.files:
            directory = file_info["directory"]
            subdirs[directory].append(file_info["path"])

        print("\n📂 Existing Structure:")
        for dir_name, files in sorted(subdirs.items()):
            print(f"  {dir_name}/: {len(files)} files")

        return dict(subdirs)

    def generate_report(self) -> str:
        """Generate complete analysis report in Markdown"""

        # Statistics
        total_files = len(self.files)
        total_lines = sum(f["lines"] for f in self.files)
        avg_lines = total_lines // total_files if total_files > 0 else 0

        report = f"""# Core Structure Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analyzed Path:** `{self.core_path}`

---

## 📊 Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | {total_files} |
| **Total Lines** | {total_lines:,} |
| **Average Lines/File** | {avg_lines} |
| **Existing Subdirectories** | {len(set(f["directory"] for f in self.files))} |

---

## 📁 Existing Directory Structure

"""

        subdirs = self.get_existing_subdirs()
        for dir_name, files in sorted(subdirs.items(), key=lambda x: len(x[1]), reverse=True):
            report += f"\n### `{dir_name}/` ({len(files)} files)\n\n"
            for file_path in sorted(files)[:10]:
                report += f"- `{Path(file_path).name}`\n"
            if len(files) > 10:
                report += f"- ... and {len(files) - 10} more\n"

        report += "\n---\n\n## 🎯 Suggested Thematic Grouping\n\n"

        grouping = self.suggest_grouping()
        for group, files in sorted(grouping.items(), key=lambda x: len(x[1]), reverse=True):
            if files:
                report += f"\n### {group.upper()} ({len(files)} files)\n\n"
                report += "| File | Lines | Classes |\n"
                report += "|------|-------|--------|\n"

                # Get file details
                file_details = []
                for file_path in files:
                    file_info = next((f for f in self.files if f["path"] == file_path), None)
                    if file_info:
                        file_details.append(file_info)

                # Sort by lines
                file_details.sort(key=lambda x: x["lines"], reverse=True)

                for file_info in file_details[:15]:
                    name = file_info["name"]
                    lines = file_info["lines"]
                    classes = ", ".join(file_info["classes"][:3])
                    if len(file_info["classes"]) > 3:
                        classes += "..."
                    report += f"| {name} | {lines} | {classes} |\n"

                if len(file_details) > 15:
                    report += f"\n*... and {len(file_details) - 15} more files*\n"

        # Duplications
        report += "\n---\n\n## 🔍 Potential Duplications\n\n"
        duplications = self.detect_duplications()

        if duplications:
            report += "Files with similar names that may need consolidation:\n\n"
            for dup in duplications[:15]:
                report += f"\n### `{dup['base_name']}` ({dup['count']} files, {dup['total_lines']} total lines)\n\n"
                for file_path in dup["files"]:
                    file_info = next((f for f in self.files if f["path"] == file_path), None)
                    if file_info:
                        report += f"- `{file_path}` ({file_info['lines']} lines)\n"
        else:
            report += "No significant duplications detected.\n"

        # Largest files
        report += "\n---\n\n## 📏 Largest Files (Top 25)\n\n"
        largest = sorted(self.files, key=lambda x: x["lines"], reverse=True)[:25]

        report += "| File | Lines | Classes | Functions |\n"
        report += "|------|-------|---------|----------|\n"

        for file_info in largest:
            name = file_info["name"]
            lines = file_info["lines"]
            classes = len(file_info["classes"])
            functions = len(file_info["functions"])
            report += f"| {name} | {lines} | {classes} | {functions} |\n"

        # Recommendations
        report += "\n---\n\n## 💡 Recommendations\n\n"

        root_files = [f for f in self.files if f["directory"] == "root"]
        report += f"1. **{len(root_files)} files in root directory** - Consider moving to thematic subdirectories\n"

        if duplications:
            report += (
                f"2. **{len(duplications)} potential duplications** - Review and consolidate\n"
            )

        large_files = [f for f in self.files if f["lines"] > 500]
        if large_files:
            report += f"3. **{len(large_files)} files > 500 lines** - Consider splitting\n"

        report += "\n---\n\n## 🚀 Migration Priority\n\n"
        report += """
Based on the analysis, recommended migration order:

1. **Platform** (config, DI, exceptions) - Foundation for other modules
2. **Observability** (health, metrics, logging) - Needed for monitoring migration
3. **Security** (auth, validation) - Independent module
4. **Retrieval** (cache, RAG, KG) - Large but cohesive
5. **Agents** (LLM, specialists) - Depends on retrieval
6. **TWS** (client, monitor) - Domain-specific, last

"""

        return report

    def save_json_data(self, output_file: str = "docs/core_analysis.json"):
        """Save complete data as JSON"""
        data = {
            "generated": datetime.now().isoformat(),
            "statistics": {
                "total_files": len(self.files),
                "total_lines": sum(f["lines"] for f in self.files),
                "avg_lines": sum(f["lines"] for f in self.files) // len(self.files)
                if self.files
                else 0,
            },
            "files": self.files,
            "grouping": self.suggest_grouping(),
            "duplications": self.detect_duplications(),
            "existing_structure": self.get_existing_subdirs(),
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n💾 JSON data saved to {output_file}")


def main():
    print("=" * 60)
    print("Core Structure Analyzer - Resync v5.4.2")
    print("=" * 60)

    analyzer = CoreAnalyzer()

    # 1. Scan files
    analyzer.scan_files()

    # 2. Generate report
    report = analyzer.generate_report()

    # 3. Save report
    os.makedirs("docs", exist_ok=True)
    output_file = "docs/CORE_ANALYSIS_REPORT.md"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to {output_file}")

    # 4. Save JSON data
    analyzer.save_json_data()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
