#!/usr/bin/env python3
"""
Script de correção usando AST - mais seguro que regex
Corrige logging_kwargs_std_logger: logger.com("msg", key=value) -> logger.com("msg", extra={"key": value})
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Verify Python version is 3.9 or higher
if sys.version_info < (3, 9):
    raise RuntimeError("This script requires Python 3.9 or higher")


class LoggingFixer(ast.NodeTransformer):
    """Corrige logger calls que usam kwargs direto ao invés de extra={}"""
    
    def __init__(self):
        self.fixes: List[Tuple[int, str, str]] = []
        super().__init__()
    
    def visit_Call(self, node: ast.Call) -> ast.Call:
        # Check if it's a logger call
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return node
        
        if node.func.attr not in ('info', 'warning', 'error', 'debug', 'critical', 'exception'):
            self.generic_visit(node)
            return node
        
        # Check if it's the standard 'logger' name
        if not isinstance(node.func.value, ast.Name):
            self.generic_visit(node)
            return node
        
        if node.func.value.id != 'logger':
            self.generic_visit(node)
            return node
        
        # Valid kwargs that should NOT be moved to extra
        valid_kwargs = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
        
        # Find kwargs that need to be moved to extra
        kwargs_to_move = []
        remaining_kwargs = []
        
        for kw in node.keywords:
            if kw.arg and kw.arg not in valid_kwargs:
                kwargs_to_move.append(kw)
            else:
                remaining_kwargs.append(kw)
        
        if not kwargs_to_move:
            self.generic_visit(node)
            return node
        
        # Check if there's already an 'extra' keyword
        has_extra = any(kw.arg == 'extra' for kw in remaining_kwargs)
        
        if has_extra:
            # Already has extra, skip
            self.generic_visit(node)
            return node
        
        # Build the extra dict
        extra_keys = [ast.Constant(value=kw.arg) for kw in kwargs_to_move]
        extra_values = [kw.value for kw in kwargs_to_move]
        
        extra_dict = ast.Dict(keys=extra_keys, values=extra_values)
        
        # Replace keywords with just 'extra'
        node.keywords = remaining_kwargs + [ast.keyword(arg='extra', value=extra_dict)]
        
        # Record the fix
        msg = "unknown"
        if node.args and isinstance(node.args[0], ast.Constant):
            msg = str(node.args[0].value)[:30]
        
        self.fixes.append((node.lineno, msg, f"Moved {len(kwargs_to_move)} kwargs to extra"))
        
        self.generic_visit(node)
        return node


def get_node_text(source: str, node: ast.AST) -> str:
    """Get the original text for an AST node from source."""
    return ast.get_source_segment(source, node) or ""


def generate_replacement(node: ast.Call, source: str) -> str:
    """Generate replacement text for a modified logger call node."""
    # Get the function name (e.g., logger.info)
    func_name = node.func.attr
    
    # Get positional arguments
    args_texts = [get_node_text(source, arg) for arg in node.args]
    args_str = ", ".join(args_texts)
    
    # Process keywords - find the extra dict if present
    keywords_parts = []
    extra_dict = None
    
    for kw in node.keywords:
        if kw.arg == 'extra':
            # Get the original extra dict text
            extra_dict = get_node_text(source, kw.value)
        else:
            kw_text = get_node_text(source, kw)
            if kw_text:
                keywords_parts.append(kw_text)
    
    # Build the replacement
    result = f"{func_name}({args_str}"
    
    if keywords_parts:
        result += ", " + ", ".join(keywords_parts)
    
    if extra_dict:
        result += f", extra={extra_dict}"
    
    result += ")"
    
    return f"logger.{result}"


def find_node_at_line(tree: ast.AST, line: int, col_offset: int = 0) -> ast.AST | None:
    """Find a node at the given line and column offset."""
    for node in ast.walk(tree):
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            if node.lineno == line and node.col_offset >= col_offset:
                return node
    return None


def fix_file_ast(file_path: str, dry_run: bool = True) -> List[Tuple[int, str, str]]:
    """Corrige um arquivo usando AST"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=file_path)
        
        fixer = LoggingFixer()
        new_tree = fixer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        if not fixer.fixes:
            return []
        
        if dry_run:
            return fixer.fixes
        
        # Apply fixes using targeted text replacements to preserve formatting
        # Collect all replacements: (start_offset, end_offset, new_text)
        replacements: List[Tuple[int, int, str]] = []
        
        # Walk both trees to find changes
        old_nodes = {id(n): n for n in ast.walk(tree)}
        new_nodes = {id(n): n for n in ast.walk(new_tree)}
        
        # Find Call nodes that were modified
        for old_node_id, old_node in old_nodes.items():
            if isinstance(old_node, ast.Call):
                new_node = new_nodes.get(old_node_id)
                if new_node is not None and id(new_node) != id(old_node):
                    # This node was modified
                    old_text = get_node_text(source, old_node)
                    if old_text:
                        # Get exact position from old node
                        start = old_node.col_offset
                        end = start + len(old_text)
                        
                        # Find the line start
                        lines = source.split('\n')
                        line_start = sum(len(l) + 1 for l in lines[:old_node.lineno - 1])
                        actual_start = line_start + start
                        actual_end = line_start + end
                        
                        # Generate new text preserving as much formatting as possible
                        new_text = generate_replacement_text(new_node, source)
                        replacements.append((actual_start, actual_end, new_text))
        
        # Sort replacements by position (descending) to apply from end to start
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply replacements
        source_chars = list(source)
        for start, end, new_text in replacements:
            source_chars[start:end] = list(new_text)
        
        new_source = ''.join(source_chars)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_source)
        
        return fixer.fixes
        
    except SyntaxError as e:
        print(f"  SyntaxError in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"  Error in {file_path}: {e}")
        return []


def generate_replacement_text(node: ast.Call, source: str) -> str:
    """Generate replacement text for a modified logger call - preserves formatting."""
    # Get the function attribute
    if not isinstance(node.func, ast.Attribute):
        return get_node_text(source, node) or ""
    
    func_name = node.func.attr
    
    # Get positional arguments with original formatting preserved
    args_parts = []
    for arg in node.args:
        arg_text = get_node_text(source, arg)
        if arg_text:
            args_parts.append(arg_text)
    
    # Process keywords
    keyword_parts = []
    extra_text = None
    
    for kw in node.keywords:
        kw_text = get_node_text(source, kw)
        if kw_text:
            if kw.arg == 'extra':
                extra_text = kw_text
            else:
                keyword_parts.append(kw_text)
    
    # Build the call with minimal changes
    parts = args_parts + keyword_parts
    if extra_text:
        parts.append(extra_text)
    
    return f"logger.{func_name}({', '.join(parts)})"


def main():
    import os
    dry_run = '--dry-run' in sys.argv
    # Use relative path from script location
    script_dir = Path(__file__).parent.resolve()
    base_path = script_dir
    
    files_to_fix = [
        'resync/api/auth_legacy.py',
        'resync/api/chat.py',
        'resync/api/core/security.py',
        'resync/api/dependencies_v2.py',
        'resync/api/enhanced_endpoints.py',
        'resync/api/middleware/cors_config.py',
        'resync/api/middleware/cors_monitoring.py',
        'resync/api/middleware/database_security_middleware.py',
        'resync/api/middleware/error_handler.py',
        'resync/api/routes/admin/config.py',
        'resync/api/routes/admin/main.py',
        'resync/api/routes/admin/prompts.py',
        'resync/api/routes/admin/semantic_cache.py',
        'resync/api/routes/admin/v2.py',
        'resync/api/routes/cache.py',
        'resync/api/routes/core/health.py',
        'resync/api/routes/monitoring/metrics_dashboard.py',
        'resync/api/routes/rag/query.py',
        'resync/api/routes/rag/upload.py',
        'resync/api/services/rag_config.py',
        'resync/api/utils/helpers.py',
        'resync/config/security.py',
        'resync/core/__init__.py',
        'resync/core/audit_db.py',
        'resync/core/audit_lock.py',
        'resync/core/cache/async_cache.py',
        'resync/core/cache/cache_hierarchy.py',
        'resync/core/cache/memory_manager.py',
        'resync/core/task_tracker.py',
        'resync/core/utils/async_bridge.py',
        'resync/core/utils/common_error_handlers.py',
        'resync/core/utils/json_commands.py',
        'resync/knowledge/ingestion/chunking_eval.py',
        'resync/knowledge/ingestion/document_converter.py',
        'resync/knowledge/ingestion/embedding_service.py',
        'resync/knowledge/ingestion/embeddings.py',
        'resync/knowledge/ingestion/ingest.py',
        'resync/knowledge/ingestion/pipeline.py',
        'resync/knowledge/kg_extraction/extractor.py',
        'resync/knowledge/kg_store/store.py',
        'resync/knowledge/retrieval/hybrid_retriever.py',
        'resync/knowledge/store/pgvector_store.py',
        'resync/services/advanced_graph_queries.py',
        'resync/services/llm_service.py',
    ]
    
    print("=" * 60)
    print("AST-based Logger Fixer")
    print("=" * 60)
    print(f"Modo: {'DRY RUN' if dry_run else 'EXECUTANDO'}")
    print()
    
    total_fixes = 0
    files_with_fixes = 0
    
    for rel_path in files_to_fix:
        file_path = base_path / rel_path
        if not file_path.exists():
            continue
        
        fixes = fix_file_ast(str(file_path), dry_run=dry_run)
        
        if fixes:
            files_with_fixes += 1
            total_fixes += len(fixes)
            print(f"  {rel_path}: {len(fixes)} correções")
            for line, msg, action in fixes[:3]:
                print(f"    Linha {line}: {action}")
            if len(fixes) > 3:
                print(f"    ... e mais {len(fixes) - 3}")
    
    print()
    print("=" * 60)
    print(f"Total: {total_fixes} correções em {files_with_fixes} arquivos")
    print("=" * 60)
    
    # Verify compilation
    if not dry_run:
        print("\nVerificando compilação...")
        import py_compile
        errors = []
        for rel_path in files_to_fix:
            file_path = base_path / rel_path
            if file_path.exists():
                try:
                    py_compile.compile(str(file_path), doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append((rel_path, str(e)))
        
        if errors:
            print(f"Erros de compilação em {len(errors)} arquivos:")
            for path, err in errors:
                print(f"  {path}: {err[:100]}")
        else:
            print("✓ Todos os arquivos compilam corretamente")


if __name__ == '__main__':
    main()
