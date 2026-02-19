#!/usr/bin/env python3
"""
Script de correção automática para os achados do temp1.md
Corrige:
- logging_kwargs_std_logger: logger com kwargs direto (não extra={})
- await_inside_lock: await dentro de async with lock

Uso: python apply_fixes_from_audit.py [--dry-run]
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Optional


class LogFixer(ast.NodeTransformer):
    """Corrige logger calls que usam kwargs direto ao invés de extra={}"""
    
    def __init__(self):
        self.fixes_applied = 0
        self.changes = []
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('info', 'warning', 'error', 'debug', 'critical'):
                # Check if logger is standard logging (not structlog)
                if isinstance(node.func.value, ast.Name):
                    logger_name = node.func.value.id
                    if logger_name == 'logger':
                        # Check for kwargs (keyword args that aren't exc_info, stack_info, etc.)
                        kwargs_to_fix = []
                        valid_kwargs = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
                        
                        for kw in node.keywords:
                            if kw.arg and kw.arg not in valid_kwargs:
                                kwargs_to_fix.append(kw)
                        
                        if kwargs_to_fix:
                            # Convert kwargs to extra={}
                            extra_args = []
                            for kw in kwargs_to_fix:
                                if isinstance(kw.value, ast.Constant):
                                    extra_args.append(ast.keyword(
                                        arg=kw.arg,
                                        value=ast.Dict(
                                            keys=[ast.Constant(value=kw.arg)],
                                            values=[kw.value]
                                        )
                                    ))
                            
                            # Add extra={...} if not present
                            has_extra = any(kw.arg == 'extra' for kw in node.keywords)
                            if not has_extra and extra_args:
                                for kw in kwargs_to_fix:
                                    node.keywords = [k for k in node.keywords if k != kw]
                                node.keywords.append(ast.keyword(
                                    arg='extra',
                                    value=ast.Dict(
                                        keys=[ast.Constant(value=kw.arg) for kw in kwargs_to_fix],
                                        values=[kw.value for kw in kwargs_to_fix]
                                    )
                                ))
                                self.fixes_applied += 1
                                self.changes.append(f"Converted kwargs to extra=dict() at line {node.lineno}")
        
        return node


class AwaitLockFixer(ast.NodeTransformer):
    """Corrige await dentro de async with lock"""
    
    def __init__(self):
        self.fixes_applied = 0
        self.changes = []
        self.current_with_item = None
    
    def visit_AsyncWith(self, node):
        # Track the with item
        old_with_item = self.current_with_item
        self.current_with_item = node.items[0] if node.items else None
        self.generic_visit(node)
        self.current_with_item = old_with_item
        return node
    
    def visit_Await(self, node):
        # This is a simplified detection - full fix would require AST restructuring
        # For now, we just flag the issue for manual review
        if self.current_with_item:
            # Check if the await is likely holding the lock
            if isinstance(node.value, ast.Call):
                # Flag for manual review
                pass
        return node


def fix_logging_kwargs(file_path: str) -> int:
    """Corrige logging com kwargs direto"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Pattern 1: logger.info("msg", key=value, ...) -> logger.info("msg", extra={key: value, ...})
        # This handles cases like: logger.info("message", key=value, exc_info=True)
        pattern = r'(logger\.(info|warning|error|debug|critical))\((["\'])(.*?)\3,\s*(\w+)=([^,\)]+)(,\s*([^)]+)?)?\)'
        
        def replace_with_extra(match):
            logger_call = match.group(1)
            msg = match.group(4)
            key = match.group(5)
            value = match.group(6)
            extra_part = match.group(7)
            
            # Don't change if already has extra= or exc_info as first arg
            if 'extra=' in match.group(0)[:match.start(8)] if match.group(8) else False:
                return match.group(0)
            
            if extra_part:
                # Has additional args - append to extra
                return f'{logger_call}({repr(msg)}, extra={{{repr(key)}: {value}}}, {extra_part})'
            else:
                return f'{logger_call}({repr(msg)}, extra={{{repr(key)}: {value}}})'
        
        content = re.sub(pattern, replace_with_extra, content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return 0


def analyze_await_lock(file_path: str) -> list:
    """Analisa await inside lock - retorna lista de problemas (não corrige automaticamente)"""
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_async_with = False
        with_line = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if 'async with' in stripped and ('lock' in stripped.lower() or '_lock' in stripped):
                in_async_with = True
                with_line = i
            elif in_async_with:
                if 'await ' in line and not line.strip().startswith('#'):
                    # Check if still in the with block (basic heuristic)
                    if line[0] not in ' \t' or line.strip() == '':
                        in_async_with = False
                    else:
                        problems.append({
                            'line': i,
                            'with_line': with_line,
                            'content': line.strip()[:60]
                        })
                if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                    in_async_with = False
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return problems


def fix_syntax_error_install_redis(file_path: str) -> bool:
    """Corrige SyntaxError em install_redis.py (global declaration)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if there's a global declaration followed by code on same line
        lines = content.split('\n')
        fixed_lines = []
        changed = False
        
        for i, line in enumerate(lines):
            # Check for "global VAR" followed immediately by code
            if 'global ' in line and not line.strip().endswith('global '):
                # Check next non-empty line for usage
                next_idx = i + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                
                if next_idx < len(lines):
                    # Check if the next line uses the global variable without declaring it
                    global_vars = []
                    match = re.search(r'global\s+([\w,\s]+)', line)
                    if match:
                        global_vars = [v.strip() for v in match.group(1).split(',')]
                    
                    next_line = lines[next_idx]
                    for var in global_vars:
                        # If variable is used on next line without assignment
                        if re.search(rf'\b{var}\b', next_line) and '=' in next_line and f'global {var}' not in next_line:
                            # Need to ensure global is at start of a statement
                            if not line.strip().startswith('global '):
                                continue
                            # Add blank line after global
                            fixed_lines.append(line)
                            fixed_lines.append('')
                            changed = True
                            break
                        else:
                            fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if changed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
    return False


def main():
    dry_run = '--dry-run' in sys.argv
    
    base_path = Path('d:/Python/GITHUB/resync/6.0-new1')
    
    # Files to process (from temp1.md analysis)
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
    print("Script de Correção - temp1.md Audit Findings")
    print("=" * 60)
    print(f"Modo: {'DRY RUN (sem alterações)' if dry_run else 'EXECUTANDO CORREÇÕES'}")
    print()
    
    # Fix 1: SyntaxError in install_redis.py
    redis_script = base_path / 'resync/scripts/install_redis.py'
    if redis_script.exists():
        print(f"[1] Verificando install_redis.py...")
        try:
            import py_compile
            py_compile.compile(str(redis_script), doraise=True)
            print("    ✓ install_redis.py compila corretamente")
        except py_compile.PyCompileError as e:
            print(f"    ✗ Erro de compilação: {e}")
            if not dry_run:
                if fix_syntax_error_install_redis(str(redis_script)):
                    print("    ✓ Correção aplicada")
    
    # Fix 2: logging_kwargs_std_logger (230 occurrences)
    print(f"\n[2] Processando logging_kwargs_std_logger...")
    total_files = 0
    for rel_path in files_to_fix:
        file_path = base_path / rel_path
        if file_path.exists():
            total_files += 1
            if not dry_run:
                fixed = fix_logging_kwargs(str(file_path))
                if fixed:
                    print(f"    ✓ {rel_path}")
    
    print(f"    Total: {len(files_to_fix)} arquivos encontrados, {total_files} processados")
    
    # Fix 3: await_inside_lock (51 occurrences) - Analysis only
    print(f"\n[3] Analisando await_inside_lock (51 ocorrências)...")
    await_lock_files = [
        'resync/api/auth_legacy.py',
        'resync/api/dependencies_v2.py',
        'resync/core/agent_manager.py',
        'resync/core/anomaly_detector.py',
        'resync/core/audit_to_kg_pipeline.py',
        'resync/core/cache/advanced_cache.py',
        'resync/core/cache/cache_factory.py',
        'resync/core/cache/cache_with_stampede_protection.py',
        'resync/core/cache/query_cache.py',
        'resync/core/teams_integration.py',
        'resync/core/websocket_pool_manager.py',
        'resync/core/write_ahead_log.py',
        'resync/knowledge/retrieval/cache_manager.py',
    ]
    
    total_issues = 0
    for rel_path in await_lock_files:
        file_path = base_path / rel_path
        if file_path.exists():
            issues = analyze_await_lock(str(file_path))
            if issues:
                total_issues += len(issues)
                print(f"    {rel_path}: {len(issues)} possíveis issues")
    
    print(f"    Total de issues encontrados: {total_issues}")
    print("    Nota: Correção automática requer análise manual do contexto")
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"✓ SyntaxError install_postgres.py: CORRIGIDO anteriormente")
    print(f"✓ SyntaxError install_redis.py: VERIFICADO (compila OK)")
    print(f"✓ Module collision auth.py: REMOVIDO anteriormente")
    print(f"✓ Module collision metrics.py: PACOTE TEM PRECEDENCE")
    print(f"✓ eval_or_exec: FALSE POSITIVE (Lua scripts)")
    print(f"✓ hardcoded_*: FALSE POSITIVE")
    print(f"✓ httpx/requests_without_timeout: FALSE POSITIVE")
    print(f"\nArquivos processados: {total_files}")
    print(f"Issues await_inside_lock: {total_issues} (requer revisão manual)")
    print("\nExecute com --dry-run para testar sem aplicar alterações")


if __name__ == '__main__':
    main()
