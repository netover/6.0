"""
Script automatizado para correção de erros E701 (multiple statements on one line).

Este script:
1. Lê o output JSON do Ruff
2. Identifica linhas com múltiplas declarações
3. Aplica correções automáticas preservando a semântica
4. Valida as correções com re-execução do Ruff

Uso:
    python fix_ruff_errors.py
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List

# Setup basic logging since structlog might not be installed in the environment yet
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_ruff_check() -> List[Dict]:
    """Executa Ruff e retorna erros em formato JSON."""
    result = subprocess.run(
        ['ruff', 'check', 'resync/', '--output-format=json'],
        capture_output=True,
        text=True
    )
    
    try:
        if not result.stdout:
            return []
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.error("Failed to parse Ruff output")
        return []


def group_errors_by_file(errors: List[Dict]) -> Dict[str, List[Dict]]:
    """Agrupa erros por arquivo."""
    files_to_fix = {}
    
    for error in errors:
        # E701: Multiple statements on one line (colon)
        # E702: Multiple statements on one line (semicolon)
        # E703: Statement ends with a semicolon
        if error['code'] in ['E701', 'E702', 'E703']:
            filepath = error['filename']
            if filepath not in files_to_fix:
                files_to_fix[filepath] = []
            files_to_fix[filepath].append(error)
    
    return files_to_fix


def fix_file(filepath: str, errors: List[Dict]) -> None:
    """Aplica correções em um arquivo específico."""
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"File not found: {filepath}")
        return
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        logger.error(f"Failed to read {filepath} with utf-8")
        return
    
    # Processar erros de trás para frente para preservar índices
    for error in sorted(errors, key=lambda e: e['location']['row'], reverse=True):
        line_num = error['location']['row'] - 1
        if line_num >= len(lines):
            continue
            
        line = lines[line_num]
        col = error['location']['column'] - 1
        
        if error['code'] == 'E701':
            fixed_line = fix_e701(line, col)
            lines[line_num] = fixed_line
        
        elif error['code'] == 'E702':
            fixed_lines = fix_e702(line)
            lines[line_num:line_num+1] = fixed_lines
        
        elif error['code'] == 'E703':
            fixed_line = fix_e703(line)
            lines[line_num] = fixed_line
    
    # Escrever arquivo corrigido
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    logger.info(f"Fixed {len(errors)} errors in {filepath}")


def fix_e701(line: str, col: int) -> str:
    """
    Corrige E701: Multiple statements on one line (colon).
    """
    before_colon = line[:col+1]  # Inclui o ':'
    after_colon = line[col+1:].lstrip()
    
    base_indent = len(line) - len(line.lstrip())
    indent = ' ' * (base_indent + 4)
    
    return f"{before_colon}\n{indent}{after_colon}"


def fix_e702(line: str) -> List[str]:
    """
    Corrige E702: Multiple statements on one line (semicolon).
    """
    statements = line.split(';')
    base_indent = len(line) - len(line.lstrip())
    indent = ' ' * base_indent
    
    fixed_lines = []
    for i, stmt in enumerate(statements):
        stmt = stmt.strip()
        if stmt:
            fixed_lines.append(f"{indent}{stmt}\n")
    
    return fixed_lines


def fix_e703(line: str) -> str:
    """
    Corrige E703: Statement ends with a semicolon.
    """
    return line.rstrip('; \t\n') + '\n'


def main():
    """Executa pipeline de correção."""
    logger.info("Scanning for Ruff errors...")
    
    errors = run_ruff_check()
    if not errors:
        logger.info("No errors found!")
        return
    
    files_to_fix = group_errors_by_file(errors)
    logger.info(f"Found {len(files_to_fix)} files with syntax errors")
    
    for filepath, file_errors in files_to_fix.items():
        logger.info(f"Fixing {filepath} ({len(file_errors)} errors)...")
        fix_file(filepath, file_errors)
    
    logger.info("Validating fixes...")
    remaining_errors = run_ruff_check()
    syntax_errors = [e for e in remaining_errors if e['code'] in ['E701', 'E702', 'E703']]
    
    if not syntax_errors:
        logger.info("All syntax errors fixed!")
    else:
        logger.warning(f"{len(syntax_errors)} errors remaining (manual review needed)")


if __name__ == '__main__':
    main()
