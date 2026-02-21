"""
Core Domain Mypy Analyzer
Analisa relatórios do MyPy especificamente para o core do Resync.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

def _parse_mypy_line(line: str, prefix: str = "resync/core/") -> str | None:
    """
    Extrai o domínio a partir de uma linha de erro do mypy.
    Retorna None se a linha for inválida ou não pertencer ao core.
    """
    if "error:" not in line:
        return None
        
    parts = line.split(":")
    
    # HARDENING (P2): Validação rigorosa da estrutura do MyPy
    # Formato esperado: filepath:line:col: severity: message
    if len(parts) < 4:
        return None
        
    filepath = parts[0].replace('\\', '/')
    if not filepath.startswith(prefix):
        return None
        
    path_parts = filepath.split('/')
    if len(path_parts) >= 3:
        return f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}"
        
    return f"{prefix.rstrip('/')} (root files)"

def _count_errors_by_domain(report_file: Path, prefix: str = "resync/core/") -> Dict[str, int]:
    """Lê o relatório de forma segura e conta erros agrupados por domínio."""
    domain_counts: Dict[str, int] = defaultdict(int)
    
    try:
        content = report_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            domain = _parse_mypy_line(line, prefix)
            if domain:
                domain_counts[domain] += 1
    except OSError as e: # Captura falhas de permissão ou de leitura (I/O)
        print(f"ERROR: Falha ao ler o relatório {report_file.name}: {e}", file=sys.stderr)
        
    return domain_counts

def _format_plan(domain_counts: Dict[str, int]) -> str:
    """Gera o texto em Markdown estruturado por quantidade de erros."""
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    
    plan_lines = [
        "# Mypy Core Remediation Task",
        "",
        "## Sub-tasks for `resync/core/`",
        "",
    ]
    
    if not sorted_domains:
        plan_lines.append("Nenhum erro encontrado! O core está 100% complacente.")
    else:
        for domain, count in sorted_domains:
            plan_lines.append(f"- [ ] `mypy` para `{domain}/` ({count} erros)")
            
    return "\n".join(plan_lines)

def generate_core_plan() -> None:
    """Função principal de orquestração."""
    report_file = Path("mypy_core_report.txt")
    
    if not report_file.exists():
        print(f"Relatório '{report_file.name}' não encontrado.")
        return
        
    domain_counts = _count_errors_by_domain(report_file)
    if not domain_counts:
        print("Nenhum erro processado ou o ficheiro estava vazio/ilegível.")
        return
        
    plan = _format_plan(domain_counts)
    print(plan)

if __name__ == "__main__":
    generate_core_plan()
