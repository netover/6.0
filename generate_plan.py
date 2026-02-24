"""
Global Mypy Plan Generator
Gera o plano geral de remediação para todo o projeto Resync.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def _parse_global_mypy_line(line: str) -> Tuple[str, str] | None:
    """
    Extrai o domínio global e o caminho do ficheiro de uma linha de erro.
    Retorna o tuplo (domain, filepath) ou None.
    """
    if "error:" not in line:
        return None

    parts = line.split(":")

    # HARDENING (P2): Garante que existem dados suficientes na divisão do array
    if len(parts) < 4:
        return None

    filepath = parts[0].replace("\\", "/")
    if not filepath.startswith("resync/"):
        return None

    path_parts = filepath.split("/")
    if len(path_parts) >= 2:
        domain = f"{path_parts[0]}/{path_parts[1]}"
        return (domain, filepath)

    return None


def _aggregate_errors(report_file: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Agrega os erros globalmente lidando com eventuais falhas de I/O."""
    domain_counts: Dict[str, int] = defaultdict(int)
    file_counts: Dict[str, int] = defaultdict(int)

    try:
        content = report_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            result = _parse_global_mypy_line(line)
            if result:
                domain, filepath = result
                domain_counts[domain] += 1
                file_counts[filepath] += 1
    except OSError as e:
        print(
            f"ERROR: Falha ao ler o relatório {report_file.name}: {e}", file=sys.stderr
        )

    return domain_counts, file_counts


def _write_plan(domain_counts: Dict[str, int], output_file: Path) -> bool:
    """Escreve o plano no ficheiro Markdown lidando com erros de I/O de escrita."""
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    plan_lines = [
        "# MYPY REMEDIATION PLAN",
        "",
        "This is the strict tracking file for achieving 100% mypy compliance.",
        "",
        "## Domain Groups",
        "",
    ]

    if not sorted_domains:
        plan_lines.append("Nenhum erro encontrado! O projeto está 100% complacente.")
    else:
        for i, (domain, count) in enumerate(sorted_domains, 1):
            plan_lines.append(f"- [ ] STEP {i}: Fix `{domain}/` ({count} erros)")

    try:
        # write_text abre, escreve e fecha o ficheiro em segurança, numa única linha
        output_file.write_text("\n".join(plan_lines), encoding="utf-8")
        return True
    except OSError as e:
        print(
            f"ERROR: Falha ao guardar o plano {output_file.name}: {e}", file=sys.stderr
        )
        return False


def generate_plan() -> None:
    """Função principal de orquestração do gerador global."""
    report_file = Path("mypy_global_report.txt")
    output_file = Path("MYPY_REMEDIATION_PLAN.md")

    if not report_file.exists():
        print(f"Relatório '{report_file.name}' não encontrado.")
        return

    domain_counts, _ = _aggregate_errors(report_file)

    # Confirmação visual de sucesso garantida apenas se o I/O for concluído sem erros
    if _write_plan(domain_counts, output_file):
        print(
            f"✅ Plano gerado com sucesso em '{output_file.absolute()}' ({len(domain_counts)} steps)."
        )


if __name__ == "__main__":
    generate_plan()
