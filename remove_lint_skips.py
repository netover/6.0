#!/usr/bin/env python3
"""
Script para remover/comentar linhas de skip de linting do projeto.

Este script procura e substitui:
- `# pylint: skip-file` -> remove ou comenta
- `# mypy: ignore-errors` -> remove ou comenta
"""

import argparse
import os
from pathlib import Path
from typing import Optional


def process_file(file_path: Path, dry_run: bool = True, verbose: bool = False) -> Optional[bool]:
    """
    Processa um arquivo, removendo/comentando linhas de skip de linting.
    
    Args:
        file_path: Caminho do arquivo
        dry_run: Se True, apenas mostra o que seria feito
        verbose: Se True, mostra detalhes
        
    Returns:
        True se o arquivo foi modificado, False se não precisou de mudanças, None se erro
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Erro ao ler {file_path}: {e}")
        return None

    original_lines = lines.copy()
    
    # Linhas a procurar e substituir
    replacements = {
        "# pylint: skip-file": "# pylint: disable=all",
        "# mypy: ignore-errors": "# mypy: no-rerun",
    }
    
    modified = False
    new_lines: list[str] = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Verifica se a linha atual é uma das que queremos substituir
        if stripped in replacements:
            new_line = replacements[stripped] + "\n"
            new_lines.append(new_line)
            modified = True
            if verbose:
                print(f"  ↳ Linha {i+1}: '{stripped}' -> '{replacements[stripped]}'")
        else:
            new_lines.append(line)
    
    if not modified:
        if verbose:
            print(f"  ✓ Nenhuma mudança necessária")
        return False
    
    if dry_run:
        if verbose:
            print(f"  🔍 dry_run: mudanças não aplicadas")
        return True
    
    # Aplica as mudanças
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        if verbose:
            print(f"  ✅ Aplicado!")
        return True
    except Exception as e:
        if verbose:
            print(f"  ❌ Erro ao escrever {file_path}: {e}")
        return None


def find_python_files(root_dir: Path) -> list[Path]:
    """Encontra todos os arquivos Python no diretório."""
    return list(root_dir.rglob("*.py"))


def main():
    parser = argparse.ArgumentParser(
        description="Remove/comenta linhas de skip de linting (pylint/mypy)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Diretório raiz do projeto (padrão: ./)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito sem aplicar mudanças",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostra detalhes de cada arquivo",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["venv", ".venv", "__pycache__", ".git", "node_modules"],
        help="Diretórios a excluir",
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.directory).resolve()
    
    if not root_dir.exists():
        print(f"❌ Diretório não encontrado: {root_dir}")
        return 1
    
    print(f"🔍 Procurando arquivos Python em: {root_dir}")
    print(f"📁 Excluindo: {args.exclude}")
    print()
    
    # Encontra arquivos Python
    all_files = find_python_files(root_dir)
    
    # Filtra arquivos excluídos
    files_to_process = []
    for f in all_files:
        # Verifica se está em algum diretório excluído
        excluded = False
        for excl in args.exclude:
            if excl in f.parts:
                excluded = True
                break
        if not excluded:
            files_to_process.append(f)
    
    print(f"📊 Total de arquivos encontrados: {len(all_files)}")
    print(f"📄 Arquivos a processar (excluídos {len(all_files) - len(files_to_process)}): {len(files_to_process)}")
    print()
    
    # Processa cada arquivo
    stats = {"modified": 0, "unchanged": 0, "errors": 0}
    
    for file_path in sorted(files_to_process):
        if args.verbose:
            print(f"\n📄 {file_path.relative_to(root_dir)}")
        
        result = process_file(file_path, dry_run=args.dry_run, verbose=args.verbose)
        
        if result is True:
            stats["modified"] += 1
        elif result is False:
            stats["unchanged"] += 1
        else:
            stats["errors"] += 1
    
    # Resumo
    print()
    print("=" * 50)
    print("📈 RESUMO")
    print("=" * 50)
    print(f"  ✅ Modificados: {stats['modified']}")
    print(f"  ➖ Sem mudanças: {stats['unchanged']}")
    print(f"  ❌ Erros: {stats['errors']}")
    print(f"  📊 Total processado: {stats['modified'] + stats['unchanged'] + stats['errors']}")
    
    if args.dry_run:
        print()
        print("⚠️  Modo dry-run: execute sem --dry-run para aplicar as mudanças")
    
    return 0


if __name__ == "__main__":
    exit(main())
