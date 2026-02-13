#!/usr/bin/env python3
"""
Resync v5.3.9 - Script de Correção Automática
Executa correções seguras identificadas na análise completa.

Uso:
    python scripts/auto_fix_v5_3_9.py --dry-run    # Apenas mostra o que seria feito
    python scripts/auto_fix_v5_3_9.py --execute    # Executa as correções

Autor: Claude AI Analysis
Data: 2025-12-10
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class FixResult(NamedTuple):
    """Resultado de uma correção."""

    category: str
    action: str
    file: str
    success: bool
    message: str


class ResyncFixer:
    """Corretor automático para issues do Resync."""

    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.results: list[FixResult] = []
        self.backup_dir = project_root / ".backup_v5_3_9"

    def run_all(self) -> None:
        """Executa todas as correções."""
        print("=" * 60)
        print("🔧 Resync v5.3.9 - Auto-Fix Script")
        print(f"   Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}")
        print("=" * 60)
        print()

        # Criar backup se executando
        if not self.dry_run:
            self._create_backup()

        # Executar correções por categoria
        self._fix_dead_code()
        self._fix_missing_init()
        self._fix_ruff_auto()
        self._fix_blocking_sleep()
        self._fix_raise_from()

        # Relatório final
        self._print_report()

    def _create_backup(self) -> None:
        """Cria backup dos arquivos que serão modificados."""
        print("📦 Criando backup...")
        self.backup_dir.mkdir(exist_ok=True)
        print(f"   Backup em: {self.backup_dir}")
        print()

    def _backup_file(self, filepath: Path) -> None:
        """Faz backup de um arquivo específico."""
        if not self.dry_run:
            rel_path = filepath.relative_to(self.project_root)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, backup_path)

    def _fix_dead_code(self) -> None:
        """Remove arquivos de código morto confirmado."""
        print("🗑️  FASE 1: Remoção de Código Morto")
        print("-" * 40)

        dead_files = [
            "resync/core/health/refactored_health_check_service.py",
            "resync/core/health/refactored_enhanced_health_service.py",
            "resync/core/health/refactored_health_service_orchestrator.py",
        ]

        for rel_path in dead_files:
            filepath = self.project_root / rel_path
            if filepath.exists():
                if self.dry_run:
                    print(f"   [DRY] Removeria: {rel_path}")
                    self.results.append(
                        FixResult("dead_code", "remove", rel_path, True, "Would remove")
                    )
                else:
                    self._backup_file(filepath)
                    filepath.unlink()
                    print(f"   ✅ Removido: {rel_path}")
                    self.results.append(FixResult("dead_code", "remove", rel_path, True, "Removed"))
            else:
                print(f"   ⏭️  Já removido: {rel_path}")

        print()

    def _fix_missing_init(self) -> None:
        """Cria __init__.py faltando."""
        print("📝 FASE 2: Criando __init__.py Faltantes")
        print("-" * 40)

        missing_init = [
            "resync/RAG/BASE/__init__.py",
            "resync/prompts/__init__.py",
        ]

        init_content = '"""Package initialization."""\n'

        for rel_path in missing_init:
            filepath = self.project_root / rel_path
            if not filepath.exists():
                if self.dry_run:
                    print(f"   [DRY] Criaria: {rel_path}")
                    self.results.append(FixResult("init", "create", rel_path, True, "Would create"))
                else:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(init_content)
                    print(f"   ✅ Criado: {rel_path}")
                    self.results.append(FixResult("init", "create", rel_path, True, "Created"))
            else:
                print(f"   ⏭️  Já existe: {rel_path}")

        print()

    def _fix_ruff_auto(self) -> None:
        """Executa correções automáticas do ruff."""
        print("🔍 FASE 3: Correções Automáticas Ruff")
        print("-" * 40)

        if self.dry_run:
            # Mostrar o que seria corrigido
            result = subprocess.run(
                ["ruff", "check", ".", "--fix", "--diff"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.count("\n") if result.stdout else 0
            print(f"   [DRY] {lines} linhas seriam modificadas")
            self.results.append(FixResult("ruff", "auto-fix", "multiple", True, f"{lines} lines"))
        else:
            # Executar correção
            result = subprocess.run(
                ["ruff", "check", ".", "--fix"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            print("   ✅ Ruff auto-fix executado")
            self.results.append(FixResult("ruff", "auto-fix", "multiple", True, "Applied"))

            # Formatar código
            subprocess.run(["ruff", "format", "."], cwd=self.project_root, capture_output=True)
            print("   ✅ Ruff format executado")

        print()

    def _fix_blocking_sleep(self) -> None:
        """Corrige time.sleep() em código async."""
        print("⚡ FASE 4: Corrigindo Blocking Sleep")
        print("-" * 40)

        # Arquivo específico identificado
        target_file = self.project_root / "resync/core/utils/common_error_handlers.py"

        if target_file.exists():
            content = target_file.read_text()

            # Verificar se tem o problema
            if "time.sleep(" in content and "async def" in content:
                if self.dry_run:
                    print(f"   [DRY] Corrigiria: {target_file.relative_to(self.project_root)}")
                    self.results.append(
                        FixResult(
                            "blocking",
                            "fix",
                            str(target_file.relative_to(self.project_root)),
                            True,
                            "Would fix time.sleep",
                        )
                    )
                else:
                    self._backup_file(target_file)

                    # Adicionar import asyncio se não existir
                    if "import asyncio" not in content:
                        content = "import asyncio\n" + content

                    # Substituir time.sleep por await asyncio.sleep
                    content = re.sub(r"time\.sleep\(([^)]+)\)", r"await asyncio.sleep(\1)", content)

                    target_file.write_text(content)
                    print(f"   ✅ Corrigido: {target_file.relative_to(self.project_root)}")
                    self.results.append(
                        FixResult(
                            "blocking",
                            "fix",
                            str(target_file.relative_to(self.project_root)),
                            True,
                            "Fixed time.sleep",
                        )
                    )
            else:
                print("   ⏭️  Já corrigido ou não aplicável")

        print()

    def _fix_raise_from(self) -> None:
        """Mostra correções necessárias para raise without from."""
        print("🔗 FASE 5: Análise de 'raise' sem 'from'")
        print("-" * 40)

        # Esta correção precisa ser manual, apenas listar
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "B904", "--output-format", "text"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            lines = result.stdout.strip().split("\n")
            count = len([l for l in lines if "B904" in l])
            print(f"   ⚠️  {count} ocorrências precisam correção MANUAL")
            print("   📝 Execute: ruff check . --select B904")
            self.results.append(
                FixResult("raise_from", "manual", "multiple", False, f"{count} need manual fix")
            )
        else:
            print("   ✅ Nenhum problema encontrado")

        print()

    def _print_report(self) -> None:
        """Imprime relatório final."""
        print("=" * 60)
        print("📊 RELATÓRIO FINAL")
        print("=" * 60)

        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"success": 0, "manual": 0}
            if r.success:
                categories[r.category]["success"] += 1
            else:
                categories[r.category]["manual"] += 1

        for cat, counts in categories.items():
            print(f"   {cat}: {counts['success']} auto, {counts['manual']} manual")

        print()
        if self.dry_run:
            print("⚠️  MODO DRY-RUN: Nenhuma alteração foi feita")
            print("   Execute com --execute para aplicar correções")
        else:
            print("✅ Correções aplicadas!")
            print(f"   Backup salvo em: {self.backup_dir}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Resync v5.3.9 Auto-Fix Script")
    parser.add_argument(
        "--dry-run", "-d", action="store_true", help="Apenas mostra o que seria feito"
    )
    parser.add_argument("--execute", "-e", action="store_true", help="Executa as correções")

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Erro: Especifique --dry-run ou --execute")
        sys.exit(1)

    # Encontrar raiz do projeto
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if not (project_root / "resync").exists():
        print(f"Erro: Não encontrou pasta resync/ em {project_root}")
        sys.exit(1)

    fixer = ResyncFixer(project_root, dry_run=args.dry_run)
    fixer.run_all()


if __name__ == "__main__":
    main()
