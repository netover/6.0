#!/usr/bin/env python3
"""
RESYNC - Database Backup Utility

Este script fornece funções de backup e restore para o banco SQLite do Resync.
Usa a API de backup do SQLite para garantir consistência mesmo com conexões ativas.

Uso:
    python backup_db.py backup              # Criar backup
    python backup_db.py backup --name test  # Backup com nome personalizado
    python backup_db.py restore <file>      # Restaurar de backup
    python backup_db.py list                # Listar backups
    python backup_db.py verify <file>       # Verificar integridade
    python backup_db.py cleanup --keep 5    # Manter apenas N backups
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


# Configurações padrão
DEFAULT_DB_PATH = "/opt/resync/shared/db/resync.db"
DEFAULT_BACKUP_DIR = "/opt/resync/backups"


class DatabaseBackup:
    """Gerencia backups do banco de dados SQLite."""

    def __init__(self, db_path: str = None, backup_dir: str = None):
        self.db_path = Path(db_path or os.environ.get("RESYNC_DB_PATH", DEFAULT_DB_PATH))
        self.backup_dir = Path(backup_dir or os.environ.get("RESYNC_BACKUP_DIR", DEFAULT_BACKUP_DIR))
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup(self, name: str = None, progress_callback=None) -> Path:
        """
        Cria backup do banco de dados usando a API do SQLite.
        
        Args:
            name: Nome personalizado para o backup (opcional)
            progress_callback: Função callback para progresso
            
        Returns:
            Path do arquivo de backup criado
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Gerar nome do backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            backup_name = f"resync_{name}_{timestamp}.db"
        else:
            backup_name = f"resync_backup_{timestamp}.db"
        
        backup_path = self.backup_dir / backup_name

        # Usar SQLite Online Backup API
        source = sqlite3.connect(str(self.db_path))
        dest = sqlite3.connect(str(backup_path))

        try:
            with dest:
                # backup() copia em páginas, permitindo progresso
                source.backup(dest, pages=100, progress=progress_callback)
            
            print(f"✓ Backup created: {backup_path}")
            
            # Verificar integridade do backup
            if not self.verify(backup_path):
                backup_path.unlink()
                raise RuntimeError("Backup failed integrity check")
            
            # Criar arquivo de metadados
            self._save_metadata(backup_path)
            
            return backup_path
            
        finally:
            source.close()
            dest.close()

    def verify(self, backup_path: Path) -> bool:
        """
        Verifica integridade de um backup.
        
        Args:
            backup_path: Caminho do arquivo de backup
            
        Returns:
            True se o backup está íntegro
        """
        try:
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()
            
            # PRAGMA integrity_check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            
            conn.close()
            
            if result == "ok":
                print(f"✓ Backup integrity verified: {backup_path.name}")
                return True
            else:
                print(f"✗ Backup integrity FAILED: {result}")
                return False
                
        except Exception as e:
            print(f"✗ Verification error: {e}")
            return False

    def restore(self, backup_path: Path, force: bool = False) -> bool:
        """
        Restaura banco de dados de um backup.
        
        Args:
            backup_path: Caminho do backup
            force: Não pedir confirmação
            
        Returns:
            True se restaurado com sucesso
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            # Tentar encontrar no diretório de backups
            backup_path = self.backup_dir / backup_path.name
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Verificar integridade do backup
        if not self.verify(backup_path):
            raise RuntimeError("Backup failed integrity check, aborting restore")

        if not force:
            response = input(f"Restore database from {backup_path.name}? [y/N] ")
            if response.lower() != 'y':
                print("Restore cancelled")
                return False

        # Fazer backup do banco atual antes de restaurar
        if self.db_path.exists():
            pre_restore_backup = self.db_path.with_suffix('.db.pre_restore')
            shutil.copy2(self.db_path, pre_restore_backup)
            print(f"✓ Current database backed up to: {pre_restore_backup.name}")

        # Restaurar
        shutil.copy2(backup_path, self.db_path)
        
        # Verificar restauração
        if self.verify(self.db_path):
            print(f"✓ Database restored from: {backup_path.name}")
            return True
        else:
            # Tentar restaurar o backup pré-restore
            if pre_restore_backup.exists():
                shutil.copy2(pre_restore_backup, self.db_path)
            raise RuntimeError("Restored database failed integrity check")

    def list_backups(self) -> list:
        """Lista todos os backups disponíveis."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("resync_*.db"), reverse=True):
            stat = backup_file.stat()
            backups.append({
                "name": backup_file.name,
                "path": str(backup_file),
                "size": stat.st_size,
                "size_human": self._human_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        return backups

    def cleanup(self, keep: int = 5) -> int:
        """
        Remove backups antigos, mantendo apenas os N mais recentes.
        
        Args:
            keep: Número de backups para manter
            
        Returns:
            Número de backups removidos
        """
        backups = sorted(self.backup_dir.glob("resync_*.db"), 
                        key=lambda p: p.stat().st_mtime, 
                        reverse=True)
        
        removed = 0
        for backup in backups[keep:]:
            backup.unlink()
            # Remover arquivo de metadados se existir
            meta_file = backup.with_suffix('.db.meta')
            if meta_file.exists():
                meta_file.unlink()
            removed += 1
            print(f"✓ Removed: {backup.name}")
        
        if removed:
            print(f"✓ Cleanup complete: {removed} backup(s) removed, {keep} kept")
        else:
            print(f"✓ No cleanup needed: {len(backups)} backup(s) found")
        
        return removed

    def _save_metadata(self, backup_path: Path):
        """Salva metadados do backup."""
        meta = {
            "source": str(self.db_path),
            "backup_date": datetime.now().isoformat(),
            "size": backup_path.stat().st_size,
            "resync_version": self._get_resync_version(),
        }
        
        meta_file = backup_path.with_suffix('.db.meta')
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

    def _get_resync_version(self) -> str:
        """Obtém versão do Resync."""
        try:
            version_file = Path("/opt/resync/current/VERSION")
            if version_file.exists():
                return version_file.read_text().strip()
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _human_size(size: int) -> str:
        """Converte bytes para formato legível."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return "{size:.1f} {unit}"
            size /= 1024
        return "{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Resync Database Backup Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s backup                    Create a backup
  %(prog)s backup --name pre_upgrade Create named backup
  %(prog)s restore backup_file.db    Restore from backup
  %(prog)s list                      List available backups
  %(prog)s verify backup_file.db     Verify backup integrity
  %(prog)s cleanup --keep 5          Keep only 5 most recent backups
        """
    )
    
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'verify', 'cleanup'],
                        help='Action to perform')
    parser.add_argument('file', nargs='?', help='Backup file (for restore/verify)')
    parser.add_argument('--name', '-n', help='Custom backup name')
    parser.add_argument('--keep', '-k', type=int, default=5, help='Number of backups to keep (cleanup)')
    parser.add_argument('--db-path', help='Database path')
    parser.add_argument('--backup-dir', help='Backup directory')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        backup_mgr = DatabaseBackup(args.db_path, args.backup_dir)
        
        if args.action == 'backup':
            backup_path = backup_mgr.backup(args.name)
            if args.json:
                print(json.dumps({"status": "success", "path": str(backup_path)}))
                
        elif args.action == 'restore':
            if not args.file:
                parser.error("restore requires a backup file")
            backup_mgr.restore(Path(args.file), args.force)
            
        elif args.action == 'list':
            backups = backup_mgr.list_backups()
            if args.json:
                print(json.dumps(backups, indent=2))
            else:
                print("\nAvailable backups:")
                print("=" * 60)
                for b in backups:
                    print("  {b['name']:<40} {b['size_human']:>8}  {b['created']}")
                print(f"\nTotal: {len(backups)} backup(s)")
                print(f"Location: {backup_mgr.backup_dir}\n")
                
        elif args.action == 'verify':
            if not args.file:
                parser.error("verify requires a backup file")
            success = backup_mgr.verify(Path(args.file))
            sys.exit(0 if success else 1)
            
        elif args.action == 'cleanup':
            backup_mgr.cleanup(args.keep)
            
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
