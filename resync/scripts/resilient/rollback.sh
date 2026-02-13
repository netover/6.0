#!/bin/bash
# =============================================================================
# RESYNC - Rollback Script
# =============================================================================
#
# Este script faz rollback para uma versão anterior do Resync.
#
# Uso:
#   sudo ./rollback.sh              # Rollback para versão anterior
#   sudo ./rollback.sh 5.9.8        # Rollback para versão específica
#   sudo ./rollback.sh list         # Listar versões disponíveis
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

APP_NAME="resync"
BASE_DIR="/opt/${APP_NAME}"
RELEASES_DIR="${BASE_DIR}/releases"
CURRENT_LINK="${BASE_DIR}/current"
SHARED_DIR="${BASE_DIR}/shared"
BACKUP_DIR="${BASE_DIR}/backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# FUNÇÕES
# =============================================================================

list_releases() {
    echo ""
    echo "Available releases:"
    echo "==================="
    
    local current=""
    if [[ -L "$CURRENT_LINK" ]]; then
        current=$(readlink -f "$CURRENT_LINK")
    fi
    
    for release in $(ls -1dt ${RELEASES_DIR}/*/ 2>/dev/null); do
        local version=$(basename "$release")
        local marker=""
        
        if [[ "$(readlink -f $release)" == "$current" ]]; then
            marker=" ${GREEN}← CURRENT${NC}"
        fi
        
        # Verificar data de criação
        local date=$(stat -c %y "$release" 2>/dev/null | cut -d. -f1)
        
        echo -e "  $version  ($date)$marker"
    done
    
    echo ""
    echo "Backups available:"
    echo "=================="
    
    for backup in $(ls -1t ${BACKUP_DIR}/*.db 2>/dev/null | head -5); do
        local name=$(basename "$backup")
        local size=$(du -h "$backup" | cut -f1)
        echo "  $name  ($size)"
    done
    
    echo ""
}

rollback_to_previous() {
    local current=""
    if [[ -L "$CURRENT_LINK" ]]; then
        current=$(readlink -f "$CURRENT_LINK")
    fi
    
    local releases=($(ls -1dt ${RELEASES_DIR}/*/ 2>/dev/null))
    
    # Encontrar primeira release que não é a atual
    for release in "${releases[@]}"; do
        if [[ "$(readlink -f $release)" != "$current" ]]; then
            rollback_to "$release"
            return
        fi
    done
    
    log_error "No previous release found to rollback to"
    exit 1
}

rollback_to() {
    local target="$1"
    
    # Se for apenas versão, construir path completo
    if [[ ! -d "$target" ]]; then
        target="${RELEASES_DIR}/${target}"
    fi
    
    if [[ ! -d "$target" ]]; then
        log_error "Release not found: $target"
        log_info "Use '$0 list' to see available releases"
        exit 1
    fi
    
    local version=$(basename "$target")
    
    log_info "=============================================="
    log_info "     RESYNC ROLLBACK"
    log_info "=============================================="
    log_info ""
    log_info "Rolling back to: $version"
    
    # Confirmar
    read -p "Are you sure you want to rollback? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    # Parar serviço
    log_info "Stopping service..."
    systemctl stop "${APP_NAME}" 2>/dev/null || true
    
    # Restaurar banco de dados se houver backup correspondente
    restore_database "$version"
    
    # Atomic symlink switch
    log_info "Switching to release: $version"
    ln -sfn "$target" "${CURRENT_LINK}.new"
    mv -Tf "${CURRENT_LINK}.new" "$CURRENT_LINK"
    
    # Iniciar serviço
    log_info "Starting service..."
    systemctl start "${APP_NAME}"
    
    # Verificar se iniciou
    sleep 3
    if systemctl is-active --quiet "${APP_NAME}"; then
        log_success "Service started successfully"
    else
        log_error "Service failed to start!"
        journalctl -u "${APP_NAME}" --no-pager -n 20
        exit 1
    fi
    
    # Atualizar histórico
    update_history "$version" "rollback"
    
    log_info ""
    log_success "Rollback completed successfully!"
    log_info "Current version: $version"
    log_info ""
}

restore_database() {
    local version="$1"
    
    # Procurar backup mais recente ANTES desta versão
    local latest_backup=$(ls -1t ${BACKUP_DIR}/resync_*.db 2>/dev/null | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log_warning "No database backup found, skipping database restore"
        return 0
    fi
    
    log_info "Found backup: $(basename $latest_backup)"
    read -p "Restore database from this backup? [y/N] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restoring database..."
        
        # Backup do banco atual antes de restaurar
        local current_db="${SHARED_DIR}/db/resync.db"
        if [[ -f "$current_db" ]]; then
            cp "$current_db" "${current_db}.pre_rollback.$(date +%Y%m%d%H%M%S)"
        fi
        
        # Restaurar
        cp "$latest_backup" "$current_db"
        
        # Verificar integridade
        if sqlite3 "$current_db" "PRAGMA integrity_check" | grep -q "ok"; then
            log_success "Database restored successfully"
        else
            log_error "Database restore failed integrity check!"
            exit 1
        fi
    else
        log_warning "Database restore skipped"
    fi
}

update_history() {
    local version="$1"
    local action="$2"
    local history_file="${BASE_DIR}/deployment_history.json"
    
    python3 << EOF 2>/dev/null || true
import json
from pathlib import Path
from datetime import datetime

history_file = Path("$history_file")
version = "$version"
action = "$action"

if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)
else:
    history = {"current_version": "", "deployments": []}

# Marcar todas como archived
for dep in history["deployments"]:
    dep["status"] = "archived"

# Adicionar rollback
history["current_version"] = version
history["deployments"].insert(0, {
    "version": version,
    "deployed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "deployed_by": "rollback.sh",
    "action": action,
    "status": "active"
})

with open(history_file, "w") as f:
    json.dump(history, f, indent=2)
EOF
}

# =============================================================================
# MAIN
# =============================================================================

# Verificar root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root or with sudo"
    exit 1
fi

case "${1:-previous}" in
    list|ls|-l)
        list_releases
        ;;
    previous|"")
        rollback_to_previous
        ;;
    *)
        rollback_to "$1"
        ;;
esac
