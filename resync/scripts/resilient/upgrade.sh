#!/bin/bash
# =============================================================================
# RESYNC - Production Upgrade Script
# =============================================================================
#
# Este script executa upgrade do Resync no servidor de produção.
# Inclui backup automático, rollback em caso de falha e health check.
#
# Uso:
#   sudo ./upgrade.sh /path/to/resync-X.Y.Z-upgrade.zip
#
# Requisitos:
#   - Executar como root ou com sudo
#   - Pacote .zip gerado pelo build_release.sh
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
PACKAGES_DIR="${BASE_DIR}/packages"
LOG_DIR="/var/log/${APP_NAME}"
KEEP_RELEASES=5
HEALTH_CHECK_URL="http://127.0.0.1:8000/health"
HEALTH_CHECK_TIMEOUT=60

# State tracking for rollback
PREVIOUS_RELEASE=""
NEW_RELEASE_DIR=""
DB_BACKUP=""
UPGRADE_SUCCESS=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# FUNÇÕES DE LOG
# =============================================================================

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# FUNÇÕES DE CLEANUP E ROLLBACK
# =============================================================================

cleanup_on_failure() {
    if [[ "$UPGRADE_SUCCESS" != "true" ]]; then
        log_error "Upgrade failed! Initiating automatic rollback..."
        rollback
    fi
}

rollback() {
    log_warning "Starting rollback procedure..."
    
    # Restaurar database
    if [[ -n "$DB_BACKUP" && -f "$DB_BACKUP" ]]; then
        log_info "Restoring database from backup..."
        cp "$DB_BACKUP" "${SHARED_DIR}/db/resync.db"
        log_success "Database restored"
    fi
    
    # Restaurar symlink para versão anterior
    if [[ -n "$PREVIOUS_RELEASE" && -d "$PREVIOUS_RELEASE" ]]; then
        log_info "Restoring previous release: $(basename $PREVIOUS_RELEASE)"
        ln -sfn "$PREVIOUS_RELEASE" "${CURRENT_LINK}.new"
        mv -Tf "${CURRENT_LINK}.new" "$CURRENT_LINK"
        log_success "Previous release restored"
    fi
    
    # Reiniciar serviço
    log_info "Restarting service..."
    systemctl start "${APP_NAME}" || true
    
    # Cleanup failed release
    if [[ -n "$NEW_RELEASE_DIR" && -d "$NEW_RELEASE_DIR" && "$NEW_RELEASE_DIR" != "$PREVIOUS_RELEASE" ]]; then
        log_info "Removing failed release directory..."
        rm -rf "$NEW_RELEASE_DIR"
    fi
    
    log_warning "Rollback completed. Please investigate the failure."
    exit 1
}

# Trap para cleanup automático em caso de erro
trap cleanup_on_failure EXIT

# =============================================================================
# FUNÇÕES DE UPGRADE
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Verificar root
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
    
    # Verificar pacote
    if [[ -z "${1:-}" ]]; then
        log_error "Usage: $0 /path/to/resync-X.Y.Z-upgrade.zip"
        exit 1
    fi
    
    if [[ ! -f "$1" ]]; then
        log_error "Package not found: $1"
        exit 1
    fi
    
    # Verificar comandos necessários
    for cmd in python3 unzip sqlite3; do
        if ! command -v $cmd &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    log_success "Prerequisites OK"
}

backup_database() {
    log_info "Backing up SQLite database..."
    
    local db_file="${SHARED_DIR}/db/resync.db"
    
    if [[ ! -f "$db_file" ]]; then
        log_warning "Database not found, skipping backup (first install?)"
        return 0
    fi
    
    # Criar diretório de backup
    mkdir -p "$BACKUP_DIR"
    
    # Nome do backup com timestamp
    DB_BACKUP="${BACKUP_DIR}/resync_pre_upgrade_$(date +%Y%m%d_%H%M%S).db"
    
    # Usar SQLite backup API para consistência
    sqlite3 "$db_file" ".backup '${DB_BACKUP}'"
    
    # Verificar integridade do backup
    if sqlite3 "$DB_BACKUP" "PRAGMA integrity_check" | grep -q "ok"; then
        log_success "Database backup verified: $(basename $DB_BACKUP)"
    else
        log_error "Backup verification failed!"
        exit 1
    fi
    
    # Manter apenas os últimos N backups
    ls -1t ${BACKUP_DIR}/resync_*.db 2>/dev/null | tail -n +$((KEEP_RELEASES + 1)) | xargs rm -f 2>/dev/null || true
}

extract_package() {
    local package_path="$1"
    
    log_info "Extracting package..."
    
    # Copiar pacote para diretório de packages
    mkdir -p "$PACKAGES_DIR"
    cp "$package_path" "$PACKAGES_DIR/"
    
    # Extrair em diretório temporário
    local temp_dir=$(mktemp -d)
    unzip -q "$package_path" -d "$temp_dir"
    
    # Encontrar diretório extraído
    local extracted_dir=$(find "$temp_dir" -maxdepth 1 -type d -name "resync-*" | head -1)
    
    if [[ -z "$extracted_dir" ]]; then
        log_error "Invalid package structure"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Ler versão do MANIFEST ou VERSION
    local version=""
    if [[ -f "${extracted_dir}/MANIFEST.json" ]]; then
        version=$(python3 -c "import json; print(json.load(open('${extracted_dir}/MANIFEST.json'))['version'])" 2>/dev/null)
    fi
    
    if [[ -z "$version" && -f "${extracted_dir}/VERSION" ]]; then
        version=$(cat "${extracted_dir}/VERSION")
    fi
    
    if [[ -z "$version" ]]; then
        version=$(date +%Y%m%d_%H%M%S)
        log_warning "Version not found in package, using timestamp: $version"
    fi
    
    # Criar diretório de release
    NEW_RELEASE_DIR="${RELEASES_DIR}/${version}"
    
    if [[ -d "$NEW_RELEASE_DIR" ]]; then
        log_warning "Version ${version} already exists, adding timestamp..."
        NEW_RELEASE_DIR="${RELEASES_DIR}/${version}_$(date +%H%M%S)"
    fi
    
    mkdir -p "$RELEASES_DIR"
    mv "$extracted_dir" "$NEW_RELEASE_DIR"
    rm -rf "$temp_dir"
    
    log_success "Extracted to: $NEW_RELEASE_DIR"
}

setup_virtualenv() {
    log_info "Setting up Python virtual environment..."
    
    local venv_dir="${NEW_RELEASE_DIR}/venv"
    local wheels_dir="${NEW_RELEASE_DIR}/wheels"
    
    # Criar venv
    python3 -m venv "$venv_dir"
    
    # Atualizar pip
    "${venv_dir}/bin/pip" install --upgrade pip wheel setuptools --quiet
    
    # Instalar dependências offline
    if [[ -d "$wheels_dir" && -n "$(ls -A $wheels_dir 2>/dev/null)" ]]; then
        log_info "Installing dependencies from bundled wheels..."
        "${venv_dir}/bin/pip" install \
            --no-index \
            --find-links="$wheels_dir" \
            -r "${NEW_RELEASE_DIR}/requirements.txt" \
            --quiet
    else
        log_warning "No wheels found, trying online installation..."
        "${venv_dir}/bin/pip" install -r "${NEW_RELEASE_DIR}/requirements.txt" \
            --extra-index-url https://download.pytorch.org/whl/cpu --quiet
    fi
    
    log_success "Virtual environment ready"
}

setup_shared_links() {
    log_info "Setting up shared resource links..."
    
    # Criar estrutura shared se não existir
    mkdir -p "${SHARED_DIR}/config"
    mkdir -p "${SHARED_DIR}/db"
    
    # Copiar .env.example se .env não existir
    if [[ ! -f "${SHARED_DIR}/config/.env" ]]; then
        if [[ -f "${NEW_RELEASE_DIR}/config/.env.example" ]]; then
            cp "${NEW_RELEASE_DIR}/config/.env.example" "${SHARED_DIR}/config/.env"
            log_warning "Created .env from example - please configure before starting!"
        elif [[ -f "${NEW_RELEASE_DIR}/.env.example" ]]; then
            cp "${NEW_RELEASE_DIR}/.env.example" "${SHARED_DIR}/config/.env"
            log_warning "Created .env from example - please configure before starting!"
        fi
    fi
    
    # Criar symlinks
    ln -sf "${SHARED_DIR}/config/.env" "${NEW_RELEASE_DIR}/.env"
    ln -sf "${SHARED_DIR}/db" "${NEW_RELEASE_DIR}/data"
    
    log_success "Shared links configured"
}

run_migrations() {
    log_info "Checking for database migrations..."
    
    local alembic_cmd="${NEW_RELEASE_DIR}/venv/bin/alembic"
    
    if [[ ! -f "${NEW_RELEASE_DIR}/alembic.ini" ]]; then
        log_warning "Alembic not configured, skipping migrations"
        return 0
    fi
    
    cd "$NEW_RELEASE_DIR"
    
    # Salvar revisão atual para possível rollback
    local current_revision=$("$alembic_cmd" current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1 || echo "base")
    echo "$current_revision" > "${NEW_RELEASE_DIR}/.previous_db_revision"
    
    # Executar migrations
    log_info "Running migrations..."
    "$alembic_cmd" upgrade head
    
    local new_revision=$("$alembic_cmd" current 2>/dev/null | grep -oE '[a-f0-9]{12}' | head -1 || echo "unknown")
    
    if [[ "$current_revision" != "$new_revision" ]]; then
        log_success "Migrations applied: $current_revision → $new_revision"
    else
        log_info "No new migrations to apply"
    fi
}

switch_release() {
    log_info "Switching to new release..."
    
    # Atomic symlink switch usando rename
    ln -sfn "$NEW_RELEASE_DIR" "${CURRENT_LINK}.new"
    mv -Tf "${CURRENT_LINK}.new" "$CURRENT_LINK"
    
    log_success "Now running: $(basename $NEW_RELEASE_DIR)"
}

restart_service() {
    log_info "Restarting Resync service..."
    
    if systemctl is-active --quiet "${APP_NAME}"; then
        systemctl restart "${APP_NAME}"
    else
        log_warning "Service not running, starting..."
        systemctl start "${APP_NAME}"
    fi
    
    sleep 2
    
    if systemctl is-active --quiet "${APP_NAME}"; then
        log_success "Service started successfully"
    else
        log_error "Service failed to start!"
        journalctl -u "${APP_NAME}" --no-pager -n 20
        exit 1
    fi
}

health_check() {
    log_info "Running health checks..."
    
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / 3))
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        sleep 3
        
        if curl -sf "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
            log_success "Health check passed (attempt $attempt)"
            return 0
        fi
        
        log_warning "Health check attempt $attempt/$max_attempts failed"
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    exit 1
}

cleanup_old_releases() {
    log_info "Cleaning up old releases..."
    
    local release_count=$(ls -1dt ${RELEASES_DIR}/*/ 2>/dev/null | wc -l)
    
    if [[ $release_count -gt $KEEP_RELEASES ]]; then
        local to_remove=$((release_count - KEEP_RELEASES))
        log_info "Removing $to_remove old release(s)..."
        ls -1dt ${RELEASES_DIR}/*/ | tail -n +$((KEEP_RELEASES + 1)) | xargs rm -rf
        log_success "Cleanup complete"
    fi
}

update_deployment_history() {
    local history_file="${BASE_DIR}/deployment_history.json"
    local version=$(basename "$NEW_RELEASE_DIR")
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Criar ou atualizar histórico
    python3 << EOF
import json
from pathlib import Path

history_file = Path("$history_file")
version = "$version"
timestamp = "$timestamp"
release_path = "$NEW_RELEASE_DIR"
previous = "$PREVIOUS_RELEASE"

# Carregar histórico existente
if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)
else:
    history = {"current_version": "", "deployments": []}

# Marcar versão anterior como archived
for dep in history["deployments"]:
    if dep["status"] == "active":
        dep["status"] = "archived"

# Adicionar nova versão
history["current_version"] = version
history["deployments"].insert(0, {
    "version": version,
    "deployed_at": timestamp,
    "deployed_by": "upgrade.sh",
    "release_path": release_path,
    "previous_version": previous.split("/")[-1] if previous else None,
    "status": "active"
})

# Manter apenas últimos 50 registros
history["deployments"] = history["deployments"][:50]

# Salvar
with open(history_file, "w") as f:
    json.dump(history, f, indent=2)
EOF

    log_success "Deployment history updated"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local package_path="${1:-}"
    
    log ""
    log "=============================================="
    log "     RESYNC UPGRADE - $(date '+%Y-%m-%d %H:%M')"
    log "=============================================="
    log ""
    
    # Guardar versão atual
    if [[ -L "$CURRENT_LINK" ]]; then
        PREVIOUS_RELEASE=$(readlink -f "$CURRENT_LINK")
        log_info "Current version: $(basename $PREVIOUS_RELEASE)"
    else
        log_info "No previous installation detected"
    fi
    
    # Executar steps
    check_prerequisites "$package_path"
    backup_database
    
    # Parar serviço
    log_info "Stopping Resync service..."
    systemctl stop "${APP_NAME}" 2>/dev/null || true
    
    extract_package "$package_path"
    setup_virtualenv
    setup_shared_links
    run_migrations
    switch_release
    
    # Ajustar permissões
    chown -R resync:resync "$BASE_DIR" 2>/dev/null || true
    chown -R resync:resync "$LOG_DIR" 2>/dev/null || true
    
    restart_service
    health_check
    cleanup_old_releases
    update_deployment_history
    
    # Marcar sucesso para desabilitar rollback automático
    UPGRADE_SUCCESS=true
    
    log ""
    log "=============================================="
    log_success "UPGRADE COMPLETED SUCCESSFULLY!"
    log "=============================================="
    log ""
    log_info "Previous version: $(basename ${PREVIOUS_RELEASE:-none})"
    log_info "New version: $(basename $NEW_RELEASE_DIR)"
    log_info "Database backup: $(basename ${DB_BACKUP:-none})"
    log ""
    log_info "To rollback if needed:"
    log_info "  sudo ${BASE_DIR}/current/scripts/rollback.sh"
    log ""
    
    # Desabilitar trap
    trap - EXIT
}

main "$@"
