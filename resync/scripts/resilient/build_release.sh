#!/bin/bash
# =============================================================================
# RESYNC - Build Release Package
# =============================================================================
#
# Este script cria um pacote .zip completo para deploy em VMs sem acesso à internet.
# Inclui todas as dependências Python como arquivos .whl
#
# Uso:
#   ./scripts/build_release.sh [version]
#
# Exemplo:
#   ./scripts/build_release.sh 5.9.10
#   ./scripts/build_release.sh  # usa versão do arquivo VERSION
#
# Requisitos (máquina de build):
#   - Python 3.11+
#   - pip com acesso à internet
#   - zip
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Diretório do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/dist/build"
OUTPUT_DIR="${PROJECT_DIR}/dist/releases"

# Versão
if [[ -n "${1:-}" ]]; then
    VERSION="$1"
else
    VERSION=$(cat "${PROJECT_DIR}/VERSION" 2>/dev/null || echo "0.0.0")
fi

# Target Python version
PYTHON_VERSION="311"
PLATFORM="linux_x86_64"

# Nome do pacote
PACKAGE_NAME="resync-${VERSION}-upgrade"
PACKAGE_DIR="${BUILD_DIR}/${PACKAGE_NAME}"

log_info "=============================================="
log_info "Building Resync Release Package"
log_info "=============================================="
log_info "Version: ${VERSION}"
log_info "Python: ${PYTHON_VERSION}"
log_info "Platform: ${PLATFORM}"
log_info ""

# Cleanup
log_info "Cleaning previous builds..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}" "${PACKAGE_DIR}"

# Criar estrutura do pacote
log_info "Creating package structure..."
mkdir -p "${PACKAGE_DIR}/"{wheels,migrations,config,scripts,static,templates}

# Copiar código fonte
log_info "Copying source code..."
cp -r "${PROJECT_DIR}/resync" "${PACKAGE_DIR}/"
cp -r "${PROJECT_DIR}/static" "${PACKAGE_DIR}/"
cp -r "${PROJECT_DIR}/templates" "${PACKAGE_DIR}/"
cp -r "${PROJECT_DIR}/alembic" "${PACKAGE_DIR}/"
cp "${PROJECT_DIR}/alembic.ini" "${PACKAGE_DIR}/"

# Copiar migrations
if [[ -d "${PROJECT_DIR}/alembic/versions" ]]; then
    cp -r "${PROJECT_DIR}/alembic/versions" "${PACKAGE_DIR}/migrations/"
    log_success "Migrations copied"
fi

# Copiar arquivos de configuração
log_info "Copying configuration files..."
cp "${PROJECT_DIR}/requirements.txt" "${PACKAGE_DIR}/"
cp "${PROJECT_DIR}/requirements-dev.txt" "${PACKAGE_DIR}/" 2>/dev/null || true
cp "${PROJECT_DIR}/.env.example" "${PACKAGE_DIR}/" 2>/dev/null || true
cp "${PROJECT_DIR}/gunicorn_config.py" "${PACKAGE_DIR}/"
cp "${PROJECT_DIR}/gunicorn.conf.py" "${PACKAGE_DIR}/" 2>/dev/null || true  # Legacy alias
cp "${PROJECT_DIR}/VERSION" "${PACKAGE_DIR}/"
cp "${PROJECT_DIR}/CHANGELOG.md" "${PACKAGE_DIR}/" 2>/dev/null || true

# Copiar scripts de deploy
cp "${PROJECT_DIR}/deploy/resync.service" "${PACKAGE_DIR}/config/"
cp "${PROJECT_DIR}/scripts/upgrade.sh" "${PACKAGE_DIR}/scripts/" 2>/dev/null || true
cp "${PROJECT_DIR}/scripts/rollback.sh" "${PACKAGE_DIR}/scripts/" 2>/dev/null || true
cp "${PROJECT_DIR}/scripts/backup_db.py" "${PACKAGE_DIR}/scripts/" 2>/dev/null || true

# Copiar .env.example
cp "${PROJECT_DIR}/.env.example" "${PACKAGE_DIR}/config/" 2>/dev/null || true

# Baixar dependências como wheels
log_info "Downloading Python dependencies as wheels..."
log_info "This may take a few minutes..."

pip download \
    -r "${PROJECT_DIR}/requirements.txt" \
    -d "${PACKAGE_DIR}/wheels" \
    --platform "${PLATFORM}" \
    --python-version "${PYTHON_VERSION}" \
    --only-binary=:all: \
    2>/dev/null || {
    log_warning "Some packages don't have binary wheels, downloading source..."
    pip download \
        -r "${PROJECT_DIR}/requirements.txt" \
        -d "${PACKAGE_DIR}/wheels" \
        2>/dev/null || true
}

WHEEL_COUNT=$(ls -1 "${PACKAGE_DIR}/wheels"/*.whl 2>/dev/null | wc -l)
log_success "Downloaded ${WHEEL_COUNT} wheel packages"

# Criar MANIFEST.json
log_info "Creating MANIFEST.json..."
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Calcular checksums dos arquivos importantes
CHECKSUMS=""
for file in "${PACKAGE_DIR}/resync"/*.py; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        checksum=$(sha256sum "$file" | cut -d' ' -f1)
        CHECKSUMS="${CHECKSUMS}    \"resync/${filename}\": \"${checksum}\",\n"
    fi
done

# Detectar se há migrations novas
HAS_MIGRATIONS="false"
if [[ -d "${PACKAGE_DIR}/migrations/versions" ]] && [[ -n "$(ls -A ${PACKAGE_DIR}/migrations/versions/*.py 2>/dev/null)" ]]; then
    HAS_MIGRATIONS="true"
fi

cat > "${PACKAGE_DIR}/MANIFEST.json" << EOF
{
    "name": "resync",
    "version": "${VERSION}",
    "build_date": "${BUILD_DATE}",
    "python_version": "3.${PYTHON_VERSION:0:1}${PYTHON_VERSION:1}",
    "platform": "${PLATFORM}",
    "requires_migration": ${HAS_MIGRATIONS},
    "min_version": "5.0.0",
    "wheel_count": ${WHEEL_COUNT},
    "checksums": {
$(echo -e "${CHECKSUMS}" | sed '$ s/,$//')
    },
    "notes": "Generated by build_release.sh"
}
EOF

log_success "MANIFEST.json created"

# Criar script de instalação
log_info "Creating install script..."
cat > "${PACKAGE_DIR}/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# =============================================================================
# RESYNC - Quick Install Script
# =============================================================================
# 
# Este script é executado automaticamente pelo upgrade.sh
# Também pode ser executado manualmente para instalação inicial
#
# Uso:
#   sudo ./install.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Installing Resync from package..."

# Verificar se está rodando como root ou com sudo
if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] This script must be run as root or with sudo"
    exit 1
fi

# Criar usuário resync se não existir
if ! id "resync" &>/dev/null; then
    echo "[INFO] Creating resync user..."
    useradd -r -s /bin/false -d /opt/resync resync
fi

# Criar estrutura de diretórios
echo "[INFO] Creating directory structure..."
mkdir -p /opt/resync/{releases,shared/config,shared/db,backups,packages}
mkdir -p /var/log/resync

# Copiar pacote atual
VERSION=$(cat "${SCRIPT_DIR}/VERSION")
RELEASE_DIR="/opt/resync/releases/${VERSION}"

if [[ -d "$RELEASE_DIR" ]]; then
    echo "[WARN] Version ${VERSION} already exists, backing up..."
    mv "$RELEASE_DIR" "${RELEASE_DIR}.backup.$(date +%Y%m%d%H%M%S)"
fi

echo "[INFO] Copying files to ${RELEASE_DIR}..."
cp -r "$SCRIPT_DIR" "$RELEASE_DIR"

# Criar virtual environment
echo "[INFO] Creating virtual environment..."
python3 -m venv "${RELEASE_DIR}/venv"

# Instalar dependências offline
echo "[INFO] Installing dependencies from wheels..."
"${RELEASE_DIR}/venv/bin/pip" install --upgrade pip wheel setuptools --quiet
"${RELEASE_DIR}/venv/bin/pip" install \
    --no-index \
    --find-links="${RELEASE_DIR}/wheels" \
    -r "${RELEASE_DIR}/requirements.txt" \
    --quiet

# Copiar .env se não existir
if [[ ! -f /opt/resync/shared/config/.env ]]; then
    if [[ -f "${RELEASE_DIR}/.env.example" ]]; then
        echo "[INFO] Creating initial .env from example..."
        cp "${RELEASE_DIR}/.env.example" /opt/resync/shared/config/.env
        echo "[WARN] Please edit /opt/resync/shared/config/.env with your settings"
    elif [[ -f "${RELEASE_DIR}/config/.env.example" ]]; then
        cp "${RELEASE_DIR}/config/.env.example" /opt/resync/shared/config/.env
        echo "[WARN] Please edit /opt/resync/shared/config/.env with your settings"
    fi
fi

# Symlinks para shared resources
ln -sf /opt/resync/shared/config/.env "${RELEASE_DIR}/.env"
ln -sf /opt/resync/shared/db "${RELEASE_DIR}/data"

# Atualizar symlink current
echo "[INFO] Updating current symlink..."
ln -sfn "$RELEASE_DIR" /opt/resync/current.new
mv -Tf /opt/resync/current.new /opt/resync/current

# Ajustar permissões
chown -R resync:resync /opt/resync
chown -R resync:resync /var/log/resync
chmod 600 /opt/resync/shared/config/.env 2>/dev/null || true

# Instalar/atualizar systemd service
if [[ -f "${RELEASE_DIR}/config/resync.service" ]]; then
    echo "[INFO] Installing systemd service..."
    cp "${RELEASE_DIR}/config/resync.service" /etc/systemd/system/
    systemctl daemon-reload
fi

echo ""
echo "[OK] Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Edit configuration: sudo nano /opt/resync/shared/config/.env"
echo "  2. Run migrations: cd /opt/resync/current && ./venv/bin/alembic upgrade head"
echo "  3. Start service: sudo systemctl start resync"
echo "  4. Enable on boot: sudo systemctl enable resync"
echo ""
INSTALL_EOF

chmod +x "${PACKAGE_DIR}/install.sh"

# Remover arquivos desnecessários
log_info "Cleaning up unnecessary files..."
find "${PACKAGE_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PACKAGE_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${PACKAGE_DIR}" -type f -name ".DS_Store" -delete 2>/dev/null || true
rm -rf "${PACKAGE_DIR}/.git" 2>/dev/null || true
rm -rf "${PACKAGE_DIR}/.pytest_cache" 2>/dev/null || true
rm -rf "${PACKAGE_DIR}/.mypy_cache" 2>/dev/null || true
rm -rf "${PACKAGE_DIR}/.ruff_cache" 2>/dev/null || true

# Criar ZIP
log_info "Creating ZIP package..."
cd "${BUILD_DIR}"
zip -r "${OUTPUT_DIR}/${PACKAGE_NAME}.zip" "${PACKAGE_NAME}" -x "*.pyc" -x "*__pycache__*"

# Calcular tamanho e checksum
ZIP_SIZE=$(du -h "${OUTPUT_DIR}/${PACKAGE_NAME}.zip" | cut -f1)
ZIP_SHA256=$(sha256sum "${OUTPUT_DIR}/${PACKAGE_NAME}.zip" | cut -d' ' -f1)

# Criar arquivo de checksum
echo "${ZIP_SHA256}  ${PACKAGE_NAME}.zip" > "${OUTPUT_DIR}/${PACKAGE_NAME}.zip.sha256"

# Cleanup
rm -rf "${BUILD_DIR}"

log_info ""
log_info "=============================================="
log_success "Build completed successfully!"
log_info "=============================================="
log_info ""
log_info "Package: ${OUTPUT_DIR}/${PACKAGE_NAME}.zip"
log_info "Size: ${ZIP_SIZE}"
log_info "SHA256: ${ZIP_SHA256}"
log_info ""
log_info "To deploy, copy the .zip file to the server and run:"
log_info "  unzip ${PACKAGE_NAME}.zip"
log_info "  cd ${PACKAGE_NAME}"
log_info "  sudo ./install.sh"
log_info ""
