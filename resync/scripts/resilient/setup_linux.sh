#!/bin/bash
# =============================================================================
# RESYNC v6.2.0 — Auto-Install Script (Linux)
# =============================================================================
# Supported: Ubuntu 22.04+, Debian 12+
# Usage:
#   sudo ./setup_linux.sh              # Full install
#   sudo ./setup_linux.sh --dev        # Development mode (skip systemd)
#   sudo ./setup_linux.sh --help       # Show this help
#
# What this script does:
#   1. Installs system packages (Python 3.11, build tools, libpq)
#   2. Installs and configures PostgreSQL 16 + pgvector extension
#   3. Creates resync database, user, and schema
#   4. Installs and configures Redis 7
#   5. Creates Python virtualenv and installs dependencies (including Docling)
#   6. Copies .env template and generates secure credentials
#   7. Runs database migrations (Alembic)
#   8. Pre-downloads Docling ML models
#   9. Installs systemd service
# =============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}    $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}      $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC}    $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC}   $1"; }
log_step()    { echo -e "\n${BOLD}── $1 ──${NC}"; }

# ── Parse arguments ──────────────────────────────────────────────────────────
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --dev)      DEV_MODE=true ;;
        --help|-h)
            echo "Usage: sudo $0 [--dev]"
            echo ""
            echo "Options:"
            echo "  --dev       Development mode (skip systemd service install)"
            echo "  --help      Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Detect project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in scripts/resilient/ — project root is two levels up
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
    log_error "Cannot find project root. Expected: $PROJECT_DIR/requirements.txt"
    log_error "Run from project root: sudo scripts/resilient/setup_linux.sh"
    exit 1
fi

cd "$PROJECT_DIR"

PYTHON_CMD="python3.11"
DB_NAME="resync"
DB_USER="resync"
DB_PASSWORD="$(openssl rand -base64 24 2>/dev/null || echo "resync_$(date +%s)")"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${BOLD}   RESYNC v6.2.0 — Auto-Install${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
log_info "Project dir:  $PROJECT_DIR"
log_info "Dev mode:     $DEV_MODE"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
log_step "1/9  Pre-flight checks"

if [[ $EUID -ne 0 && "$DEV_MODE" == "false" ]]; then
    log_error "Production install requires root. Use: sudo $0"
    log_info  "Or use --dev for development mode (no root needed)."
    exit 1
fi

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    log_info "OS: $PRETTY_NAME"
fi

log_success "Pre-flight checks passed"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — System packages
# ─────────────────────────────────────────────────────────────────────────────
log_step "2/9  System packages"

if [[ $EUID -eq 0 ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq

    # Python 3.11
    if ! command -v $PYTHON_CMD &>/dev/null; then
        log_info "Installing Python 3.11..."
        apt-get install -y software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
        apt-get update -qq
        apt-get install -y python3.11 python3.11-venv python3.11-dev
    fi
    log_success "Python $($PYTHON_CMD --version 2>&1)"

    # Build tools
    apt-get install -y --no-install-recommends \
        build-essential libpq-dev libffi-dev libssl-dev \
        git curl wget unzip jq lsb-release gnupg2 \
        2>/dev/null
    log_success "Build tools installed"
else
    log_warning "Not root — skipping system packages"
    # Try to find Python 3.11
    if command -v python3.11 &>/dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
        log_warning "Using $(python3 --version) — Python 3.11+ recommended"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — PostgreSQL + pgvector
# ─────────────────────────────────────────────────────────────────────────────
log_step "3/9  PostgreSQL + pgvector"

SKIP_DB=false

if [[ $EUID -eq 0 ]]; then
    # Install PostgreSQL
    if ! command -v psql &>/dev/null; then
        log_info "Installing PostgreSQL..."
        apt-get install -y postgresql postgresql-contrib
    fi

    # Install pgvector
    PG_VERSION=$(psql --version 2>/dev/null | grep -oP '\d+' | head -1 || echo "16")
    apt-get install -y "postgresql-${PG_VERSION}-pgvector" 2>/dev/null || \
        apt-get install -y postgresql-pgvector 2>/dev/null || {
        log_warning "pgvector not in repos — building from source..."
        if [[ ! -d /tmp/pgvector ]]; then
            git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git /tmp/pgvector
        fi
        cd /tmp/pgvector && make && make install && cd "$PROJECT_DIR"
        log_success "pgvector built from source"
    }

    # Start PostgreSQL
    systemctl enable postgresql
    systemctl start postgresql

    if systemctl is-active --quiet postgresql; then
        log_success "PostgreSQL is running"
    else
        log_error "PostgreSQL failed to start"
        SKIP_DB=true
    fi

    # Create database and user
    if [[ "$SKIP_DB" == "false" ]]; then
        log_info "Creating database '${DB_NAME}' and user '${DB_USER}'..."

        sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
            sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';"

        sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
            sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

        sudo -u postgres psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null && \
            log_success "pgvector extension enabled" || \
            log_warning "pgvector extension not available — install manually later"

        sudo -u postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" 2>/dev/null
        sudo -u postgres psql -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER};" 2>/dev/null

        log_success "Database '${DB_NAME}' ready (user: ${DB_USER})"
    fi
else
    log_warning "Not root — skipping PostgreSQL setup"
    log_info "Set DATABASE_URL in .env manually"
    SKIP_DB=true
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Redis
# ─────────────────────────────────────────────────────────────────────────────
log_step "4/9  Redis"

if [[ $EUID -eq 0 ]]; then
    if ! command -v redis-server &>/dev/null; then
        log_info "Installing Redis..."
        apt-get install -y redis-server
    fi

    systemctl enable redis-server
    systemctl start redis-server

    if systemctl is-active --quiet redis-server; then
        log_success "Redis is running"
    else
        log_warning "Redis failed to start (optional — set RESYNC_DISABLE_REDIS=true)"
    fi
else
    log_warning "Not root — skipping Redis"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Python virtualenv + core dependencies
# ─────────────────────────────────────────────────────────────────────────────
log_step "5/9  Python virtual environment"

VENV_DIR="${PROJECT_DIR}/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Creating virtualenv..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools --quiet

log_info "Installing dependencies (this may take 5-10 minutes — includes Docling ML)..."
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --quiet 2>&1 | tail -1 || \
    pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

PKG_COUNT=$(pip list --format=freeze 2>/dev/null | wc -l)
log_success "Core dependencies installed (${PKG_COUNT} packages)"

# Dev dependencies
if [[ "$DEV_MODE" == "true" && -f "requirements-dev.txt" ]]; then
    log_info "Installing dev dependencies..."
    pip install -r requirements-dev.txt --quiet
    log_success "Dev dependencies installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Docling ML models pre-download
# ─────────────────────────────────────────────────────────────────────────────
log_step "6/9  Docling ML models"

if python -c "import docling" 2>/dev/null; then
    log_success "Docling installed"
    log_info "Pre-downloading ML models (layout analysis + TableFormer)..."
    python -c "
from docling.document_converter import DocumentConverter
try:
    DocumentConverter()
    print('Models cached successfully')
except Exception as e:
    print(f'Models will download on first use: {e}')
" 2>/dev/null || log_warning "Model pre-download failed — will download on first conversion"
else
    log_error "Docling not installed — check requirements.txt installation above"
    log_info "Try: pip install docling --extra-index-url https://download.pytorch.org/whl/cpu"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Configuration (.env)
# ─────────────────────────────────────────────────────────────────────────────
log_step "7/9  Configuration"

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
    if [[ -f "${PROJECT_DIR}/.env.example" ]]; then
        cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
        chmod 600 "${PROJECT_DIR}/.env"

        # Auto-configure database URL
        if [[ "$SKIP_DB" == "false" ]]; then
            DB_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@localhost:5432/${DB_NAME}"
            sed -i "s|^DATABASE_URL=.*|DATABASE_URL=${DB_URL}|" "${PROJECT_DIR}/.env"
        fi

        # Generate secure SECRET_KEY
        SECRET_KEY="$(openssl rand -base64 48 2>/dev/null || python -c 'import secrets; print(secrets.token_urlsafe(48))')"
        sed -i "s|^SECRET_KEY=.*|SECRET_KEY=${SECRET_KEY}|" "${PROJECT_DIR}/.env"

        log_success ".env created with auto-generated credentials"
        log_warning "⚠️  You still need to set LLM_API_KEY and ADMIN_PASSWORD in .env"
    else
        log_error ".env.example not found — create .env manually"
    fi
else
    log_success ".env already exists (not overwritten)"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 — Database migrations
# ─────────────────────────────────────────────────────────────────────────────
log_step "8/9  Database migrations (Alembic)"

if [[ "$SKIP_DB" == "false" ]]; then
    # Source .env for DATABASE_URL
    set -a
    source "${PROJECT_DIR}/.env" 2>/dev/null || true
    set +a

    if [[ -f "${PROJECT_DIR}/alembic.ini" ]]; then
        log_info "Running Alembic migrations..."
        cd "$PROJECT_DIR"
        "${VENV_DIR}/bin/python" -m alembic upgrade head 2>&1 && \
            log_success "Database schema up to date" || \
            log_warning "Migration failed — run manually: alembic upgrade head"
    else
        log_warning "alembic.ini not found — skipping migrations"
    fi
else
    log_warning "Skipping migrations (database not configured locally)"
    log_info "Run manually after configuring DATABASE_URL:"
    log_info "  source .venv/bin/activate && alembic upgrade head"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 — Systemd service
# ─────────────────────────────────────────────────────────────────────────────
log_step "9/9  Systemd service"

if [[ "$DEV_MODE" == "true" ]]; then
    log_info "Dev mode — skipping systemd"
elif [[ $EUID -eq 0 ]]; then
    # Create resync user
    if ! id "resync" &>/dev/null; then
        useradd -r -s /bin/false -d "$PROJECT_DIR" resync
        log_info "Created system user 'resync'"
    fi

    # Write service file
    cat > /etc/systemd/system/resync.service << SERVICEEOF
[Unit]
Description=Resync - TWS/HWA AI Operations Platform
After=network.target postgresql.service redis-server.service
Wants=postgresql.service redis-server.service

[Service]
Type=notify
User=resync
Group=resync
WorkingDirectory=${PROJECT_DIR}
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin
Environment=PYTHONUNBUFFERED=1
Environment=ENVIRONMENT=production
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${VENV_DIR}/bin/gunicorn -c ${PROJECT_DIR}/gunicorn_config.py resync.main:app
Restart=always
RestartSec=10
TimeoutStartSec=180
TimeoutStopSec=30
LimitNOFILE=65536
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=${PROJECT_DIR} /var/log/resync /tmp

[Install]
WantedBy=multi-user.target
SERVICEEOF

    chown -R resync:resync "$PROJECT_DIR" 2>/dev/null || true
    mkdir -p /var/log/resync && chown resync:resync /var/log/resync
    systemctl daemon-reload

    log_success "Systemd service installed (not started yet)"
else
    log_warning "Not root — skipping systemd"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}   ✓ Installation Complete${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo ""

# Status table
printf "  %-18s %s\n" "Component" "Status"
printf "  %-18s %s\n" "──────────────" "──────────────────────"

PY_VER=$(python --version 2>&1)
printf "  %-18s ${GREEN}✓${NC} %s\n" "Python" "$PY_VER"

if systemctl is-active --quiet postgresql 2>/dev/null; then
    printf "  %-18s ${GREEN}✓${NC} %s\n" "PostgreSQL" "running (db: ${DB_NAME})"
elif [[ "$SKIP_DB" == "true" ]]; then
    printf "  %-18s ${YELLOW}○${NC} %s\n" "PostgreSQL" "not configured"
fi

if [[ "$SKIP_DB" == "false" ]]; then
    if sudo -u postgres psql -d "${DB_NAME}" -tc "SELECT 1 FROM pg_extension WHERE extname='vector'" 2>/dev/null | grep -q 1; then
        printf "  %-18s ${GREEN}✓${NC} %s\n" "pgvector" "enabled"
    else
        printf "  %-18s ${YELLOW}○${NC} %s\n" "pgvector" "check manually"
    fi
fi

if redis-cli ping &>/dev/null 2>&1; then
    printf "  %-18s ${GREEN}✓${NC} %s\n" "Redis" "running"
else
    printf "  %-18s ${YELLOW}○${NC} %s\n" "Redis" "not running (optional)"
fi

if python -c "import docling" 2>/dev/null; then
    printf "  %-18s ${GREEN}✓${NC} %s\n" "Docling" "ML document conversion"
else
    printf "  %-18s ${RED}✗${NC} %s\n" "Docling" "FAILED — reinstall with: pip install docling"
fi

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    printf "  %-18s ${GREEN}✓${NC} %s\n" "Configuration" ".env exists"
fi

printf "  %-18s ${GREEN}✓${NC} %s\n" "Dependencies" "${PKG_COUNT:-?} packages"

echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Configure secrets in .env:"
echo "     nano ${PROJECT_DIR}/.env"
echo "       → LLM_API_KEY=sk-your-key-here"
echo "       → ADMIN_PASSWORD=your-secure-password"
echo ""

if [[ "$DEV_MODE" == "true" ]]; then
    echo "  2. Start (dev mode with auto-reload):"
    echo "     source .venv/bin/activate"
    echo "     uvicorn resync.main:app --reload --port 8000"
    echo ""
    echo "     Or production mode locally:"
    echo "     gunicorn -c gunicorn_config.py resync.main:app"
else
    echo "  2. Start the service:"
    echo "     sudo systemctl start resync"
    echo "     sudo systemctl enable resync"
    echo ""
    echo "  3. Verify:"
    echo "     curl http://localhost:8000/health"
    echo ""
    echo "  Production command (same as systemd runs):"
    echo "     gunicorn -c gunicorn_config.py resync.main:app"
fi

if [[ "$SKIP_DB" == "false" ]]; then
    echo ""
    echo -e "  ${YELLOW}Database credentials (save these!):${NC}"
    echo "    User:     ${DB_USER}"
    echo "    Password: ${DB_PASSWORD}"
    echo "    Database: ${DB_NAME}"
fi

echo ""
