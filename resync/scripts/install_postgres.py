#!/usr/bin/env python3
"""
PostgreSQL Setup Script for Resync v6.2.0

Este script:
1. Instala PostgreSQL (Windows/Linux/macOS)
2. Cria o banco de dados e usuário
3. Instala extensões necessárias (pgvector, pg_trgm)
4. Cria todas as tabelas do projeto
5. Configura o banco para produção

Usage:
    python install_postgres.py --action install    # Instala PostgreSQL
    python install_postgres.py --action setup      # Configura banco
    python install_postgres.py --action all        # Instala + configura
    python install_postgres.py --action migrate    # Executa migrações
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DB_NAME = "resync"
DB_USER = "resync"
# SECURITY: Read password from environment variable, never hardcode in production
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = "localhost"
DB_PORT = 5432

SCHEMA_SQL = """
-- =============================================================================
-- RESYNC v6.2.0 - PostgreSQL Schema
-- Execute este script após criar o banco de dados
-- =============================================================================

-- =============================================================================
-- EXTENSÕES NECESSÁRIAS
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- USUARIOS E AUTENTICAÇÃO
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- =============================================================================
-- AUDIT LOGS
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- =============================================================================
-- TWS (IBM Tivoli Workload Scheduler)
-- =============================================================================
CREATE TABLE IF NOT EXISTS tws_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workstation VARCHAR(255) NOT NULL,
    job_name VARCHAR(255),
    job_id VARCHAR(255),
    status VARCHAR(50),
    description TEXT,
    submitted_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    exit_code INTEGER,
    workstation_status JSONB DEFAULT '{}'::jsonb,
    raw_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tws_job_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workstation VARCHAR(255) NOT NULL,
    job_name VARCHAR(255) NOT NULL,
    job_id VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    description TEXT,
    scheduled_time TIMESTAMP WITH TIME ZONE,
    actual_start TIMESTAMP WITH TIME ZONE,
    actual_end TIMESTAMP WITH TIME ZONE,
    exit_code INTEGER,
    is_running BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tws_workstation_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workstation VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL,
    ip_address VARCHAR(50),
    cpu_usage REAL,
    memory_usage REAL,
    disk_usage REAL,
    active_jobs INTEGER DEFAULT 0,
    queued_jobs INTEGER DEFAULT 0,
    last_contact TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tws_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workstation VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50),
    message TEXT,
    job_name VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS tws_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    pattern VARCHAR(500) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tws_problem_solutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_pattern VARCHAR(500) NOT NULL,
    solution_description TEXT NOT NULL,
    solution_steps JSONB DEFAULT '[]'::jsonb,
    success_rate REAL DEFAULT 0.0,
    times_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para TWS
CREATE INDEX IF NOT EXISTS idx_tws_snapshots_workstation ON tws_snapshots(workstation);
CREATE INDEX IF NOT EXISTS idx_tws_snapshots_created ON tws_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_tws_job_status_workstation ON tws_job_status(workstation);
CREATE INDEX IF NOT EXISTS idx_tws_job_status_status ON tws_job_status(status);
CREATE INDEX IF NOT EXISTS idx_tws_workstation_status ON tws_workstation_status(workstation);
CREATE INDEX IF NOT EXISTS idx_tws_events_workstation ON tws_events(workstation);
CREATE INDEX IF NOT EXISTS idx_tws_events_timestamp ON tws_events(timestamp);

-- =============================================================================
-- WORKSTATION METRICS HISTORY
-- =============================================================================
CREATE TABLE IF NOT EXISTS workstation_metrics_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workstation VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    unit VARCHAR(50),
    tags JSONB DEFAULT '{}'::jsonb,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workstation_metrics_workstation ON workstation_metrics_history(workstation);
CREATE INDEX IF NOT EXISTS idx_workstation_metrics_collected ON workstation_metrics_history(collected_at);
CREATE INDEX IF NOT EXISTS idx_workstation_metrics_name ON workstation_metrics_history(metric_name);

-- =============================================================================
-- CONVERSATIONS & CONTEXT
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS context_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_context_session ON context_content(session_id);

-- =============================================================================
-- AUDIT & QUEUE
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    data JSONB DEFAULT '{}'::jsonb,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entry_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_error TEXT,
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_entries_timestamp ON audit_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_queue_status ON audit_queue(status);

-- =============================================================================
-- USER PROFILES & SESSIONS
-- =============================================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    preferences JSONB DEFAULT '{}'::jsonb,
    theme VARCHAR(50) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    notification_settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    login_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    logout_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_session_history_session ON session_history(session_id);

-- =============================================================================
-- FEEDBACK & LEARNING
-- =============================================================================
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    message_hash VARCHAR(64),
    feedback_type VARCHAR(50) NOT NULL,
    rating INTEGER,
    feedback_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_thresholds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) UNIQUE NOT NULL,
    threshold_value REAL NOT NULL,
    confidence_level REAL DEFAULT 0.95,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS active_learning_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    query_text TEXT NOT NULL,
    response_text TEXT,
    uncertainty_score REAL NOT NULL,
    is_labeled BOOLEAN DEFAULT FALSE,
    label_value TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_learning_candidates_session ON active_learning_candidates(session_id);

-- =============================================================================
-- METRICS DATA
-- =============================================================================
CREATE TABLE IF NOT EXISTS metric_data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value REAL NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS metric_aggregations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    aggregation_type VARCHAR(50) NOT NULL,
    value REAL NOT NULL,
    time_bucket VARCHAR(50) NOT NULL,
    bucket_start TIMESTAMP WITH TIME ZONE NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metric_data_timestamp ON metric_data_points(timestamp);
CREATE INDEX IF NOT EXISTS idx_metric_aggregations_bucket ON metric_aggregations(bucket_start);

-- =============================================================================
-- TEAMS WEBHOOK
-- =============================================================================
CREATE TABLE IF NOT EXISTS teams_webhook_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    teams_id VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams_webhook_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    teams_user_id UUID REFERENCES teams_webhook_users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- TEAMS NOTIFICATIONS
-- =============================================================================
CREATE TABLE IF NOT EXISTS teams_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id VARCHAR(255) UNIQUE NOT NULL,
    channel_name VARCHAR(255),
    webhook_url TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams_job_mappings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_pattern VARCHAR(255) NOT NULL,
    channel_id UUID REFERENCES teams_channels(id) ON DELETE CASCADE,
    notification_types JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams_pattern_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    pattern VARCHAR(500) NOT NULL,
    severity_filter VARCHAR(50),
    channel_id UUID REFERENCES teams_channels(id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams_notification_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID REFERENCES teams_channels(id) ON DELETE CASCADE,
    config_key VARCHAR(100) NOT NULL,
    config_value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS teams_notification_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID REFERENCES teams_channels(id) ON DELETE SET NULL,
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- KNOWLEDGE GRAPH (DKG)
-- =============================================================================
CREATE TABLE IF NOT EXISTS kg_nodes (
    tenant TEXT NOT NULL DEFAULT 'default',
    graph_version INTEGER NOT NULL DEFAULT 1,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant, graph_version, node_id)
);

CREATE TABLE IF NOT EXISTS kg_edges (
    tenant TEXT NOT NULL DEFAULT 'default',
    graph_version INTEGER NOT NULL DEFAULT 1,
    edge_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant, graph_version, edge_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_name_trgm ON kg_nodes USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON kg_nodes (tenant, graph_version, node_type);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges (tenant, graph_version, source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges (tenant, graph_version, target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_relation ON kg_edges (tenant, graph_version, relation_type);
CREATE INDEX IF NOT EXISTS idx_kg_edges_evidence_gin ON kg_edges USING GIN (evidence);

-- =============================================================================
-- LANGGRAPH CHECKPOINTER (para persistência de estado)
-- =============================================================================
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS langgraph_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (namespace, key)
);

CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_thread ON langgraph_checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoint_writes_thread ON langgraph_checkpoint_writes(thread_id);
CREATE INDEX IF NOT EXISTS idx_langgraph_store_namespace ON langgraph_store(namespace);
"""


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Executa um comando shell."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def detect_os() -> str:
    """Detecta o sistema operacional."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"


def install_postgres_windows() -> None:
    """Instala PostgreSQL no Windows via Chocolatey ou manual."""
    print("\n=== Instalando PostgreSQL no Windows ===")

    # Verifica se já está instalado
    try:
        result = run_command(["psql", "--version"], check=False)
        if result.returncode == 0:
            print(f"PostgreSQL já instalado: {result.stdout.strip()}")
            return
    except FileNotFoundError:
        pass

    print("""
Para instalar PostgreSQL no Windows:

1. Baixe PostgreSQL em: https://www.postgresql.org/download/windows/
2. Execute o instalador
3. Durante a instalação:
   - Escolha a senha para o usuário 'postgres'
   - Porta: 5432 (padrão)
   - Locale: Default

Ou use Chocolatey:
    choco install postgresql -y

Ou use Winget:
    winget install PostgreSQL.PostgreSQL
""")


def install_postgres_linux() -> None:
    """Instala PostgreSQL no Linux."""
    print("\n=== Instalando PostgreSQL no Linux ===")

    # Detecta distribuição
    if os.path.exists("/etc/debian_version"):
        # Debian/Ubuntu
        run_command(["sudo", "apt", "update"])
        run_command(["sudo", "apt", "install", "-y", "postgresql", "postgresql-contrib", "postgresql-16-pgvector"])
    elif os.path.exists("/etc/redhat-release"):
        # RHEL/CentOS/Fedora
        run_command(["sudo", "yum", "install", "-y", "postgresql-server", "postgresql-contrib"])
        run_command(["sudo", "postgresql-setup", "initdb"])
    else:
        print("Distribuição não suportada. Instale manualmente.")


def install_postgres_macos() -> None:
    """Instala PostgreSQL no macOS."""
    print("\n=== Instalando PostgreSQL no macOS ===")
    print("""
Use Homebrew:
    brew install postgresql@16
    brew services start postgresql@16

Ou use Docker:
    docker run -d --name postgres -e POSTGRES_PASSWORD=resync_password -p 5432:5432 postgres:16
""")


def setup_database() -> None:
    """Configura o banco de dados."""
    print("\n=== Configurando Banco de Dados ===")

    # Verifica se PostgreSQL está rodando
    try:
        result = run_command([
            "psql", "-h", DB_HOST, "-p", str(DB_PORT),
            "-U", "postgres", "-c", "SELECT version();"
        ], check=False)
        if result.returncode != 0:
            print("Erro ao conectar ao PostgreSQL. Verifique se está rodando.")
            return
    except FileNotFoundError:
        print("Cliente psql não encontrado. Adicione PostgreSQL ao PATH.")
        return

    # Criar usuário e banco
    commands = [
        # Criar usuário
        f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';",
        # Criar banco
        f"CREATE DATABASE {DB_NAME} OWNER {DB_USER};",
        # Conceder privilégios
        f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};",
        # Conceder privilégios no schema
        f"GRANT ALL ON SCHEMA public TO {DB_USER};",
    ]

    for cmd in commands:
        try:
            run_command([
                "psql", "-h", DB_HOST, "-p", str(DB_PORT),
                "-U", "postgres", "-c", cmd
            ], check=False)
        except Exception as e:
            print(f"Nota: {e}")

    # Executar schema
    print("\nCriando tabelas...")
    schema_file = Path(__file__).parent / "schema.sql"
    schema_file.write_text(SCHEMA_SQL)

    run_command([
        "psql", "-h", DB_HOST, "-p", str(DB_PORT),
        "-U", "postgres", "-d", DB_NAME, "-f", str(schema_file)
    ])

    print(f"""
=== Banco de Dados Configurado ===

Database: {DB_NAME}
User: {DB_USER}
Host: {DB_HOST}
Port: {DB_PORT}

Configure o arquivo .env:
    DATABASE_URL=postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}
""")


def run_migrations() -> None:
    """Executa migrações SQLAlchemy (se houver)."""
    print("\n=== Executando Migrações ===")
    print("Para criar tabelas via SQLAlchemy, execute:")
    print("    python -c \"from resync.core.database import engine; from resync.core.database.models import *; import resync.core.database.models_registry; from resync.core.database.schema import Base; Base.metadata.create_all(engine)\"")


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Setup for Resync")
    parser.add_argument(
        "--action",
        choices=["install", "setup", "all", "migrate"],
        default="all",
        help="Ação a executar"
    )
    parser.add_argument("--db-name", default=DB_NAME, help="Nome do banco de dados")
    parser.add_argument("--db-user", default=DB_USER, help="Usuário do banco")
    parser.add_argument("--db-password", default=DB_PASSWORD, help="Senha do banco")
    parser.add_argument("--db-host", default=DB_HOST, help="Host do banco")
    parser.add_argument("--db-port", type=int, default=DB_PORT, help="Porta do banco")

    args = parser.parse_args()

    # Atualiza variáveis globais
    global DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
    DB_NAME = args.db_name
    DB_USER = args.db_user
    DB_PASSWORD = args.db_password
    DB_HOST = args.db_host
    DB_PORT = args.db_port

    os_name = detect_os()

    if args.action in ("install", "all"):
        if os_name == "windows":
            install_postgres_windows()
        elif os_name == "linux":
            install_postgres_linux()
        elif os_name == "macos":
            install_postgres_macos()

    if args.action in ("setup", "all"):
        setup_database()

    if args.action == "migrate":
        run_migrations()


if __name__ == "__main__":
    main()
