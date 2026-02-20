#!/usr/bin/env python3
"""
Setup Completo para Resync v6.2.0

Este script instala e configura todos os pré-requisitos:
1. PostgreSQL
2. Redis
3. Dependências Python

Usage:
    python setup_environment.py              # Setup completo
    python setup_environment.py --skip-db   # Pula banco de dados
    python setup_environment.py --docker   # Usa Docker para tudo
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Constants for file paths
DOCKER_COMPOSE_FILE = "docker-compose.resync.yml"


def run_command(
    cmd: list[str], check: bool = True, cwd: str | None = None
) -> subprocess.CompletedProcess:
    """Executa um comando shell."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(result.stdout)
    return result


def check_python_version() -> bool:
    """Verifica versão do Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 14):
        print(
            f"❌ Python 3.14+ necessário. Versão atual: {version.major}.{version.minor}"
        )
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_python_deps() -> None:
    """Instala dependências Python."""
    print("\n" + "=" * 60)
    print("INSTALANDO DEPENDÊNCIAS PYTHON")
    print("=" * 60)

    requirements_file = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt não encontrado!")
        return

    # Criar ambiente virtual se não existir
    venv_path = Path(__file__).parent.parent / ".venv"

    if not venv_path.exists():
        print("\nCriando ambiente virtual...")
        run_command([sys.executable, "-m", "venv", str(venv_path)])

    # Determinar pip e python do venv
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    # Atualizar pip
    print("\nAtualizando pip...")
    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    # Instalar dependências
    print("\nInstalando dependências...")
    try:
        run_command([str(pip_path), "install", "-r", str(requirements_file)])
        print("✅ Dependências instaladas!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Erro ao instalar dependências: {e}")
        print("   Continuando mesmo assim...")


def setup_postgres() -> None:
    """Configura PostgreSQL."""
    print("\n" + "=" * 60)
    print("CONFIGURANDO POSTGRESQL")
    print("=" * 60)

    script_path = Path(__file__).parent / "install_postgres.py"
    if script_path.exists():
        run_command([sys.executable, str(script_path), "--action", "setup"])
    else:
        print("❌ Script install_postgres.py não encontrado")


def setup_redis() -> None:
    """Configura Redis."""
    print("\n" + "=" * 60)
    print("CONFIGURANDO REDIS")
    print("=" * 60)

    script_path = Path(__file__).parent / "install_redis.py"
    if script_path.exists():
        run_command([str(script_path), "--action", "setup"])
    else:
        print("❌ Script install_redis.py não encontrado")


def setup_docker() -> None:
    """Setup usando Docker."""
    print("\n" + "=" * 60)
    print("CONFIGURANDO COM DOCKER")
    print("=" * 60)

    # Docker Compose para PostgreSQL e Redis
    docker_compose = """
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: resync-postgres
    environment:
      POSTGRES_USER: resync
      POSTGRES_PASSWORD: resync_password
      POSTGRES_DB: resync
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U resync"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: resync-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  postgres_data:
  redis_data:
"""

    docker_file = Path(DOCKER_COMPOSE_FILE)
    docker_file.write_text(docker_compose)
    print(f"Docker Compose salvo em: {docker_file}")

    print("\nIniciando containers...")
    try:
        run_command(["docker-compose", "-f", DOCKER_COMPOSE_FILE, "up", "-d"])
        print("✅ Containers PostgreSQL e Redis iniciados!")
    except FileNotFoundError:
        # Tenta docker compose v2
        try:
            run_command(["docker", "compose", "-f", DOCKER_COMPOSE_FILE, "up", "-d"])
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"❌ Docker não disponível: {e}")


def create_env_file() -> None:
    """Cria arquivo .env de exemplo."""
    print("\n" + "=" * 60)
    print("CRIANDO ARQUIVO .env")
    print("=" * 60)

    env_example = """# =============================================================================
# RESYNC v6.2.0 - Environment Configuration
# Copie este arquivo para .env e configure os valores
# =============================================================================

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_URL=postgresql+asyncpg://resync:resync_password@localhost:5432/resync
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# =============================================================================
# REDIS
# =============================================================================
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_URL=redis://localhost:6379/1

# =============================================================================
# AUTH
# =============================================================================
AUTH_SECRET_KEY=change-this-to-a-random-secret-key
AUTH_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# =============================================================================
# LLM PROVIDERS
# =============================================================================
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
LITELLM_API_KEY=your-litellm-api-key

# =============================================================================
# LANGFUSE (Observability)
# =============================================================================
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# =============================================================================
# LANGGRAPH
# =============================================================================
LANGGRAPH_ENABLED=true
LANGGRAPH_CHECKPOINT_TTL_HOURS=24
LANGGRAPH_MAX_RETRIES=3
LANGGRAPH_REQUIRE_APPROVAL=true

# =============================================================================
# ENVIRONMENT
# =============================================================================
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# =============================================================================
# CORS
# =============================================================================
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
"""

    env_file = Path(".env")
    if env_file.exists():
        print(f"⚠️  {env_file} já existe. Pulando...")
    else:
        env_file.write_text(env_example)
        print(f"✅ {env_file} criado!")
        print("   Copie para .env e configure os valores")


def verify_setup() -> None:
    """Verifica se tudo está configurado."""
    print("\n" + "=" * 60)
    print("VERIFICANDO CONFIGURAÇÃO")
    print("=" * 60)

    # Verificar PostgreSQL
    try:
        result = subprocess.run(
            [
                "psql",
                "-h",
                "localhost",
                "-p",
                "5432",
                "-U",
                "resync",
                "-c",
                "SELECT 1;",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("✅ PostgreSQL conectado")
        else:
            print("❌ PostgreSQL não conectado")
    except FileNotFoundError:
        print("⚠️  PostgreSQL: cliente não encontrado")

    # Verificar Redis
    try:
        result = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            print("✅ Redis conectado")
        else:
            print("❌ Redis não conectado")
    except FileNotFoundError:
        print("⚠️  Redis: cliente não encontrado")

    # Verificar Python deps
    venv_python = (
        Path(__file__).parent.parent
        / ".venv"
        / ("python.exe" if sys.platform == "win32" else "bin/python")
    )
    if venv_python.exists():
        print("✅ Ambiente virtual criado")
    else:
        print("⚠️  Ambiente virtual não criado")


def main():
    parser = argparse.ArgumentParser(description="Setup Completo para Resync")
    parser.add_argument(
        "--skip-db", action="store_true", help="Pula configuração de banco de dados"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Usa Docker para PostgreSQL e Redis"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Pula instalação de dependências Python",
    )

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RESYNC v6.2.0 - SETUP COMPLETO                         ║
║                                                                              ║
║  Este script instalará e configurará todos os pré-requisitos              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Verificar Python
    if not check_python_version():
        sys.exit(1)

    # Instalar dependências Python
    if not args.skip_deps:
        install_python_deps()

    # Configurar bancos de dados
    if not args.skip_db:
        if args.docker:
            setup_docker()
        else:
            setup_postgres()
            setup_redis()

    # Criar arquivo .env
    create_env_file()

    # Verificar setup
    verify_setup()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SETUP CONCLUÍDO!                                   ║
║                                                                              ║
║  Próximos passos:                                                          ║
║  1. Configure o arquivo .env com suas credenciais                        ║
║  2. Execute: python -m uvicorn resync.main:app --reload                   ║
║  3. Acesse: http://localhost:8000                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
