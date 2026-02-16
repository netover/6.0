#!/usr/bin/env python3
"""
Redis Setup Script for Resync v6.2.0

Este script:
1. Instala Redis (Windows/Linux/macOS)
2. Configura Redis para produção
3. Configura persistência e segurança
4. Cria configuração recomendada

Usage:
    python install_redis.py --action install    # Instala Redis
    python install_redis.py --action setup      # Configura Redis
    python install_redis.py --action all       # Instala + configura
    python install_redis.py --action start      # Inicia Redis
    python install_redis.py --action docker     # Inicia via Docker
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_DB = 0

REDIS_CONFIG = """
# =============================================================================
# Redis Configuration for Resync v6.2.0
# =============================================================================

# Rede
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300

# Memória
maxmemory 256mb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistência RDB
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./data

# Persistência AOF
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Segurança
# Descomente para habilitar senha:
# requirepass sua_senha_aqui

# Conexões
maxclients 10000
timeout 0
tcp-keepalive 300

# Logs
loglevel notice
logfile ""

# Snapshot de memória
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Performance
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
rdb-save-incremental-fsync yes
"""

REDIS_DOCKER_CONFIG = """
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: resync-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  redis_data:
    driver: local
"""


def run_command(cmd: list[str], check: bool = True, cwd: str = None) -> subprocess.CompletedProcess:
    """Executa um comando shell."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=cwd)


def detect_os() -> str:
    """Detecta o sistema operacional."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"


def install_redis_windows() -> None:
    """Instala Redis no Windows."""
    print("\n=== Instalando Redis no Windows ===")

    # Verifica se já está instalado
    try:
        result = run_command(["redis-cli", "--version"], check=False)
        if result.returncode == 0:
            print(f"Redis já instalado: {result.stdout.strip()}")
            return
    except FileNotFoundError:
        pass

    print("""
Para instalar Redis no Windows:

1. Use WSL2 (recomendado):
   - Instale WSL2: wsl --install
   - Use Ubuntu e instale Redis: sudo apt install redis-server

2. Ou use Memurai (fork nativo):
   - Baixe em: https://www.memurai.com/
   - Instale e inicie o serviço

3. Ou use Docker (mais fácil):
   - Instale Docker Desktop
   - Execute: docker run -d --name resync-redis -p 6379:6379 redis:7-alpine
""")


def install_redis_linux() -> None:
    """Instala Redis no Linux."""
    print("\n=== Instalando Redis no Linux ===")

    if os.path.exists("/etc/debian_version"):
        # Debian/Ubuntu
        run_command(["sudo", "apt", "update"])
        run_command(["sudo", "apt", "install", "-y", "redis-server"])
    elif os.path.exists("/etc/redhat-release"):
        # RHEL/CentOS/Fedora
        run_command(["sudo", "yum", "install", "-y", "redis"])
    else:
        print("Instalando via snap...")
        run_command(["sudo", "snap", "install", "redis"])


def install_redis_macos() -> None:
    """Instala Redis no macOS."""
    print("\n=== Instalando Redis no macOS ===")
    print("""
Use Homebrew:
    brew install redis
    brew services start redis

Ou use Docker:
    docker run -d --name resync-redis -p 6379:6379 redis:7-alpine
""")


def setup_redis() -> None:
    """Configura Redis."""
    print("\n=== Configurando Redis ===")

    # Criar diretório de dados
    data_dir = Path("redis_data")
    data_dir.mkdir(exist_ok=True)

    # Criar configuração
    config_path = Path("redis.conf")
    config_path.write_text(REDIS_CONFIG)
    print(f"Configuração salva em: {config_path}")

    # Criar configuração Docker
    docker_path = Path("docker-compose.redis.yml")
    docker_path.write_text(REDIS_DOCKER_CONFIG)
    print(f"Docker Compose salvo em: {docker_path}")

    print(f"""
=== Redis Configurado ===

Host: {REDIS_HOST}
Port: {REDIS_PORT}
Data dir: {data_dir.absolute()}

Configure o arquivo .env:
    REDIS_URL=redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}

Para iniciar Redis:
    - Linux/Mac: redis-server redis.conf
    - Docker: docker-compose -f docker-compose.redis.yml up -d
    - Windows: Use WSL2 ou Docker
""")


def start_redis() -> None:
    """Inicia Redis."""
    print("\n=== Iniciando Redis ===")

    # Tenta iniciar via redis-server
    try:
        run_command(["redis-server", "--version"], check=False)
        run_command(["redis-server", "redis.conf"], check=False)
        print("Redis iniciado com sucesso!")
        return
    except FileNotFoundError:
        pass

    # Tenta via Docker
    try:
        result = run_command(["docker", "--version"], check=False)
        if result.returncode == 0:
            print("Iniciando Redis via Docker...")
            run_command(["docker", "run", "-d", "--name", "resync-redis",
                        "-p", "6379:6379", "-v", "redis_data:/data",
                        "redis:7-alpine"], check=False)
            print("Redis iniciado via Docker!")
            return
    except FileNotFoundError:
        pass

    print("""
Não foi possível iniciar Redis automaticamente.
Options:
1. Use Docker: docker run -d --name resync-redis -p 6379:6379 redis:7-alpine
2. Use WSL2 no Windows
3. Instale Redis manualmente
""")


def start_docker() -> None:
    """Inicia Redis via Docker Compose."""
    print("\n=== Iniciando Redis via Docker ===")

    docker_path = Path("docker-compose.redis.yml")
    if not docker_path.exists():
        docker_path.write_text(REDIS_DOCKER_CONFIG)

    try:
        run_command(["docker-compose", "-f", "docker-compose.redis.yml", "up", "-d"])
        print("Redis iniciado via Docker Compose!")
    except FileNotFoundError:
        # Tenta docker compose (v2)
        try:
            run_command(["docker", "compose", "-f", "docker-compose.redis.yml", "up", "-d"])
        except Exception:
            print("Docker Compose não disponível.")


def test_redis() -> None:
    """Testa conexão com Redis."""
    print("\n=== Testando Conexão com Redis ===")

    try:
        result = run_command([
            "redis-cli", "-h", REDIS_HOST, "-p", str(REDIS_PORT), "ping"
        ], check=False)

        if result.returncode == 0:
            print(f"✅ Redis conectado: {result.stdout.strip()}")
        else:
            print(f"❌ Erro: {result.stderr.strip()}")
    except FileNotFoundError:
        print("redis-cli não encontrado. Redis pode não estar instalado.")


def main():
    global REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
    parser = argparse.ArgumentParser(description="Redis Setup for Resync")
    parser.add_argument(
        "--action",
        choices=["install", "setup", "all", "start", "docker", "test"],
        default="all",
        help="Ação a executar"
    )
    parser.add_argument("--host", default=REDIS_HOST, help="Host do Redis")
    parser.add_argument("--port", type=int, default=REDIS_PORT, help="Porta do Redis")
    parser.add_argument("--password", default=REDIS_PASSWORD, help="Senha do Redis")

    args = parser.parse_args()

    REDIS_HOST = args.host
    REDIS_PORT = args.port
    REDIS_PASSWORD = args.password

    os_name = detect_os()

    if args.action == "install":
        if os_name == "windows":
            install_redis_windows()
        elif os_name == "linux":
            install_redis_linux()
        elif os_name == "macos":
            install_redis_macos()

    elif args.action == "setup":
        setup_redis()

    elif args.action == "all":
        if os_name == "windows":
            install_redis_windows()
        elif os_name == "linux":
            install_redis_linux()
        elif os_name == "macos":
            install_redis_macos()
        setup_redis()

    elif args.action == "start":
        start_redis()

    elif args.action == "docker":
        start_docker()

    elif args.action == "test":
        test_redis()


if __name__ == "__main__":
    main()
