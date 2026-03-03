#!/usr/bin/env python3
"""
Valkey 9 Setup Script for Resync

Este script:
1. Instala dependências de compilação
2. Baixa e compila Valkey 9 a partir do código fonte
3. Configura Valkey para produção
4. Cria serviço systemd
5. Inicia o Valkey

Usage:
    python install_valkey.py --action install    # Instala Valkey
    python install_valkey.py --action setup     # Configura Valkey
    python install_valkey.py --action all       # Instala + configura + inicia
    python install_valkey.py --action start     # Inicia Valkey
    python install_valkey.py --action stop      # Para Valkey
    python install_valkey.py --action restart    # Reinicia Valkey
    python install_valkey.py --action test       # Testa conexão
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Configuration
VALKEY_VERSION = "9.0.3"
VALKEY_HOST = "localhost"
VALKEY_PORT = 6379
VALKEY_PASSWORD = os.environ.get("VALKEY_PASSWORD", "default")
VALKEY_DB = 0

# Install paths
INSTALL_PREFIX = "/usr/local"
VALKEY_USER = "valkey"
VALKEY_GROUP = "valkey"
VALKEY_DATA_DIR = "/var/lib/valkey"
VALKEY_LOG_DIR = "/var/log/valkey"
VALKEY_CONF_DIR = "/etc/valkey"
VALKEY_CONF_FILE = f"{VALKEY_CONF_DIR}/valkey.conf"
VALKEY_SERVICE_FILE = "/etc/systemd/system/valkey.service"

VALKEY_CONFIG = """# =============================================================================
# Valkey Configuration for Resync v6.2.0
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
dir /var/lib/valkey

# Persistência AOF
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Segurança
requirepass default

# Conexões
maxclients 10000
timeout 0

# Logs
loglevel notice
logfile /var/log/valkey/valkey.log

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

# Daemon
daemonize yes
pidfile /var/run/valkey/valkey.pid
"""

VALKEY_SYSTEMD_SERVICE = f"""[Unit]
Description=Valkey In-Memory Data Store
Documentation=https://valkey.io/
After=network.target

[Service]
Type=simple
User={VALKEY_USER}
Group={VALKEY_GROUP}
ExecStart={INSTALL_PREFIX}/bin/valkey-server {VALKEY_CONF_FILE}
Restart=always
RestartSec=3
RuntimeDirectory=valkey
RuntimeDirectoryMode=0755
LimitNOFILE=65536
TimeoutStartSec=30
TimeoutStopSec=30

# Hardening
PrivateTmp=yes
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=yes

[Install]
WantedBy=multi-user.target
"""


def run_command(cmd: list[str], check: bool = True, cwd: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Executa um comando shell."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=cwd, env=env)


def run_command_shell(cmd: str, check: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Executa um comando shell (string)."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True, cwd=cwd)


def check_valkey_installed() -> bool:
    """Verifica se Valkey já está instalado."""
    try:
        result = run_command([f"{INSTALL_PREFIX}/bin/valkey-server", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_dependencies() -> None:
    """Instala dependências de compilação."""
    print("\n=== Instalando dependências de compilação ===")
    
    packages = [
        "build-essential",
        "pkg-config",
        "libsystemd-dev",
        "libssl-dev",
        "tcl",
        "tcl-dev",
    ]
    
    for pkg in packages:
        result = run_command_shell(f"dpkg -l | grep -q '^{pkg} '", check=False)
        if result.returncode != 0:
            run_command_shell(f"apt-get install -y {pkg}")
    
    print("Dependências instaladas!")


def download_valkey() -> Path:
    """Baixa o código fonte do Valkey via git."""
    print(f"\n=== Baixando Valkey {VALKEY_VERSION} ===")
    
    temp_dir = Path("/tmp/valkey_build")
    temp_dir.mkdir(exist_ok=True)
    
    build_dir = temp_dir / f"valkey-{VALKEY_VERSION}"
    
    if build_dir.exists():
        print(f"Diretório já existe: {build_dir}")
        return build_dir
    
    print(f"Clonando do GitHub (tag {VALKEY_VERSION})...")
    run_command_shell(
        f"git clone --depth 1 --branch {VALKEY_VERSION} https://github.com/valkey-io/valkey.git {build_dir}",
        cwd=str(temp_dir)
    )
    
    print(f"Download concluído: {build_dir}")
    return build_dir


def compile_valkey(build_dir: Path) -> None:
    """Compila o Valkey."""
    print("\n=== Compilando Valkey ===")
    
    # Compila
    print("Compilando (make -j$(nproc))...")
    num_cpus = os.cpu_count() or 4
    run_command_shell(f"make -j{num_cpus}", cwd=str(build_dir))
    
    print("Instalando...")
    run_command_shell(f"make install PREFIX={INSTALL_PREFIX}", cwd=str(build_dir))
    
    print(f"Valkey instalado em: {INSTALL_PREFIX}/bin/valkey-server")


def create_user_and_dirs() -> None:
    """Cria usuário e diretórios."""
    print("\n=== Criando usuário e diretórios ===")
    
    # Cria usuário se não existir
    result = run_command_shell(f"id {VALKEY_USER} 2>/dev/null", check=False)
    if result.returncode != 0:
        run_command_shell(f"useradd --system --shell /usr/sbin/nologin --home-dir {VALKEY_DATA_DIR} {VALKEY_USER}")
        print(f"Usuário '{VALKEY_USER}' criado")
    
    # Cria diretórios
    for directory in [VALKEY_DATA_DIR, VALKEY_LOG_DIR, VALKEY_CONF_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        run_command_shell(f"chown {VALKEY_USER}:{VALKEY_GROUP} {directory}")
    
    print("Diretórios criados!")


def configure_valkey() -> None:
    """Configura o Valkey."""
    print("\n=== Configurando Valkey ===")
    
    # Cria diretório de configuração
    Path(VALKEY_CONF_DIR).mkdir(parents=True, exist_ok=True)
    
    # Escreve configuração
    Path(VALKEY_CONF_FILE).write_text(VALKEY_CONFIG)
    print(f"Configuração salva em: {VALKEY_CONF_FILE}")
    
    # Cria serviço systemd
    Path(VALKEY_SERVICE_FILE).write_text(VALKEY_SYSTEMD_SERVICE)
    print(f"Serviço systemd salvo em: {VALKEY_SERVICE_FILE}")
    
    # Permissões
    run_command_shell(f"chown {VALKEY_USER}:{VALKEY_GROUP} {VALKEY_CONF_FILE}")
    run_command_shell(f"chmod 640 {VALKEY_CONF_FILE}")
    
    print("Configuração concluída!")


def start_valkey() -> None:
    """Inicia o Valkey via systemd."""
    print("\n=== Iniciando Valkey ===")
    
    # Recarrega systemd
    run_command_shell("systemctl daemon-reload")
    
    # Habilita e inicia
    run_command_shell(f"systemctl enable valkey")
    run_command_shell(f"systemctl start valkey")
    
    # Verifica status
    result = run_command_shell("systemctl is-active valkey", check=False)
    if result.returncode == 0 and "active" in result.stdout:
        print("✅ Valkey iniciado com sucesso!")
    else:
        print("❌ Erro ao iniciar Valkey")
        run_command_shell("systemctl status valkey --no-pager")


def stop_valkey() -> None:
    """Para o Valkey."""
    print("\n=== Parando Valkey ===")
    run_command_shell("systemctl stop valkey")
    print("Valkey parado!")


def restart_valkey() -> None:
    """Reinicia o Valkey."""
    print("\n=== Reiniciando Valkey ===")
    run_command_shell("systemctl restart valkey")
    print("Valkey reiniciado!")


def test_valkey() -> None:
    """Testa conexão com Valkey."""
    print("\n=== Testando Conexão com Valkey ===")
    
    try:
        result = run_command_shell(
            f"{INSTALL_PREFIX}/bin/valkey-cli -h {VALKEY_HOST} -p {VALKEY_PORT} -a {VALKEY_PASSWORD} ping",
            check=False
        )
        
        if result.returncode == 0 and "PONG" in result.stdout:
            print(f"✅ Valkey conectado: {result.stdout.strip()}")
        else:
            print(f"❌ Erro: {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"{INSTALL_PREFIX}/bin/valkey-cli não encontrado.")


def main() -> None:
    global VALKEY_HOST, VALKEY_PORT, VALKEY_PASSWORD, VALKEY_VERSION
    
    parser = argparse.ArgumentParser(description="Valkey 9 Setup for Resync")
    parser.add_argument(
        "--action",
        choices=["install", "setup", "all", "start", "stop", "restart", "test"],
        default="all",
        help="Ação a executar",
    )
    parser.add_argument("--host", default=VALKEY_HOST, help="Host do Valkey")
    parser.add_argument("--port", type=int, default=VALKEY_PORT, help="Porta do Valkey")
    parser.add_argument("--password", default=VALKEY_PASSWORD, help="Senha do Valkey")
    parser.add_argument("--version", default=VALKEY_VERSION, help="Versão do Valkey")
    
    args = parser.parse_args()
    
    VALKEY_HOST = args.host
    VALKEY_PORT = args.port
    VALKEY_PASSWORD = args.password
    VALKEY_VERSION = args.version
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║           Valkey 9 Setup Script for Resync                        ║
║                    Versão: {VALKEY_VERSION:<39}║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    if args.action == "install":
        if check_valkey_installed():
            print(f"✅ Valkey já está instalado!")
        else:
            install_dependencies()
            tarball = download_valkey()
            compile_valkey(tarball)
            create_user_and_dirs()
            print("✅ Valkey instalado com sucesso!")
    
    elif args.action == "setup":
        create_user_and_dirs()
        configure_valkey()
        print("✅ Valkey configurado com sucesso!")
    
    elif args.action == "all":
        # Instalação
        if check_valkey_installed():
            print(f"✅ Valkey já está instalado!")
        else:
            install_dependencies()
            build_dir = download_valkey()
            compile_valkey(build_dir)
        
        # Configuração
        create_user_and_dirs()
        configure_valkey()
        
        # Início
        start_valkey()
        test_valkey()
        
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   Instalação Concluída!                            ║
╚═══════════════════════════════════════════════════════════════════╝

Valkey está instalado e rodando!

Configurações:
  - Host: {host}
  - Porta: {port}
  - Senha: {password}
  - Data: {data_dir}
  - Log: {log_dir}

Comandos úteis:
  - Status:  systemctl status valkey
  - Iniciar: systemctl start valkey
  - Parar:   systemctl stop valkey
  - Reiniciar: systemctl restart valkey
  - CLI:     {install_prefix}/bin/valkey-cli -a {password}

Configure o .env:
  VALKEY_URL=redis://:{password}@{host}:{port}/{db}
""".format(
            host=VALKEY_HOST,
            port=VALKEY_PORT,
            password=VALKEY_PASSWORD,
            data_dir=VALKEY_DATA_DIR,
            log_dir=VALKEY_LOG_DIR,
            install_prefix=INSTALL_PREFIX,
            db=VALKEY_DB
        ))
    
    elif args.action == "start":
        start_valkey()
    
    elif args.action == "stop":
        stop_valkey()
    
    elif args.action == "restart":
        restart_valkey()
    
    elif args.action == "test":
        test_valkey()


if __name__ == "__main__":
    main()
