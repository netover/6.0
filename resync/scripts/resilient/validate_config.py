#!/usr/bin/env python3
"""Script para validar configuração do .env"""

import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from resync.core.exceptions import ConfigurationError
from resync.settings import get_settings


def main():
    """Valida configuração e exibe resultado."""

    print("\n[*] Validando configuracao Resync...")
    print(f"Diretorio: {project_root}")

    # Verificar se .env existe
    env_file = project_root / ".env"
    if not env_file.exists():
        print(f"\n[X] Arquivo .env nao encontrado em: {env_file}")
        print("\n   Crie um arquivo .env com as variaveis necessarias")
        return 1

    print(f"[OK] Arquivo .env encontrado: {env_file}")

    # Validar configuração
    try:
        settings = get_settings()

        print("\n[OK] CONFIGURACAO VALIDA!\n")
        print("Resumo:")
        print(f"   Ambiente: {settings.environment}")
        print(f"   Admin User: {settings.admin_username}")
        print(f"   Redis: {settings.redis_url.split('@')[-1]}")
        print(f"   TWS: {settings.tws_host}:{settings.tws_port}")
        print(f"   TWS User: {settings.tws_user}")

        if settings.llm_endpoint:
            print(f"   LLM: {settings.llm_endpoint}")

        print()
        return 0

    except ConfigurationError as e:
        print("\n[X] CONFIGURACAO INVALIDA:")
        print(f"   {e.message}\n")

        if e.details.get("errors"):
            print("   Erros encontrados:")
            for error in e.details["errors"]:
                if error.strip():
                    print(f"   • {error}")

        print("\n   Exemplo de .env valido:")
        print("   ADMIN_USERNAME=admin")
        print("   ADMIN_PASSWORD=MinhaS3nh@Forte")
        print("   SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')")
        print("   REDIS_URL=redis://localhost:6379")
        print("   TWS_HOST=localhost")
        print("   TWS_PORT=31111")
        print("   TWS_USER=twsuser")
        print("   TWS_PASSWORD=twspass")
        print()

        return 1


if __name__ == "__main__":
    sys.exit(main())
