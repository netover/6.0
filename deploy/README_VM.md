# Deploy em VM (sem Load Balancer / sem Nginx / sem Docker)

Este projeto é compatível com Nginx/Docker, mas este guia assume **VM em rede privada**,
executando via **systemd + Gunicorn (process manager) + Uvicorn workers**.

## 1) Pré-requisitos
- Python 3.14 (venv recomendado)
- Dependências do sistema conforme suas libs (ex.: build-essential, libpq-dev, etc.)

## 2) Instalação (exemplo)
```bash
sudo useradd --system --home /opt/resync --shell /usr/sbin/nologin resync
sudo mkdir -p /opt/resync /var/log/resync /etc/resync
sudo chown -R resync:resync /opt/resync /var/log/resync

# copie o código para /opt/resync
cd /opt/resync
python3.14 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 3) Configurar environment
Copie `deploy/systemd/resync.env.example` para `/etc/resync/resync.env` e ajuste:

- `TRUSTED_HOSTS`: hosts permitidos (protege contra Host header attacks).
- `PROXY_HEADERS=false`: **sem load balancer/proxy**, mantenha falso.
- Rate limiting: habilitado por padrão, com limites internos conservadores.
- `JWT_LEEWAY_SECONDS`: tolerância de clock skew.

## 4) Systemd service
```bash
sudo cp deploy/systemd/resync.service.example /etc/systemd/system/resync.service
sudo systemctl daemon-reload
sudo systemctl enable --now resync
sudo systemctl status resync
```

## 5) Healthchecks
- Liveness: `/liveness`
- Readiness: `/readiness`

## 6) Atualização (go-live seguro)
- Envie nova versão para `/opt/resync`
- Rode smoke checks (compileall, testes)
- `sudo systemctl restart resync`
- Verifique `/readiness` e logs no `journalctl -u resync -f`

## Referências
- Uvicorn deployment (nota: `uvicorn.workers` deprecated; use `uvicorn-worker`):
  https://uvicorn.dev/deployment/
- FastAPI middleware (TrustedHost/Proxy headers):
  https://fastapi.tiangolo.com/advanced/behind-a-proxy/
