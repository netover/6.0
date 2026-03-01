# VM Setup (no Docker)

## 1) Install system packages (Ubuntu/Debian)
- Python 3.14
- build-essential, python3.14-venv, libpq-dev
- PostgreSQL 14+ and Redis 6+

## 2) Configure Postgres & Redis
```bash
export RESYNC_DB_PASS='strong_password'
./deploy/vm/configure_postgres.sh

export RESYNC_REDIS_PASS='strong_redis_password'
./deploy/vm/configure_redis.sh
```

## 3) Configure environment
```bash
export APP_DATABASE_URL='postgresql+asyncpg://resync:strong_password@localhost:5432/resync'
export APP_REDIS_URL='redis://:strong_redis_password@127.0.0.1:6379/0'
```

## 4) Install + init schema
```bash
./deploy/vm/install.sh
```

## 5) Run
```bash
source .venv/bin/activate
uvicorn resync.main:app --host 0.0.0.0 --port 8000
```

## Optional: systemd service
```bash
cd /opt/resync
./deploy/vm/systemd/install_systemd.sh
sudo systemctl start resync
sudo systemctl status resync
```

## Deploy/Run helper
```bash
./deploy/vm/run_prod.sh
```

## Production server (Gunicorn + uvicorn-worker)
This VM deployment uses Gunicorn with the uvicorn-worker class as recommended by Uvicorn docs.

