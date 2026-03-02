# Resync — First Production Deployment Checklist

## Prerequisites
- PostgreSQL 15+ running and accessible
- Redis 7+ running and accessible
- Python 3.14+ or Docker

---

## Step-by-step: First Deploy

### 1. Configure environment
```bash
cp .env.example .env
chmod 600 .env
# Edit .env with real values — especially:
#   APP_SECRET_KEY      → python -c "import secrets; print(secrets.token_hex(32))"
#   APP_ADMIN_PASSWORD  → strong password (min 8 chars)
#   APP_DATABASE_URL    → postgresql+asyncpg://user:pass@host:5432/dbname
#   APP_REDIS_URL       → redis://:password@host:6379/0
#   APP_CORS_ALLOWED_ORIGINS → https://yourdomain.com
#   TRUSTED_HOSTS       → yourdomain.com
```

### 2. Initialize the database
```bash
# Option A: Run init script directly
python scripts/init_db.py

# Option B: Via Docker
docker run --rm --env-file .env resync:prod python scripts/init_db.py

# Option C: Via alembic (after schema exists)
alembic upgrade head
```

### 3. Build and start
```bash
# Docker Compose (recommended)
docker compose -f deploy/docker-compose.yml --env-file .env up -d

# Or Gunicorn directly
gunicorn -c deploy/gunicorn_conf.py resync.main:app
```

### 4. Verify health
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
```

---

## Required environment variables (minimum)

| Variable | Description | Example |
|---|---|---|
| `APP_SECRET_KEY` | JWT signing key (≥32 chars) | `secrets.token_hex(32)` |
| `APP_ADMIN_PASSWORD` | Admin panel password (≥8 chars) | `MyStr0ngP@ss` |
| `APP_DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://...` |
| `APP_REDIS_URL` | Redis connection URL | `redis://:pass@host:6379/0` |
| `APP_CORS_ALLOWED_ORIGINS` | Allowed frontend origins | `https://app.example.com` |
| `TRUSTED_HOSTS` | Allowed Host header values | `app.example.com` |
| `PROMETHEUS_MULTIPROC_DIR` | Prometheus metrics dir | `/tmp/prometheus_multiproc` |

---

## Security hardening (production)

- [ ] `APP_SECRET_KEY` is a random 32+ char secret (not the example value)
- [ ] `APP_ADMIN_PASSWORD` is a strong password (not "admin" or "password")
- [ ] `APP_CORS_ALLOWED_ORIGINS` does NOT include wildcard `*`
- [ ] `TRUSTED_HOSTS` is set to your actual domain(s)
- [ ] `FORWARDED_ALLOW_IPS` is set to your nginx/LB IP only
- [ ] `.env` file has `chmod 600` permissions
- [ ] `.env` is in `.gitignore` (never committed)
- [ ] PostgreSQL password is not the default
- [ ] Redis requires authentication (`requirepass` in redis.conf)
- [ ] Container runs as non-root user (Dockerfile: `USER resync`)
- [ ] HTTPS is configured in nginx (port 443 + TLS)

---

## Troubleshooting

**App fails to start: "relation does not exist"**
→ Database schema not initialized. Run: `python scripts/init_db.py`

**App fails to start: "AUTH_SECRET_KEY must be set in production"**
→ Set `APP_SECRET_KEY` in `.env`

**Prometheus metrics incorrect with multiple workers**
→ Set `PROMETHEUS_MULTIPROC_DIR` to a shared writable directory

**CORS errors in browser**
→ Add your frontend URL to `APP_CORS_ALLOWED_ORIGINS`
