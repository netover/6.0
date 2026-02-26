# Resync Go/No-Go Checklist (VM • Python 3.14)

This checklist is meant to be executed **before the first production deployment** and before each release.

## 1) Build & Import-Time Gates (must pass)

- [ ] **Install deps** (prod + dev):
  - `python3.14 -m pip install -r requirements.txt -r requirements-dev.txt`
- [ ] **Syntax check**: `python3.14 -m compileall -q resync`
- [ ] **Import-time wiring**: `python3.14 -c "from resync.main import app; print('import ok')"`
- [ ] **Run smoke script**: `./scripts/smoke_import_check.sh`

## 2) Security & Config Gates (must pass)

- [ ] `TRUSTED_HOSTS` set (even on private networks).
- [ ] Secrets present for auth/JWT (no auto-generated production secrets).
- [ ] Monitoring endpoints require auth (or are firewalled to admin networks).
- [ ] Rate limits enabled for `/auth/*` and WS connect.

## 3) Runtime Gates (must pass)

- [ ] `/liveness` returns 200.
- [ ] `/readiness` returns 200 (or 503 only when a required dependency is down).
- [ ] `/metrics` reachable by Prometheus (and protected if required).
- [ ] WS connect/disconnect works for authorized users.

## 4) Operations Gates (recommended)

- [ ] systemd service installed and enabled.
- [ ] `LimitNOFILE` configured (WebSockets consume file descriptors).
- [ ] graceful shutdown verified (SIGTERM → clean stop without hanging tasks).

## 5) CI / Supply Chain Gates (recommended)

- [ ] `ruff` passes.
- [ ] `mypy` / `pyright` passes.
- [ ] `pytest` passes with minimum coverage threshold.
- [ ] `pip-audit` (SCA) passes.
- [ ] SBOM generated and archived.
