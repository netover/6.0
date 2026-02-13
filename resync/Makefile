SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ── Variables ────────────────────────────────────────────────────────────────
VENV       := .venv
PIP        := $(VENV)/bin/pip
PYTHON     := $(VENV)/bin/python
UVICORN    := $(VENV)/bin/uvicorn
ALEMBIC    := $(VENV)/bin/alembic

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "RESYNC v6.2.0 — Development Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install all dependencies (including Docling)"
	@echo "    make install-dev      Install dev/test dependencies"
	@echo ""
	@echo "  Run:"
	@echo "    make dev              Start dev server (auto-reload)"
	@echo "    make run              Start production server"
	@echo ""
	@echo "  Database:"
	@echo "    make migrate          Run Alembic migrations"
	@echo "    make migrate-new      Create new migration (MSG=description)"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint             Run ruff linter"
	@echo "    make typecheck        Run mypy"
	@echo "    make test             Run unit tests"
	@echo "    make test-int         Run integration tests"
	@echo "    make check            Run lint + typecheck + test"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
.PHONY: venv install install-dev

venv:
	@test -d $(VENV) || python3.11 -m venv $(VENV)
	@$(PIP) install --upgrade pip wheel setuptools --quiet

install: venv
	@echo "Installing dependencies (includes Docling — may take 5-10 min)..."
	@$(PIP) install -r requirements.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu --quiet
	@echo "✓ All dependencies installed"

install-dev: venv
	@echo "Installing dev dependencies..."
	@$(PIP) install -r requirements-dev.txt --quiet
	@echo "✓ Dev dependencies installed"

# ── Run ──────────────────────────────────────────────────────────────────────
.PHONY: dev run

dev: venv
	$(UVICORN) resync.main:app --reload --host 0.0.0.0 --port 8000

run: venv
	$(VENV)/bin/gunicorn -c gunicorn_config.py resync.main:app

# ── Database ─────────────────────────────────────────────────────────────────
.PHONY: migrate migrate-new

migrate: venv
	$(ALEMBIC) upgrade head
	@echo "✓ Migrations applied"

migrate-new: venv
	@test -n "$(MSG)" || (echo "Usage: make migrate-new MSG='description'" && exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(MSG)"

# ── Quality ──────────────────────────────────────────────────────────────────
.PHONY: lint typecheck test test-int check

lint:
	ruff check .

typecheck:
	mypy resync/core/wiring.py resync/api

test: venv
	$(PYTHON) -m pytest -q

test-int: venv
	$(PYTHON) -m pytest -q -m integration

check: lint typecheck test
	@echo "✓ All checks passed"
