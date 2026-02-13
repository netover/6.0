# Database migrations (Alembic)

This project uses **Alembic** for relational database schema migrations.

Why:
- versioned, reviewable schema changes
- repeatable upgrades/downgrades across environments
- supports autogenerate from SQLAlchemy models

## Configure

Set `DATABASE_URL` (recommended):

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/dbname"
```

## Common commands

Create a new revision (autogenerate):

```bash
alembic revision --autogenerate -m "describe change"
```

Apply migrations:

```bash
alembic upgrade head
```

Downgrade one step:

```bash
alembic downgrade -1
```

## Async setup

`alembic/env.py` is based on Alembic's **official async template** and uses
`async_engine_from_config` + `run_sync` to execute migrations. See Alembic docs
for details.
