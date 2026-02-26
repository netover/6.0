"""
SQLAlchemy Eager Loading Guide - N+1 Query Prevention

Este arquivo documenta como usar eager loading corretamente para prevenir
N+1 queries no Resync.

PROBLEMA: N+1 Queries
=====================
Sem eager loading, acessar relationships gera queries adicionais:

```python
# ❌ PROBLEMA: Gera 1 query + N queries (N+1)
snapshots = await session.execute(select(TWSSnapshot))
for snapshot in snapshots.scalars():
    print(snapshot.job_statuses)  # Query SQL adicional por snapshot!
```

Com 100 snapshots, isso gera 101 queries SQL!

SOLUÇÃO: Eager Loading
======================

1. Para one-to-many (1:N): Use selectinload()
```python
from sqlalchemy.orm import selectinload

# ✅ CORRETO: 2 queries total (1 para snapshots + 1 IN clause para statuses)
result = await session.execute(
    select(TWSSnapshot)
    .options(selectinload(TWSSnapshot.job_statuses))
    .limit(100)
)
snapshots = result.scalars().all()
```

2. Para many-to-one (N:1): Use joinedload()
```python
from sqlalchemy.orm import joinedload

# ✅ CORRETO: 1 query com JOIN
result = await session.execute(
    select(TWSJobStatus)
    .options(joinedload(TWSJobStatus.snapshot))
    .limit(100)
)
statuses = result.unique().scalars().all()  # IMPORTANTE: .unique()!
```

3. Para relacionamentos aninhados: Combine strategies
```python
from sqlalchemy.orm import selectinload, joinedload

# ✅ CORRETO: Carrega snapshot -> job_statuses -> workstation em 3 queries
result = await session.execute(
    select(TWSSnapshot)
    .options(
        selectinload(TWSSnapshot.job_statuses).joinedload(
            TWSJobStatus.workstation
        )
    )
)
```

DETECÇÃO AUTOMÁTICA: lazy="raise"
=================================

Todos os relationships em `stores.py` estão configurados com `lazy="raise"`.
Isso força ERRO se você tentar acessar relationship sem eager loading:

```python
snapshot = await session.get(TWSSnapshot, 1)
print(snapshot.job_statuses)  # ❌ ERRO: "relationship is not eager loaded"
```

Isso é INTENCIONAL! Previne N+1 queries em produção.

Para acessar, use eager loading:

```python
result = await session.execute(
    select(TWSSnapshot)
    .where(TWSSnapshot.id == 1)
    .options(selectinload(TWSSnapshot.job_statuses))
)
snapshot = result.scalar_one()
print(snapshot.job_statuses)  # ✅ OK: eager loaded
```

HELPERS ÚTEIS
============

Use essas funções helper para queries comuns:

```python
from resync.core.database.eager_loading import (
    with_job_statuses,
    with_snapshot,
)

# Query snapshots com job_statuses eager loaded
snapshots = await session.execute(
    select(TWSSnapshot).options(with_job_statuses())
)

# Query job_statuses com snapshot eager loaded
statuses = await session.execute(
    select(TWSJobStatus).options(with_snapshot())
)
```

PERFORMANCE TIPS
===============

1. **Limite resultados ANTES de eager loading**
```python
# ✅ BOM: Limita antes
select(TWSSnapshot).limit(100).options(selectinload(...))

# ❌ RUIM: Carrega tudo, depois limita na app
result = select(TWSSnapshot).options(selectinload(...))
snapshots[:100]  # Muito tarde!
```

2. **Use contains_eager() quando já tem JOIN**
```python
from sqlalchemy.orm import contains_eager

# Se já tem JOIN na query:
result = await session.execute(
    select(TWSJobStatus)
    .join(TWSJobStatus.snapshot)
    .options(contains_eager(TWSJobStatus.snapshot))
)
```

3. **Evite eager loading desnecessário**
```python
# Se não vai acessar relationships, não carregue:
snapshots = await session.execute(
    select(TWSSnapshot).limit(10)
)  # OK se não acessar job_statuses
```

REFERÊNCIAS
===========
- SQLAlchemy Docs: https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html
- N+1 Problem: https://stackoverflow.com/questions/97197/what-is-the-n1-selects-problem
- Relationship Loading Techniques: https://docs.sqlalchemy.org/en/20/orm/loading_relationships.html

ÚLTIMA ATUALIZAÇÃO
==================
Data: 23 Dezembro 2024
Versão: v5.9.6
"""

# Helpers para eager loading
from sqlalchemy.orm import joinedload, selectinload

def with_job_statuses():
    """Helper para eager load job_statuses em TWSSnapshot."""
    from resync.core.database.models.stores import TWSSnapshot

    return selectinload(TWSSnapshot.job_statuses)

def with_snapshot():
    """Helper para eager load snapshot em TWSJobStatus."""
    from resync.core.database.models.stores import TWSJobStatus

    return joinedload(TWSJobStatus.snapshot)

def with_full_snapshot_data():
    """Helper para eager load snapshot completo com todos relacionamentos."""
    from resync.core.database.models.stores import TWSSnapshot

    return selectinload(TWSSnapshot.job_statuses)

# Exemplo de uso em repository:
"""
from resync.core.database.eager_loading import with_job_statuses

class TWSRepository:
    async def get_snapshots_with_statuses(self, limit: int = 100):
        result = await self.session.execute(
            select(TWSSnapshot)
            .options(with_job_statuses())
            .order_by(TWSSnapshot.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()
"""
