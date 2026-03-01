from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None

_SCHEMAS = ["tws", "context", "audit", "analytics", "learning", "metrics"]

def upgrade() -> None:
    for s in _SCHEMAS:
        op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {s}"))
    bind = op.get_bind()
    from resync.core.database.models.base import Base
    from resync.core.database.models import get_all_models
    get_all_models()
    Base.metadata.create_all(bind=bind)
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_kg_edges_source ON learning.kg_edges (source_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_kg_edges_target ON learning.kg_edges (target_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_kg_edges_relation ON learning.kg_edges (relation_type)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_kg_nodes_type ON learning.kg_nodes (node_type)"))

def downgrade() -> None:
    pass
