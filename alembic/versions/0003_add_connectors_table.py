"""add connectors table

Revision ID: 0003_add_connectors_table
Revises: 0002_add_tws_pattern_unique
Create Date: 2026-03-03 12:00:00.000000
"""

from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_add_connectors_table"
down_revision: str | None = "0002_add_tws_pattern_unique"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    # Check if table exists
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    
    if "connectors" not in tables:
        op.create_table(
            "connectors",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("type", sa.String(length=50), nullable=False),
            sa.Column("host", sa.String(length=255), nullable=True),
            sa.Column("port", sa.Integer(), nullable=True),
            sa.Column("username", sa.String(length=255), nullable=True),
            sa.Column("password", sa.Text(), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("status", sa.String(length=50), nullable=False, server_default="unknown"),
            sa.Column("last_check", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.UniqueConstraint("name", name="uq_connector_name")
        )

def downgrade() -> None:
    op.drop_table("connectors")
