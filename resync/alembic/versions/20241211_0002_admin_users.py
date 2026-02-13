"""Add admin_users table

Revision ID: 20241211_0002
Revises: 20241210_0001_001_initial_schema_initial_schema
Create Date: 2024-12-11 14:30:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "20241211_0002_admin_users"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create core schema if not exists
    op.execute("CREATE SCHEMA IF NOT EXISTS core")

    # Create admin_users table
    op.create_table(
        "admin_users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=50), server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("is_verified", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("failed_login_attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
        schema="core",
    )

    # Create indexes
    op.create_index("idx_admin_users_username", "admin_users", ["username"], schema="core")
    op.create_index("idx_admin_users_email", "admin_users", ["email"], schema="core")
    op.create_index("idx_admin_users_role", "admin_users", ["role"], schema="core")
    op.create_index("idx_admin_users_is_active", "admin_users", ["is_active"], schema="core")


def downgrade() -> None:
    op.drop_index("idx_admin_users_is_active", table_name="admin_users", schema="core")
    op.drop_index("idx_admin_users_role", table_name="admin_users", schema="core")
    op.drop_index("idx_admin_users_email", table_name="admin_users", schema="core")
    op.drop_index("idx_admin_users_username", table_name="admin_users", schema="core")
    op.drop_table("admin_users", schema="core")
