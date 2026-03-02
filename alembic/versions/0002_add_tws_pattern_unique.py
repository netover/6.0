"""add tws pattern unique constraint

Revision ID: 0002_add_tws_pattern_unique
Revises: 0001_init_baseline
Create Date: 2026-03-02 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_tws_pattern_unique"
down_revision: str | None = "0001_init_baseline"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_tws_pattern_job",
        "tws_patterns",
        ["pattern_type", "job_name"],
        schema="tws",
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_tws_pattern_job",
        "tws_patterns",
        schema="tws",
        type_="unique",
    )
