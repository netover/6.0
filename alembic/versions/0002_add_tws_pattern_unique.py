"""add tws pattern unique constraint

Revision ID: 0002_add_tws_pattern_unique
Revises: 0001_init_baseline
Create Date: 2026-03-02 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_tws_pattern_unique"
down_revision: str | None = "0001_init"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    unique_constraints = inspector.get_unique_constraints("tws_patterns", schema="tws")
    constraint_names = {constraint.get("name") for constraint in unique_constraints}

    if "uq_tws_pattern_job" not in constraint_names:
        op.create_unique_constraint(
            "uq_tws_pattern_job",
            "tws_patterns",
            ["pattern_type", "job_name"],
            schema="tws",
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    unique_constraints = inspector.get_unique_constraints("tws_patterns", schema="tws")
    constraint_names = {constraint.get("name") for constraint in unique_constraints}

    if "uq_tws_pattern_job" in constraint_names:
        op.drop_constraint(
            "uq_tws_pattern_job",
            "tws_patterns",
            schema="tws",
            type_="unique",
        )
