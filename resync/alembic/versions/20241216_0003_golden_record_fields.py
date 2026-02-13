"""Add Golden Record fields to feedback table.

v5.2.3.20: Implements the Golden Record flow for knowledge incorporation.

Adds:
- user_correction: Expert's corrected answer
- curation_status: Status of the feedback curation (pending/approved/rejected/incorporated)
- approved_by: ID of the reviewer who approved
- approved_at: Timestamp of approval
- incorporated_doc_id: Document ID in vector store (if incorporated)

Revision ID: 20241216_0003
Revises: 20241211_0002
Create Date: 2024-12-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20241216_0003'
down_revision: Union[str, None] = '20241211_0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add Golden Record fields to feedback table."""
    # Add new columns
    op.add_column(
        'feedback',
        sa.Column('user_correction', sa.Text(), nullable=True),
        schema='learning'
    )
    op.add_column(
        'feedback',
        sa.Column('curation_status', sa.String(50), nullable=False, server_default='pending'),
        schema='learning'
    )
    op.add_column(
        'feedback',
        sa.Column('approved_by', sa.String(255), nullable=True),
        schema='learning'
    )
    op.add_column(
        'feedback',
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        schema='learning'
    )
    op.add_column(
        'feedback',
        sa.Column('incorporated_doc_id', sa.String(255), nullable=True),
        schema='learning'
    )
    
    # Add index for curation_status for efficient filtering
    op.create_index(
        'idx_feedback_curation_status',
        'feedback',
        ['curation_status'],
        schema='learning'
    )


def downgrade() -> None:
    """Remove Golden Record fields from feedback table."""
    # Drop index first
    op.drop_index('idx_feedback_curation_status', table_name='feedback', schema='learning')
    
    # Drop columns
    op.drop_column('feedback', 'incorporated_doc_id', schema='learning')
    op.drop_column('feedback', 'approved_at', schema='learning')
    op.drop_column('feedback', 'approved_by', schema='learning')
    op.drop_column('feedback', 'curation_status', schema='learning')
    op.drop_column('feedback', 'user_correction', schema='learning')
