"""Create workstation_metrics_history table

Revision ID: add_workstation_metrics
Revises: previous_revision
Create Date: 2024-12-25 10:00:00.000000

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = 'add_workstation_metrics'
down_revision = 'previous_revision'  # AJUSTAR para última migration
branch_labels = None
depends_on = None


def upgrade():
    """Create workstation_metrics_history table."""

    # Criar tabela
    op.create_table(
        'workstation_metrics_history',

        # Primary key
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),

        # Identificação
        sa.Column('workstation', sa.String(length=100), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),

        # Métricas principais (required)
        sa.Column('cpu_percent', sa.Float(), nullable=False),
        sa.Column('memory_percent', sa.Float(), nullable=False),
        sa.Column('disk_percent', sa.Float(), nullable=False),

        # Métricas adicionais (optional)
        sa.Column('load_avg_1min', sa.Float(), nullable=True),
        sa.Column('cpu_count', sa.Integer(), nullable=True),
        sa.Column('total_memory_gb', sa.Integer(), nullable=True),
        sa.Column('total_disk_gb', sa.Integer(), nullable=True),

        # Metadata
        sa.Column('os_type', sa.String(length=50), nullable=True),
        sa.Column('hostname', sa.String(length=255), nullable=True),
        sa.Column('collector_version', sa.String(length=20), nullable=True),

        # Audit
        sa.Column('received_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),

        # Constraints
        sa.PrimaryKeyConstraint('id')
    )

    # Indexes simples
    op.create_index(
        'ix_workstation_metrics_history_workstation',
        'workstation_metrics_history',
        ['workstation'],
        unique=False
    )

    op.create_index(
        'ix_workstation_metrics_history_timestamp',
        'workstation_metrics_history',
        ['timestamp'],
        unique=False
    )

    op.create_index(
        'ix_workstation_metrics_history_received_at',
        'workstation_metrics_history',
        ['received_at'],
        unique=False
    )

    # Index composto para queries comuns (workstation + time range)
    op.create_index(
        'ix_workstation_timestamp',
        'workstation_metrics_history',
        ['workstation', 'timestamp'],
        unique=False,
        postgresql_using='btree'
    )

    # Comment na tabela
    op.execute("""
        COMMENT ON TABLE workstation_metrics_history IS
        'Histórico de métricas de CPU, memory e disk das workstations TWS coletadas via scripts bash'
    """)

    # Comments nas colunas
    op.execute("""
        COMMENT ON COLUMN workstation_metrics_history.workstation IS
        'Nome/identificador da workstation TWS (ex: WS-PROD-01)'
    """)

    op.execute("""
        COMMENT ON COLUMN workstation_metrics_history.timestamp IS
        'Timestamp quando métricas foram coletadas na workstation (UTC)'
    """)

    op.execute("""
        COMMENT ON COLUMN workstation_metrics_history.received_at IS
        'Timestamp quando Resync recebeu as métricas (UTC)'
    """)


def downgrade():
    """Drop workstation_metrics_history table."""

    # Drop indexes
    op.drop_index('ix_workstation_timestamp', table_name='workstation_metrics_history')
    op.drop_index('ix_workstation_metrics_history_received_at', table_name='workstation_metrics_history')
    op.drop_index('ix_workstation_metrics_history_timestamp', table_name='workstation_metrics_history')
    op.drop_index('ix_workstation_metrics_history_workstation', table_name='workstation_metrics_history')

    # Drop table
    op.drop_table('workstation_metrics_history')
