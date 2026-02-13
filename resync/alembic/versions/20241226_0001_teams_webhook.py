"""Teams webhook tables

Revision ID: 20241226_0001
Revises: 
Create Date: 2024-12-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20241226_0001'
down_revision = None  # Ajustar para última migration
branch_labels = None
depends_on = None


def upgrade():
    # TeamsWebhookUser table
    op.create_table(
        'teams_webhook_users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('aad_object_id', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='viewer'),
        sa.Column('can_execute_commands', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email', name='uq_teams_webhook_users_email')
    )
    op.create_index('idx_teams_webhook_users_email', 'teams_webhook_users', ['email'])
    op.create_index('idx_teams_webhook_users_aad', 'teams_webhook_users', ['aad_object_id'])
    
    # TeamsWebhookAudit table
    op.create_table(
        'teams_webhook_audit',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_email', sa.String(length=255), nullable=True),
        sa.Column('user_name', sa.String(length=255), nullable=True),
        sa.Column('channel_id', sa.String(length=255), nullable=True),
        sa.Column('channel_name', sa.String(length=255), nullable=True),
        sa.Column('message_text', sa.Text(), nullable=True),
        sa.Column('command_type', sa.String(length=50), nullable=True),
        sa.Column('was_authorized', sa.Boolean(), nullable=True),
        sa.Column('response_sent', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_teams_webhook_audit_email', 'teams_webhook_audit', ['user_email'])
    op.create_index('idx_teams_webhook_audit_timestamp', 'teams_webhook_audit', ['timestamp'])


def downgrade():
    op.drop_index('idx_teams_webhook_audit_timestamp', table_name='teams_webhook_audit')
    op.drop_index('idx_teams_webhook_audit_email', table_name='teams_webhook_audit')
    op.drop_table('teams_webhook_audit')
    
    op.drop_index('idx_teams_webhook_users_aad', table_name='teams_webhook_users')
    op.drop_index('idx_teams_webhook_users_email', table_name='teams_webhook_users')
    op.drop_table('teams_webhook_users')
