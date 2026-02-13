"""Teams notifications tables

Revision ID: 20241226_0002
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20241226_0002'
down_revision = '20241226_0001'

def upgrade():
    op.create_table('teams_channels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('webhook_url', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('color', sa.String(20), server_default='#0078D4'),
        sa.Column('icon', sa.String(20), server_default='📢'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('last_notification_sent', sa.DateTime()),
        sa.Column('notification_count', sa.Integer(), server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    op.create_table('teams_job_mappings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_name', sa.String(255), nullable=False),
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('priority', sa.Integer(), server_default='0'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_name')
    )
    
    op.create_table('teams_pattern_rules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pattern', sa.String(255), nullable=False),
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.String(500)),
        sa.Column('priority', sa.Integer(), server_default='0'),
        sa.Column('pattern_type', sa.String(20), server_default='glob'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('match_count', sa.Integer(), server_default='0'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('teams_notification_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('notify_on_status', postgresql.JSON(), server_default='["ABEND", "ERROR", "FAILED"]'),
        sa.Column('quiet_hours_enabled', sa.Boolean(), server_default='false'),
        sa.Column('quiet_hours_start', sa.String(5)),
        sa.Column('quiet_hours_end', sa.String(5)),
        sa.Column('rate_limit_enabled', sa.Boolean(), server_default='true'),
        sa.Column('max_notifications_per_job', sa.Integer(), server_default='5'),
        sa.Column('rate_limit_window_minutes', sa.Integer(), server_default='60'),
        sa.Column('default_channel_id', sa.Integer()),
        sa.Column('include_mention_on_critical', sa.Boolean(), server_default='false'),
        sa.Column('mention_text', sa.String(100), server_default='@Operations'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('teams_notification_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.Integer()),
        sa.Column('channel_name', sa.String(100)),
        sa.Column('job_name', sa.String(255)),
        sa.Column('job_status', sa.String(50)),
        sa.Column('instance_name', sa.String(100)),
        sa.Column('return_code', sa.Integer()),
        sa.Column('error_message', sa.Text()),
        sa.Column('notification_sent', sa.Boolean(), server_default='false'),
        sa.Column('response_status', sa.Integer()),
        sa.Column('error', sa.Text()),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_teams_notification_log_timestamp', 'teams_notification_log', ['timestamp'])

def downgrade():
    op.drop_table('teams_notification_log')
    op.drop_table('teams_notification_config')
    op.drop_table('teams_pattern_rules')
    op.drop_table('teams_job_mappings')
    op.drop_table('teams_channels')
