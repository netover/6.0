#!/bin/bash
# Teams Webhook Quick Setup Script

echo "=========================================="
echo "Teams Webhook Setup"
echo "=========================================="
echo ""

# Check if database is configured
if [ -z "$DATABASE_URL" ]; then
    echo "⚠️  DATABASE_URL não configurado!"
    echo "Configure no .env antes de continuar."
    exit 1
fi

# Run migration
echo "🔧 Aplicando migration..."
alembic upgrade head

# Create admin user
echo "👤 Criando usuário admin de exemplo..."
python3 << PYEND
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from resync.core.database.models.teams import TeamsWebhookUser
from resync.settings import settings
import os

engine = create_engine(os.getenv("DATABASE_URL"))
session = Session(engine)

# Create example admin
admin = TeamsWebhookUser(
    email="admin@example.com",
    name="Admin User",
    role="admin",
    can_execute_commands=True,
    is_active=True
)
session.add(admin)
session.commit()
print("✅ Admin user created: admin@example.com")
PYEND

echo ""
echo "✅ Setup completo!"
echo ""
echo "Próximos passos:"
echo "1. Configure o webhook no Teams"
echo "2. Copie o security token para o .env"
echo "3. Reinicie o servidor"
