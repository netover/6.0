#!/bin/bash
# scripts/create_core_structure.sh
# Creates the new core directory structure for refactoring

set -e

echo "📁 Creating new core structure..."
echo "================================="

BASE="resync/core"

# Platform
mkdir -p $BASE/platform/config
mkdir -p $BASE/platform/container
mkdir -p $BASE/platform/resilience
mkdir -p $BASE/platform/redis

# Observability
mkdir -p $BASE/observability/logging
mkdir -p $BASE/observability/alerting
mkdir -p $BASE/observability/tracing

# Security
mkdir -p $BASE/security/auth
mkdir -p $BASE/security/validation

# Retrieval
mkdir -p $BASE/retrieval/rag
mkdir -p $BASE/retrieval/context

# Agents
mkdir -p $BASE/agents/router
mkdir -p $BASE/agents/llm

# TWS
mkdir -p $BASE/tws/client
mkdir -p $BASE/tws/monitor
mkdir -p $BASE/tws/queries

# Shared
mkdir -p $BASE/shared/types
mkdir -p $BASE/shared/interfaces

# Create __init__.py files
echo "📝 Creating __init__.py files..."
find $BASE -type d -exec sh -c 'touch "$1/__init__.py" 2>/dev/null || true' _ {} \;

echo ""
echo "✅ Structure created successfully!"
echo ""
echo "📂 New directories:"
find $BASE -type d -name "platform" -o -name "observability" -o -name "security" -o -name "retrieval" -o -name "agents" -o -name "tws" -o -name "shared" 2>/dev/null | head -20

echo ""
echo "📋 Next steps:"
echo "1. Run: python scripts/refactor_helper.py validate"
echo "2. Start migrating files with: python scripts/refactor_helper.py move"
echo "3. Update imports after each batch"
