#!/bin/bash
# =============================================================================
# Resync v5.3.17 - Complete Semantic Cache Deployment Script
# =============================================================================
#
# This script deploys the complete semantic cache system including:
# 1. Redis Stack setup
# 2. Python dependencies (sentence-transformers)
# 3. Environment configuration
# 4. Model preloading
# 5. Verification
#
# Usage:
#   chmod +x scripts/deploy_semantic_cache.sh
#   ./scripts/deploy_semantic_cache.sh
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Resync v5.3.17 - Semantic Cache Deployment${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# Check if running in correct directory
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo -e "${RED}Error: Run this script from the resync project directory${NC}"
    exit 1
fi

# =============================================================================
# Step 1: Install Python Dependencies
# =============================================================================
install_python_deps() {
    echo -e "\n${BLUE}[1/5] Installing Python dependencies...${NC}"
    
    # Check if sentence-transformers is installed
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        echo -e "  ${GREEN}✓ sentence-transformers already installed${NC}"
    else
        echo "  Installing sentence-transformers..."
        pip install sentence-transformers --break-system-packages 2>/dev/null || \
        pip install sentence-transformers --user 2>/dev/null || \
        pip install sentence-transformers
        echo -e "  ${GREEN}✓ sentence-transformers installed${NC}"
    fi
    
    # Check for redis
    if python3 -c "import redis" 2>/dev/null; then
        echo -e "  ${GREEN}✓ redis-py already installed${NC}"
    else
        echo "  Installing redis..."
        pip install redis --break-system-packages 2>/dev/null || \
        pip install redis --user 2>/dev/null || \
        pip install redis
        echo -e "  ${GREEN}✓ redis installed${NC}"
    fi
}

# =============================================================================
# Step 2: Check/Setup Redis Stack
# =============================================================================
setup_redis() {
    echo -e "\n${BLUE}[2/5] Checking Redis Stack...${NC}"
    
    # Check if Redis is running
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e "  ${GREEN}✓ Redis is running${NC}"
        
        # Check for RediSearch module
        if redis-cli MODULE LIST 2>/dev/null | grep -qi "search\|ft"; then
            echo -e "  ${GREEN}✓ RediSearch module available${NC}"
        else
            echo -e "  ${YELLOW}⚠ RediSearch module not found - fallback mode will be used${NC}"
            echo -e "  ${YELLOW}  Run: sudo ./scripts/setup_redis_stack.sh${NC}"
        fi
    else
        echo -e "  ${YELLOW}⚠ Redis is not running${NC}"
        
        # Try to start Redis
        if command -v redis-server &> /dev/null; then
            echo "  Starting Redis..."
            redis-server --daemonize yes 2>/dev/null || true
            sleep 2
            
            if redis-cli ping 2>/dev/null | grep -q "PONG"; then
                echo -e "  ${GREEN}✓ Redis started${NC}"
            else
                echo -e "  ${RED}✗ Failed to start Redis${NC}"
                echo -e "  ${YELLOW}  Install Redis Stack: sudo ./scripts/setup_redis_stack.sh${NC}"
            fi
        else
            echo -e "  ${RED}✗ Redis not installed${NC}"
            echo -e "  ${YELLOW}  Install Redis Stack: sudo ./scripts/setup_redis_stack.sh${NC}"
        fi
    fi
}

# =============================================================================
# Step 3: Configure Environment
# =============================================================================
configure_env() {
    echo -e "\n${BLUE}[3/5] Configuring environment...${NC}"
    
    # Check if .env exists
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
            echo -e "  ${GREEN}✓ Created .env from .env.example${NC}"
        else
            touch "$ENV_FILE"
            echo -e "  ${YELLOW}⚠ Created empty .env file${NC}"
        fi
    fi
    
    # Check/add semantic cache config
    CACHE_VARS=(
        "SEMANTIC_CACHE_ENABLED=true"
        "SEMANTIC_CACHE_THRESHOLD=0.25"
        "SEMANTIC_CACHE_TTL=86400"
        "SEMANTIC_CACHE_MAX_ENTRIES=10000"
        "SEMANTIC_CACHE_REDIS_DB=3"
    )
    
    for var in "${CACHE_VARS[@]}"; do
        key="${var%%=*}"
        if ! grep -q "^$key=" "$ENV_FILE" 2>/dev/null; then
            echo "$var" >> "$ENV_FILE"
            echo -e "  ${GREEN}✓ Added $key${NC}"
        else
            echo -e "  ${GREEN}✓ $key already configured${NC}"
        fi
    done
    
    # Check/add RAG cross-encoder config
    RAG_VARS=(
        "RAG_CROSS_ENCODER_ON=true"
        "RAG_CROSS_ENCODER_MODEL=BAAI/bge-reranker-small"
        "RAG_CROSS_ENCODER_TOP_K=5"
        "RAG_CROSS_ENCODER_THRESHOLD=0.3"
    )
    
    for var in "${RAG_VARS[@]}"; do
        key="${var%%=*}"
        if ! grep -q "^$key=" "$ENV_FILE" 2>/dev/null; then
            echo "$var" >> "$ENV_FILE"
            echo -e "  ${GREEN}✓ Added $key${NC}"
        else
            echo -e "  ${GREEN}✓ $key already configured${NC}"
        fi
    done
}

# =============================================================================
# Step 4: Preload Models
# =============================================================================
preload_models() {
    echo -e "\n${BLUE}[4/5] Preloading AI models...${NC}"
    
    cd "$PROJECT_DIR"
    
    python3 << 'PYTHON_SCRIPT'
import sys
import time

print("  Loading embedding model for semantic cache...")
start = time.time()
try:
    from resync.core.cache.embedding_model import preload_model
    preload_model()
    print(f"  ✓ Embedding model loaded ({time.time() - start:.1f}s)")
except Exception as e:
    print(f"  ⚠ Embedding model: {e}")

print("  Loading cross-encoder for cache reranking...")
start = time.time()
try:
    from resync.core.cache.reranker import preload_reranker
    preload_reranker()
    print(f"  ✓ Cache reranker loaded ({time.time() - start:.1f}s)")
except Exception as e:
    print(f"  ⚠ Cache reranker: {e}")

print("  Loading cross-encoder for RAG reranking...")
start = time.time()
try:
    from resync.RAG.microservice.core.rag_reranker import preload_cross_encoder
    preload_cross_encoder()
    print(f"  ✓ RAG reranker loaded ({time.time() - start:.1f}s)")
except Exception as e:
    print(f"  ⚠ RAG reranker: {e}")

print("  Model preloading complete!")
PYTHON_SCRIPT
}

# =============================================================================
# Step 5: Verify Installation
# =============================================================================
verify_installation() {
    echo -e "\n${BLUE}[5/5] Verifying installation...${NC}"
    
    cd "$PROJECT_DIR"
    
    python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')

print("  Checking components...")

# Check semantic cache
try:
    from resync.core.cache.semantic_cache import SemanticCache
    print("  ✓ Semantic Cache module OK")
except Exception as e:
    print(f"  ✗ Semantic Cache: {e}")

# Check cache reranker
try:
    from resync.core.cache.reranker import is_reranker_available
    available = is_reranker_available()
    print(f"  ✓ Cache Reranker: {'Available' if available else 'Fallback mode'}")
except Exception as e:
    print(f"  ✗ Cache Reranker: {e}")

# Check RAG reranker
try:
    from resync.RAG.microservice.core.rag_reranker import is_cross_encoder_available
    available = is_cross_encoder_available()
    print(f"  ✓ RAG Reranker: {'Available' if available else 'Fallback mode'}")
except Exception as e:
    print(f"  ✗ RAG Reranker: {e}")

# Check Redis connection
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=3)
    r.ping()
    print("  ✓ Redis connection OK")
except Exception as e:
    print(f"  ⚠ Redis: {e}")

print("\n  Deployment verification complete!")
PYTHON_SCRIPT
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo -e "\n${GREEN}=============================================${NC}"
    echo -e "${GREEN}  Semantic Cache Deployment Complete!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
    echo "  Components deployed:"
    echo "    • Semantic Cache (Redis + sentence-transformers)"
    echo "    • Cache Reranker (bge-reranker-small)"
    echo "    • RAG Cross-Encoder (bge-reranker-small)"
    echo ""
    echo "  Configuration:"
    echo "    • Environment: $ENV_FILE"
    echo "    • Redis DB: 3"
    echo "    • Similarity threshold: 0.25"
    echo ""
    echo -e "${YELLOW}  Next steps:${NC}"
    echo "    1. Restart the Resync application"
    echo "    2. Access Admin > Semantic Cache to verify"
    echo "    3. Monitor cache hit rate in dashboard"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    install_python_deps
    setup_redis
    configure_env
    preload_models
    verify_installation
    print_summary
}

main "$@"
