#!/bin/bash
# ============================================================================
# Validação v5.3.18 - Script de Verificação Pré-Deploy
# ============================================================================

echo "========================================================================"
echo "  VALIDAÇÃO v5.3.18 - Resync Project"
echo "========================================================================"
echo ""
echo "Data: $(date)"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contadores
PASSED=0
FAILED=0
WARNINGS=0

# Função para log de sucesso
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    PASSED=$((PASSED + 1))
}

# Função para log de falha
fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

# Função para log de aviso
warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    WARNINGS=$((WARNINGS + 1))
}

# ============================================================================
# 1. VERIFICAÇÃO DE SINTAXE
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "1. Verificando sintaxe Python..."
echo "────────────────────────────────────────────────────────────────────────"

SYNTAX_ERRORS=$(find resync/ -name "*.py" -exec python3 -m py_compile {} \; 2>&1 || true)

if [ -z "$SYNTAX_ERRORS" ]; then
    pass "Sintaxe Python OK - nenhum erro encontrado"
else
    fail "Erros de sintaxe encontrados:"
    echo "$SYNTAX_ERRORS"
fi

# ============================================================================
# 2. VERIFICAÇÃO DE IMPORTS CRÍTICOS
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "2. Verificando imports críticos..."
echo "────────────────────────────────────────────────────────────────────────"

python3 -c "
from resync.core.utils.common_error_handlers import handle_parsing_errors
print('   common_error_handlers: OK')
" && pass "common_error_handlers importado" || fail "common_error_handlers falhou"

python3 -c "
from resync.core.utils.json_parser import parse_llm_json_response
print('   json_parser: OK')
" && pass "json_parser importado" || fail "json_parser falhou"

python3 -c "
from resync.core.ia_auditor import analyze_and_flag_memories
print('   ia_auditor: OK')
" 2>/dev/null && pass "ia_auditor importado" || warn "ia_auditor não disponível (dependências)"

python3 -c "
from resync.models.tws_validators import validate_job_status
print('   tws_validators: OK')
" && pass "tws_validators importado" || fail "tws_validators falhou"

python3 -c "
from resync.core.embedding_router import EmbeddingRouter
print('   embedding_router: OK')
" && pass "embedding_router importado" || fail "embedding_router falhou"

python3 -c "
from resync.core.unified_retrieval import UnifiedRetrievalService
print('   unified_retrieval: OK')
" && pass "unified_retrieval importado" || fail "unified_retrieval falhou"

python3 -c "
from resync.RAG.microservice.core.config import CFG
print('   RAG config: OK')
print(f'     - cross_encoder_enabled: {CFG.enable_cross_encoder}')
print(f'     - cross_encoder_model: {CFG.cross_encoder_model}')
" && pass "RAG config importado" || fail "RAG config falhou"

# ============================================================================
# 3. VALIDAÇÃO DE EXEMPLOS DE INTENT
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "3. Validando exemplos de intent..."
echo "────────────────────────────────────────────────────────────────────────"

INTENT_COUNT=$(python3 -c "
from resync.core.intent_examples_expanded import get_example_stats
stats = get_example_stats()
print(stats['total_examples'])
")

echo "   Total de exemplos: $INTENT_COUNT"

if [ "$INTENT_COUNT" -ge 200 ]; then
    pass "Exemplos de intent OK ($INTENT_COUNT >= 200)"
else
    fail "Exemplos insuficientes ($INTENT_COUNT < 200)"
fi

# ============================================================================
# 4. VERIFICAÇÃO DO CACHE WARMER
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "4. Verificando Cache Warmer..."
echo "────────────────────────────────────────────────────────────────────────"

python3 -c "
from resync.core.cache.cache_warmer import CacheWarmer, get_cache_warmer
warmer = CacheWarmer()
counts = warmer.get_static_queries_count()
print(f'   Queries estáticas: {counts[\"total\"]}')
print(f'     - Priority 1 (alta): {counts[\"priority_1\"]}')
print(f'     - Priority 2 (média): {counts[\"priority_2\"]}')
print(f'     - Priority 3 (baixa): {counts[\"priority_3\"]}')
" && pass "Cache warmer OK" || fail "Cache warmer falhou"

# ============================================================================
# 5. VERIFICAÇÃO DO DASHBOARD
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "5. Verificando Dashboard de Métricas..."
echo "────────────────────────────────────────────────────────────────────────"

python3 -c "
from resync.fastapi_app.api.v1.routes.admin_metrics_dashboard import router, DashboardMetrics
print(f'   Router prefix: {router.prefix}')
print(f'   Endpoints: {len(router.routes)}')
" && pass "Dashboard routes OK" || fail "Dashboard routes falhou"

# ============================================================================
# 6. TESTES DE INTEGRAÇÃO
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "6. Executando testes de integração v5.3.17..."
echo "────────────────────────────────────────────────────────────────────────"

TEST_RESULT=$(python3 -m pytest tests/integration/test_v5317_integration.py tests/integration/test_full_pipeline.py -v --tb=short 2>&1 | tail -5)
echo "$TEST_RESULT"

if echo "$TEST_RESULT" | grep -q "passed"; then
    TESTS_PASSED=$(echo "$TEST_RESULT" | grep -oP '\d+(?= passed)')
    pass "Testes de integração OK ($TESTS_PASSED passed)"
else
    fail "Testes de integração falharam"
fi

# ============================================================================
# 7. VERIFICAÇÃO DE CONFIGURAÇÕES
# ============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo "7. Verificando configurações..."
echo "────────────────────────────────────────────────────────────────────────"

# Verificar .env.example
if [ -f ".env.example" ]; then
    # Verificar variáveis novas do v5.3.17/18
    if grep -q "RAG_CROSS_ENCODER" .env.example; then
        pass ".env.example contém RAG_CROSS_ENCODER"
    else
        warn ".env.example não contém RAG_CROSS_ENCODER"
    fi
    
    if grep -q "SEMANTIC_CACHE" .env.example; then
        pass ".env.example contém SEMANTIC_CACHE"
    else
        warn ".env.example não contém SEMANTIC_CACHE"
    fi
else
    warn ".env.example não encontrado"
fi

# ============================================================================
# RESUMO
# ============================================================================
echo ""
echo "========================================================================"
echo "  RESUMO DA VALIDAÇÃO"
echo "========================================================================"
echo ""
echo -e "  ${GREEN}✅ Passou:${NC}   $PASSED"
echo -e "  ${RED}❌ Falhou:${NC}   $FAILED"
echo -e "  ${YELLOW}⚠️  Avisos:${NC}  $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ VALIDAÇÃO v5.3.18 COMPLETA - PRONTO PARA DEPLOY!${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ VALIDAÇÃO v5.3.18 FALHOU - CORRIGIR ERROS ANTES DO DEPLOY${NC}"
    echo -e "${RED}══════════════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
