#!/bin/bash
################################################################################
# Test Script - Workstation Metrics Simulator
# 
# Simula múltiplas FTAs enviando métricas para Resync
# Útil para testar o endpoint antes de fazer deployment real
#
# Uso:
#   ./test_metrics_simulator.sh <RESYNC_URL> <API_KEY> <NUM_WORKSTATIONS>
#
# Exemplo:
#   ./test_metrics_simulator.sh https://resync.company.com rsk_abc123 5
#
# Versão: 1.0.0
################################################################################

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

RESYNC_URL="${1:-https://resync.company.com/api/v1/metrics/workstation}"
API_KEY="${2:-your-api-key-here}"
NUM_WORKSTATIONS="${3:-3}"

# ============================================================================
# CORES
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNÇÕES
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

generate_random_metrics() {
    # CPU: 10-95%
    CPU=$(awk -v min=10 -v max=95 'BEGIN{srand(); print min+rand()*(max-min)}')
    
    # Memory: 20-90%
    MEMORY=$(awk -v min=20 -v max=90 'BEGIN{srand(); print min+rand()*(max-min)}')
    
    # Disk: 30-85%
    DISK=$(awk -v min=30 -v max=85 'BEGIN{srand(); print min+rand()*(max-min)}')
    
    # Load Average: 0.5-8.0
    LOAD=$(awk -v min=0.5 -v max=8.0 'BEGIN{srand(); print min+rand()*(max-min)}')
    
    # Round to 2 decimals
    CPU=$(printf "%.2f" $CPU)
    MEMORY=$(printf "%.2f" $MEMORY)
    DISK=$(printf "%.2f" $DISK)
    LOAD=$(printf "%.2f" $LOAD)
    
    echo "$CPU|$MEMORY|$DISK|$LOAD"
}

send_metrics() {
    local ws_name=$1
    local cpu=$2
    local memory=$3
    local disk=$4
    local load=$5
    
    # Timestamp UTC
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # JSON payload
    JSON_PAYLOAD=$(cat <<EOF
{
  "workstation": "$ws_name",
  "timestamp": "$TIMESTAMP",
  "metrics": {
    "cpu_percent": $cpu,
    "memory_percent": $memory,
    "disk_percent": $disk,
    "load_avg_1min": $load,
    "cpu_count": 8,
    "total_memory_gb": 32,
    "total_disk_gb": 500
  },
  "metadata": {
    "os_type": "linux-gnu",
    "hostname": "$ws_name.test.local",
    "collector_version": "1.0.0-test"
  }
}
EOF
)
    
    # Send HTTP POST
    HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        --max-time 10 \
        --data "$JSON_PAYLOAD" \
        "$RESYNC_URL" 2>&1)
    
    HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        log_success "$ws_name: CPU=${cpu}% MEM=${memory}% DISK=${disk}% (HTTP $HTTP_CODE)"
        return 0
    else
        log_error "$ws_name: Failed (HTTP $HTTP_CODE)"
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo ""
    echo "============================================="
    echo "  Workstation Metrics Simulator"
    echo "============================================="
    echo ""
    
    log_info "Configuration:"
    echo "  Resync URL:      $RESYNC_URL"
    echo "  API Key:         ${API_KEY:0:20}..."
    echo "  Workstations:    $NUM_WORKSTATIONS"
    echo ""
    
    # Test API connectivity
    log_info "Testing API connectivity..."
    HEALTH_CHECK=$(curl -s "${RESYNC_URL/\/workstation/\/health}" 2>&1)
    
    if echo "$HEALTH_CHECK" | grep -q "healthy"; then
        log_success "API is reachable and healthy"
    else
        log_error "API is not reachable or unhealthy"
        log_error "Response: $HEALTH_CHECK"
        exit 1
    fi
    
    echo ""
    log_info "Sending metrics from $NUM_WORKSTATIONS workstations..."
    echo ""
    
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    
    # Send metrics from multiple workstations
    for i in $(seq 1 $NUM_WORKSTATIONS); do
        WS_NAME=$(printf "TEST-WS-%02d" $i)
        
        # Generate random metrics
        METRICS=$(generate_random_metrics)
        CPU=$(echo "$METRICS" | cut -d'|' -f1)
        MEMORY=$(echo "$METRICS" | cut -d'|' -f2)
        DISK=$(echo "$METRICS" | cut -d'|' -f3)
        LOAD=$(echo "$METRICS" | cut -d'|' -f4)
        
        # Send
        if send_metrics "$WS_NAME" "$CPU" "$MEMORY" "$DISK" "$LOAD"; then
            ((SUCCESS_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
        
        # Small delay between requests
        sleep 0.5
    done
    
    echo ""
    echo "============================================="
    echo "  Results"
    echo "============================================="
    echo ""
    log_success "Successful: $SUCCESS_COUNT"
    if [ $FAILED_COUNT -gt 0 ]; then
        log_error "Failed:     $FAILED_COUNT"
    else
        echo "  Failed:     0"
    fi
    echo ""
    
    if [ $SUCCESS_COUNT -eq $NUM_WORKSTATIONS ]; then
        log_success "All metrics sent successfully! ✅"
        echo ""
        log_info "Next steps:"
        echo "  1. Check database:"
        echo "     psql -U resync -d resync -c \"SELECT * FROM workstation_metrics_history WHERE workstation LIKE 'TEST-WS-%' ORDER BY received_at DESC LIMIT 10;\""
        echo ""
        echo "  2. Query metrics:"
        echo "     curl -H 'X-API-Key: $API_KEY' '${RESYNC_URL/\/workstation/\/workstation\/TEST-WS-01}?hours=1'"
        echo ""
        exit 0
    else
        log_error "Some metrics failed to send"
        exit 1
    fi
}

# ============================================================================
# EXECUTION
# ============================================================================

main
