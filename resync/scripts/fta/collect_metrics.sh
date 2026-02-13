#!/bin/bash
################################################################################
# TWS FTA Metrics Collector
# 
# Coleta métricas de CPU, Memory e Disk da workstation TWS e envia para Resync
#
# Instalação:
#   1. Copiar para: /opt/tws/scripts/collect_metrics.sh
#   2. Dar permissão: chmod +x /opt/tws/scripts/collect_metrics.sh
#   3. Configurar cron: */5 * * * * /opt/tws/scripts/collect_metrics.sh
#
# Versão: 1.0.0
# Data: 2024-12-25
################################################################################

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# URL do Resync API (AJUSTAR PARA SEU AMBIENTE!)
RESYNC_URL="https://resync.company.com/api/v1/metrics/workstation"

# API Key para autenticação (AJUSTAR!)
API_KEY="your-api-key-here"

# Identificador da workstation (auto-detecta ou usar fixo)
WORKSTATION_NAME=$(hostname -s)

# Log file
LOG_FILE="/var/log/tws_metrics_collector.log"

# Timeout para HTTP request (segundos)
TIMEOUT=10

# ============================================================================
# FUNÇÕES
# ============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_cpu_usage() {
    # Método 1: top (mais compatível)
    if command -v top &> /dev/null; then
        # Linux
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            CPU=$(top -bn2 -d 0.5 | grep "Cpu(s)" | tail -1 | awk '{print $2}' | cut -d'%' -f1)
        # macOS
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            CPU=$(top -l 2 | grep "CPU usage" | tail -1 | awk '{print $3}' | cut -d'%' -f1)
        # AIX
        elif [[ "$OSTYPE" == "aix"* ]]; then
            CPU=$(topas -d 1 -n 2 | grep "User" | tail -1 | awk '{print $2}')
        fi
    fi
    
    # Método 2: mpstat (se disponível)
    if [ -z "$CPU" ] && command -v mpstat &> /dev/null; then
        CPU=$(mpstat 1 1 | tail -1 | awk '{print 100 - $NF}')
    fi
    
    # Método 3: sar (fallback)
    if [ -z "$CPU" ] && command -v sar &> /dev/null; then
        CPU=$(sar 1 1 | tail -1 | awk '{print 100 - $NF}')
    fi
    
    # Validação
    if [ -z "$CPU" ]; then
        log_message "ERROR: Could not detect CPU usage"
        CPU=0
    fi
    
    # Arredonda para 2 casas decimais
    CPU=$(printf "%.2f" "$CPU")
    echo "$CPU"
}

get_memory_usage() {
    # Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v free &> /dev/null; then
            MEMORY=$(free | grep Mem | awk '{printf "%.2f", ($3/$2) * 100.0}')
        fi
    
    # macOS
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v vm_stat &> /dev/null; then
            # Cálculo complexo para macOS
            VM_STATS=$(vm_stat)
            PAGES_FREE=$(echo "$VM_STATS" | grep "Pages free" | awk '{print $3}' | tr -d '.')
            PAGES_ACTIVE=$(echo "$VM_STATS" | grep "Pages active" | awk '{print $3}' | tr -d '.')
            PAGES_INACTIVE=$(echo "$VM_STATS" | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
            PAGES_WIRED=$(echo "$VM_STATS" | grep "Pages wired down" | awk '{print $4}' | tr -d '.')
            
            PAGE_SIZE=4096
            TOTAL_MEM=$((($PAGES_FREE + $PAGES_ACTIVE + $PAGES_INACTIVE + $PAGES_WIRED) * $PAGE_SIZE))
            USED_MEM=$((($PAGES_ACTIVE + $PAGES_WIRED) * $PAGE_SIZE))
            
            MEMORY=$(awk -v used=$USED_MEM -v total=$TOTAL_MEM 'BEGIN {printf "%.2f", (used/total) * 100}')
        fi
    
    # AIX
    elif [[ "$OSTYPE" == "aix"* ]]; then
        if command -v svmon &> /dev/null; then
            MEMORY=$(svmon -G | grep memory | awk '{printf "%.2f", ($3/$2) * 100.0}')
        fi
    fi
    
    # Validação
    if [ -z "$MEMORY" ]; then
        log_message "ERROR: Could not detect memory usage"
        MEMORY=0
    fi
    
    echo "$MEMORY"
}

get_disk_usage() {
    # Filesystem raiz (/)
    if command -v df &> /dev/null; then
        DISK=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    fi
    
    # Validação
    if [ -z "$DISK" ]; then
        log_message "ERROR: Could not detect disk usage"
        DISK=0
    fi
    
    echo "$DISK"
}

get_additional_metrics() {
    # Load average (1 min)
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    
    # Número de CPUs
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CPU_COUNT=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CPU_COUNT=$(sysctl -n hw.ncpu)
    elif [[ "$OSTYPE" == "aix"* ]]; then
        CPU_COUNT=$(lsdev -Cc processor | wc -l)
    else
        CPU_COUNT=1
    fi
    
    # Memória total (GB)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        TOTAL_MEM=$(free -g | grep Mem | awk '{print $2}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
    elif [[ "$OSTYPE" == "aix"* ]]; then
        TOTAL_MEM=$(lsattr -El sys0 -a realmem | awk '{printf "%.0f", $2/1024/1024}')
    else
        TOTAL_MEM=0
    fi
    
    # Disk total (GB)
    TOTAL_DISK=$(df -BG / 2>/dev/null | tail -1 | awk '{print $2}' | cut -d'G' -f1)
    if [ -z "$TOTAL_DISK" ]; then
        TOTAL_DISK=$(df -k / | tail -1 | awk '{printf "%.0f", $2/1024/1024}')
    fi
    
    echo "$LOAD_AVG|$CPU_COUNT|$TOTAL_MEM|$TOTAL_DISK"
}

send_metrics() {
    local cpu=$1
    local memory=$2
    local disk=$3
    local load_avg=$4
    local cpu_count=$5
    local total_mem=$6
    local total_disk=$7
    
    # Timestamp ISO 8601 UTC
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # JSON payload
    JSON_PAYLOAD=$(cat <<EOF
{
  "workstation": "$WORKSTATION_NAME",
  "timestamp": "$TIMESTAMP",
  "metrics": {
    "cpu_percent": $cpu,
    "memory_percent": $memory,
    "disk_percent": $disk,
    "load_avg_1min": $load_avg,
    "cpu_count": $cpu_count,
    "total_memory_gb": $total_mem,
    "total_disk_gb": $total_disk
  },
  "metadata": {
    "os_type": "$OSTYPE",
    "hostname": "$(hostname)",
    "collector_version": "1.0.0"
  }
}
EOF
)
    
    # Log do payload (debug)
    # echo "$JSON_PAYLOAD" >> "$LOG_FILE"
    
    # HTTP POST com curl
    if command -v curl &> /dev/null; then
        HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            -H "User-Agent: TWS-Metrics-Collector/1.0.0" \
            --max-time $TIMEOUT \
            --data "$JSON_PAYLOAD" \
            "$RESYNC_URL" 2>&1)
        
        HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
        HTTP_BODY=$(echo "$HTTP_RESPONSE" | head -n -1)
        
        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
            log_message "SUCCESS: Metrics sent (HTTP $HTTP_CODE)"
            return 0
        else
            log_message "ERROR: HTTP $HTTP_CODE - $HTTP_BODY"
            return 1
        fi
    else
        log_message "ERROR: curl not found"
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    log_message "Starting metrics collection for $WORKSTATION_NAME"
    
    # Coleta métricas
    CPU=$(get_cpu_usage)
    MEMORY=$(get_memory_usage)
    DISK=$(get_disk_usage)
    
    # Métricas adicionais
    ADDITIONAL=$(get_additional_metrics)
    LOAD_AVG=$(echo "$ADDITIONAL" | cut -d'|' -f1)
    CPU_COUNT=$(echo "$ADDITIONAL" | cut -d'|' -f2)
    TOTAL_MEM=$(echo "$ADDITIONAL" | cut -d'|' -f3)
    TOTAL_DISK=$(echo "$ADDITIONAL" | cut -d'|' -f4)
    
    log_message "Collected: CPU=${CPU}% MEM=${MEMORY}% DISK=${DISK}% LOAD=${LOAD_AVG}"
    
    # Envia para Resync
    if send_metrics "$CPU" "$MEMORY" "$DISK" "$LOAD_AVG" "$CPU_COUNT" "$TOTAL_MEM" "$TOTAL_DISK"; then
        log_message "Metrics collection completed successfully"
        exit 0
    else
        log_message "Metrics collection failed"
        exit 1
    fi
}

# ============================================================================
# EXECUTION
# ============================================================================

# Verifica se já está rodando (evita duplicação)
LOCKFILE="/var/lock/tws_metrics_collector.lock"
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        log_message "Already running (PID: $PID)"
        exit 0
    else
        rm -f "$LOCKFILE"
    fi
fi

# Cria lockfile
echo $$ > "$LOCKFILE"

# Executa
main

# Remove lockfile
rm -f "$LOCKFILE"
