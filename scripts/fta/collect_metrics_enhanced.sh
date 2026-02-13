#!/bin/bash
################################################################################
# TWS FTA Enhanced Metrics Collector
# 
# Coleta métricas expandidas incluindo:
# - CPU, Memory, Disk (básico)
# - Latência para TWS Master (20 pings)
# - Network connectivity
# - Disk I/O
# - Process count
# - System load
#
# Instalação:
#   1. Copiar para: /opt/tws/scripts/collect_metrics_enhanced.sh
#   2. Dar permissão: chmod +x /opt/tws/scripts/collect_metrics_enhanced.sh
#   3. Configurar cron: */5 * * * * /opt/tws/scripts/collect_metrics_enhanced.sh
#
# Versão: 2.0.0
# Data: 2024-12-25
################################################################################

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# URL do Resync API (AJUSTAR!)
RESYNC_URL="https://resync.company.com/api/v1/metrics/workstation"

# API Key (AJUSTAR!)
API_KEY="your-api-key-here"

# TWS Master hostname/IP (AJUSTAR!)
TWS_MASTER_HOST="tws-master.company.com"
TWS_MASTER_PORT=31116

# Identificador da workstation
WORKSTATION_NAME=$(hostname -s)

# Log file
LOG_FILE="/var/log/tws_metrics_collector.log"

# Timeout para HTTP request (segundos)
TIMEOUT=10

# Número de pings para latência
PING_COUNT=20

# ============================================================================
# FUNÇÕES BÁSICAS
# ============================================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_cpu_usage() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CPU=$(top -bn2 -d 0.5 | grep "Cpu(s)" | tail -1 | awk '{print $2}' | cut -d'%' -f1)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CPU=$(top -l 2 | grep "CPU usage" | tail -1 | awk '{print $3}' | cut -d'%' -f1)
    elif [[ "$OSTYPE" == "aix"* ]]; then
        CPU=$(topas -d 1 -n 2 | grep "User" | tail -1 | awk '{print $2}')
    fi
    
    if [ -z "$CPU" ] && command -v mpstat &> /dev/null; then
        CPU=$(mpstat 1 1 | tail -1 | awk '{print 100 - $NF}')
    fi
    
    if [ -z "$CPU" ]; then
        log_message "ERROR: Could not detect CPU usage"
        CPU=0
    fi
    
    CPU=$(printf "%.2f" "$CPU")
    echo "$CPU"
}

get_memory_usage() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v free &> /dev/null; then
            MEMORY=$(free | grep Mem | awk '{printf "%.2f", ($3/$2) * 100.0}')
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v vm_stat &> /dev/null; then
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
    elif [[ "$OSTYPE" == "aix"* ]]; then
        if command -v svmon &> /dev/null; then
            MEMORY=$(svmon -G | grep memory | awk '{printf "%.2f", ($3/$2) * 100.0}')
        fi
    fi
    
    if [ -z "$MEMORY" ]; then
        log_message "ERROR: Could not detect memory usage"
        MEMORY=0
    fi
    
    echo "$MEMORY"
}

get_disk_usage() {
    if command -v df &> /dev/null; then
        DISK=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    fi
    
    if [ -z "$DISK" ]; then
        log_message "ERROR: Could not detect disk usage"
        DISK=0
    fi
    
    echo "$DISK"
}

# ============================================================================
# FUNÇÕES AVANÇADAS (NOVAS!)
# ============================================================================

get_latency_to_master() {
    """
    Testa latência para TWS Master com 20 pings.
    
    Retorna: min|avg|max|packet_loss
    """
    log_message "Testing latency to TWS Master: $TWS_MASTER_HOST"
    
    # Ping command varia por OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PING_CMD="ping -c $PING_COUNT -i 0.2"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PING_CMD="ping -c $PING_COUNT -i 0.2"
    elif [[ "$OSTYPE" == "aix"* ]]; then
        PING_CMD="ping -c $PING_COUNT"
    else
        PING_CMD="ping -n $PING_COUNT"
    fi
    
    # Execute ping
    PING_OUTPUT=$($PING_CMD $TWS_MASTER_HOST 2>&1)
    PING_EXIT_CODE=$?
    
    if [ $PING_EXIT_CODE -ne 0 ]; then
        log_message "ERROR: Ping to TWS Master failed"
        echo "0|0|0|100"
        return 1
    fi
    
    # Parse results (Linux/macOS format)
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Extract stats: min/avg/max/mdev
        STATS=$(echo "$PING_OUTPUT" | grep "min/avg/max" | awk -F'/' '{print $4"|"$5"|"$6}')
        
        # Extract packet loss
        PACKET_LOSS=$(echo "$PING_OUTPUT" | grep "packet loss" | awk '{print $6}' | cut -d'%' -f1)
        
        # Combine
        LATENCY_RESULT="${STATS}|${PACKET_LOSS}"
        
    # Parse results (AIX format - different)
    elif [[ "$OSTYPE" == "aix"* ]]; then
        # AIX ping output is different, adjust parsing
        MIN=$(echo "$PING_OUTPUT" | grep "round-trip" | awk '{print $4}' | cut -d'/' -f1)
        AVG=$(echo "$PING_OUTPUT" | grep "round-trip" | awk '{print $4}' | cut -d'/' -f2)
        MAX=$(echo "$PING_OUTPUT" | grep "round-trip" | awk '{print $4}' | cut -d'/' -f3)
        PACKET_LOSS=$(echo "$PING_OUTPUT" | grep "packet loss" | awk '{print $6}' | cut -d'%' -f1)
        
        LATENCY_RESULT="${MIN}|${AVG}|${MAX}|${PACKET_LOSS}"
    else
        LATENCY_RESULT="0|0|0|0"
    fi
    
    log_message "Latency results: $LATENCY_RESULT"
    echo "$LATENCY_RESULT"
}

test_tws_master_connectivity() {
    """
    Testa conectividade TCP para porta do TWS Master.
    
    Retorna: 1 (success) ou 0 (failed)
    """
    log_message "Testing TCP connectivity to $TWS_MASTER_HOST:$TWS_MASTER_PORT"
    
    # Try netcat (nc)
    if command -v nc &> /dev/null; then
        nc -z -w 3 $TWS_MASTER_HOST $TWS_MASTER_PORT &> /dev/null
        if [ $? -eq 0 ]; then
            log_message "TCP connectivity: SUCCESS"
            echo "1"
            return 0
        fi
    fi
    
    # Try telnet (fallback)
    if command -v telnet &> /dev/null; then
        (echo quit | telnet $TWS_MASTER_HOST $TWS_MASTER_PORT 2>&1 | grep -q "Connected") && {
            log_message "TCP connectivity: SUCCESS (via telnet)"
            echo "1"
            return 0
        }
    fi
    
    # Try bash TCP (last resort)
    timeout 3 bash -c "cat < /dev/null > /dev/tcp/$TWS_MASTER_HOST/$TWS_MASTER_PORT" 2>/dev/null && {
        log_message "TCP connectivity: SUCCESS (via bash)"
        echo "1"
        return 0
    }
    
    log_message "TCP connectivity: FAILED"
    echo "0"
    return 1
}

get_disk_io() {
    """
    Mede I/O de disco (reads/writes per second).
    
    Retorna: reads|writes (em KB/s)
    """
    # Linux: iostat
    if command -v iostat &> /dev/null && [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Sample 1 second
        IO_STATS=$(iostat -d -k 1 2 | tail -n 2 | head -n 1)
        READS=$(echo "$IO_STATS" | awk '{print $3}')
        WRITES=$(echo "$IO_STATS" | awk '{print $4}')
        
        echo "${READS}|${WRITES}"
        return 0
    fi
    
    # macOS: iostat (different format)
    if command -v iostat &> /dev/null && [[ "$OSTYPE" == "darwin"* ]]; then
        IO_STATS=$(iostat -d -K 1 2 | tail -n 2 | head -n 1)
        READS=$(echo "$IO_STATS" | awk '{print $2}')
        WRITES=$(echo "$IO_STATS" | awk '{print $3}')
        
        echo "${READS}|${WRITES}"
        return 0
    fi
    
    # AIX: iostat
    if command -v iostat &> /dev/null && [[ "$OSTYPE" == "aix"* ]]; then
        IO_STATS=$(iostat 1 2 | tail -1)
        READS=$(echo "$IO_STATS" | awk '{print $5}')
        WRITES=$(echo "$IO_STATS" | awk '{print $6}')
        
        echo "${READS}|${WRITES}"
        return 0
    fi
    
    # Not available
    echo "0|0"
}

get_process_count() {
    """
    Conta processos em execução.
    
    Retorna: total_processes
    """
    if command -v ps &> /dev/null; then
        PROC_COUNT=$(ps aux | wc -l)
        # Subtract header line
        PROC_COUNT=$((PROC_COUNT - 1))
        echo "$PROC_COUNT"
    else
        echo "0"
    fi
}

get_load_average() {
    """
    Load average (1, 5, 15 minutos).
    
    Retorna: load1|load5|load15
    """
    if command -v uptime &> /dev/null; then
        LOAD=$(uptime | awk -F'load average:' '{print $2}' | sed 's/ //g')
        # Parse: 1.23, 2.34, 3.45 → 1.23|2.34|3.45
        LOAD_1=$(echo "$LOAD" | cut -d',' -f1)
        LOAD_5=$(echo "$LOAD" | cut -d',' -f2)
        LOAD_15=$(echo "$LOAD" | cut -d',' -f3)
        
        echo "${LOAD_1}|${LOAD_5}|${LOAD_15}"
    else
        echo "0|0|0"
    fi
}

get_network_stats() {
    """
    Network RX/TX bytes (delta from last run).
    
    Retorna: rx_bytes|tx_bytes
    
    Note: Requires storing last values in temp file
    """
    TEMP_FILE="/tmp/tws_network_stats_${WORKSTATION_NAME}"
    
    # Linux: /proc/net/dev
    if [ -f "/proc/net/dev" ]; then
        # Get total RX/TX for all interfaces
        CURRENT_RX=$(cat /proc/net/dev | grep -v "lo:" | awk '{sum+=$2} END {print sum}')
        CURRENT_TX=$(cat /proc/net/dev | grep -v "lo:" | awk '{sum+=$10} END {print sum}')
        
        if [ -f "$TEMP_FILE" ]; then
            LAST_STATS=$(cat "$TEMP_FILE")
            LAST_RX=$(echo "$LAST_STATS" | cut -d'|' -f1)
            LAST_TX=$(echo "$LAST_STATS" | cut -d'|' -f2)
            
            DELTA_RX=$((CURRENT_RX - LAST_RX))
            DELTA_TX=$((CURRENT_TX - LAST_TX))
            
            # Convert to KB/s (assuming 5 min interval = 300s)
            RX_KBS=$((DELTA_RX / 1024 / 300))
            TX_KBS=$((DELTA_TX / 1024 / 300))
            
            echo "${RX_KBS}|${TX_KBS}"
        else
            echo "0|0"
        fi
        
        # Store current values
        echo "${CURRENT_RX}|${CURRENT_TX}" > "$TEMP_FILE"
        return 0
    fi
    
    # Not available
    echo "0|0"
}

# ============================================================================
# METADATA
# ============================================================================

get_additional_metrics() {
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CPU_COUNT=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CPU_COUNT=$(sysctl -n hw.ncpu)
    elif [[ "$OSTYPE" == "aix"* ]]; then
        CPU_COUNT=$(lsdev -Cc processor | wc -l)
    else
        CPU_COUNT=1
    fi
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        TOTAL_MEM=$(free -g | grep Mem | awk '{print $2}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
    elif [[ "$OSTYPE" == "aix"* ]]; then
        TOTAL_MEM=$(lsattr -El sys0 -a realmem | awk '{printf "%.0f", $2/1024/1024}')
    else
        TOTAL_MEM=0
    fi
    
    TOTAL_DISK=$(df -BG / 2>/dev/null | tail -1 | awk '{print $2}' | cut -d'G' -f1)
    if [ -z "$TOTAL_DISK" ]; then
        TOTAL_DISK=$(df -k / | tail -1 | awk '{printf "%.0f", $2/1024/1024}')
    fi
    
    echo "$LOAD_AVG|$CPU_COUNT|$TOTAL_MEM|$TOTAL_DISK"
}

# ============================================================================
# SEND METRICS
# ============================================================================

send_metrics() {
    local cpu=$1
    local memory=$2
    local disk=$3
    local load_avg=$4
    local cpu_count=$5
    local total_mem=$6
    local total_disk=$7
    local latency_min=$8
    local latency_avg=$9
    local latency_max=${10}
    local packet_loss=${11}
    local tcp_connectivity=${12}
    local disk_io_read=${13}
    local disk_io_write=${14}
    local process_count=${15}
    local load_1=${16}
    local load_5=${17}
    local load_15=${18}
    local net_rx=${19}
    local net_tx=${20}
    
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
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
    "total_disk_gb": $total_disk,
    "latency_min_ms": $latency_min,
    "latency_avg_ms": $latency_avg,
    "latency_max_ms": $latency_max,
    "packet_loss_percent": $packet_loss,
    "tcp_connectivity": $tcp_connectivity,
    "disk_io_read_kbs": $disk_io_read,
    "disk_io_write_kbs": $disk_io_write,
    "process_count": $process_count,
    "load_avg_1": $load_1,
    "load_avg_5": $load_5,
    "load_avg_15": $load_15,
    "network_rx_kbs": $net_rx,
    "network_tx_kbs": $net_tx
  },
  "metadata": {
    "os_type": "$OSTYPE",
    "hostname": "$(hostname)",
    "collector_version": "2.0.0-enhanced",
    "tws_master_host": "$TWS_MASTER_HOST",
    "tws_master_port": $TWS_MASTER_PORT
  }
}
EOF
)
    
    HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -H "User-Agent: TWS-Metrics-Collector/2.0.0" \
        --max-time $TIMEOUT \
        --data "$JSON_PAYLOAD" \
        "$RESYNC_URL" 2>&1)
    
    HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
    
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        log_message "SUCCESS: Metrics sent (HTTP $HTTP_CODE)"
        return 0
    else
        log_message "ERROR: HTTP $HTTP_CODE"
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    log_message "=== Starting ENHANCED metrics collection for $WORKSTATION_NAME ==="
    
    # Basic metrics
    CPU=$(get_cpu_usage)
    MEMORY=$(get_memory_usage)
    DISK=$(get_disk_usage)
    
    # Advanced metrics
    LATENCY_RESULT=$(get_latency_to_master)
    LATENCY_MIN=$(echo "$LATENCY_RESULT" | cut -d'|' -f1)
    LATENCY_AVG=$(echo "$LATENCY_RESULT" | cut -d'|' -f2)
    LATENCY_MAX=$(echo "$LATENCY_RESULT" | cut -d'|' -f3)
    PACKET_LOSS=$(echo "$LATENCY_RESULT" | cut -d'|' -f4)
    
    TCP_CONNECTIVITY=$(test_tws_master_connectivity)
    
    DISK_IO=$(get_disk_io)
    DISK_IO_READ=$(echo "$DISK_IO" | cut -d'|' -f1)
    DISK_IO_WRITE=$(echo "$DISK_IO" | cut -d'|' -f2)
    
    PROCESS_COUNT=$(get_process_count)
    
    LOAD_AVG_FULL=$(get_load_average)
    LOAD_1=$(echo "$LOAD_AVG_FULL" | cut -d'|' -f1)
    LOAD_5=$(echo "$LOAD_AVG_FULL" | cut -d'|' -f2)
    LOAD_15=$(echo "$LOAD_AVG_FULL" | cut -d'|' -f3)
    
    NETWORK=$(get_network_stats)
    NET_RX=$(echo "$NETWORK" | cut -d'|' -f1)
    NET_TX=$(echo "$NETWORK" | cut -d'|' -f2)
    
    # Metadata
    ADDITIONAL=$(get_additional_metrics)
    LOAD_AVG=$(echo "$ADDITIONAL" | cut -d'|' -f1)
    CPU_COUNT=$(echo "$ADDITIONAL" | cut -d'|' -f2)
    TOTAL_MEM=$(echo "$ADDITIONAL" | cut -d'|' -f3)
    TOTAL_DISK=$(echo "$ADDITIONAL" | cut -d'|' -f4)
    
    log_message "Collected: CPU=${CPU}% MEM=${MEMORY}% DISK=${DISK}% LATENCY=${LATENCY_AVG}ms LOSS=${PACKET_LOSS}%"
    
    # Send to Resync
    if send_metrics "$CPU" "$MEMORY" "$DISK" "$LOAD_AVG" "$CPU_COUNT" "$TOTAL_MEM" "$TOTAL_DISK" \
                    "$LATENCY_MIN" "$LATENCY_AVG" "$LATENCY_MAX" "$PACKET_LOSS" "$TCP_CONNECTIVITY" \
                    "$DISK_IO_READ" "$DISK_IO_WRITE" "$PROCESS_COUNT" \
                    "$LOAD_1" "$LOAD_5" "$LOAD_15" "$NET_RX" "$NET_TX"; then
        log_message "=== Metrics collection completed successfully ==="
        exit 0
    else
        log_message "=== Metrics collection failed ==="
        exit 1
    fi
}

# ============================================================================
# EXECUTION
# ============================================================================

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

echo $$ > "$LOCKFILE"
main
rm -f "$LOCKFILE"
