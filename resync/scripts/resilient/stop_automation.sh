#!/bin/bash
# =============================================================================
# Stop Resync Automation Systems
# =============================================================================

set -e

echo "🛑 Stopping Resync Automation Systems..."
echo ""

stop_system() {
    local name=$1
    local pid_file="/tmp/resync_${name}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p "$pid" > /dev/null 2>&1; then
                    break
                fi
                sleep 0.5
            done
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  Force killing..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            echo "✅ $name stopped"
        else
            echo "⚠️  $name not running (stale PID file)"
        fi
        
        rm -f "$pid_file"
    else
        echo "⚠️  $name PID file not found"
    fi
    
    echo ""
}

# Stop systems
stop_system "self_healing"
stop_system "code_guardian"

echo "✅ All systems stopped"
