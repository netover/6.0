#!/bin/bash
# =============================================================================
# Resync Automation Systems Launcher
# =============================================================================
# Starts self-healing and code quality monitoring systems
# =============================================================================

set -e

echo "🚀 Starting Resync Automation Systems..."
echo ""

# Check if running in virtual env
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected"
    echo "    Attempting to use uv..."
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# Function to start system in background
start_system() {
    local name=$1
    local script=$2
    local pid_file="/tmp/resync_${name}.pid"
    
    echo "Starting $name..."
    
    # Kill existing if running
    if [ -f "$pid_file" ]; then
        old_pid=$(cat "$pid_file")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            echo "  Killing existing process $old_pid"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f "$pid_file"
    fi
    
    # Start new process
    $PYTHON_CMD "$script" > "logs/${name}.log" 2>&1 &
    pid=$!
    echo "$pid" > "$pid_file"
    
    echo "✅ $name started (PID: $pid)"
    echo "   Logs: logs/${name}.log"
    echo ""
}

# Create logs directory
mkdir -p logs

# Start systems
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Self-Healing System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
start_system "self_healing" "resync/tools/self_healing.py"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Code Quality Guardian"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
start_system "code_guardian" "resync/tools/code_quality_guardian.py"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All Systems Running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/self_healing.log"
echo "  tail -f logs/code_guardian.log"
echo ""
echo "Stop systems:"
echo "  ./scripts/stop_automation.sh"
echo ""
