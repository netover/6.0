#!/bin/bash
# =============================================================================
# Redis Stack Installation and Configuration Script
# Resync v5.3.16 - Semantic Cache
# =============================================================================
# 
# This script helps you:
# 1. Backup existing Redis data
# 2. Install Redis Stack (or verify installation)
# 3. Configure Redis for semantic caching
# 4. Verify the installation
#
# Usage:
#   chmod +x scripts/setup_redis_stack.sh
#   ./scripts/setup_redis_stack.sh
#
# Prerequisites:
#   - Ubuntu/Debian Linux
#   - sudo access
#   - Internet connection (for package download)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REDIS_CONF_DIR="/etc/redis"
REDIS_DATA_DIR="/var/lib/redis"
BACKUP_DIR="/tmp/redis_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Redis Stack Setup for Semantic Cache${NC}"
echo -e "${BLUE}  Resync v5.3.16${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# =============================================================================
# STEP 1: Check if running as root or with sudo
# =============================================================================
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${YELLOW}This script requires sudo privileges for installation.${NC}"
        echo -e "${YELLOW}Please run: sudo $0${NC}"
        exit 1
    fi
}

# =============================================================================
# STEP 2: Backup existing Redis data
# =============================================================================
backup_redis() {
    echo -e "\n${BLUE}[1/5] Backing up existing Redis data...${NC}"
    
    mkdir -p "$BACKUP_DIR"
    
    # Check if Redis is running and backup
    if systemctl is-active --quiet redis-server 2>/dev/null || systemctl is-active --quiet redis 2>/dev/null; then
        echo "  Redis is running, triggering BGSAVE..."
        redis-cli BGSAVE 2>/dev/null || true
        sleep 2
        
        # Copy RDB file
        if [ -f "$REDIS_DATA_DIR/dump.rdb" ]; then
            cp "$REDIS_DATA_DIR/dump.rdb" "$BACKUP_DIR/"
            echo -e "  ${GREEN}✓ Backed up dump.rdb${NC}"
        fi
        
        # Copy AOF file if exists
        if [ -f "$REDIS_DATA_DIR/appendonly.aof" ]; then
            cp "$REDIS_DATA_DIR/appendonly.aof" "$BACKUP_DIR/"
            echo -e "  ${GREEN}✓ Backed up appendonly.aof${NC}"
        fi
        
        # Copy config
        if [ -f "$REDIS_CONF_DIR/redis.conf" ]; then
            cp "$REDIS_CONF_DIR/redis.conf" "$BACKUP_DIR/"
            echo -e "  ${GREEN}✓ Backed up redis.conf${NC}"
        fi
    else
        echo -e "  ${YELLOW}No Redis instance running, skipping backup.${NC}"
    fi
    
    echo -e "  ${GREEN}Backup location: $BACKUP_DIR${NC}"
}

# =============================================================================
# STEP 3: Install Redis Stack
# =============================================================================
install_redis_stack() {
    echo -e "\n${BLUE}[2/5] Installing Redis Stack...${NC}"
    
    # Check if Redis Stack is already installed
    if command -v redis-stack-server &> /dev/null; then
        echo -e "  ${GREEN}✓ Redis Stack already installed${NC}"
        redis-stack-server --version 2>/dev/null || true
        return 0
    fi
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        echo -e "  ${RED}Cannot detect OS. Please install Redis Stack manually.${NC}"
        echo -e "  ${YELLOW}Visit: https://redis.io/docs/install/install-stack/linux/${NC}"
        exit 1
    fi
    
    echo "  Detected OS: $OS $VERSION"
    
    case $OS in
        ubuntu|debian)
            echo "  Installing for Ubuntu/Debian..."
            
            # Add Redis repository
            curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
            
            # Update and install
            apt-get update
            apt-get install -y redis-stack-server
            
            echo -e "  ${GREEN}✓ Redis Stack installed${NC}"
            ;;
            
        centos|rhel|fedora)
            echo "  Installing for CentOS/RHEL/Fedora..."
            
            # Add Redis repository
            curl -fsSL https://packages.redis.io/gpg > /tmp/redis.key
            rpm --import /tmp/redis.key
            
            cat > /etc/yum.repos.d/redis.repo << 'EOF'
[Redis]
name=Redis
baseurl=https://packages.redis.io/rpm/
enabled=1
gpgcheck=1
EOF
            
            # Install
            yum install -y redis-stack-server || dnf install -y redis-stack-server
            
            echo -e "  ${GREEN}✓ Redis Stack installed${NC}"
            ;;
            
        *)
            echo -e "  ${RED}Unsupported OS: $OS${NC}"
            echo -e "  ${YELLOW}Please install Redis Stack manually:${NC}"
            echo -e "  ${YELLOW}https://redis.io/docs/install/install-stack/linux/${NC}"
            exit 1
            ;;
    esac
}

# =============================================================================
# STEP 4: Configure Redis Stack
# =============================================================================
configure_redis() {
    echo -e "\n${BLUE}[3/5] Configuring Redis Stack...${NC}"
    
    REDIS_STACK_CONF="/etc/redis-stack.conf"
    
    # Create configuration if it doesn't exist
    if [ ! -f "$REDIS_STACK_CONF" ]; then
        REDIS_STACK_CONF="/etc/redis/redis-stack.conf"
    fi
    
    # Generate a random password if not set
    if [ -z "$REDIS_PASSWORD" ]; then
        REDIS_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
        echo -e "  ${YELLOW}Generated Redis password (save this!):${NC}"
        echo -e "  ${GREEN}$REDIS_PASSWORD${NC}"
        echo ""
        echo "  Add to your .env file:"
        echo "  REDIS_PASSWORD=$REDIS_PASSWORD"
    fi
    
    # Create optimized configuration
    cat > /tmp/redis-semantic-cache.conf << EOF
# =============================================================================
# Redis Stack Configuration for Semantic Cache
# Generated by Resync v5.3.16 setup script
# =============================================================================

# Network
bind 127.0.0.1
port 6379
protected-mode yes
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Security
requirepass $REDIS_PASSWORD

# Memory Management
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence - RDB Snapshots
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb
dir /var/lib/redis

# Persistence - AOF
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Logging
loglevel notice
logfile /var/log/redis/redis-stack.log

# Clients
maxclients 10000

# Performance
activerehashing yes
hz 10
dynamic-hz yes

# Modules (loaded automatically by Redis Stack)
# RediSearch, ReJSON, etc. are included

# Slow Log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Latency Monitoring
latency-monitor-threshold 100
EOF

    # Backup existing config and use new one
    if [ -f "$REDIS_STACK_CONF" ]; then
        cp "$REDIS_STACK_CONF" "${REDIS_STACK_CONF}.backup"
    fi
    
    cp /tmp/redis-semantic-cache.conf "$REDIS_STACK_CONF"
    chmod 640 "$REDIS_STACK_CONF"
    
    # Create log directory
    mkdir -p /var/log/redis
    chown redis:redis /var/log/redis 2>/dev/null || true
    
    echo -e "  ${GREEN}✓ Configuration written to $REDIS_STACK_CONF${NC}"
}

# =============================================================================
# STEP 5: Start Redis Stack
# =============================================================================
start_redis() {
    echo -e "\n${BLUE}[4/5] Starting Redis Stack...${NC}"
    
    # Stop old Redis if running
    systemctl stop redis-server 2>/dev/null || true
    systemctl stop redis 2>/dev/null || true
    systemctl disable redis-server 2>/dev/null || true
    systemctl disable redis 2>/dev/null || true
    
    # Start Redis Stack
    systemctl enable redis-stack-server 2>/dev/null || systemctl enable redis-stack 2>/dev/null || true
    systemctl start redis-stack-server 2>/dev/null || systemctl start redis-stack 2>/dev/null || {
        # Try direct start if systemd fails
        redis-stack-server /etc/redis-stack.conf --daemonize yes
    }
    
    # Wait for startup
    sleep 2
    
    # Check if running
    if redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q "PONG"; then
        echo -e "  ${GREEN}✓ Redis Stack is running${NC}"
    else
        echo -e "  ${RED}✗ Redis Stack failed to start${NC}"
        echo -e "  ${YELLOW}Check logs: /var/log/redis/redis-stack.log${NC}"
        exit 1
    fi
}

# =============================================================================
# STEP 6: Verify Installation
# =============================================================================
verify_installation() {
    echo -e "\n${BLUE}[5/5] Verifying installation...${NC}"
    
    AUTH_ARG=""
    if [ -n "$REDIS_PASSWORD" ]; then
        AUTH_ARG="-a $REDIS_PASSWORD"
    fi
    
    # Check connection
    echo -n "  Checking connection... "
    if redis-cli $AUTH_ARG ping 2>/dev/null | grep -q "PONG"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
    
    # Check modules
    echo -n "  Checking RediSearch module... "
    if redis-cli $AUTH_ARG MODULE LIST 2>/dev/null | grep -qi "search\|ft"; then
        echo -e "${GREEN}✓ Available${NC}"
    else
        echo -e "${YELLOW}⚠ Not found (fallback mode will be used)${NC}"
    fi
    
    echo -n "  Checking ReJSON module... "
    if redis-cli $AUTH_ARG MODULE LIST 2>/dev/null | grep -qi "json\|rejson"; then
        echo -e "${GREEN}✓ Available${NC}"
    else
        echo -e "${YELLOW}⚠ Not found${NC}"
    fi
    
    # Check memory
    echo -n "  Memory usage: "
    redis-cli $AUTH_ARG INFO memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 || echo "unknown"
    
    # Check version
    echo -n "  Redis version: "
    redis-cli $AUTH_ARG INFO server 2>/dev/null | grep "redis_version" | cut -d: -f2 || echo "unknown"
}

# =============================================================================
# STEP 7: Print Summary
# =============================================================================
print_summary() {
    echo -e "\n${GREEN}=============================================${NC}"
    echo -e "${GREEN}  Redis Stack Setup Complete!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
    echo "  Configuration:"
    echo "    Host: localhost"
    echo "    Port: 6379"
    echo "    Password: ${REDIS_PASSWORD:-'(not set)'}"
    echo "    Max Memory: 4GB"
    echo ""
    echo "  Add to your .env file:"
    echo "    REDIS_HOST=localhost"
    echo "    REDIS_PORT=6379"
    echo "    REDIS_PASSWORD=$REDIS_PASSWORD"
    echo ""
    echo "  Useful commands:"
    echo "    redis-cli -a \$REDIS_PASSWORD ping"
    echo "    redis-cli -a \$REDIS_PASSWORD MODULE LIST"
    echo "    redis-cli -a \$REDIS_PASSWORD INFO memory"
    echo ""
    echo "  Backup location: $BACKUP_DIR"
    echo ""
    echo -e "${YELLOW}  Next steps:${NC}"
    echo "    1. Add Redis password to your .env file"
    echo "    2. Restart Resync application"
    echo "    3. Access Admin > Semantic Cache to verify"
    echo ""
}

# =============================================================================
# Main execution
# =============================================================================
main() {
    check_privileges
    backup_redis
    install_redis_stack
    configure_redis
    start_redis
    verify_installation
    print_summary
}

# Run main function
main "$@"
