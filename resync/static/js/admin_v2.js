/**
 * Admin Interface 2.0 - Enhanced JavaScript Module
 * 
 * Provides real-time monitoring and control for:
 * - Health status (connected to UnifiedHealthService)
 * - Resilience controls (Circuit Breakers, Redis Strategy)
 * - RAG configuration (Chunking strategies)
 * - System operations
 * 
 * Part of Resync v5.4.2
 */

const AdminV2 = {
    // Configuration
    API_BASE: '/api/v1/admin',
    REFRESH_INTERVAL: 5000, // 5 seconds
    
    // State
    refreshTimers: {},
    charts: {},
    
    /**
     * Initialize all Admin 2.0 features
     */
    init() {
        console.log('Admin Interface 2.0 initializing...');
        
        // Initialize sections
        this.initHealthMonitoring();
        this.initResilienceControls();
        this.initRAGConfig();
        this.initSystemOperations();
        
        // Start periodic refresh
        this.startAutoRefresh();
        
        console.log('Admin Interface 2.0 ready');
    },
    
    // =========================================================================
    // Health Monitoring
    // =========================================================================
    
    /**
     * Initialize real-time health monitoring
     */
    async initHealthMonitoring() {
        const container = document.getElementById('health-realtime');
        if (!container) return;
        
        // Initial load
        await this.refreshHealth();
    },
    
    /**
     * Refresh health status from real endpoint
     */
    async refreshHealth() {
        try {
            const response = await fetch(`${this.API_BASE}/health/realtime`);
            const data = await response.json();
            
            this.updateHealthUI(data);
        } catch (error) {
            console.error('Failed to fetch health:', error);
            this.showHealthError(error);
        }
    },
    
    /**
     * Update health UI with real data
     */
    updateHealthUI(data) {
        // Update overall status
        const overallBadge = document.getElementById('health-overall-status');
        if (overallBadge) {
            overallBadge.className = `badge bg-${this.getStatusColor(data.overall_status)}`;
            overallBadge.textContent = data.overall_status.toUpperCase();
        }
        
        // Update uptime
        const uptimeEl = document.getElementById('health-uptime');
        if (uptimeEl) {
            uptimeEl.textContent = this.formatUptime(data.uptime_seconds);
        }
        
        // Update individual services
        const servicesContainer = document.getElementById('health-services');
        if (servicesContainer && data.services) {
            servicesContainer.innerHTML = data.services.map(service => `
                <div class="col-md-6 col-lg-3 mb-3">
                    <div class="card h-100">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center">
                                <h6 class="mb-0">${service.name.toUpperCase()}</h6>
                                <span class="badge bg-${this.getStatusColor(service.status)}">
                                    ${service.status}
                                </span>
                            </div>
                            ${service.latency_ms ? `
                                <small class="text-muted">
                                    Latency: ${service.latency_ms.toFixed(1)}ms
                                </small>
                            ` : ''}
                            ${service.message ? `
                                <small class="text-warning d-block">
                                    ${service.message}
                                </small>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        // Update timestamp
        const timestampEl = document.getElementById('health-timestamp');
        if (timestampEl) {
            timestampEl.textContent = new Date(data.timestamp).toLocaleString();
        }
    },
    
    // =========================================================================
    // Resilience Controls
    // =========================================================================
    
    /**
     * Initialize resilience controls section
     */
    async initResilienceControls() {
        const container = document.getElementById('resilience-panel');
        if (!container) return;
        
        await this.refreshResilienceStatus();
    },
    
    /**
     * Refresh resilience status
     */
    async refreshResilienceStatus() {
        try {
            // Get circuit breakers
            const cbResponse = await fetch(`${this.API_BASE}/resilience/breakers`);
            const cbData = await cbResponse.json();
            this.updateCircuitBreakersUI(cbData);
            
            // Get overall resilience status
            const statusResponse = await fetch(`${this.API_BASE}/resilience/status`);
            const statusData = await statusResponse.json();
            this.updateRedisStrategyUI(statusData.redis_strategy);
            
        } catch (error) {
            console.error('Failed to fetch resilience status:', error);
        }
    },
    
    /**
     * Update circuit breakers UI
     */
    updateCircuitBreakersUI(data) {
        const container = document.getElementById('circuit-breakers-list');
        if (!container) return;
        
        // Update summary
        const summaryEl = document.getElementById('cb-summary');
        if (summaryEl) {
            summaryEl.innerHTML = `
                <span class="badge bg-secondary me-2">Total: ${data.total}</span>
                <span class="badge bg-${data.open_count > 0 ? 'danger' : 'success'} me-2">
                    Open: ${data.open_count}
                </span>
                <span class="badge bg-${data.critical_open_count > 0 ? 'danger' : 'success'}">
                    Critical Open: ${data.critical_open_count}
                </span>
            `;
        }
        
        // Update table
        container.innerHTML = `
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>State</th>
                        <th>Failures</th>
                        <th>Threshold</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.breakers.map(cb => `
                        <tr class="${cb.state === 'OPEN' ? 'table-danger' : ''}">
                            <td>
                                ${cb.name}
                                ${cb.is_critical ? '<span class="badge bg-warning ms-1">Critical</span>' : ''}
                            </td>
                            <td>
                                <span class="badge bg-${this.getCBStateColor(cb.state)}">
                                    ${cb.state}
                                </span>
                            </td>
                            <td>${cb.failure_count} / ${cb.threshold}</td>
                            <td>${cb.recovery_timeout}s</td>
                            <td>
                                <button class="btn btn-sm btn-outline-warning" 
                                        onclick="AdminV2.resetCircuitBreaker('${cb.name}')"
                                        ${cb.state === 'CLOSED' ? 'disabled' : ''}>
                                    <i class="fas fa-redo"></i> Reset
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    },
    
    /**
     * Reset a circuit breaker
     */
    async resetCircuitBreaker(name) {
        if (!confirm(`Reset circuit breaker "${name}"? This will force a reconnection attempt.`)) {
            return;
        }
        
        try {
            const response = await fetch(`${this.API_BASE}/resilience/breaker/${name}/reset`, {
                method: 'POST',
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`Circuit breaker "${name}" reset successfully`, 'success');
                await this.refreshResilienceStatus();
            } else {
                this.showNotification(`Failed to reset: ${data.message}`, 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    },
    
    /**
     * Update Redis strategy UI
     */
    updateRedisStrategyUI(data) {
        const container = document.getElementById('redis-strategy');
        if (!container || !data) return;
        
        container.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h6 class="mb-0">Redis Fail-Fast Strategy</h6>
                        <div class="form-check form-switch">
                            <input class="form-check-input" type="checkbox" 
                                   id="failFastToggle" 
                                   ${data.enabled ? 'checked' : ''}
                                   onchange="AdminV2.toggleFailFast(this.checked)">
                            <label class="form-check-label" for="failFastToggle">
                                ${data.enabled ? 'Enabled' : 'Disabled'}
                            </label>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <small class="text-muted">Mode</small>
                            <p class="mb-1">
                                <span class="badge bg-${data.mode === 'normal' ? 'success' : 'warning'}">
                                    ${data.mode}
                                </span>
                            </p>
                        </div>
                        <div class="col-md-6">
                            <small class="text-muted">Timeout</small>
                            <p class="mb-1">${data.fail_fast_timeout}s</p>
                        </div>
                    </div>
                    
                    ${data.degraded_endpoints && data.degraded_endpoints.length > 0 ? `
                        <div class="mt-3">
                            <small class="text-muted">Degraded Endpoints</small>
                            <ul class="list-unstyled mb-0">
                                ${data.degraded_endpoints.map(ep => `
                                    <li><i class="fas fa-exclamation-triangle text-warning"></i> ${ep}</li>
                                `).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },
    
    /**
     * Toggle fail-fast mode
     */
    async toggleFailFast(enabled) {
        try {
            const response = await fetch(`${this.API_BASE}/resilience/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fail_fast_enabled: enabled }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(
                    `Fail-fast ${enabled ? 'enabled' : 'disabled'}`,
                    'success'
                );
                
                if (data.restart_required) {
                    this.showRestartBanner();
                }
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    },
    
    // =========================================================================
    // RAG Configuration
    // =========================================================================
    
    /**
     * Initialize RAG configuration section
     */
    async initRAGConfig() {
        const container = document.getElementById('rag-config');
        if (!container) return;
        
        await this.refreshRAGConfig();
    },
    
    /**
     * Refresh RAG configuration
     */
    async refreshRAGConfig() {
        try {
            const response = await fetch(`${this.API_BASE}/rag/chunking`);
            const data = await response.json();
            this.updateRAGConfigUI(data);
        } catch (error) {
            console.error('Failed to fetch RAG config:', error);
        }
    },
    
    /**
     * Update RAG config UI
     */
    updateRAGConfigUI(data) {
        const container = document.getElementById('rag-config-form');
        if (!container) return;
        
        container.innerHTML = `
            <form onsubmit="AdminV2.saveRAGConfig(event)">
                <div class="mb-3">
                    <label class="form-label">Chunking Strategy</label>
                    <select class="form-select" id="chunkingStrategy">
                        <option value="tws_optimized" ${data.strategy === 'tws_optimized' ? 'selected' : ''}>
                            TWS Optimized (Recommended)
                        </option>
                        <option value="hierarchical" ${data.strategy === 'hierarchical' ? 'selected' : ''}>
                            Hierarchical
                        </option>
                        <option value="semantic" ${data.strategy === 'semantic' ? 'selected' : ''}>
                            Semantic
                        </option>
                        <option value="fixed" ${data.strategy === 'fixed' ? 'selected' : ''}>
                            Fixed Size
                        </option>
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label">Chunk Size (tokens)</label>
                    <input type="range" class="form-range" id="chunkSize" 
                           min="128" max="4096" step="64" value="${data.chunk_size}">
                    <div class="d-flex justify-content-between">
                        <small>128</small>
                        <strong id="chunkSizeValue">${data.chunk_size}</strong>
                        <small>4096</small>
                    </div>
                </div>
                
                <div class="mb-3">
                    <label class="form-label">Chunk Overlap (tokens)</label>
                    <input type="range" class="form-range" id="chunkOverlap" 
                           min="0" max="512" step="10" value="${data.chunk_overlap}">
                    <div class="d-flex justify-content-between">
                        <small>0</small>
                        <strong id="chunkOverlapValue">${data.chunk_overlap}</strong>
                        <small>512</small>
                    </div>
                </div>
                
                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="preserveStructure"
                           ${data.preserve_structure ? 'checked' : ''}>
                    <label class="form-check-label" for="preserveStructure">
                        Preserve Document Structure
                    </label>
                </div>
                
                <div class="form-check mb-3">
                    <input class="form-check-input" type="checkbox" id="extractMetadata"
                           ${data.extract_metadata ? 'checked' : ''}>
                    <label class="form-check-label" for="extractMetadata">
                        Extract Metadata
                    </label>
                </div>
                
                <div class="d-grid gap-2">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Save Configuration
                    </button>
                    <button type="button" class="btn btn-warning" onclick="AdminV2.startReindex()">
                        <i class="fas fa-sync"></i> Rebuild Index
                    </button>
                </div>
            </form>
        `;
        
        // Add slider event listeners
        document.getElementById('chunkSize').addEventListener('input', (e) => {
            document.getElementById('chunkSizeValue').textContent = e.target.value;
        });
        document.getElementById('chunkOverlap').addEventListener('input', (e) => {
            document.getElementById('chunkOverlapValue').textContent = e.target.value;
        });
    },
    
    /**
     * Save RAG configuration
     */
    async saveRAGConfig(event) {
        event.preventDefault();
        
        const config = {
            strategy: document.getElementById('chunkingStrategy').value,
            chunk_size: parseInt(document.getElementById('chunkSize').value),
            chunk_overlap: parseInt(document.getElementById('chunkOverlap').value),
            preserve_structure: document.getElementById('preserveStructure').checked,
            extract_metadata: document.getElementById('extractMetadata').checked,
        };
        
        try {
            const response = await fetch(`${this.API_BASE}/rag/chunking`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('RAG configuration saved', 'success');
                
                if (data.restart_required) {
                    this.showRestartBanner();
                }
            } else {
                this.showNotification(data.message || 'Save failed', 'error');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    },
    
    /**
     * Start knowledge base reindex
     */
    async startReindex() {
        if (!confirm('This will reprocess all documents with the current configuration. Continue?')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.API_BASE}/rag/reindex`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            
            const data = await response.json();
            
            if (data.job_id) {
                this.showNotification(`Reindex job started: ${data.job_id}`, 'info');
                this.trackReindexProgress(data.job_id);
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    },
    
    /**
     * Track reindex progress
     */
    async trackReindexProgress(jobId) {
        const progressContainer = document.getElementById('reindex-progress');
        if (!progressContainer) return;
        
        progressContainer.innerHTML = `
            <div class="alert alert-info">
                <h6>Reindex in Progress (${jobId})</h6>
                <div class="progress">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         id="reindex-progress-bar" style="width: 0%">0%</div>
                </div>
                <small id="reindex-status">Starting...</small>
            </div>
        `;
        
        const checkProgress = async () => {
            try {
                const response = await fetch(`${this.API_BASE}/rag/reindex/${jobId}`);
                const data = await response.json();
                
                const progressBar = document.getElementById('reindex-progress-bar');
                const statusEl = document.getElementById('reindex-status');
                
                if (progressBar) {
                    const pct = Math.round(data.progress * 100);
                    progressBar.style.width = `${pct}%`;
                    progressBar.textContent = `${pct}%`;
                }
                
                if (statusEl) {
                    statusEl.textContent = `${data.documents_processed} / ${data.documents_total} documents`;
                }
                
                if (data.status === 'completed') {
                    progressContainer.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> Reindex completed successfully!
                        </div>
                    `;
                } else if (data.status === 'failed') {
                    progressContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-times-circle"></i> Reindex failed: ${data.error}
                        </div>
                    `;
                } else {
                    // Continue checking
                    setTimeout(checkProgress, 1000);
                }
            } catch (error) {
                console.error('Progress check failed:', error);
            }
        };
        
        setTimeout(checkProgress, 1000);
    },
    
    // =========================================================================
    // System Operations
    // =========================================================================
    
    /**
     * Initialize system operations
     */
    initSystemOperations() {
        // Check for pending restart
        this.checkRestartRequired();
    },
    
    /**
     * Check if restart is required
     */
    async checkRestartRequired() {
        try {
            const response = await fetch(`${this.API_BASE}/system/restart-required`);
            const data = await response.json();
            
            if (data.restart_required) {
                this.showRestartBanner(data.urgency, data.pending_changes);
            }
        } catch (error) {
            console.error('Failed to check restart status:', error);
        }
    },
    
    /**
     * Show restart required banner
     */
    showRestartBanner(urgency = 'graceful', changes = []) {
        let existingBanner = document.getElementById('restart-banner');
        if (existingBanner) existingBanner.remove();
        
        const bannerClass = urgency === 'immediate' ? 'alert-danger' : 'alert-warning';
        
        const banner = document.createElement('div');
        banner.id = 'restart-banner';
        banner.className = `alert ${bannerClass} alert-dismissible fade show m-3`;
        banner.innerHTML = `
            <strong><i class="fas fa-exclamation-triangle"></i> Restart Required</strong>
            <p class="mb-2">Configuration changes require a restart to take effect.</p>
            ${changes.length > 0 ? `
                <small>Changed: ${changes.join(', ')}</small>
            ` : ''}
            <div class="mt-2">
                <button class="btn btn-sm btn-outline-dark" onclick="AdminV2.dismissRestartBanner()">
                    Dismiss
                </button>
            </div>
            <button type="button" class="btn-close" onclick="AdminV2.dismissRestartBanner()"></button>
        `;
        
        document.body.insertBefore(banner, document.body.firstChild);
    },
    
    /**
     * Dismiss restart banner
     */
    dismissRestartBanner() {
        const banner = document.getElementById('restart-banner');
        if (banner) banner.remove();
    },
    
    /**
     * Toggle maintenance mode
     */
    async toggleMaintenanceMode(enabled) {
        const message = enabled 
            ? prompt('Enter maintenance message:', 'System is under maintenance')
            : null;
        
        if (enabled && !message) return;
        
        try {
            const response = await fetch(`${this.API_BASE}/system/maintenance`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled, message: message || '' }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(
                    enabled ? 'Maintenance mode enabled' : 'Maintenance mode disabled',
                    enabled ? 'warning' : 'success'
                );
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    },
    
    // =========================================================================
    // Utility Functions
    // =========================================================================
    
    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        this.refreshTimers.health = setInterval(() => this.refreshHealth(), this.REFRESH_INTERVAL);
        this.refreshTimers.resilience = setInterval(() => this.refreshResilienceStatus(), this.REFRESH_INTERVAL);
    },
    
    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        Object.values(this.refreshTimers).forEach(timer => clearInterval(timer));
        this.refreshTimers = {};
    },
    
    /**
     * Get status color for badges
     */
    getStatusColor(status) {
        const colors = {
            'healthy': 'success',
            'degraded': 'warning',
            'unhealthy': 'danger',
            'unknown': 'secondary',
        };
        return colors[status?.toLowerCase()] || 'secondary';
    },
    
    /**
     * Get circuit breaker state color
     */
    getCBStateColor(state) {
        const colors = {
            'CLOSED': 'success',
            'OPEN': 'danger',
            'HALF_OPEN': 'warning',
        };
        return colors[state] || 'secondary';
    },
    
    /**
     * Format uptime
     */
    formatUptime(seconds) {
        if (!seconds) return 'Unknown';
        
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        
        let result = '';
        if (days > 0) result += `${days}d `;
        if (hours > 0) result += `${hours}h `;
        result += `${mins}m`;
        
        return result;
    },
    
    /**
     * Show notification toast
     */
    showNotification(message, type = 'info') {
        // Use existing toast system or create simple notification
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto dismiss
        setTimeout(() => toast.remove(), 5000);
    },
    
    /**
     * Create toast container
     */
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        document.body.appendChild(container);
        return container;
    },
    
    /**
     * Show health error
     */
    showHealthError(error) {
        const container = document.getElementById('health-services');
        if (container) {
            container.innerHTML = `
                <div class="col-12">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle"></i>
                        Failed to fetch health status: ${error.message}
                    </div>
                </div>
            `;
        }
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    AdminV2.init();
});

// Export for use in other scripts
window.AdminV2 = AdminV2;
