/**
 * Resync Admin Panel - JavaScript Controller
 * 
 * This script manages all administrative interface interactions,
 * connecting the UI with backend REST APIs.
 * 
 * Features:
 * - Configuration management (Teams, TWS, System)
 * - Real-time validation
 * - Toast notifications
 * - Auto-save functionality
 * - Error handling with user feedback
 */

class AdminPanel {
    constructor() {
        this.apiBase = '/admin';
        this.currentSection = 'teams-config';
        this.unsavedChanges = false;
        
        // Debounce timer for auto-save
        this.autoSaveTimer = null;
        this.autoSaveDelay = 2000; // 2 seconds
        
        this.init();
    }
    
    /**
     * Initialize the admin panel
     */
    init() {
        console.log('Initializing Resync Admin Panel...');
        
        // Load initial configuration
        this.loadCurrentConfig();
        
        // Setup navigation
        this.setupNavigation();
        
        // Setup save buttons
        this.setupSaveButtons();
        
        // Setup test buttons
        this.setupTestButtons();
        
        // Setup cache management
        this.setupCacheManagement();
        
        // Setup backup/restore
        this.setupBackupRestore();
        
        // Setup logs viewer
        this.setupLogsViewer();
        
        // Setup form change detection
        this.setupFormChangeDetection();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Setup auto-refresh
        this.setupAutoRefresh();
        
        console.log('Admin Panel initialized successfully');
    }
    
    /**
     * Load current configuration from server
     */
    async loadCurrentConfig() {
        try {
            this.showLoadingOverlay('Loading configuration...');
            
            const response = await fetch(`${this.apiBase}/config`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const config = await response.json();
            
            // Populate forms
            this.populateTeamsForm(config.teams);
            this.populateTwsForm(config.tws);
            this.populateSystemForm(config.system);
            
            this.unsavedChanges = false;
            
        } catch (error) {
            console.error('Failed to load configuration:', error);
            this.showToast('error', 'Failed to load configuration: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Populate Teams Integration form
     */
    populateTeamsForm(config) {
        if (!config) return;
        
        this.setCheckbox('teamsEnabled', config.enabled);
        this.setValue('webhookUrl', config.webhook_url);
        this.setValue('channelName', config.channel_name);
        this.setValue('botName', config.bot_name);
        this.setValue('avatarUrl', config.avatar_url);
        this.setCheckbox('conversationLearning', config.enable_conversation_learning);
        this.setCheckbox('jobNotifications', config.enable_job_notifications);
        
        // Populate monitored instances
        if (config.monitored_tws_instances) {
            this.populateInstancesList(config.monitored_tws_instances);
        }
        
        // Populate job status filters
        if (config.job_status_filters) {
            this.setMultiSelect('jobStatusFilters', config.job_status_filters);
        }
    }
    
    /**
     * Populate TWS Configuration form
     */
    populateTwsForm(config) {
        if (!config) return;
        
        this.setValue('twsHost', config.host);
        this.setValue('twsPort', config.port);
        this.setValue('twsUser', config.user);
        this.setCheckbox('twsVerifySsl', config.verify_ssl);
        this.setCheckbox('twsMockMode', config.mock_mode);
        
        // Populate monitored instances
        if (config.monitored_instances) {
            this.populateTwsInstancesList(config.monitored_instances);
        }
    }
    
    /**
     * Populate System Settings form
     */
    populateSystemForm(config) {
        if (!config) return;
        
        this.setValue('envSelect', config.environment);
        this.setCheckbox('debugMode', config.debug);
        this.setCheckbox('sslEnabled', config.ssl_enabled);
        this.setCheckbox('cspEnabled', config.csp_enabled);
        this.setCheckbox('corsEnabled', config.cors_enabled);
    }
    
    /**
     * Save Teams configuration
     */
    async saveTeamsConfig() {
        try {
            this.showLoadingOverlay('Saving Teams configuration...');
            
            const config = {
                enabled: this.getCheckbox('teamsEnabled'),
                webhook_url: this.getValue('webhookUrl'),
                channel_name: this.getValue('channelName'),
                bot_name: this.getValue('botName'),
                avatar_url: this.getValue('avatarUrl'),
                enable_conversation_learning: this.getCheckbox('conversationLearning'),
                enable_job_notifications: this.getCheckbox('jobNotifications'),
                monitored_tws_instances: this.getInstancesList(),
                job_status_filters: this.getMultiSelect('jobStatusFilters'),
                notification_types: this.getNotificationTypes()
            };
            
            // Validate
            const validation = this.validateTeamsConfig(config);
            if (!validation.valid) {
                this.showToast('error', 'Validation failed: ' + validation.errors.join(', '));
                return;
            }
            
            const response = await fetch(`${this.apiBase}/config/teams`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save configuration');
            }
            
            this.showToast('success', 'Teams configuration saved successfully!');
            this.unsavedChanges = false;
            
            // Reload configuration to ensure sync
            await this.loadCurrentConfig();
            
        } catch (error) {
            console.error('Failed to save Teams configuration:', error);
            this.showToast('error', 'Failed to save: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Save TWS configuration
     */
    async saveTwsConfig() {
        try {
            this.showLoadingOverlay('Saving TWS configuration...');
            
            const config = {
                host: this.getValue('twsHost'),
                port: parseInt(this.getValue('twsPort')),
                user: this.getValue('twsUser'),
                password: this.getValue('twsPassword'), // if provided
                verify_ssl: this.getCheckbox('twsVerifySsl'),
                mock_mode: this.getCheckbox('twsMockMode'),
                monitored_instances: this.getTwsInstancesList()
            };
            
            // Remove empty values
            Object.keys(config).forEach(key => {
                if (config[key] === '' || config[key] === null || config[key] === undefined) {
                    delete config[key];
                }
            });
            
            const response = await fetch(`${this.apiBase}/config/tws`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save configuration');
            }
            
            this.showToast('success', 'TWS configuration saved successfully!');
            this.unsavedChanges = false;
            
            await this.loadCurrentConfig();
            
        } catch (error) {
            console.error('Failed to save TWS configuration:', error);
            this.showToast('error', 'Failed to save: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Save System configuration
     */
    async saveSystemConfig() {
        try {
            this.showLoadingOverlay('Saving system configuration...');
            
            const config = {
                environment: this.getValue('envSelect'),
                debug: this.getCheckbox('debugMode'),
                ssl_enabled: this.getCheckbox('sslEnabled'),
                csp_enabled: this.getCheckbox('cspEnabled'),
                cors_enabled: this.getCheckbox('corsEnabled')
            };
            
            const response = await fetch(`${this.apiBase}/config/system`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save configuration');
            }
            
            this.showToast('success', 'System configuration saved successfully!');
            this.showToast('warning', 'Note: Some changes require application restart to take effect', 5000);
            this.unsavedChanges = false;
            
            await this.loadCurrentConfig();
            
        } catch (error) {
            console.error('Failed to save system configuration:', error);
            this.showToast('error', 'Failed to save: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Test Teams notification
     */
    async testTeamsNotification() {
        try {
            this.showLoadingOverlay('Sending test notification...');
            
            const response = await fetch(`${this.apiBase}/config/teams/test-notification`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to send test notification');
            }
            
            this.showToast('success', 'Test notification sent to Teams!');
            
        } catch (error) {
            console.error('Failed to send test notification:', error);
            this.showToast('error', 'Failed to send test notification: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Clear cache
     */
    async clearCache(cacheType = 'all') {
        if (!confirm(`Are you sure you want to clear ${cacheType} cache?`)) {
            return;
        }
        
        try {
            this.showLoadingOverlay('Clearing cache...');
            
            const response = await fetch(`${this.apiBase}/cache/clear`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ cache_type: cacheType })
            });
            
            if (!response.ok) {
                throw new Error('Failed to clear cache');
            }
            
            const result = await response.json();
            this.showToast('success', `Cache cleared: ${result.cleared.join(', ')}`);
            
        } catch (error) {
            console.error('Failed to clear cache:', error);
            this.showToast('error', 'Failed to clear cache: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Create backup
     */
    async createBackup() {
        try {
            this.showLoadingOverlay('Creating backup...');
            
            const response = await fetch(`${this.apiBase}/backup`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to create backup');
            }
            
            const result = await response.json();
            this.showToast('success', `Backup created: ${result.backup_file}`);
            
            // Reload backups list
            await this.loadBackupsList();
            
        } catch (error) {
            console.error('Failed to create backup:', error);
            this.showToast('error', 'Failed to create backup: ' + error.message);
        } finally {
            this.hideLoadingOverlay();
        }
    }
    
    /**
     * Load logs
     */
    async loadLogs(lines = 100, level = null, search = null) {
        try {
            let url = `${this.apiBase}/logs?lines=${lines}`;
            if (level) url += `&level=${level}`;
            if (search) url += `&search=${encodeURIComponent(search)}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error('Failed to load logs');
            }
            
            const result = await response.json();
            this.displayLogs(result.logs, result.count, result.total_lines);
            
        } catch (error) {
            console.error('Failed to load logs:', error);
            this.showToast('error', 'Failed to load logs: ' + error.message);
        }
    }
    
    /**
     * Display logs in the UI
     */
    displayLogs(logs, count, totalLines) {
        const logsContainer = document.getElementById('logsContainer');
        if (!logsContainer) return;
        
        logsContainer.innerHTML = '';
        
        if (logs.length === 0) {
            logsContainer.innerHTML = '<p class="text-muted">No logs found</p>';
            return;
        }
        
        const pre = document.createElement('pre');
        pre.className = 'logs-display';
        pre.textContent = logs.join('');
        
        logsContainer.appendChild(pre);
        
        // Update count
        const countEl = document.getElementById('logsCount');
        if (countEl) {
            countEl.textContent = `Showing ${count} of ${totalLines} lines`;
        }
    }
    
    // ============================================================================
    // HELPER METHODS
    // ============================================================================
    
    setValue(id, value) {
        const el = document.getElementById(id);
        if (el) el.value = value || '';
    }
    
    getValue(id) {
        const el = document.getElementById(id);
        return el ? el.value : '';
    }
    
    setCheckbox(id, checked) {
        const el = document.getElementById(id);
        if (el) el.checked = !!checked;
    }
    
    getCheckbox(id) {
        const el = document.getElementById(id);
        return el ? el.checked : false;
    }
    
    setMultiSelect(id, values) {
        const el = document.getElementById(id);
        if (!el) return;
        
        Array.from(el.options).forEach(option => {
            option.selected = values.includes(option.value);
        });
    }
    
    getMultiSelect(id) {
        const el = document.getElementById(id);
        if (!el) return [];
        
        return Array.from(el.selectedOptions).map(option => option.value);
    }
    
    /**
     * Show toast notification
     */
    showToast(type, message, duration = 3000) {
        const container = document.querySelector('.toast-container') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast ${type} show`;
        toast.innerHTML = `
            <div class="toast-header">
                <i class="fas fa-${this.getToastIcon(type)} me-2"></i>
                <strong class="me-auto">${this.getToastTitle(type)}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;
        
        container.appendChild(toast);
        
        // Auto-remove after duration
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
        
        // Close button
        toast.querySelector('.btn-close')?.addEventListener('click', () => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        });
    }
    
    createToastContainer() {
        const container = document.createElement('div');
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }
    
    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    getToastTitle(type) {
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Information'
        };
        return titles[type] || 'Notification';
    }
    
    /**
     * Show loading overlay
     */
    showLoadingOverlay(message = 'Loading...') {
        let overlay = document.getElementById('loadingOverlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner-border text-primary" role="status"></div>
                    <p class="mt-3" id="loadingMessage">${message}</p>
                </div>
            `;
            document.body.appendChild(overlay);
        } else {
            document.getElementById('loadingMessage').textContent = message;
        }
        
        overlay.style.display = 'flex';
    }
    
    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
    
    /**
     * Validate Teams configuration
     */
    validateTeamsConfig(config) {
        const errors = [];
        
        if (config.enabled && !config.webhook_url) {
            errors.push('Webhook URL is required when Teams integration is enabled');
        }
        
        if (config.webhook_url && !config.webhook_url.startsWith('https://')) {
            errors.push('Webhook URL must start with https://');
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }
    
    // ============================================================================
    // SETUP METHODS
    // ============================================================================
    
    setupNavigation() {
        document.querySelectorAll('.nav-link[data-target]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = link.getAttribute('data-target');
                this.switchSection(target);
            });
        });
    }
    
    switchSection(sectionId) {
        // Update nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-target="${sectionId}"]`)?.classList.add('active');
        
        // Update sections
        document.querySelectorAll('.config-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionId)?.classList.add('active');
        
        this.currentSection = sectionId;
        
        // Load section-specific data
        if (sectionId === 'graphrag' && typeof refreshGraphRAGStats === 'function') {
            refreshGraphRAGStats();
        }
    }
    
    setupSaveButtons() {
        document.getElementById('saveTeamsConfig')?.addEventListener('click', () => {
            this.saveTeamsConfig();
        });
        
        document.getElementById('saveTwsConfig')?.addEventListener('click', () => {
            this.saveTwsConfig();
        });
        
        document.getElementById('saveSystemSettings')?.addEventListener('click', () => {
            this.saveSystemConfig();
        });
    }
    
    setupTestButtons() {
        document.getElementById('testNotificationBtn')?.addEventListener('click', () => {
            this.testTeamsNotification();
        });
        
        document.getElementById('refreshHealthBtn')?.addEventListener('click', () => {
            this.loadCurrentConfig();
        });
    }
    
    setupCacheManagement() {
        document.getElementById('clearCacheBtn')?.addEventListener('click', () => {
            this.clearCache('all');
        });
    }
    
    setupBackupRestore() {
        document.getElementById('createBackupBtn')?.addEventListener('click', () => {
            this.createBackup();
        });
        
        document.getElementById('loadBackupsBtn')?.addEventListener('click', () => {
            this.loadBackupsList();
        });
    }
    
    setupLogsViewer() {
        document.getElementById('loadLogsBtn')?.addEventListener('click', () => {
            const lines = parseInt(document.getElementById('logLines')?.value || '100');
            const level = document.getElementById('logLevel')?.value || null;
            const search = document.getElementById('logSearch')?.value || null;
            this.loadLogs(lines, level, search);
        });
    }
    
    setupFormChangeDetection() {
        document.querySelectorAll('input, select, textarea').forEach(element => {
            element.addEventListener('change', () => {
                this.unsavedChanges = true;
            });
        });
        
        // Warn before leaving with unsaved changes
        window.addEventListener('beforeunload', (e) => {
            if (this.unsavedChanges) {
                e.preventDefault();
                e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S to save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.saveCurrentSection();
            }
            
            // Ctrl/Cmd + R to refresh
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.loadCurrentConfig();
            }
        });
    }
    
    saveCurrentSection() {
        if (this.currentSection === 'teams-config') {
            this.saveTeamsConfig();
        } else if (this.currentSection === 'tws-config') {
            this.saveTwsConfig();
        } else if (this.currentSection === 'system-config') {
            this.saveSystemConfig();
        }
    }
    
    setupAutoRefresh() {
        // Auto-refresh health status every 30 seconds
        setInterval(() => {
            this.refreshHealthStatus();
        }, 30000);
    }
    
    async refreshHealthStatus() {
        try {
            const response = await fetch(`${this.apiBase}/status`);
            if (response.ok) {
                const status = await response.json();
                this.updateHealthIndicators(status);
            }
        } catch (error) {
            console.error('Failed to refresh health status:', error);
        }
    }
    
    updateHealthIndicators(status) {
        // Update health indicators in the UI
        // Implementation depends on your specific health status structure
        console.log('Health status updated:', status);
    }
    
    // Placeholder methods for specific UI interactions
    populateInstancesList(instances) {
        // Implement based on your HTML structure
        console.log('Populating instances list:', instances);
    }
    
    getInstancesList() {
        // Implement based on your HTML structure
        return [];
    }
    
    populateTwsInstancesList(instances) {
        // Implement based on your HTML structure
        console.log('Populating TWS instances list:', instances);
    }
    
    getTwsInstancesList() {
        // Implement based on your HTML structure
        return [];
    }
    
    getNotificationTypes() {
        // Get selected notification types
        const types = [];
        if (document.getElementById('notifyJobStatus')?.checked) types.push('job_status');
        if (document.getElementById('notifyAlerts')?.checked) types.push('alerts');
        if (document.getElementById('notifyPerformance')?.checked) types.push('performance');
        return types;
    }
    
    async loadBackupsList() {
        // Implement backup list loading
        console.log('Loading backups list...');
    }
    
    // =========================================================================
    // PROACTIVE MONITORING FUNCTIONS
    // =========================================================================
    
    initProactiveMonitoring() {
        // Slider de intervalo de polling
        const pollingSlider = document.getElementById('pollingInterval');
        const pollingValue = document.getElementById('pollingIntervalValue');
        if (pollingSlider && pollingValue) {
            pollingSlider.addEventListener('input', () => {
                pollingValue.textContent = pollingSlider.value;
            });
        }
        
        // Modo de polling - mostrar/ocultar config de schedule
        const pollingMode = document.getElementById('pollingMode');
        const scheduledConfig = document.getElementById('scheduledPollingConfig');
        if (pollingMode && scheduledConfig) {
            pollingMode.addEventListener('change', () => {
                scheduledConfig.classList.toggle('d-none', pollingMode.value !== 'scheduled');
            });
        }
        
        // Botões de controle
        document.getElementById('startPollerBtn')?.addEventListener('click', () => this.startPoller());
        document.getElementById('stopPollerBtn')?.addEventListener('click', () => this.stopPoller());
        document.getElementById('forcePollBtn')?.addEventListener('click', () => this.forcePoll());
        document.getElementById('detectPatternsBtn')?.addEventListener('click', () => this.detectPatterns());
        document.getElementById('testNotification')?.addEventListener('click', () => this.testNotification());
        document.getElementById('saveProactiveConfig')?.addEventListener('click', () => this.saveProactiveConfig());
        
        // Carregar status inicial
        this.loadProactiveStatus();
        
        // Auto-refresh do status a cada 10s
        setInterval(() => this.loadProactiveStatus(), 10000);
    }
    
    async loadProactiveStatus() {
        try {
            const response = await fetch('/api/v1/monitoring/stats');
            if (response.ok) {
                const data = await response.json();
                this.updateProactiveStatus(data);
            }
        } catch (error) {
            console.error('Failed to load proactive status:', error);
        }
    }
    
    updateProactiveStatus(data) {
        // Atualiza métricas
        const pollerMetrics = data.poller || {};
        const eventBusMetrics = data.event_bus || {};
        
        document.getElementById('pollerStatus').textContent = pollerMetrics.polls_count || '0';
        document.getElementById('eventsGenerated').textContent = pollerMetrics.events_generated || '0';
        document.getElementById('wsClients').textContent = eventBusMetrics.websocket_clients || '0';
        document.getElementById('patternsDetected').textContent = data.patterns_count || '0';
        
        // Atualiza indicador de status
        const indicator = document.getElementById('pollerStatusIndicator');
        const statusText = document.getElementById('pollerStatusText');
        
        if (pollerMetrics.is_running) {
            indicator.className = 'status-indicator status-connected';
            statusText.textContent = `Poller ativo (intervalo: ${pollerMetrics.polling_interval}s)`;
        } else {
            indicator.className = 'status-indicator status-disconnected';
            statusText.textContent = 'Poller parado';
        }
    }
    
    async startPoller() {
        try {
            const response = await fetch('/api/v1/monitoring/poller/start', { method: 'POST' });
            if (response.ok) {
                this.showNotification('Poller iniciado com sucesso', 'success');
                this.loadProactiveStatus();
            }
        } catch (error) {
            this.showNotification('Erro ao iniciar poller: ' + error.message, 'danger');
        }
    }
    
    async stopPoller() {
        try {
            const response = await fetch('/api/v1/monitoring/poller/stop', { method: 'POST' });
            if (response.ok) {
                this.showNotification('Poller parado com sucesso', 'success');
                this.loadProactiveStatus();
            }
        } catch (error) {
            this.showNotification('Erro ao parar poller: ' + error.message, 'danger');
        }
    }
    
    async forcePoll() {
        try {
            const response = await fetch('/api/v1/monitoring/poller/poll', { method: 'POST' });
            if (response.ok) {
                this.showNotification('Poll forçado executado', 'success');
                this.loadProactiveStatus();
            }
        } catch (error) {
            this.showNotification('Erro ao forçar poll: ' + error.message, 'danger');
        }
    }
    
    async detectPatterns() {
        try {
            const response = await fetch('/api/v1/monitoring/patterns/detect', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                this.showNotification(`${data.patterns_detected || 0} padrões detectados`, 'info');
                this.loadProactiveStatus();
            }
        } catch (error) {
            this.showNotification('Erro ao detectar padrões: ' + error.message, 'danger');
        }
    }
    
    async testNotification() {
        try {
            // Solicita permissão de notificação se necessário
            if (Notification.permission === 'default') {
                await Notification.requestPermission();
            }
            
            const response = await fetch('/api/v1/monitoring/test-notification', { method: 'POST' });
            if (response.ok) {
                // Mostra notificação local de teste
                if (Notification.permission === 'granted') {
                    new Notification('Resync TWS Monitor', {
                        body: 'Esta é uma notificação de teste!',
                        icon: '/favicon.ico',
                        tag: 'test-notification'
                    });
                }
                this.showNotification('Notificação de teste enviada', 'success');
            }
        } catch (error) {
            this.showNotification('Erro ao testar notificação: ' + error.message, 'danger');
        }
    }
    
    async saveProactiveConfig() {
        const config = {
            // Poller config
            polling_enabled: document.getElementById('pollerEnabled')?.checked ?? true,
            polling_interval_seconds: parseInt(document.getElementById('pollingInterval')?.value) || 30,
            polling_mode: document.getElementById('pollingMode')?.value || 'fixed',
            polling_schedule: document.getElementById('pollingSchedule')?.value || '06:00-22:00',
            
            // Thresholds
            job_stuck_threshold_minutes: parseInt(document.getElementById('jobStuckThreshold')?.value) || 60,
            job_late_threshold_minutes: parseInt(document.getElementById('jobLateThreshold')?.value) || 30,
            anomaly_failure_rate_threshold: (parseInt(document.getElementById('anomalyThreshold')?.value) || 10) / 100,
            
            // WebSocket config
            websocket_enabled: document.getElementById('websocketEnabled')?.checked ?? true,
            ws_filter_jobs: document.getElementById('wsFilterJobs')?.checked ?? true,
            ws_filter_workstations: document.getElementById('wsFilterWs')?.checked ?? true,
            ws_filter_system: document.getElementById('wsFilterSystem')?.checked ?? true,
            ws_filter_critical_only: document.getElementById('wsFilterCritical')?.checked ?? false,
            ws_min_severity: document.getElementById('wsMinSeverity')?.value || 'error',
            ws_job_filter_regex: document.getElementById('wsJobFilter')?.value || '',
            
            // Notifications
            webpush_enabled: document.getElementById('webpushEnabled')?.checked ?? true,
            notify_abend: document.getElementById('notifyAbend')?.checked ?? true,
            notify_ws_offline: document.getElementById('notifyWsOffline')?.checked ?? true,
            notify_stuck: document.getElementById('notifyStuck')?.checked ?? false,
            notify_pattern: document.getElementById('notifyPattern')?.checked ?? false,
            sound_enabled: document.getElementById('soundEnabled')?.checked ?? false,
            
            // Retention
            retention_days_full: parseInt(document.getElementById('retentionFull')?.value) || 7,
            retention_days_summary: parseInt(document.getElementById('retentionSummary')?.value) || 30,
            retention_days_patterns: parseInt(document.getElementById('retentionPatterns')?.value) || 90,
            
            // Pattern detection
            pattern_detection_enabled: document.getElementById('patternDetectionEnabled')?.checked ?? true,
            pattern_detection_interval_minutes: parseInt(document.getElementById('patternInterval')?.value) || 15,
            pattern_min_confidence: (parseInt(document.getElementById('patternConfidence')?.value) || 70) / 100,
        };
        
        try {
            const response = await fetch('/api/v1/monitoring/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showNotification('Configurações salvas com sucesso!', 'success');
            } else {
                const error = await response.json();
                this.showNotification('Erro ao salvar: ' + (error.detail || 'Erro desconhecido'), 'danger');
            }
        } catch (error) {
            this.showNotification('Erro ao salvar configurações: ' + error.message, 'danger');
        }
    }
    
    async loadProactiveConfig() {
        try {
            const response = await fetch('/api/v1/monitoring/config');
            if (response.ok) {
                const config = await response.json();
                this.populateProactiveConfig(config);
            }
        } catch (error) {
            console.error('Failed to load proactive config:', error);
        }
    }
    
    populateProactiveConfig(config) {
        // Poller config
        document.getElementById('pollerEnabled').checked = config.polling_enabled ?? true;
        document.getElementById('pollingInterval').value = config.polling_interval_seconds || 30;
        document.getElementById('pollingIntervalValue').textContent = config.polling_interval_seconds || 30;
        document.getElementById('pollingMode').value = config.polling_mode || 'fixed';
        document.getElementById('pollingSchedule').value = config.polling_schedule || '06:00-22:00';
        
        // Thresholds
        document.getElementById('jobStuckThreshold').value = config.job_stuck_threshold_minutes || 60;
        document.getElementById('jobLateThreshold').value = config.job_late_threshold_minutes || 30;
        document.getElementById('anomalyThreshold').value = (config.anomaly_failure_rate_threshold || 0.1) * 100;
        
        // WebSocket config
        document.getElementById('websocketEnabled').checked = config.websocket_enabled ?? true;
        document.getElementById('wsFilterJobs').checked = config.ws_filter_jobs ?? true;
        document.getElementById('wsFilterWs').checked = config.ws_filter_workstations ?? true;
        document.getElementById('wsFilterSystem').checked = config.ws_filter_system ?? true;
        document.getElementById('wsFilterCritical').checked = config.ws_filter_critical_only ?? false;
        document.getElementById('wsMinSeverity').value = config.ws_min_severity || 'error';
        document.getElementById('wsJobFilter').value = config.ws_job_filter_regex || '';
        
        // Notifications
        document.getElementById('webpushEnabled').checked = config.webpush_enabled ?? true;
        document.getElementById('notifyAbend').checked = config.notify_abend ?? true;
        document.getElementById('notifyWsOffline').checked = config.notify_ws_offline ?? true;
        document.getElementById('notifyStuck').checked = config.notify_stuck ?? false;
        document.getElementById('notifyPattern').checked = config.notify_pattern ?? false;
        document.getElementById('soundEnabled').checked = config.sound_enabled ?? false;
        
        // Retention
        document.getElementById('retentionFull').value = config.retention_days_full || 7;
        document.getElementById('retentionSummary').value = config.retention_days_summary || 30;
        document.getElementById('retentionPatterns').value = config.retention_days_patterns || 90;
        
        // Pattern detection
        document.getElementById('patternDetectionEnabled').checked = config.pattern_detection_enabled ?? true;
        document.getElementById('patternInterval').value = config.pattern_detection_interval_minutes || 15;
        document.getElementById('patternConfidence').value = (config.pattern_min_confidence || 0.7) * 100;
    }
}

// Initialize admin panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.adminPanel = new AdminPanel();

    // Inicializa monitoramento proativo quando a seção é mostrada
    document.querySelectorAll('[data-target="proactive-monitoring"]').forEach(el => {
        el.addEventListener('click', () => {
            window.adminPanel.initProactiveMonitoring();
            window.adminPanel.loadProactiveConfig();
        });
    });

    // Inicializa auto-tuning quando a seção é mostrada
    document.querySelectorAll('[data-target="auto-tuning"]').forEach(el => {
        el.addEventListener('click', () => {
            window.autoTuning.loadDashboard();
        });
    });

    // Initialize auto-tuning module
    window.autoTuning = new AutoTuningController();
});

// Expose for console debugging
window.AdminPanel = AdminPanel;


/**
 * Auto-Tuning Controller
 * Manages the auto-tuning UI for Active Learning thresholds
 */
class AutoTuningController {
    constructor() {
        this.apiBase = '/api/v1/admin/auto-tuning';
        this.currentLevel = 'off';
        this.originalLevel = 'off';
        this.levelMap = ['off', 'low', 'mid', 'high'];
        this.levelColors = {
            'off': 'secondary',
            'low': 'info',
            'mid': 'warning',
            'high': 'success'
        };
        this.levelAlerts = {
            'off': 'secondary',
            'low': 'info',
            'mid': 'warning',
            'high': 'success'
        };
        this.levelDescriptions = {
            'off': { title: 'Desativado', description: 'Thresholds estáticos. Sem coleta de métricas.' },
            'low': { title: 'Observação', description: 'Coleta métricas de eficácia. Sem ajustes.' },
            'mid': { title: 'Sugestões', description: 'Sugere ajustes. Requer aprovação manual.' },
            'high': { title: 'Auto-Ajuste', description: 'Ajusta automaticamente com rollback.' }
        };

        this.init();
    }

    init() {
        this.setupSlider();
        this.setupButtons();
    }

    setupSlider() {
        const slider = document.getElementById('autoTuningSlider');
        if (!slider) return;

        slider.addEventListener('input', (e) => {
            const levelIndex = parseInt(e.target.value);
            const level = this.levelMap[levelIndex];
            this.updateLevelDisplay(level);

            // Enable apply button if level changed
            const applyBtn = document.getElementById('applyAutoTuningLevel');
            if (applyBtn) {
                applyBtn.disabled = (level === this.originalLevel);
            }
        });

        // Highlight labels on hover/select
        document.querySelectorAll('.slider-label').forEach(label => {
            label.addEventListener('click', () => {
                const level = parseInt(label.dataset.level);
                slider.value = level;
                slider.dispatchEvent(new Event('input'));
            });
        });
    }

    setupButtons() {
        document.getElementById('applyAutoTuningLevel')?.addEventListener('click', () => {
            this.applyLevel();
        });

        document.getElementById('resetThresholds')?.addEventListener('click', () => {
            this.resetThresholds();
        });

        document.getElementById('refreshAutoTuning')?.addEventListener('click', () => {
            this.loadDashboard();
        });
    }

    updateLevelDisplay(level) {
        this.currentLevel = level;
        const info = this.levelDescriptions[level];
        const alertClass = this.levelAlerts[level];

        // Update info box
        const levelInfo = document.getElementById('autoTuningLevelInfo');
        if (levelInfo) {
            levelInfo.className = `alert alert-${alertClass} mb-0`;
        }
        document.getElementById('levelTitle').textContent = info.title;
        document.getElementById('levelDescription').textContent = info.description;

        // Update slider labels
        document.querySelectorAll('.slider-label').forEach(label => {
            const labelLevel = this.levelMap[parseInt(label.dataset.level)];
            if (labelLevel === level) {
                label.classList.add('opacity-100');
                label.style.transform = 'scale(1.1)';
            } else {
                label.classList.remove('opacity-100');
                label.style.transform = 'scale(1)';
            }
        });

        // Show/hide suggestions card
        const suggestionsCard = document.getElementById('suggestionsCard');
        if (suggestionsCard) {
            suggestionsCard.style.display = (level === 'mid') ? 'block' : 'none';
        }
    }

    async loadDashboard() {
        try {
            const response = await fetch(`${this.apiBase}/dashboard`);
            if (!response.ok) throw new Error('Failed to load dashboard');

            const data = await response.json();
            this.populateDashboard(data);
        } catch (error) {
            console.error('Failed to load auto-tuning dashboard:', error);
            this.showNotification('Erro ao carregar dados de auto-tuning', 'danger');
        }
    }

    populateDashboard(data) {
        // Set current level
        const level = data.config?.level || 'off';
        this.currentLevel = level;
        this.originalLevel = level;

        const slider = document.getElementById('autoTuningSlider');
        if (slider) {
            slider.value = this.levelMap.indexOf(level);
        }
        this.updateLevelDisplay(level);

        // Update sidebar badge
        const badge = document.getElementById('autoTuningLevelBadge');
        if (badge) {
            badge.textContent = level.toUpperCase();
            badge.className = `badge bg-${this.levelColors[level]}`;
        }

        // Disable apply button (no changes yet)
        const applyBtn = document.getElementById('applyAutoTuningLevel');
        if (applyBtn) applyBtn.disabled = true;

        // Populate metrics
        const metrics = data.metrics || {};
        document.getElementById('metricReviewRate').textContent =
            `${(metrics.review_rate || 0).toFixed(1)}%`;
        document.getElementById('metricFPRate').textContent =
            `${(metrics.false_positive_rate || 0).toFixed(1)}%`;
        document.getElementById('metricFNRate').textContent =
            `${(metrics.false_negative_rate || 0).toFixed(1)}%`;
        document.getElementById('metricF1Score').textContent =
            `${(metrics.f1_score || 0).toFixed(1)}%`;

        // Populate interpretation
        this.populateInterpretation(data.metrics_interpretation);

        // Populate thresholds table
        this.populateThresholdsTable(data.config?.thresholds, data.config?.bounds);

        // Populate suggestions (if in MID mode)
        if (level === 'mid' && data.suggestions?.length > 0) {
            this.populateSuggestions(data.suggestions);
        }

        // Populate history
        this.populateHistory(data.recent_adjustments || []);
    }

    populateInterpretation(interpretation) {
        const container = document.getElementById('metricsInterpretation');
        if (!container || !interpretation?.items) return;

        const typeClasses = {
            'success': 'alert-success',
            'info': 'alert-info',
            'warning': 'alert-warning',
            'critical': 'alert-danger'
        };

        container.innerHTML = interpretation.items.map(item => `
            <div class="alert ${typeClasses[item.type] || 'alert-secondary'} mb-2 py-2">
                <strong>${item.message}</strong>
                <p class="mb-0 small">${item.suggestion}</p>
            </div>
        `).join('');
    }

    populateThresholdsTable(thresholds, bounds) {
        const tbody = document.getElementById('thresholdsTable');
        if (!tbody || !thresholds) return;

        const thresholdNames = {
            'min_classification_confidence': 'Confiança Mín. Classificação',
            'min_rag_similarity': 'Similaridade RAG Mínima',
            'min_entity_count': 'Contagem Mín. Entidades',
            'error_similarity_threshold': 'Threshold Similaridade Erro'
        };

        tbody.innerHTML = Object.entries(thresholds).map(([key, value]) => {
            const bound = bounds?.[key] || {};
            return `
                <tr>
                    <td>${thresholdNames[key] || key}</td>
                    <td><strong>${value}</strong></td>
                    <td>${bound.min ?? '-'}</td>
                    <td>${bound.max ?? '-'}</td>
                    <td>${bound.default ?? '-'}</td>
                </tr>
            `;
        }).join('');
    }

    populateSuggestions(suggestions) {
        const container = document.getElementById('suggestionsList');
        if (!container) return;

        if (!suggestions || suggestions.length === 0) {
            container.innerHTML = '<p class="text-muted">Nenhuma sugestão disponível no momento.</p>';
            return;
        }

        container.innerHTML = suggestions.map(s => `
            <div class="card mb-2">
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${s.threshold_name}</h6>
                            <p class="mb-1 small">${s.reason}</p>
                            <span class="badge bg-secondary">${s.current_value} → ${s.suggested_value}</span>
                            <span class="badge bg-info">Confiança: ${s.confidence}%</span>
                        </div>
                        <div>
                            <button class="btn btn-sm btn-success me-1" onclick="autoTuning.approveSuggestion('${s.threshold_name}')">
                                <i class="fas fa-check"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="autoTuning.rejectSuggestion('${s.threshold_name}')">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    populateHistory(adjustments) {
        const tbody = document.getElementById('adjustmentHistory');
        if (!tbody) return;

        if (!adjustments || adjustments.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Nenhum ajuste registrado</td></tr>';
            return;
        }

        tbody.innerHTML = adjustments.map(adj => {
            const timestamp = new Date(adj.timestamp).toLocaleString('pt-BR');
            const typeLabel = adj.auto_applied ? 'Automático' : 'Manual';
            const typeBadge = adj.auto_applied ? 'bg-info' : 'bg-primary';
            const statusLabel = adj.rolled_back ? 'Revertido' : 'Ativo';
            const statusBadge = adj.rolled_back ? 'bg-warning' : 'bg-success';

            return `
                <tr>
                    <td>${timestamp}</td>
                    <td>${adj.threshold_name}</td>
                    <td>${adj.old_value}</td>
                    <td>${adj.new_value}</td>
                    <td><span class="badge ${typeBadge}">${typeLabel}</span></td>
                    <td><span class="badge ${statusBadge}">${statusLabel}</span></td>
                </tr>
            `;
        }).join('');
    }

    async applyLevel() {
        const level = this.currentLevel;

        try {
            const response = await fetch(`${this.apiBase}/level`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ level: level })
            });

            if (!response.ok) throw new Error('Failed to apply level');

            const result = await response.json();
            this.originalLevel = level;
            document.getElementById('applyAutoTuningLevel').disabled = true;

            // Update sidebar badge
            const badge = document.getElementById('autoTuningLevelBadge');
            if (badge) {
                badge.textContent = level.toUpperCase();
                badge.className = `badge bg-${this.levelColors[level]}`;
            }

            this.showNotification(`Nível alterado para ${level.toUpperCase()}`, 'success');
            this.loadDashboard();
        } catch (error) {
            console.error('Failed to apply level:', error);
            this.showNotification('Erro ao aplicar nível', 'danger');
        }
    }

    async resetThresholds() {
        if (!confirm('Tem certeza que deseja resetar todos os thresholds para os valores padrão?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/reset`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to reset thresholds');

            this.showNotification('Thresholds resetados com sucesso', 'success');
            this.loadDashboard();
        } catch (error) {
            console.error('Failed to reset thresholds:', error);
            this.showNotification('Erro ao resetar thresholds', 'danger');
        }
    }

    async approveSuggestion(thresholdName) {
        try {
            const response = await fetch(`${this.apiBase}/suggestions/${thresholdName}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approved_by: 'admin' })
            });

            if (!response.ok) throw new Error('Failed to approve suggestion');

            this.showNotification('Sugestão aprovada e aplicada', 'success');
            this.loadDashboard();
        } catch (error) {
            console.error('Failed to approve suggestion:', error);
            this.showNotification('Erro ao aprovar sugestão', 'danger');
        }
    }

    async rejectSuggestion(thresholdName) {
        try {
            const response = await fetch(`${this.apiBase}/suggestions/${thresholdName}/reject`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to reject suggestion');

            this.showNotification('Sugestão rejeitada', 'info');
            this.loadDashboard();
        } catch (error) {
            console.error('Failed to reject suggestion:', error);
            this.showNotification('Erro ao rejeitar sugestão', 'danger');
        }
    }

    showNotification(message, type = 'info') {
        // Use AdminPanel's notification if available
        if (window.adminPanel?.showNotification) {
            window.adminPanel.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Expose for console debugging
window.AutoTuningController = AutoTuningController;
