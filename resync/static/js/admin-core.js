/**
 * Admin APP - Main controller for the Resync Administrative SPA.
 */
class AdminApp {
    constructor() {
        this.api = new AdminAPIClient();
        this.router = new AdminRouter(this);
        this.config = new AdminConfig(this);
        this.resources = new AdminResources(this);
        this.backup = new AdminBackup(this); // Init Backup
        this.audit = new AdminAudit(this);   // Init Audit
        this.feedback = new AdminFeedback(this); // Init Feedback
        this.rag = new AdminRAG(this);       // Init RAG
        this.tuning = new AdminTuning(this); // Init Tuning
        this.init();
    }

    init() {
        if (!this.isAuthenticated()) {
            this.showLogin();
        } else {
            this.setupApp();
        }
    }

    isAuthenticated() {
        return sessionStorage.getItem('admin_auth') !== null;
    }

    showLogin() {
        document.getElementById('loginModal').style.display = 'flex';
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const user = document.getElementById('username').value;
            const pass = document.getElementById('password').value;
            const auth = btoa(`${user}:${pass}`);
            sessionStorage.setItem('admin_auth', auth);

            try {
                await this.api.get('/health'); // Test auth
                location.reload();
            } catch (err) {
                alert('Login failed: Invalid credentials');
                sessionStorage.removeItem('admin_auth');
            }
        });
    }

    setupApp() {
        document.getElementById('app').style.display = 'grid';
        this.setupRoutes();
        this.setupEventListeners();

        // Handle initial route
        const hash = window.location.hash.slice(1) || 'health';
        this.router.navigate(hash);

        window.addEventListener('hashchange', () => {
            const hash = window.location.hash.slice(1) || 'health';
            this.router.navigate(hash);
        });
    }

    setupRoutes() {
        this.router.addRoute('health', () => this.loadHealthView());
        this.router.addRoute('routing', () => this.loadRoutingView());

        // Configuration Routes (via AdminConfig)
        this.router.addRoute('teams-config', () => this.config.loadTeamsView());
        this.router.addRoute('tws-config', () => this.config.loadSettingsView('tws'));
        this.router.addRoute('tws-instances', () => this.resources.loadTWSInstancesView());
        this.router.addRoute('users', () => this.resources.loadUsersView());
        this.router.addRoute('api-keys', () => this.resources.loadAPIKeysView());
        this.router.addRoute('litellm-config', () => this.config.loadSettingsView('llm'));

        // Specialized Routes
        this.router.addRoute('graphrag', () => this.config.loadSettingsView('langgraph'));
        this.router.addRoute('rag-reranker', () => this.rag.loadRAGView());
        this.router.addRoute('threshold-tuning', () => this.tuning.loadTuningView());
        this.router.addRoute('backup', () => this.backup.loadBackupView());
        this.router.addRoute('backup', () => this.backup.loadBackupView());
        this.router.addRoute('audit', () => this.audit.loadAuditView());
        this.router.addRoute('feedback', () => this.feedback.loadFeedbackView());

        // Enterprise
        this.router.addRoute('enterprise-overview', () => this.config.loadSettingsView('enterprise'));
    }

    setupEventListeners() {
        document.getElementById('logoutBtn').addEventListener('click', () => {
            sessionStorage.removeItem('admin_auth');
            location.reload();
        });
    }

    async loadHealthView() {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="card"><div class="stat-label">Carregando Health...</div></div>';

        try {
            const stats = await this.api.get('/health');
            content.innerHTML = `
                <div class="header-title">
                    <h1>System Health</h1>
                    <p>Estado atual de todos os componentes do sistema</p>
                </div>
                <div class="dashboard-grid">
                    <div class="card stat-card">
                        <span class="stat-label">Status Global</span>
                        <div class="stat-value">${stats.status || 'UNKNOWN'}</div>
                        <span class="badge ${stats.status === 'healthy' ? 'badge-success' : 'badge-error'}">${stats.status}</span>
                    </div>
                    <div class="card">
                        <div class="card-title">Componentes</div>
                        <div class="data-list">
                            ${Object.entries(stats.components || {}).map(([name, data]) => `
                                <div class="data-item">
                                    <div class="data-item-info">
                                        <span class="data-item-title">${name}</span>
                                        <span class="data-item-meta">${data.message || 'Sem detalhes'}</span>
                                    </div>
                                    <span class="badge ${data.status === 'healthy' ? 'badge-success' : 'badge-error'}">${data.status}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        } catch (err) {
            content.innerHTML = `<div class="card"><div class="stat-label">Erro ao carregar health: ${err.message}</div></div>`;
        }
    }

    async loadRoutingView() {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="card"><div class="stat-label">Carregando Routing Monitoring...</div></div>';

        try {
            const [stats, recent] = await Promise.all([
                this.api.get('/admin/routing/stats'),
                this.api.get('/admin/routing/recent')
            ]);

            content.innerHTML = `
                <div class="header-title">
                    <h1>Routing Decisions</h1>
                    <p>Monitoramento em tempo real de classificação e roteamento</p>
                </div>
                <div class="dashboard-grid">
                    <div class="card stat-card">
                        <span class="stat-label">Total Decisões</span>
                        <div class="stat-value">${stats.total_decisions || 0}</div>
                    </div>
                    <div class="card stat-card">
                        <span class="stat-label">Latência Média</span>
                        <div class="stat-value">${Math.round(stats.avg_latency_ms || 0)}ms</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Decisões Recentes</div>
                    <div class="data-list">
                        ${recent.decisions.map(d => `
                            <div class="data-item">
                                <div class="data-item-info">
                                    <span class="data-item-title">"${d.message || 'N/A'}"</span>
                                    <span class="data-item-meta">
                                        Intent: <strong>${d.intent}</strong> (${Math.round(d.confidence * 100)}%) | 
                                        Path: <strong>${d.mode}</strong> | 
                                        Latency: ${d.latency_ms}ms
                                    </span>
                                </div>
                                <span class="badge badge-success">${d.mode}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } catch (err) {
            content.innerHTML = `<div class="card"><div class="stat-label">Erro ao carregar routing: ${err.message}</div></div>`;
        }
    }

    renderPlaceholder(title) {
        const content = document.getElementById('content');
        content.innerHTML = `
            <div class="header-title">
                <h1>${title}</h1>
                <p>Seção em desenvolvimento para a Fase 4</p>
            </div>
            <div class="card">
                <p>Esta funcionalidade será implementada no próximo sprint.</p>
            </div>
        `;
    }
}

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AdminApp();
});
