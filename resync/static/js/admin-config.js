/**
 * Admin Configuration Module
 * Handles configuration views for Teams, TWS, AI/LLM, etc.
 */

class AdminConfig {
    constructor(app) {
        this.app = app;
        this.api = app.api;
    }

    // ========================================================================
    // TEAMS INTEGRATION (Specialized View)
    // ========================================================================
    async loadTeamsView() {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="card"><div class="stat-label">Carregando Configuração Teams...</div></div>';

        try {
            const [config, health] = await Promise.all([
                this.api.get('/admin/config'),
                this.api.get('/admin/config/teams/health')
            ]);

            const teams = config.teams;

            content.innerHTML = `
                <div class="header-title">
                    <h1>Microsoft Teams Integration</h1>
                    <p>Configuração do bot e notificações</p>
                </div>

                <div class="dashboard-grid">
                    <div class="card stat-card">
                        <span class="stat-label">Status Conexão</span>
                        <div class="stat-value" style="font-size: 1.5rem; display: flex; align-items: center; gap: 10px;">
                            ${health.status.healthy ?
                    '<span style="color: var(--success);">Online</span> <div class="icon-heartbeat" style="background: var(--success); width: 10px; height: 10px; border-radius: 50%;"></div>' :
                    '<span style="color: var(--error);">Offline</span> <div class="icon-heartbeat" style="background: var(--error); width: 10px; height: 10px; border-radius: 50%;"></div>'
                }
                        </div>
                        <span class="stat-label" style="font-size: 0.7rem;">Last Check: ${new Date(health.timestamp).toLocaleTimeString()}</span>
                    </div>
                     <div class="card stat-card">
                        <span class="stat-label">Webhook URL</span>
                        <div class="stat-value" style="font-size: 1rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${teams.webhook_url ? 'Configurado' : 'Não Configurado'}
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">
                        Configurações Gerais
                        <button class="btn btn-neu" onclick="window.configModule.testTeamsNotification()">
                            🔔 Testar Notificação
                        </button>
                    </div>
                    <form id="teamsForm">
                        <div class="form-group">
                            <label class="form-label">Habilitar Integração</label>
                            <input type="checkbox" id="teams_enabled" ${teams.enabled ? 'checked' : ''}>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Webhook URL</label>
                            <input type="text" id="teams_webhook_url" class="form-input" value="${teams.webhook_url || ''}">
                        </div>

                        <div class="form-group">
                            <label class="form-label">Bot Name</label>
                            <input type="text" id="teams_bot_name" class="form-input" value="${teams.bot_name || 'Resync Bot'}">
                        </div>

                        <div class="form-group">
                            <label class="form-label">Avatar URL</label>
                            <input type="text" id="teams_avatar_url" class="form-input" value="${teams.avatar_url || ''}">
                        </div>

                        <button type="submit" class="btn btn-primary">Salvar Configuração</button>
                    </form>
                </div>
            `;

            document.getElementById('teamsForm').addEventListener('submit', (e) => this.handleTeamsSubmit(e));

            // Expose for onclick handlers
            window.configModule = this;

        } catch (err) {
            content.innerHTML = `<div class="card"><div class="stat-label">Erro ao carregar Teams: ${err.message}</div></div>`;
        }
    }

    async handleTeamsSubmit(e) {
        e.preventDefault();
        const data = {
            enabled: document.getElementById('teams_enabled').checked,
            webhook_url: document.getElementById('teams_webhook_url').value,
            bot_name: document.getElementById('teams_bot_name').value,
            avatar_url: document.getElementById('teams_avatar_url').value
        };

        try {
            await this.api.put('/admin/config/teams', data);
            alert('Configuração salva com sucesso!');
            this.loadTeamsView(); // Reload to refresh health status
        } catch (err) {
            alert(`Erro ao salvar: ${err.message}`);
        }
    }

    async testTeamsNotification() {
        try {
            const res = await this.api.post('/admin/config/teams/test-notification');
            if (res.status === 'success') {
                alert('Notificação de teste enviada!');
            } else {
                alert('Falha ao enviar notificação de teste.');
            }
        } catch (err) {
            alert(`Erro no teste: ${err.message}`);
        }
    }

    // ========================================================================
    // GENERIC SCHEMA-DRIVEN SETTINGS (TWS, LLM, Enterprise, etc.)
    // ========================================================================
    async loadSettingsView(sectionKey) {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="card"><div class="stat-label">Carregando Configurações...</div></div>';

        try {
            // Updated path to match app_factory.py registration (/api/v1/admin + /settings)
            const response = await this.api.get(`/api/v1/admin/settings/section/${sectionKey}`);
            // Fallback if the direct API fails or returns different structure (handling potential proxy/routing issues)
            const section = response.title ? response : response.section;

            content.innerHTML = `
                <div class="header-title">
                    <h1>${response.title}</h1>
                    <p>${response.description}</p>
                </div>
                
                <div class="card">
                    <form id="settingsForm">
                        ${this.renderFields(response.fields, response.values)}
                        <button type="submit" class="btn btn-primary" style="margin-top: 1.5rem;">
                            Salvar Alterações
                        </button>
                    </form>
                </div>
            `;

            document.getElementById('settingsForm').addEventListener('submit', (e) => this.handleSettingsSubmit(e, response.fields));

        } catch (err) {
            console.error(err);
            content.innerHTML = `<div class="card"><div class="stat-label">Erro ao carregar seção ${sectionKey}: ${err.message}</div></div>`;
        }
    }

    renderFields(fields, values) {
        return Object.entries(fields).map(([key, config]) => {
            const value = values[key];
            const helpText = config.hot_reload ?
                '<span class="badge badge-success" style="font-size: 0.6rem;">Hot Reload</span>' :
                '<span class="badge badge-warning" style="font-size: 0.6rem;">Requires Restart</span>';

            let inputHtml = '';

            if (config.type === 'boolean') {
                inputHtml = `
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="${key}" ${value ? 'checked' : ''} ${config.readonly ? 'disabled' : ''}>
                        <span>${config.label}</span>
                    </div>`;
            } else if (config.type === 'select') {
                inputHtml = `
                    <label class="form-label">${config.label}</label>
                    <select id="${key}" class="form-input" ${config.readonly ? 'disabled' : ''}>
                        ${config.options.map(opt => `<option value="${opt}" ${opt === value ? 'selected' : ''}>${opt}</option>`).join('')}
                    </select>`;
            } else {
                inputHtml = `
                    <label class="form-label">${config.label}</label>
                    <input type="${config.type === 'number' ? 'number' : 'text'}" 
                           id="${key}" 
                           class="form-input" 
                           value="${value !== null ? value : ''}"
                           ${config.readonly ? 'readonly' : ''}
                           ${config.min !== undefined ? `min="${config.min}"` : ''}
                           ${config.max !== undefined ? `max="${config.max}"` : ''}
                           ${config.step !== undefined ? `step="${config.step}"` : ''}>`;
            }

            return `
                <div class="form-group">
                    ${inputHtml}
                    <div style="margin-top: 5px; display: flex; justify-content: space-between;">
                        <small style="color: var(--text-secondary);">${config.description || ''}</small>
                        ${helpText}
                    </div>
                </div>
             `;
        }).join('');
    }

    async handleSettingsSubmit(e, fields) {
        e.preventDefault();

        const updates = {};
        for (const key of Object.keys(fields)) {
            const field = fields[key];
            const element = document.getElementById(key);

            let value;
            if (field.type === 'boolean') {
                value = element.checked;
            } else if (field.type === 'number') {
                value = Number(element.value);
            } else {
                value = element.value;
            }

            // Only send if not readonly
            if (!field.readonly) {
                updates[key] = value;
            }
        }

        try {
            // Updated path to match app_factory.py registration
            const res = await this.api.put('/api/v1/admin/settings/bulk-update', { settings: updates });

            let msg = 'Configurações salvas!';
            if (res.requires_restart_count > 0) {
                msg += '\nAlgumas alterações requerem reinicialização do servidor.';
            }
            alert(msg);

        } catch (err) {
            alert(`Erro ao salvar: ${err.message}`);
        }
    }
}
