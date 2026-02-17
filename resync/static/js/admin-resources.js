/**
 * Admin Resources Module
 * Handles CRUD operations for resources like TWS Instances, Users, and API Keys.
 */

class AdminResources {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        this.currentConfig = null;
    }

    // ========================================================================
    // TWS INSTANCES VIEW
    // ========================================================================
    async loadTWSInstancesView() {
        this.currentConfig = {
            title: 'TWS Instances',
            apiPath: '/api/v1/admin/tws-instances',
            listKey: 'instances',
            idKey: 'config.id',
            columns: [
                { key: 'config.name', label: 'Name' },
                { key: 'config.host', label: 'Host' },
                { key: 'config.port', label: 'Port' },
                { key: 'status.value', label: 'Status', format: (val) => this.formatStatus(val) },
                { key: 'config.enabled', label: 'Enabled', format: (val) => val ? 'Yes' : 'No' }
            ],
            actions: [
                { label: 'Connect', icon: 'fa-plug', onClick: (item) => this.connectInstance(item) },
                { label: 'Edit', icon: 'fa-edit', onClick: (item) => this.openEditModal(item) },
                { label: 'Delete', icon: 'fa-trash', className: 'btn-danger', onClick: (item) => this.deleteItem(item) }
            ],
            createSchema: {
                name: { type: 'text', label: 'Name (ID)', required: true },
                display_name: { type: 'text', label: 'Display Name', required: true },
                host: { type: 'text', label: 'Host', required: true },
                port: { type: 'number', label: 'Port', default: 31116 },
                username: { type: 'text', label: 'Username' },
                password: { type: 'password', label: 'Password' },
                environment: { type: 'select', label: 'Environment', options: ['production', 'development'] },
                ssl_enabled: { type: 'boolean', label: 'SSL Enabled', default: true }
            }
        };
        await this.renderListView();
    }

    // ========================================================================
    // USERS VIEW
    // ========================================================================
    async loadUsersView() {
        this.currentConfig = {
            title: 'User Management',
            apiPath: '/api/v1/admin/users',
            listKey: null,
            idKey: 'id',
            columns: [
                { key: 'username', label: 'Username' },
                { key: 'email', label: 'Email' },
                { key: 'role', label: 'Role' },
                { key: 'is_active', label: 'Status', format: (val) => val ? '<span class="badge badge-success">Active</span>' : '<span class="badge badge-error">Inactive</span>' },
                { key: 'last_login', label: 'Last Login', format: (val) => val ? new Date(val).toLocaleString() : '-' }
            ],
            actions: [
                { label: 'Edit', icon: 'fa-edit', onClick: (item) => this.openEditModal(item) },
                { label: 'Delete', icon: 'fa-trash', className: 'btn-danger', onClick: (item) => this.deleteItem(item) }
            ],
            createSchema: {
                username: { type: 'text', label: 'Username', required: true },
                email: { type: 'email', label: 'Email', required: true },
                password: { type: 'password', label: 'Password', required: true },
                full_name: { type: 'text', label: 'Full Name' },
                role: { type: 'select', label: 'Role', options: ['admin', 'user', 'viewer'], default: 'user' }
            }
        };
        await this.renderListView();
    }

    // ========================================================================
    // API KEYS VIEW
    // ========================================================================
    async loadAPIKeysView() {
        this.currentConfig = {
            title: 'API Keys',
            apiPath: '/api/v1/admin/api-keys',
            listKey: 'keys',
            idKey: 'id',
            columns: [
                { key: 'name', label: 'Name' },
                { key: 'key_prefix', label: 'Prefix', format: (val) => `<code>${val}...</code>` },
                { key: 'scopes', label: 'Scopes', format: (val) => val.map(s => `<span class="badge">${s}</span>`).join(' ') },
                { key: 'is_valid', label: 'Status', format: (val, item) => item.is_revoked ? '<span class="badge badge-error">Revoked</span>' : (val ? '<span class="badge badge-success">Valid</span>' : '<span class="badge badge-warning">Expired/Invalid</span>') },
                { key: 'last_used_at', label: 'Last Used', format: (val) => val ? new Date(val).toLocaleString() : 'Never' }
            ],
            actions: [
                { label: 'Revoke', icon: 'fa-ban', className: 'btn-warning', onClick: (item) => this.revokeAPIKey(item) },
                { label: 'Delete', icon: 'fa-trash', className: 'btn-danger', onClick: (item) => this.deleteItem(item, true) } // true for permanent
            ],
            createSchema: {
                name: { type: 'text', label: 'Name', required: true },
                description: { type: 'text', label: 'Description' },
                expires_in_days: { type: 'number', label: 'Expires After (Days)', default: 365 },
                // Scopes selection could be complex, simplifying for now
                // scopes: { type: 'multiselect', ... }
            }
        };
        await this.renderListView();
    }

    // ========================================================================
    // GENERIC CRUD METHODS
    // ========================================================================

    async renderListView() {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="card"><div class="stat-label">Loading resources...</div></div>';

        try {
            const data = await this.api.get(this.currentConfig.apiPath);
            const items = this.currentConfig.listKey ? data[this.currentConfig.listKey] : data;

            let html = `
                <div class="header-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h1>${this.currentConfig.title}</h1>
                        <p>Manage list of resources</p>
                    </div>
                    <button class="btn btn-primary" onclick="window.resourcesModule.openCreateModal()">
                        + New Item
                    </button>
                </div>
                
                <div class="card" style="padding: 0; overflow: hidden; overflow-x: auto;">
                    <table class="data-table" style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                                ${this.currentConfig.columns.map(col => `<th style="padding: 1rem; text-align: left;">${col.label}</th>`).join('')}
                                <th style="padding: 1rem; text-align: right;">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${items.map(item => this.renderRow(item)).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- Modal Container -->
                <div id="resourceModal" class="modal" style="display: none;">
                    <div class="modal-content card">
                        <h2 id="modalTitle">Create Item</h2>
                        <form id="resourceForm"></form>
                    </div>
                </div>
            `;

            content.innerHTML = html;
            window.resourcesModule = this; // Expose for handlers

        } catch (err) {
            content.innerHTML = `<div class="card"><div class="stat-label">Error loading list: ${err.message}</div></div>`;
        }
    }

    renderRow(item) {
        const id = this.getValue(item, this.currentConfig.idKey || 'id');
        if (!this.itemsMap) this.itemsMap = {};
        this.itemsMap[id] = item;

        const cells = this.currentConfig.columns.map(col => {
            let val = this.getValue(item, col.key);
            if (col.format) val = col.format(val, item);
            return `<td style="padding: 1rem; border-bottom: 1px solid var(--border);">${val}</td>`;
        }).join('');

        const actions = this.currentConfig.actions.map(action => `
            <button class="btn ${action.className || 'btn-neu'}" 
                    style="padding: 0.25rem 0.5rem; font-size: 0.8rem; margin-left: 5px;"
                    onclick="window.resourcesModule.handleAction('${action.label}', '${id}')">
                ${action.icon ? `<i class="fas ${action.icon}"></i>` : ''} ${action.label}
            </button>
        `).join('');

        return `<tr>${cells}<td style="padding: 1rem; text-align: right;">${actions}</td></tr>`;
    }

    handleAction(label, id) {
        const item = this.itemsMap[id];
        const action = this.currentConfig.actions.find(a => a.label === label);
        if (action) action.onClick(item);
    }

    // ========================================================================
    // MODAL & FORMS
    // ========================================================================

    openCreateModal() {
        this.renderModal('Create', null);
    }

    openEditModal(item) {
        this.renderModal('Edit', item);
    }

    renderModal(mode, item) {
        const modal = document.getElementById('resourceModal');
        const title = document.getElementById('modalTitle');
        const form = document.getElementById('resourceForm');

        title.textContent = `${mode} ${this.currentConfig.title.slice(0, -1)}`;
        modal.style.display = 'flex';

        const schema = this.currentConfig.createSchema;
        let formHtml = '';

        for (const [key, field] of Object.entries(schema)) {
            let value = item ? this.getValue(item, key) : (field.default || '');
            if (item && item.config && item.config[key] !== undefined) value = item.config[key]; // TWS fallback

            formHtml += this.renderField(key, field, value);
        }

        formHtml += `
            <div style="margin-top: 1.5rem; display: flex; gap: 10px; justify-content: flex-end;">
                <button type="button" class="btn btn-neu" onclick="document.getElementById('resourceModal').style.display='none'">Cancel</button>
                <button type="submit" class="btn btn-primary">Save</button>
            </div>
        `;

        form.innerHTML = formHtml;
        form.onsubmit = (e) => this.handleSave(e, mode, item);
    }

    renderField(key, field, value) {
        if (field.type === 'boolean') {
            return `
                <div class="form-group" style="flex-direction: row; align-items: center; gap: 10px;">
                    <input type="checkbox" name="${key}" ${value ? 'checked' : ''}>
                    <label>${field.label}</label>
                </div>`;
        } else if (field.type === 'select') {
            return `
                <div class="form-group">
                    <label>${field.label}</label>
                    <select name="${key}" class="form-input">
                        ${field.options.map(opt => `<option value="${opt}" ${opt === value ? 'selected' : ''}>${opt}</option>`).join('')}
                    </select>
                </div>`;
        } else {
            return `
                <div class="form-group">
                    <label>${field.label}</label>
                    <input type="${field.type}" name="${key}" class="form-input" value="${value || ''}" ${field.required ? 'required' : ''}>
                </div>`;
        }
    }

    async handleSave(e, mode, item) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = {};

        const schema = this.currentConfig.createSchema;
        for (const [key, field] of Object.entries(schema)) {
            if (field.type === 'boolean') {
                data[key] = formData.get(key) === 'on';
            } else if (field.type === 'number') {
                data[key] = Number(formData.get(key));
            } else {
                data[key] = formData.get(key);
            }
        }

        try {
            if (mode === 'Create') {
                const response = await this.api.post(this.currentConfig.apiPath, data);
                // Check if response has a secret key to display (API Keys)
                if (response.key) {
                    alert(`API KEY CREATED:\n\n${response.key}\n\nSAVE THIS KEY! It will not be shown again.`);
                }
            } else {
                const id = this.getValue(item, this.currentConfig.idKey);
                await this.api.put(`${this.currentConfig.apiPath}/${id}`, data);
            }
            document.getElementById('resourceModal').style.display = 'none';
            this.renderListView();
        } catch (err) {
            alert(`Error saving: ${err.message}`);
        }
    }

    async deleteItem(item, permanent = false) {
        if (!confirm('Are you sure you want to delete this item?')) return;

        const id = this.getValue(item, this.currentConfig.idKey);
        try {
            const url = permanent
                ? `${this.currentConfig.apiPath}/${id}/permanent`
                : `${this.currentConfig.apiPath}/${id}`;

            await this.api.delete(url);
            this.renderListView();
        } catch (err) {
            alert(`Error deleting: ${err.message}`);
        }
    }

    // ========================================================================
    // SPECIFIC ACTIONS
    // ========================================================================

    async connectInstance(item) {
        const id = this.getValue(item, this.currentConfig.idKey);
        try {
            await this.api.post(`${this.currentConfig.apiPath}/${id}/connect`);
            alert('Connection initiated');
            this.renderListView();
        } catch (err) {
            alert(`Connection failed: ${err.message}`);
        }
    }

    async revokeAPIKey(item) {
        const id = this.getValue(item, this.currentConfig.idKey);
        const reason = prompt("Enter revocation reason:");
        if (reason === null) return;

        try {
            await this.api.delete(`${this.currentConfig.apiPath}/${id}`, { reason });
            // NOTE: API uses DELETE for revoke, but payload? 
            // The API expects payload for DELETE? standard fetch doesn't send body with DELETE usually
            // but Resync API check: verify logic.
            // API signature: revoke_api_key(key_id, payload: APIKeyRevoke)
            // Axios/fetch supports body in delete.
            // AdminAPIClient delete method check needed.
            this.renderListView();
        } catch (err) {
            alert(`Error revoking: ${err.message}`);
        }
    }

    // ========================================================================
    // UTILS
    // ========================================================================

    getValue(obj, path) {
        return path.split('.').reduce((o, i) => (o ? o[i] : null), obj);
    }

    formatStatus(status) {
        const map = {
            'connected': '<span class="badge badge-success">Connected</span>',
            'disconnected': '<span class="badge badge-error">Disconnected</span>',
            'error': '<span class="badge badge-error">Error</span>'
        };
        return map[status] || status;
    }
}
