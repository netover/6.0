/**
 * Admin Backup Module
 * Handles Backups and Schedules directly via API.
 */

class AdminBackup {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        // The backup router has a prefix '/admin/backup' and is mounted under '/api/v1/admin'
        // resulting in '/api/v1/admin/admin/backup'. 
        // If this 404s, try '/api/v1/admin/backup' (if backend changed).
        this.basePath = '/api/v1/admin/admin/backup';
    }

    async loadBackupView() {
        const content = document.getElementById('content');

        content.innerHTML = `
            <div class="header-title" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>Backup & Restore</h1>
                    <p>Manage system backups and schedules</p>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-neu" onclick="window.backupModule.loadSchedulesTab()">Schedules</button>
                    <button class="btn btn-primary" onclick="window.backupModule.openCreateBackupModal()">
                        + Create Backup
                    </button>
                </div>
            </div>

            <div class="card" style="margin-bottom: 2rem;">
                <div style="display: flex; border-bottom: 1px solid var(--border); margin-bottom: 1rem;">
                    <button class="tab-btn active" id="tab-backups" onclick="window.backupModule.loadBackupsTab()">Backups</button>
                    <button class="tab-btn" id="tab-schedules" onclick="window.backupModule.loadSchedulesTab()">Schedules</button>
                </div>
                <div id="backup-content">
                    <div class="stat-label">Loading...</div>
                </div>
            </div>

            <!-- Create Backup Modal -->
            <div id="createBackupModal" class="modal" style="display: none;">
                <div class="modal-content card">
                    <h2>Create Backup</h2>
                    <form id="createBackupForm">
                        <div class="form-group">
                            <label>Type</label>
                            <select name="type" class="form-input">
                                <option value="database">Database (SQL)</option>
                                <option value="config">Config (Env/Settings)</option>
                                <option value="full">Full (DB + Config)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Description</label>
                            <textarea name="description" class="form-input" rows="3"></textarea>
                        </div>
                        <div style="margin-top: 1.5rem; display: flex; gap: 10px; justify-content: flex-end;">
                            <button type="button" class="btn btn-neu" onclick="document.getElementById('createBackupModal').style.display='none'">Cancel</button>
                            <button type="submit" class="btn btn-primary">Create</button>
                        </div>
                    </form>
                </div>
            </div>
        `;

        window.backupModule = this;
        this.loadBackupsTab();
    }

    // ========================================================================
    // BACKUPS TAB
    // ========================================================================

    async loadBackupsTab() {
        this.setActiveTab('tab-backups');
        const container = document.getElementById('backup-content');
        container.innerHTML = '<div class="stat-label">Loading backups...</div>';

        try {
            const data = await this.api.get(`${this.basePath}/list`);
            const backups = data.backups || [];

            if (backups.length === 0) {
                container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No backups found.</div>';
                return;
            }

            container.innerHTML = `
                <table class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                            <th style="padding: 1rem; text-align: left;">Filename</th>
                            <th style="padding: 1rem; text-align: left;">Type</th>
                            <th style="padding: 1rem; text-align: left;">Size</th>
                            <th style="padding: 1rem; text-align: left;">Created</th>
                            <th style="padding: 1rem; text-align: right;">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${backups.map(b => this.renderBackupRow(b)).join('')}
                    </tbody>
                </table>
            `;
        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error: ${err.message}</div>`;
        }
    }

    renderBackupRow(backup) {
        return `
            <tr>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border);">
                    <div style="font-weight: 500;">${backup.filename}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">${backup.status}</div>
                </td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border);"><span class="badge">${backup.type}</span></td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${backup.size_human}</td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${new Date(backup.created_at).toLocaleString()}</td>
                <td style="padding: 1rem; text-align: right; border-bottom: 1px solid var(--border);">
                    <button class="btn btn-neu" onclick="window.backupModule.downloadBackup('${backup.id}')" title="Download"><i class="fas fa-download"></i></button>
                    <button class="btn btn-danger" onclick="window.backupModule.deleteBackup('${backup.id}')" title="Delete"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `;
    }

    // ========================================================================
    // SCHEDULES TAB
    // ========================================================================

    async loadSchedulesTab() {
        this.setActiveTab('tab-schedules');
        const container = document.getElementById('backup-content');
        container.innerHTML = '<div class="stat-label">Loading schedules...</div>';

        try {
            const data = await this.api.get(`${this.basePath}/schedules`);
            const schedules = data.schedules || [];

            let html = `
                <div style="margin-bottom: 1rem; text-align: right;">
                     <button class="btn btn-sm btn-primary" onclick="alert('Not implemented in SPA yet')">+ New Schedule</button>
                </div>
            `;

            if (schedules.length === 0) {
                html += '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No active schedules.</div>';
            } else {
                html += `
                    <table class="data-table" style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                                <th style="padding: 1rem; text-align: left;">Name</th>
                                <th style="padding: 1rem; text-align: left;">Type</th>
                                <th style="padding: 1rem; text-align: left;">Cron</th>
                                <th style="padding: 1rem; text-align: left;">Next Run</th>
                                <th style="padding: 1rem; text-align: right;">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${schedules.map(s => `
                                <tr>
                                    <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${s.name}</td>
                                    <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${s.backup_type}</td>
                                    <td style="padding: 1rem; border-bottom: 1px solid var(--border);"><code>${s.cron_expression}</code></td>
                                    <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${s.next_run ? new Date(s.next_run).toLocaleString() : '-'}</td>
                                    <td style="padding: 1rem; text-align: right; border-bottom: 1px solid var(--border);">
                                        <button class="btn btn-danger" onclick="window.backupModule.deleteSchedule('${s.id}')"><i class="fas fa-trash"></i></button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            container.innerHTML = html;

        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error: ${err.message}</div>`;
        }
    }

    // ========================================================================
    // ACTIONS
    // ========================================================================

    openCreateBackupModal() {
        document.getElementById('createBackupModal').style.display = 'flex';
        document.getElementById('createBackupForm').onsubmit = (e) => this.handleCreateBackup(e);
    }

    async handleCreateBackup(e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const type = formData.get('type');
        const description = formData.get('description');

        try {
            await this.api.post(`${this.basePath}/${type}`, { description });
            document.getElementById('createBackupModal').style.display = 'none';
            alert('Backup started!');
            this.loadBackupsTab();
        } catch (err) {
            alert(`Error creating backup: ${err.message}`);
        }
    }

    async downloadBackup(id) {
        // Direct download link
        window.open(`${this.basePath}/${id}/download`, '_blank');
    }

    async deleteBackup(id) {
        if (!confirm('Delete this backup?')) return;
        try {
            await this.api.delete(`${this.basePath}/${id}`);
            this.loadBackupsTab();
        } catch (err) {
            alert(`Error deleting: ${err.message}`);
        }
    }

    async deleteSchedule(id) {
        if (!confirm('Delete this schedule?')) return;
        try {
            await this.api.delete(`${this.basePath}/schedules/${id}`);
            this.loadSchedulesTab();
        } catch (err) {
            alert(`Error deleting: ${err.message}`);
        }
    }

    // ========================================================================
    // UTILS
    // ========================================================================

    setActiveTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
    }
}
