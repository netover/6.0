/**
 * Admin Threshold Tuning Module
 * Handles Auto-Tuning of system thresholds.
 */

class AdminTuning {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        // Prefix from threshold_tuning.py is /threshold-tuning, mounted at /api/v1/admin
        this.basePath = '/api/v1/admin/threshold-tuning';
    }

    async loadTuningView() {
        const content = document.getElementById('content');

        content.innerHTML = `
            <div class="header-title" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>Threshold Auto-Tuning</h1>
                    <p>Manage dynamic system thresholds and active learning</p>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-neu" onclick="window.tuningModule.loadAuditTab()">Audit Log</button>
                    <button class="btn btn-primary" onclick="window.tuningModule.loadOverviewTab()">Overview</button>
                </div>
            </div>

            <div class="card" style="margin-bottom: 2rem;">
                <div style="display: flex; border-bottom: 1px solid var(--border); margin-bottom: 1rem;">
                    <button class="tab-btn active" id="tab-overview" onclick="window.tuningModule.loadOverviewTab()">Overview</button>
                    <button class="tab-btn" id="tab-thresholds" onclick="window.tuningModule.loadThresholdsTab()">Thresholds</button>
                </div>
                <div id="tuning-content">
                    <div class="stat-label">Loading tuning data...</div>
                </div>
            </div>

            <!-- Edit Threshold Modal -->
             <div id="editThresholdModal" class="modal" style="display: none;">
                <div class="modal-content card">
                    <h2>Edit Threshold</h2>
                    <form id="editThresholdForm">
                        <input type="hidden" id="editThresholdName">
                        <div class="form-group">
                            <label id="editThresholdLabel"></label>
                            <input type="number" step="0.01" id="editThresholdValue" class="form-input" required>
                        </div>
                        <div class="form-group">
                            <label>Reason</label>
                            <input type="text" id="editThresholdReason" class="form-input" placeholder="Why are you changing this?" required>
                        </div>
                        <div style="margin-top: 1.5rem; display: flex; gap: 10px; justify-content: flex-end;">
                            <button type="button" class="btn btn-neu" onclick="document.getElementById('editThresholdModal').style.display='none'">Cancel</button>
                            <button type="submit" class="btn btn-primary">Save</button>
                        </div>
                    </form>
                </div>
            </div>
        `;

        window.tuningModule = this;
        this.loadOverviewTab();
    }

    // ========================================================================
    // OVERVIEW TAB
    // ========================================================================

    async loadOverviewTab() {
        this.setActiveTab('tab-overview');
        const container = document.getElementById('tuning-content');
        container.innerHTML = '<div class="stat-label">Loading status...</div>';

        try {
            const res = await this.api.get(`${this.basePath}/status`);
            const data = res.data;

            let recommendationsHtml = '<p style="color: var(--text-secondary)">No pending recommendations.</p>';
            if (data.pending_recommendations && data.pending_recommendations.length > 0) {
                recommendationsHtml = `
                    <table class="data-table" style="width: 100%; border-collapse: collapse;">
                         <thead>
                            <tr style="background: rgba(0,0,0,0.02);">
                                <th style="padding: 0.5rem; text-align: left;">Threshold</th>
                                <th style="padding: 0.5rem; text-align: left;">Current</th>
                                <th style="padding: 0.5rem; text-align: left;">Recommended</th>
                                <th style="padding: 0.5rem; text-align: right;">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.pending_recommendations.map(r => this.renderRecommendationRow(r)).join('')}
                        </tbody>
                    </table>
                `;
            }

            // Escape mode for safe display
            const mode = escapeHtml(data.mode);
            const circuitBreakerActive = data.circuit_breaker_active;

            container.innerHTML = `
                <div class="dashboard-grid">
                    <div class="card stat-card">
                        <span class="stat-label">Current Mode</span>
                        <div class="stat-value" style="text-transform: capitalize;">${mode}</div>
                        <div style="margin-top: 10px;">
                            <select onchange="window.tuningModule.setMode(this.value)" class="form-input" style="padding: 2px;">
                                <option value="off" ${data.mode === 'off' ? 'selected' : ''}>Off</option>
                                <option value="low" ${data.mode === 'low' ? 'selected' : ''}>Low (Safe)</option>
                                <option value="mid" ${data.mode === 'mid' ? 'selected' : ''}>Mid (Balanced)</option>
                                <option value="high" ${data.mode === 'high' ? 'selected' : ''}>High (Aggressive)</option>
                            </select>
                        </div>
                    </div>
                     <div class="card stat-card">
                        <span class="stat-label">Circuit Breaker</span>
                        <div class="stat-value">
                            ${circuitBreakerActive ? '<span style="color:var(--error)">TRIPPED</span>' : '<span style="color:var(--success)">OK</span>'}
                        </div>
                        ${circuitBreakerActive ? `<button class="btn btn-sm btn-warning" style="margin-top:5px;" onclick="window.tuningModule.resetBreaker()">Reset</button>` : ''}
                    </div>
                </div>

                <div class="card" style="margin-top: 1rem;">
                    <div class="card-title">Pending Recommendations</div>
                    ${recommendationsHtml}
                    <div style="margin-top: 1rem; text-align: right;">
                        <button class="btn btn-neu" onclick="window.tuningModule.generateRecs()">Generate Now</button>
                    </div>
                </div>
            `;

        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error: ${escapeHtml(err.message)}</div>`;
        }
    }

    renderRecommendationRow(r) {
        // Escape all user-provided data
        const thresholdName = escapeHtml(r.threshold_name);
        const currentValue = escapeHtml(r.current_value);
        const recommendedValue = escapeHtml(r.recommended_value);
        const id = escapeHtml(r.id);

        return `
            <tr>
                <td style="padding: 0.5rem;">${thresholdName}</td>
                <td style="padding: 0.5rem;">${currentValue}</td>
                <td style="padding: 0.5rem; font-weight: bold; color: var(--primary);">${recommendedValue}</td>
                <td style="padding: 0.5rem; text-align: right;">
                    <button class="btn btn-sm btn-primary" onclick="window.tuningModule.approveRec('${id}')">Approve</button>
                    <button class="btn btn-sm btn-danger" onclick="window.tuningModule.rejectRec('${id}')">Reject</button>
                </td>
            </tr>
        `;
    }

    // ========================================================================
    // THRESHOLDS TAB
    // ========================================================================

    async loadThresholdsTab() {
        this.setActiveTab('tab-thresholds');
        const container = document.getElementById('tuning-content');
        container.innerHTML = '<div class="stat-label">Loading thresholds...</div>';

        try {
            const res = await this.api.get(`${this.basePath}/thresholds`);
            const items = res.thresholds || {};

            container.innerHTML = `
                <table class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                            <th style="padding: 1rem; text-align: left;">Name</th>
                            <th style="padding: 1rem; text-align: left;">Value</th>
                            <th style="padding: 1rem; text-align: left;">Description</th>
                             <th style="padding: 1rem; text-align: right;">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(items).map(([key, val]) => this.renderThresholdRow(key, val)).join('')}
                    </tbody>
                </table>
                 <div style="margin-top: 1rem; text-align: right;">
                    <button class="btn btn-warning" onclick="window.tuningModule.resetDefaults()">Reset All to Defaults</button>
                </div>
            `;

        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error: ${escapeHtml(err.message)}</div>`;
        }
    }

    renderThresholdRow(key, val) {
        // Escape all data
        const name = escapeHtml(key);
        const value = escapeHtml(val.value);
        const description = escapeHtml(val.description || '');

        return `
            <tr>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); font-weight: 500;">${name}</td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); font-family: monospace;">${value}</td>
                <td style="padding: 1rem; border-bottom: 1px solid var(--border); color: var(--text-secondary);">${description}</td>
                <td style="padding: 1rem; text-align: right; border-bottom: 1px solid var(--border);">
                    <button class="btn btn-neu" onclick="window.tuningModule.openEditModal('${name}', ${value})"><i class="fas fa-edit"></i></button>
                </td>
            </tr>
        `;
    }

    // ========================================================================
    // ACTIONS
    // ========================================================================

    async setMode(mode) {
        if (!confirm(`Change auto-tuning mode to ${mode.toUpperCase()}?`)) return;
        try {
            await this.api.put(`${this.basePath}/mode`, { mode });
            this.loadOverviewTab();
        } catch (err) {
            alert(`Error setting mode: ${err.message}`);
        }
    }

    async approveRec(id) {
        try {
            await this.api.post(`${this.basePath}/recommendations/${id}/approve`, {});
            this.loadOverviewTab();
        } catch (err) {
            alert(`Error approving: ${err.message}`);
        }
    }

    async rejectRec(id) {
        try {
            await this.api.post(`${this.basePath}/recommendations/${id}/reject`, {});
            this.loadOverviewTab();
        } catch (err) {
            alert(`Error rejecting: ${err.message}`);
        }
    }

    async generateRecs() {
        try {
            const res = await this.api.post(`${this.basePath}/recommendations/generate`);
            alert(`Generated ${res.generated} recommendations.`);
            this.loadOverviewTab();
        } catch (err) {
            alert(`Error generating: ${err.message}`);
        }
    }

    async resetBreaker() {
        if (!confirm('Reset Circuit Breaker?')) return;
        try {
            await this.api.post(`${this.basePath}/circuit-breaker/reset`, {});
            this.loadOverviewTab();
        } catch (err) {
            alert(`Error resetting: ${err.message}`);
        }
    }

    async resetDefaults() {
        if (!confirm('Reset ALL thresholds to defaults? This cannot be undone.')) return;
        try {
            await this.api.post(`${this.basePath}/reset`, {});
            this.loadThresholdsTab();
        } catch (err) {
            alert(`Error resetting: ${err.message}`);
        }
    }

    // ========================================================================
    // MODAL
    // ========================================================================

    openEditModal(name, currentValue) {
        document.getElementById('editThresholdName').value = name;
        document.getElementById('editThresholdValue').value = currentValue;
        document.getElementById('editThresholdLabel').textContent = `Set value for ${name}`;
        document.getElementById('editThresholdModal').style.display = 'flex';

        document.getElementById('editThresholdForm').onsubmit = (e) => this.handleEditSubmit(e);
    }

    async handleEditSubmit(e) {
        e.preventDefault();
        const name = document.getElementById('editThresholdName').value;
        const value = Number(document.getElementById('editThresholdValue').value);
        const reason = document.getElementById('editThresholdReason').value;

        try {
            await this.api.put(`${this.basePath}/thresholds/${name}`, { value, reason });
            document.getElementById('editThresholdModal').style.display = 'none';
            this.loadThresholdsTab();
        } catch (err) {
            alert(`Error updating: ${err.message}`);
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
