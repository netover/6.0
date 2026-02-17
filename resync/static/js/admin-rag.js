/**
 * Admin RAG Reranker Module
 * Handles Reranker Gating configuration and monitoring.
 */

class AdminRAG {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        // Prefix from rag_reranker.py
        this.basePath = '/api/v1/admin/rag-reranker';
    }

    async loadRAGView() {
        const content = document.getElementById('content');
        content.innerHTML = '<div class="stat-label">Loading RAG Reranker status...</div>';

        try {
            const status = await this.api.get(`${this.basePath}/status`);
            this.renderDashboard(status, content);
        } catch (err) {
            content.innerHTML = `<div class="card"><div class="stat-label">Error loading status: ${err.message}</div></div>`;
        }
    }

    renderDashboard(data, container) {
        const r = data.reranker;
        const g = data.gating;
        const c = g.config;

        container.innerHTML = `
            <div class="header-title" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>RAG Reranker</h1>
                    <p>Optimize retrieval precision and performance</p>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-neu" onclick="window.ragModule.preloadModel()">Preload Model</button>
                    <button class="btn btn-primary" onclick="window.ragModule.saveConfig()">Save Config</button>
                </div>
            </div>

            <div class="dashboard-grid">
                <!-- Reranker Status -->
                <div class="card stat-card">
                    <span class="stat-label">Reranker Status</span>
                    <div class="stat-value" style="font-size: 1.2rem;">
                        ${r.enabled ? '<span style="color:var(--success)">Enabled</span>' : '<span style="color:var(--text-secondary)">Disabled</span>'}
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 5px;">
                        Model: ${r.model || 'N/A'}<br>
                        Type: ${r.type}
                    </div>
                </div>

                <!-- Gating Stats -->
                <div class="card stat-card">
                    <span class="stat-label">Rerank Rate</span>
                    <div class="stat-value">${(g.rerank_rate * 100).toFixed(1)}%</div>
                    <div style="font-size: 0.8rem; margin-top: 5px;">
                        ${g.rerank_activated} / ${g.total_decisions} queries
                    </div>
                </div>

                <!-- Latency -->
                <div class="card stat-card">
                    <span class="stat-label">Avg Latency</span>
                    <div class="stat-value">${r.avg_latency_ms ? r.avg_latency_ms.toFixed(1) + 'ms' : '-'}</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Gating Configuration</div>
                <form id="ragConfigForm" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    
                    <div class="form-group" style="grid-column: 1 / -1; flex-direction: row; align-items: center; gap: 10px;">
                        <input type="checkbox" name="enabled" id="gatingEnabled" ${c.enabled ? 'checked' : ''}>
                        <label for="gatingEnabled" style="font-weight: 600;">Enable Gating Strategy (Smart Rerank)</label>
                    </div>

                    <div class="form-group">
                        <label>Score Low Threshold (0.0 - 1.0)</label>
                        <input type="number" step="0.01" min="0" max="1" name="score_low_threshold" value="${c.score_low_threshold}" class="form-input">
                        <small style="color: var(--text-secondary)">Rerank if top result score is below this.</small>
                    </div>

                    <div class="form-group">
                        <label>Margin Threshold (0.0 - 1.0)</label>
                        <input type="number" step="0.01" min="0" max="1" name="margin_threshold" value="${c.margin_threshold}" class="form-input">
                        <small style="color: var(--text-secondary)">Rerank if gap between top 1 and 2 is below this.</small>
                    </div>

                    <div class="form-group">
                        <label>Max Candidates</label>
                        <input type="number" min="1" max="50" name="max_candidates" value="${c.max_candidates}" class="form-input">
                    </div>

                    <div class="form-group">
                        <label>Entropy Threshold</label>
                        <input type="number" step="0.1" name="entropy_threshold" value="${c.entropy_threshold}" class="form-input" disabled title="Read-only in UI">
                        <small style="color: var(--text-secondary)">Entropy check enabled: ${c.entropy_check_enabled}</small>
                    </div>

                </form>
            </div>
        `;

        window.ragModule = this;
    }

    async saveConfig() {
        const form = document.getElementById('ragConfigForm');
        const formData = new FormData(form);

        const payload = {
            enabled: document.getElementById('gatingEnabled').checked,
            score_low_threshold: Number(formData.get('score_low_threshold')),
            margin_threshold: Number(formData.get('margin_threshold')),
            max_candidates: Number(formData.get('max_candidates'))
        };

        try {
            await this.api.put(`${this.basePath}/gating/config`, payload);
            alert('Configuration saved!');
            this.loadRAGView(); // Refresh
        } catch (err) {
            alert(`Error saving config: ${err.message}`);
        }
    }

    async preloadModel() {
        try {
            const res = await this.api.post(`${this.basePath}/reranker/preload`);
            alert(res.message);
            this.loadRAGView();
        } catch (err) {
            alert(`Error preloading: ${err.message}`);
        }
    }
}
