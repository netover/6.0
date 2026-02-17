/**
 * Admin Audit Module
 * Handles viewing of Audit Logs.
 */

class AdminAudit {
    constructor(app) {
        this.app = app;
        this.api = app.api;
        // The audit router is typically mounted at /api/audit
        this.basePath = '/api/audit';
    }

    async loadAuditView() {
        const content = document.getElementById('content');

        content.innerHTML = `
            <div class="header-title">
                <h1>Audit Logs</h1>
                <p>View system activity logs</p>
            </div>

            <div class="card">
                <div id="audit-content">
                    <div class="stat-label">Loading logs...</div>
                </div>
                <!-- Pagination Controls could go here -->
            </div>
        `;

        this.loadLogs();
    }

    async loadLogs() {
        const container = document.getElementById('audit-content');
        try {
            const data = await this.api.get(`${this.basePath}/logs?limit=50`);
            // API returns list directly, or empty list
            const logs = Array.isArray(data) ? data : (data.logs || []);

            if (logs.length === 0) {
                container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No audit logs found.</div>';
                return;
            }

            container.innerHTML = `
                <table class="data-table" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: rgba(0,0,0,0.02); border-bottom: 1px solid var(--border);">
                            <th style="padding: 1rem; text-align: left;">Timestamp</th>
                            <th style="padding: 1rem; text-align: left;">User</th>
                            <th style="padding: 1rem; text-align: left;">Action</th>
                            <th style="padding: 1rem; text-align: left;">Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${logs.map(log => `
                            <tr>
                                <td style="padding: 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;">${new Date(log.timestamp).toLocaleString()}</td>
                                <td style="padding: 1rem; border-bottom: 1px solid var(--border);">${log.user_id}</td>
                                <td style="padding: 1rem; border-bottom: 1px solid var(--border);"><span class="badge">${log.action}</span></td>
                                <td style="padding: 1rem; border-bottom: 1px solid var(--border); font-family: monospace; font-size: 0.85rem;">
                                    ${this.formatDetails(log.details)}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } catch (err) {
            container.innerHTML = `<div class="stat-label">Error loading logs: ${err.message}</div>`;
        }
    }

    formatDetails(details) {
        if (!details) return '-';
        const str = JSON.stringify(details);
        return str.length > 100 ? str.substring(0, 100) + '...' : str;
    }
}
