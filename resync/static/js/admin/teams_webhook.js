// Teams Webhook Admin Interface
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadUsers();
    loadAuditLogs();
});

async function loadStats() {
    const response = await fetch('/api/admin/teams-webhook/stats');
    const stats = await response.json();
    document.getElementById('stats').innerHTML = `
        <div class="stats-grid">
            <div class="stat">
                <h3>${stats.total_users}</h3>
                <p>Total Users</p>
            </div>
            <div class="stat">
                <h3>${stats.active_users}</h3>
                <p>Active Users</p>
            </div>
            <div class="stat">
                <h3>${stats.total_interactions}</h3>
                <p>Total Interactions</p>
            </div>
        </div>
    `;
}

async function loadUsers() {
    const response = await fetch('/api/admin/teams-webhook/users');
    const users = await response.json();
    // Render users table...
}

async function loadAuditLogs() {
    const response = await fetch('/api/admin/teams-webhook/audit-logs');
    const logs = await response.json();
    // Render audit table...
}