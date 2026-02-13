// Teams Notifications Admin
document.addEventListener('DOMContentLoaded', () => {
    loadAll();
});

async function loadAll() {
    await Promise.all([
        loadStats(),
        loadChannels(),
        loadMappings(),
        loadRules(),
        loadConfig()
    ]);
}

async function loadStats() {
    const res = await fetch('/api/admin/teams-notifications/stats');
    const stats = await res.json();
    document.getElementById('stats').innerHTML = `
        <div class="stats-grid">
            <div class="stat"><h3>${stats.total_channels}</h3><p>Canais</p></div>
            <div class="stat"><h3>${stats.active_channels}</h3><p>Ativos</p></div>
            <div class="stat"><h3>${stats.notifications_sent_today}</h3><p>Enviadas Hoje</p></div>
            <div class="stat"><h3>${stats.notifications_failed_today}</h3><p>Falhas Hoje</p></div>
        </div>
    `;
}

async function loadChannels() {
    const res = await fetch('/api/admin/teams-notifications/channels');
    const channels = await res.json();
    document.getElementById('channels-list').innerHTML = channels.map(ch => `
        <div class="channel-card">
            <span class="icon">${ch.icon}</span>
            <strong>${ch.name}</strong>
            <span class="badge ${ch.is_active?'active':'inactive'}">${ch.is_active?'Ativo':'Inativo'}</span>
            <button onclick="testChannel(${ch.id})">🧪 Testar</button>
            <button onclick="editChannel(${ch.id})">✏️</button>
            <button onclick="deleteChannel(${ch.id})">🗑️</button>
        </div>
    `).join('');
}

async function testChannel(id) {
    const res = await fetch(`/api/admin/teams-notifications/channels/${id}/test`, {method: 'POST'});
    if (res.ok) alert('Notificação de teste enviada!');
    else alert('Erro ao enviar!');
}

function addChannel() {
    const name = prompt('Nome do canal:');
    const webhook = prompt('Webhook URL:');
    if (name && webhook) {
        fetch('/api/admin/teams-notifications/channels', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, webhook_url: webhook})
        }).then(() => loadChannels());
    }
}

async function loadMappings() {
    const res = await fetch('/api/admin/teams-notifications/mappings');
    const mappings = await res.json();
    document.getElementById('mappings-list').innerHTML = mappings.map(m => `
        <div class="mapping-item">
            <code>${m.job_name}</code> → <strong>${m.channel_name}</strong>
            <button onclick="deleteMapping(${m.id})">🗑️</button>
        </div>
    `).join('');
}

async function loadRules() {
    const res = await fetch('/api/admin/teams-notifications/rules');
    const rules = await res.json();
    document.getElementById('rules-list').innerHTML = rules.map(r => `
        <div class="rule-item">
            <code>${r.pattern}</code> (${r.pattern_type}) → <strong>${r.channel_name}</strong>
            <span class="badge">Prioridade: ${r.priority}</span>
            <button onclick="deleteRule(${r.id})">🗑️</button>
        </div>
    `).join('');
}

async function loadConfig() {
    const res = await fetch('/api/admin/teams-notifications/config');
    const cfg = await res.json();
    document.getElementById('config-form').innerHTML = `
        <label><input type="checkbox" ${cfg.rate_limit_enabled?'checked':''}> Rate Limiting</label><br>
        <label><input type="checkbox" ${cfg.quiet_hours_enabled?'checked':''}> Horários Silenciosos</label><br>
        <label>Status para notificar: <input value="${cfg.notify_on_status.join(', ')}"></label>
    `;
}
